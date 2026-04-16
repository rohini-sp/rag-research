"""
Run entity / relation extraction over the chunked corpus with a single LLM.

Usage examples:
    python code/run_extraction.py --provider openai --model gpt-5-mini \\
        --display-name "GPT-5 mini"

    python code/run_extraction.py --provider anthropic --model claude-sonnet-4-6 \\
        --display-name "Claude Sonnet 4.6"

    python code/run_extraction.py ... --prompt-variant schema_hint
    python code/run_extraction.py ... --prompt-variant cot
    python code/run_extraction.py ... --chunks 105 110 112  # only these chunks

Outputs:
    results/extracted_triples_<slug>.csv
    logs/extraction_<slug>.jsonl
    Appends one row to results/cost_accuracy.csv

Where <slug> = display-name lower-snake-case (e.g. "gpt_5_mini").

Notes:
    - --gold-only restricts extraction to chunk_id 112 (the gold slice). Use
      this for cheap dry-runs and for the precision/recall metrics. Run the
      full corpus only when you're ready to commit budget.
"""
from __future__ import annotations
import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from config import CORPUS, RESULTS, LOGS
from llm_clients import chat
from prompts import (
    EXTRACTION_SYSTEM_DEFAULT,
    EXTRACTION_SYSTEM_SCHEMA_HINT,
    EXTRACTION_SYSTEM_COT,
    EXTRACTION_USER_TEMPLATE,
)

PROMPT_VARIANTS = {
    "default":     EXTRACTION_SYSTEM_DEFAULT,
    "schema_hint": EXTRACTION_SYSTEM_SCHEMA_HINT,
    "cot":         EXTRACTION_SYSTEM_COT,
}


def slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def parse_json_response(text: str) -> list[dict]:
    """Best-effort extraction of the {triples: [...]} block from LLM output.

    Handles:
        - Markdown code fences
        - Truncated JSON (missing closing brackets — common with small models)
        - Extra prose before/after the JSON
    """
    text = text.strip()
    # Strip markdown fences if present
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```\s*$", "", text)

    # Locate the start of the JSON object
    start = text.find("{")
    if start == -1:
        raise ValueError(f"No JSON object found in response: {text[:200]}")
    raw = text[start:]

    # First try: parse as-is
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        # Second try: truncated output — find the last complete triple
        # by looking for the last "}" that ends a triple object,
        # then close the array and outer object.
        last_obj_end = raw.rfind("}")
        if last_obj_end == -1:
            raise ValueError(f"Unrecoverable JSON: {raw[:300]}")
        # Walk backwards to find a plausible closure point
        # (last complete triple ends with })
        truncated = raw[:last_obj_end + 1]
        # Count open brackets to figure out what to close
        open_braces  = truncated.count("{") - truncated.count("}")
        open_brackets = truncated.count("[") - truncated.count("]")
        truncated += "]" * max(open_brackets, 0)
        truncated += "}" * max(open_braces, 0)
        try:
            obj = json.loads(truncated)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON even after repair: {e}\n"
                             f"Raw (first 400 chars): {raw[:400]}")

    if "triples" not in obj or not isinstance(obj["triples"], list):
        raise ValueError(f"Response missing 'triples' list: {list(obj.keys())}")
    return obj["triples"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", required=True,
                    choices=["openai", "anthropic", "gemini", "groq", "kimi", "openrouter"])
    ap.add_argument("--model",   required=True, help="provider-specific model id")
    ap.add_argument("--display-name", required=True,
                    help="friendly name used in output filenames + result tables")
    ap.add_argument("--prompt-variant", default="default",
                    choices=list(PROMPT_VARIANTS))
    ap.add_argument("--gold-only", action="store_true",
                    help="extract only chunk 112 (the gold slice)")
    ap.add_argument("--chunks", type=int, nargs="+", default=None,
                    help="restrict to specific chunk_ids (overrides --gold-only)")
    ap.add_argument("--max-chunks", type=int, default=None,
                    help="hard cap on number of chunks (for cost control)")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-output-tokens", type=int, default=2000)
    args = ap.parse_args()

    chunks = json.loads((CORPUS / "chunks.json").read_text(encoding="utf-8"))
    if args.chunks:
        chunks = [c for c in chunks if c["chunk_id"] in set(args.chunks)]
    elif args.gold_only:
        chunks = [c for c in chunks if c["chunk_id"] == 112]
    if args.max_chunks:
        chunks = chunks[:args.max_chunks]
    if not chunks:
        print("No chunks selected.")
        return 1

    system = PROMPT_VARIANTS[args.prompt_variant]
    slug   = slugify(args.display_name)
    if args.prompt_variant != "default":
        slug += f"__{args.prompt_variant}"

    print(f"Extracting with {args.display_name} ({args.provider}/{args.model}) "
          f"prompt={args.prompt_variant}")
    print(f"  chunks: {len(chunks)}  ({chunks[0]['chunk_id']}..{chunks[-1]['chunk_id']})\n")

    log_path = LOGS / f"extraction_{slug}.jsonl"
    log_f = log_path.open("w", encoding="utf-8")

    all_triples: list[dict] = []
    total_in = total_out = 0
    total_cost = 0.0
    t_start = time.time()

    for c in chunks:
        user = EXTRACTION_USER_TEMPLATE.format(text=c["text"])
        try:
            resp = chat(
                provider=args.provider,
                model_id=args.model,
                system=system,
                user=user,
                max_tokens=args.max_output_tokens,
                temperature=args.temperature,
            )
        except Exception as e:
            print(f"  [chunk {c['chunk_id']}] FAILED: {e}")
            log_f.write(json.dumps({
                "chunk_id": c["chunk_id"], "error": str(e)
            }) + "\n")
            continue

        try:
            triples = parse_json_response(resp.text)
        except ValueError as e:
            print(f"  [chunk {c['chunk_id']}] PARSE ERROR: {e}")
            log_f.write(json.dumps({
                "chunk_id": c["chunk_id"], "parse_error": str(e),
                "raw_response": resp.text
            }) + "\n")
            continue

        for t in triples:
            t["chunk_id"]       = c["chunk_id"]
            t["source_article"] = c["source_article"]
            all_triples.append(t)

        total_in   += resp.n_input_tokens
        total_out  += resp.n_output_tokens
        total_cost += resp.cost_usd

        log_f.write(json.dumps({
            "chunk_id":      c["chunk_id"],
            "n_in":          resp.n_input_tokens,
            "n_out":         resp.n_output_tokens,
            "cost":          resp.cost_usd,
            "elapsed_sec":   resp.elapsed_sec,
            "n_triples":     len(triples),
            "raw_response":  resp.text,
        }) + "\n")

        print(f"  chunk {c['chunk_id']:>3}: {len(triples):>3} triples, "
              f"{resp.n_input_tokens:>5}+{resp.n_output_tokens:>4} tok, "
              f"{resp.elapsed_sec}s, ${resp.cost_usd:.4f}")

    log_f.close()
    elapsed = round(time.time() - t_start, 1)

    # Persist triples
    out_csv = RESULTS / f"extracted_triples_{slug}.csv"
    if all_triples:
        df = pd.DataFrame(all_triples)
        # Ensure consistent column order
        cols = ["chunk_id", "source_article", "subject", "subject_type",
                "predicate", "object", "object_type", "evidence_span"]
        for c in cols:
            if c not in df.columns:
                df[c] = ""
        df = df[cols + [c for c in df.columns if c not in cols]]
        df.to_csv(out_csv, index=False)
    else:
        out_csv.write_text("chunk_id,source_article,subject,subject_type,"
                           "predicate,object,object_type,evidence_span\n",
                           encoding="utf-8")

    # Append cost / accuracy summary
    cost_row = {
        "llm":              args.display_name,
        "provider":         args.provider,
        "model_id":         args.model,
        "prompt_variant":   args.prompt_variant,
        "n_chunks":         len(chunks),
        "n_triples":        len(all_triples),
        "wall_clock_sec":   elapsed,
        "n_input_tokens":   total_in,
        "n_output_tokens":  total_out,
        "cost_usd":         round(total_cost, 4),
        "qa_correct_out_of_10": "",   # filled in by qa_eval.py
        "evaluated_at":     datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    cost_csv = RESULTS / "cost_accuracy.csv"
    if cost_csv.exists():
        cdf = pd.read_csv(cost_csv)
        cdf = cdf[cdf["llm"] != args.display_name]
        cdf = pd.concat([cdf, pd.DataFrame([cost_row])], ignore_index=True)
    else:
        cdf = pd.DataFrame([cost_row])
    cdf.to_csv(cost_csv, index=False)

    print(f"\n{len(all_triples)} triples extracted.")
    print(f"  Wall clock: {elapsed}s")
    print(f"  Tokens:     {total_in:,} in / {total_out:,} out")
    print(f"  Cost:       ${total_cost:.4f}")
    print(f"  Triples:    {out_csv}")
    print(f"  Per-chunk log: {log_path}")
    print(f"  Cost row appended to {cost_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
