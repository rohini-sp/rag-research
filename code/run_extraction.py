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
import shutil
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


def _clean_llm_text(text: str) -> str:
    """Strip common non-JSON wrappers (<think>, markdown fences, prose)."""
    t = (text or "").strip()

    # Remove any number of leading <think>...</think> blocks.
    while True:
        t2 = re.sub(r"^\s*<think>.*?</think>\s*", "", t, flags=re.DOTALL | re.IGNORECASE)
        if t2 == t:
            break
        t = t2.strip()

    # Remove markdown code fences (including ```json)
    t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s*```\s*$", "", t)
    return t.strip()


def _normalize_predicate(value: object) -> str:
    """Normalize relation labels to snake_case with trimmed separators."""
    s = str(value or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def _normalize_triples(triples: list[dict]) -> list[dict]:
    out: list[dict] = []
    for t in triples:
        if not isinstance(t, dict):
            continue
        tt = dict(t)
        tt["predicate"] = _normalize_predicate(tt.get("predicate", ""))
        out.append(tt)
    return out


def parse_json_response(text: str) -> list[dict]:
    """Best-effort extraction of the {triples: [...]} block from LLM output.

    Handles:
        - Markdown code fences
        - Truncated JSON (missing closing brackets — common with small models)
        - Extra prose before/after the JSON
    """
    text = _clean_llm_text(text)

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


def _snapshot_checkpoint(
    *,
    checkpoint_dir: Path,
    slug: str,
    out_csv: Path,
    log_path: Path,
    chunk_id: int,
    key_index: int | None = None,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    csv_snap = checkpoint_dir / f"extracted_triples_{slug}__chunk_{chunk_id}__{ts}.csv"
    log_snap = checkpoint_dir / f"extraction_{slug}__chunk_{chunk_id}__{ts}.jsonl"
    if out_csv.exists():
        shutil.copy2(out_csv, csv_snap)
    if log_path.exists():
        shutil.copy2(log_path, log_snap)

    progress = {
        "slug": slug,
        "last_chunk": chunk_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if key_index is not None:
        progress["key_index"] = key_index

    (checkpoint_dir / f"progress_{slug}.json").write_text(
        json.dumps(progress, indent=2), encoding="utf-8"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", required=True,
                    choices=["openai", "anthropic", "gemini", "groq", "kimi", "openrouter", "claude_cli"])
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
    ap.add_argument("--resume", action="store_true",
                    help="resume from existing extraction log/csv if present")
    ap.add_argument("--checkpoint-every", type=int, default=0,
                    help="snapshot CSV+JSONL every N successful chunks (0 disables)")
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
    out_csv = RESULTS / f"extracted_triples_{slug}.csv"
    checkpoint_dir = Path("runtime/checkpoints")

    completed_chunk_ids: set[int] = set()
    if args.resume and log_path.exists():
        for line in log_path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except Exception:
                continue
            if "n_triples" in row and "chunk_id" in row:
                completed_chunk_ids.add(int(row["chunk_id"]))

    if completed_chunk_ids:
        chunks = [c for c in chunks if c["chunk_id"] not in completed_chunk_ids]
        print(f"  resume: skipping {len(completed_chunk_ids)} already completed chunks")

    # keep prior successful triples when resuming
    all_triples: list[dict] = []
    if args.resume and out_csv.exists():
        try:
            prev_df = pd.read_csv(out_csv)
            all_triples = prev_df.to_dict(orient="records")
            print(f"  resume: loaded {len(all_triples)} prior triples from {out_csv}")
        except Exception as e:
            print(f"  [warn] could not read existing CSV for resume: {e}")

    if not chunks:
        print("All selected chunks already completed.")
        return 0

    log_f = log_path.open("a" if args.resume else "w", encoding="utf-8")

    total_in = total_out = 0
    total_cost = 0.0
    t_start = time.time()
    successful_since_checkpoint = 0

    for c in chunks:
        user = EXTRACTION_USER_TEMPLATE.format(text=c["text"])
        is_gpt_model = args.model.startswith("gpt-")

        # Per-chunk parse retries (in addition to provider/client retries).
        max_parse_attempts = 3
        resp = None
        triples = None
        parse_err: Exception | None = None

        for parse_attempt in range(1, max_parse_attempts + 1):
            try:
                resp = chat(
                    provider=args.provider,
                    model_id=args.model,
                    system=system,
                    user=user,
                    max_tokens=args.max_output_tokens,
                    temperature=args.temperature,
                    retries=1 if is_gpt_model else 3,
                )
            except Exception as e:
                print(f"  [chunk {c['chunk_id']}] FAILED: {e}")
                log_f.write(json.dumps({
                    "chunk_id": c["chunk_id"], "error": str(e)
                }) + "\n")
                resp = None
                break

            try:
                triples = _normalize_triples(parse_json_response(resp.text))
                parse_err = None
                break
            except ValueError as e:
                parse_err = e
                if parse_attempt < max_parse_attempts:
                    print(
                        f"  [chunk {c['chunk_id']}] PARSE ERROR on attempt "
                        f"{parse_attempt}/{max_parse_attempts}: {e}; retrying chunk"
                    )
                    time.sleep(1)
                else:
                    print(f"  [chunk {c['chunk_id']}] PARSE ERROR: {e}")

        if resp is None:
            continue

        if triples is None:
            log_f.write(json.dumps({
                "chunk_id": c["chunk_id"],
                "parse_error": str(parse_err) if parse_err else "unknown parse error",
                "raw_response": resp.text,
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
        log_f.flush()

        # Persist partial CSV every successful chunk so restarts lose almost nothing.
        df = pd.DataFrame(all_triples)
        cols = ["chunk_id", "source_article", "subject", "subject_type",
                "predicate", "object", "object_type", "evidence_span"]
        for ccol in cols:
            if ccol not in df.columns:
                df[ccol] = ""
        df = df[cols + [ccol for ccol in df.columns if ccol not in cols]]
        df.to_csv(out_csv, index=False)

        successful_since_checkpoint += 1
        if args.checkpoint_every > 0 and successful_since_checkpoint >= args.checkpoint_every:
            key_index = None
            if args.provider == "groq":
                try:
                    from llm_clients import _GROQ_KEY_INDEX  # type: ignore
                    key_index = int(_GROQ_KEY_INDEX) + 1
                except Exception:
                    key_index = None
            _snapshot_checkpoint(
                checkpoint_dir=checkpoint_dir,
                slug=slug,
                out_csv=out_csv,
                log_path=log_path,
                chunk_id=c["chunk_id"],
                key_index=key_index,
            )
            successful_since_checkpoint = 0

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
