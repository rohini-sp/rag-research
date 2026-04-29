"""
Reconstruct extracted_triples_<slug>.csv from a logs/extraction_<slug>.jsonl
file when run_extraction.py aborted before writing the CSV.

Usage:
    python code/recover_from_jsonl.py logs/extraction_llama_3_3_70b.jsonl \\
        --llm "Llama 3.3 70B"
"""
from __future__ import annotations
import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

from config import RESULTS


def slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", type=Path)
    ap.add_argument("--llm", required=True,
                    help="display name used in output CSV filename")
    args = ap.parse_args()

    if not args.jsonl.exists():
        print(f"Not found: {args.jsonl}")
        return 1

    # Need chunks.json to reattach source article names
    chunks = {c["chunk_id"]: c for c in json.loads(
        (Path(__file__).parent.parent / "corpus" / "chunks.json")
        .read_text(encoding="utf-8")
    )}

    # Parse each log entry and re-run the JSON parser
    from run_extraction import parse_json_response

    all_triples: list[dict] = []
    n_ok = 0
    n_parse_err = 0
    n_api_err = 0

    for line in args.jsonl.read_text(encoding="utf-8").splitlines():
        entry = json.loads(line)
        if "error" in entry:
            n_api_err += 1
            continue
        if "raw_response" not in entry:
            continue
        chunk_id = entry["chunk_id"]
        try:
            triples = parse_json_response(entry["raw_response"])
        except ValueError:
            n_parse_err += 1
            continue
        chunk = chunks.get(chunk_id, {})
        for t in triples:
            t["chunk_id"]       = chunk_id
            t["source_article"] = chunk.get("source_article", "")
            all_triples.append(t)
        n_ok += 1

    if not all_triples:
        print("No triples recovered.")
        return 1

    df = pd.DataFrame(all_triples)
    cols = ["chunk_id", "source_article", "subject", "subject_type",
            "predicate", "object", "object_type", "evidence_span"]
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    df = df[cols + [c for c in df.columns if c not in cols]]

    slug = slugify(args.llm)
    out = RESULTS / f"extracted_triples_{slug}.csv"
    df.to_csv(out, index=False)

    unique_chunks = df["chunk_id"].nunique()
    print(f"Recovered {len(df)} triples from {n_ok} chunks "
          f"({unique_chunks} unique chunk_ids).")
    print(f"  Parse errors: {n_parse_err}")
    print(f"  API errors:   {n_api_err}")
    print(f"Written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
