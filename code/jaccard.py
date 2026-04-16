"""
Compute pairwise entity-overlap (Jaccard similarity) between every pair of
LLMs that have produced an extracted_triples_<llm>.csv in results/.

Usage:
    python code/jaccard.py
        --inputs results/extracted_triples_*.csv

Each input CSV must have at least: subject, object.

Outputs:
    - Prints the matrix
    - Writes results/jaccard_overlap.csv (long form: llm_a, llm_b, jaccard)
"""
from __future__ import annotations
import argparse
import glob
import sys
from pathlib import Path

import pandas as pd

from config import RESULTS
from metrics import normalise


def entity_set(df: pd.DataFrame) -> set[str]:
    """Union of all subject + object entity names (normalised)."""
    return {normalise(x) for x in pd.concat([df["subject"], df["object"]])}


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+",
                    default=sorted(glob.glob(str(RESULTS / "extracted_triples_*.csv"))))
    args = ap.parse_args()

    if len(args.inputs) < 2:
        print("Need at least 2 input CSVs in results/extracted_triples_*.csv")
        return 1

    name_to_set = {}
    for p in args.inputs:
        path = Path(p)
        # Strip "extracted_triples_" prefix if present
        llm = path.stem.replace("extracted_triples_", "")
        df = pd.read_csv(path)
        name_to_set[llm] = entity_set(df)

    llms = sorted(name_to_set)
    rows = []
    for i, a in enumerate(llms):
        for b in llms[i + 1:]:
            j = jaccard(name_to_set[a], name_to_set[b])
            rows.append({"llm_a": a, "llm_b": b, "jaccard": round(j, 4)})

    df = pd.DataFrame(rows)
    out = RESULTS / "jaccard_overlap.csv"
    df.to_csv(out, index=False)

    print(f"\nPairwise entity Jaccard overlap (n={len(llms)} LLMs):\n")
    matrix = pd.DataFrame(index=llms, columns=llms, dtype=float)
    for llm in llms:
        matrix.loc[llm, llm] = 1.0
    for r in rows:
        matrix.loc[r["llm_a"], r["llm_b"]] = r["jaccard"]
        matrix.loc[r["llm_b"], r["llm_a"]] = r["jaccard"]
    print(matrix.round(3).to_string())
    print(f"\nWritten to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
