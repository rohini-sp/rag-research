"""
After you've manually graded the `score` column in any qa_<slug>.csv,
this script tallies the score and writes it back into
results/cost_accuracy.csv.

Usage:
    python code/finalise_qa.py            # all qa_*.csv files in results/
    python code/finalise_qa.py --llm "GPT-5 mini"
"""
from __future__ import annotations
import argparse
import re
import sys
from pathlib import Path

import pandas as pd

from config import RESULTS


def slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", default=None)
    args = ap.parse_args()

    files = (sorted(RESULTS.glob(f"qa_{slugify(args.llm)}.csv"))
             if args.llm
             else sorted(RESULTS.glob("qa_*.csv")))
    files = [f for f in files if not f.stem.endswith("_grading")]

    if not files:
        print("No qa_<slug>.csv files found.")
        return 1

    cost_csv = RESULTS / "cost_accuracy.csv"
    if not cost_csv.exists():
        print(f"{cost_csv} does not exist; run run_extraction.py first.")
        return 1
    cost_df = pd.read_csv(cost_csv)

    for f in files:
        slug = f.stem.replace("qa_", "")
        df = pd.read_csv(f)
        if "score" not in df.columns:
            print(f"  {f.name}: no 'score' column — skipping")
            continue
        scores = pd.to_numeric(df["score"], errors="coerce").fillna(-1)
        graded = scores[scores >= 0]
        if len(graded) == 0:
            print(f"  {f.name}: no rows graded yet — skipping")
            continue
        n_correct = int(graded.sum())
        n_graded  = len(graded)

        # Match the LLM display name back to cost_accuracy.csv
        candidates = cost_df[cost_df["llm"].apply(lambda x: slugify(x) == slug)]
        if candidates.empty:
            print(f"  {f.name}: no matching row in cost_accuracy.csv — skipping")
            continue
        idx = candidates.index[0]
        cost_df.loc[idx, "qa_correct_out_of_10"] = f"{n_correct}/{n_graded}"
        print(f"  {f.name}: {n_correct}/{n_graded} -> "
              f"{cost_df.loc[idx, 'llm']}")

    cost_df.to_csv(cost_csv, index=False)
    print(f"\nUpdated {cost_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
