"""
Semi-automated QA grading.

For each qa_<slug>.csv in results/, scores each answer against the
expected_answer using three methods:
    1. Exact substring match (case-insensitive)
    2. Key-entity overlap (what fraction of named entities in the expected
       answer appear in the LLM's answer)
    3. Suggested score (1 if either method is confident, 0 otherwise)

The suggested_score column is written alongside the manual score column.
You still review and override — this just saves time on obvious cases.

Usage:
    python code/auto_grade_qa.py                  # all qa_*.csv
    python code/auto_grade_qa.py --llm "GPT-5"    # one model
"""
from __future__ import annotations
import argparse
import re
import sys
from pathlib import Path

import pandas as pd

from config import RESULTS


def normalise(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).lower().strip())


def extract_key_entities(expected: str) -> list[str]:
    """Pull out the key answer terms (split on ;, /, comma, 'and')."""
    parts = re.split(r"[;/,]|\band\b", expected)
    entities = []
    for p in parts:
        p = p.strip().strip(".")
        # Remove parenthetical clarifications
        p = re.sub(r"\(.*?\)", "", p).strip()
        if len(p) >= 2:
            entities.append(normalise(p))
    return entities


def score_answer(answer: str, expected: str) -> dict:
    ans_norm = normalise(answer)
    exp_norm = normalise(expected)

    # Method 1: substring match
    substring_match = exp_norm in ans_norm or any(
        part.strip() in ans_norm
        for part in exp_norm.split(";")
        if len(part.strip()) >= 3
    )

    # Method 2: key-entity overlap
    key_entities = extract_key_entities(expected)
    if key_entities:
        found = sum(1 for e in key_entities if e in ans_norm)
        entity_overlap = found / len(key_entities)
    else:
        entity_overlap = 0.0

    # Suggested score
    if "answer not supported" in ans_norm or "[error]" in ans_norm:
        suggested = 0
    elif substring_match or entity_overlap >= 0.5:
        suggested = 1
    else:
        suggested = 0

    return {
        "substring_match": substring_match,
        "entity_overlap":  round(entity_overlap, 3),
        "suggested_score": suggested,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", default=None)
    args = ap.parse_args()

    if args.llm:
        slug = re.sub(r"[^a-z0-9]+", "_", args.llm.lower()).strip("_")
        files = sorted(RESULTS.glob(f"qa_{slug}.csv"))
    else:
        files = sorted(RESULTS.glob("qa_*.csv"))
        files = [f for f in files if not f.stem.endswith("_grading")]

    if not files:
        print("No qa_*.csv files found.")
        return 1

    for f in files:
        df = pd.read_csv(f)
        if "expected_answer" not in df.columns or "answer" not in df.columns:
            print(f"  {f.name}: missing expected_answer/answer columns — skip")
            continue

        scores = df.apply(
            lambda row: score_answer(str(row["answer"]), str(row["expected_answer"])),
            axis=1, result_type="expand",
        )
        df["substring_match"]  = scores["substring_match"]
        df["entity_overlap"]   = scores["entity_overlap"]
        df["suggested_score"]  = scores["suggested_score"]

        # Don't overwrite manual scores if already present
        if "score" not in df.columns:
            df["score"] = ""

        df.to_csv(f, index=False)
        n_suggested = scores["suggested_score"].sum()
        print(f"  {f.name}: {n_suggested}/{len(df)} suggested correct")

    print("\nReview the 'suggested_score' column and copy to 'score' if you agree.")
    print("Then run: python code/finalise_qa.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
