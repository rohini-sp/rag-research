"""
Validate gold/gold_triples.csv against the schema described in
gold/SLICE_README.md.

Checks:
    - Required columns are present and non-empty
    - subject_type / object_type are in the controlled vocabulary
    - triple_id is unique and sequential
    - Detects likely entity-name duplicates (case-insensitive, whitespace-insensitive)
    - Detects rare predicates (used only once) — flagged but not errors
"""
from __future__ import annotations
import sys
from collections import Counter

import pandas as pd

from config import GOLD

REQUIRED_COLUMNS = {"triple_id", "subject", "subject_type", "predicate",
                    "object", "object_type", "evidence_span"}
ENTITY_TYPES = {"gene", "drug", "disease", "pathway", "organisation",
                "chemical", "other"}


def normalise(name: str) -> str:
    return " ".join(str(name).lower().split())


def main() -> int:
    path = GOLD / "gold_triples.csv"
    if not path.exists():
        print(f"[ERROR] {path} does not exist.")
        return 1

    df = pd.read_csv(path)
    n = len(df)
    errors = []
    warnings = []

    # 1. Columns
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        errors.append(f"Missing required columns: {sorted(missing)}")
        print("\n".join(errors))
        return 1

    # 2. Required cells non-empty
    for col in REQUIRED_COLUMNS:
        empty = df[col].isna() | (df[col].astype(str).str.strip() == "")
        if empty.any():
            errors.append(f"Column '{col}' has empty values in rows: "
                          f"{df.loc[empty, 'triple_id'].tolist()}")

    # 3. triple_id sequential and unique
    ids = df["triple_id"].tolist()
    if len(set(ids)) != len(ids):
        errors.append(f"Duplicate triple_id values: "
                      f"{[k for k, v in Counter(ids).items() if v > 1]}")
    expected = list(range(1, n + 1))
    if ids != expected:
        warnings.append(f"triple_id is not sequential 1..{n}.")

    # 4. Entity types in vocabulary
    for col in ("subject_type", "object_type"):
        bad = df[~df[col].isin(ENTITY_TYPES)]
        if len(bad):
            errors.append(f"Column '{col}' has unknown values in rows "
                          f"{bad['triple_id'].tolist()}: "
                          f"{bad[col].unique().tolist()}")

    # 5. Likely entity-name duplicates
    all_names = list(df["subject"]) + list(df["object"])
    norm_to_originals: dict[str, set[str]] = {}
    for name in all_names:
        norm_to_originals.setdefault(normalise(name), set()).add(str(name))
    likely_dupes = {k: v for k, v in norm_to_originals.items() if len(v) > 1}
    if likely_dupes:
        warnings.append("Likely entity-name duplicates "
                        "(differing only in case/whitespace):")
        for k, v in likely_dupes.items():
            warnings.append(f"  '{k}' appears as {sorted(v)}")

    # 6. Rare predicates
    pred_counts = Counter(df["predicate"])
    rare = [p for p, c in pred_counts.items() if c == 1]
    if rare:
        warnings.append(f"Predicates used only once "
                        f"({len(rare)} of {len(pred_counts)}): "
                        f"{rare}")

    # ---- Summary ----
    print(f"Validating {path}")
    print(f"  Rows:         {n}")
    print(f"  Predicates:   {len(pred_counts)} unique")
    print(f"  Subjects:     {df['subject'].nunique()} unique")
    print(f"  Objects:      {df['object'].nunique()} unique")
    print(f"  Subject types: {dict(Counter(df['subject_type']))}")
    print(f"  Object types:  {dict(Counter(df['object_type']))}")

    if warnings:
        print("\n[WARNINGS]")
        for w in warnings:
            print(f"  {w}")

    if errors:
        print("\n[ERRORS]")
        for e in errors:
            print(f"  {e}")
        return 1

    print("\nGold file is valid.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
