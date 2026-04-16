"""
Compare an LLM-extracted triples CSV against gold/gold_triples.csv.

Usage:
    python code/metrics.py path/to/extracted_triples.csv [--llm <name>]

The extracted CSV must have at minimum: subject, predicate, object.
(subject_type / object_type are optional but used if present.)

Matching strategy:
    - Names are normalised: lowercase, whitespace-collapsed, surrounding
      punctuation stripped.
    - A predicted triple matches a gold triple if (subject, predicate, object)
      all match after normalisation.
    - We also report a "lenient" match (entity-pair only, ignoring predicate)
      so we can separate two failure modes: missing the relation entirely vs.
      labelling it wrong.

Outputs:
    - Prints summary to stdout
    - Appends one row to results/extraction_quality.csv
"""
from __future__ import annotations
import argparse
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from config import GOLD, RESULTS


def normalise(s: str) -> str:
    s = str(s).lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"^[\W_]+|[\W_]+$", "", s)
    return s


def to_triple_set(df: pd.DataFrame, include_predicate: bool = True) -> set[tuple]:
    if include_predicate:
        return {(normalise(r.subject), normalise(r.predicate), normalise(r.object))
                for r in df.itertuples()}
    return {(normalise(r.subject), normalise(r.object))
            for r in df.itertuples()}


def evaluate(extracted_csv: Path, llm_name: str | None = None) -> dict:
    gold = pd.read_csv(GOLD / "gold_triples.csv")
    pred = pd.read_csv(extracted_csv)

    for col in ("subject", "predicate", "object"):
        if col not in pred.columns:
            raise ValueError(f"Extracted CSV is missing required column: {col}")

    g_strict = to_triple_set(gold, include_predicate=True)
    p_strict = to_triple_set(pred, include_predicate=True)
    g_lenient = to_triple_set(gold, include_predicate=False)
    p_lenient = to_triple_set(pred, include_predicate=False)

    tp_strict = len(g_strict & p_strict)
    fp_strict = len(p_strict - g_strict)
    fn_strict = len(g_strict - p_strict)
    precision = tp_strict / (tp_strict + fp_strict) if (tp_strict + fp_strict) else 0.0
    recall    = tp_strict / (tp_strict + fn_strict) if (tp_strict + fn_strict) else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    tp_lenient = len(g_lenient & p_lenient)
    lenient_precision = tp_lenient / len(p_lenient) if p_lenient else 0.0
    lenient_recall    = tp_lenient / len(g_lenient) if g_lenient else 0.0
    lenient_f1        = (2 * lenient_precision * lenient_recall
                         / (lenient_precision + lenient_recall)) if (lenient_precision + lenient_recall) else 0.0

    return {
        "llm":              llm_name or extracted_csv.stem,
        "n_gold":           len(g_strict),
        "n_predicted":      len(p_strict),
        "tp_strict":        tp_strict,
        "fp_strict":        fp_strict,
        "fn_strict":        fn_strict,
        "precision":        round(precision, 4),
        "recall":           round(recall, 4),
        "f1":               round(f1, 4),
        "lenient_precision": round(lenient_precision, 4),
        "lenient_recall":    round(lenient_recall, 4),
        "lenient_f1":        round(lenient_f1, 4),
        "evaluated_at":     datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def append_to_results(row: dict) -> Path:
    out = RESULTS / "extraction_quality.csv"
    df_new = pd.DataFrame([row])
    if out.exists():
        df = pd.read_csv(out)
        df = df[df["llm"] != row["llm"]]            # overwrite same-LLM row
        df = pd.concat([df, df_new], ignore_index=True)
    else:
        df = df_new
    df.to_csv(out, index=False)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("extracted_csv", type=Path)
    ap.add_argument("--llm", default=None)
    args = ap.parse_args()

    if not args.extracted_csv.exists():
        print(f"Not found: {args.extracted_csv}")
        return 1

    row = evaluate(args.extracted_csv, args.llm)

    print(f"Extraction quality for: {row['llm']}")
    print(f"  Gold triples:         {row['n_gold']}")
    print(f"  Predicted triples:    {row['n_predicted']}")
    print(f"  TP / FP / FN (strict): {row['tp_strict']} / {row['fp_strict']} / {row['fn_strict']}")
    print(f"  STRICT  precision:    {row['precision']:.3f}")
    print(f"  STRICT  recall:       {row['recall']:.3f}")
    print(f"  STRICT  F1:           {row['f1']:.3f}")
    print(f"  LENIENT precision:    {row['lenient_precision']:.3f}  (entity pair only)")
    print(f"  LENIENT recall:       {row['lenient_recall']:.3f}")
    print(f"  LENIENT F1:           {row['lenient_f1']:.3f}")

    out = append_to_results(row)
    print(f"\nAppended to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
