"""
Run the full pipeline (extraction + metrics + topology + QA) for every
model listed in config.MODELS.

Usage:
    python code/run_all.py                  # full corpus
    python code/run_all.py --gold-only      # cheap dry run on gold slice
    python code/run_all.py --skip-qa        # extraction + metrics only

After each model finishes, the script also runs jaccard.py and visualize.py
so you always have up-to-date plots.
"""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

from config import MODELS, RESULTS

CODE = Path(__file__).parent
PY   = sys.executable


def run(cmd: list[str]) -> int:
    print(f"\n$ {' '.join(cmd)}")
    return subprocess.run(cmd).returncode


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold-only", action="store_true")
    ap.add_argument("--skip-qa",   action="store_true")
    ap.add_argument("--prompt-variant", default="default",
                    choices=["default", "schema_hint", "cot"])
    args = ap.parse_args()

    if not MODELS:
        print("config.MODELS is empty. Edit code/config.py to register the LLMs you want to run.")
        return 1

    # 0. Save reproducibility manifest
    run([PY, str(CODE / "run_manifest.py"), "--append"])

    for display, provider, model_id in MODELS:
        print(f"\n{'=' * 70}\n  {display}  ({provider}/{model_id})\n{'=' * 70}")

        # 1. Extraction
        cmd = [PY, str(CODE / "run_extraction.py"),
               "--provider", provider, "--model", model_id,
               "--display-name", display,
               "--prompt-variant", args.prompt_variant]
        if args.gold_only:
            cmd.append("--gold-only")
        if run(cmd) != 0:
            print(f"[skip] extraction failed for {display}")
            continue

        # 2. Metrics
        slug = display.lower().replace(" ", "_").replace(".", "_").replace("-", "_")
        triples_csv = RESULTS / f"extracted_triples_{slug}.csv"
        # The actual filename uses run_extraction's slugify; let's just
        # find it by glob to be robust to slight differences.
        candidates = sorted(RESULTS.glob(f"extracted_triples_*{slug.split('_')[0]}*.csv"))
        triples_csv = candidates[-1] if candidates else triples_csv
        if triples_csv.exists():
            run([PY, str(CODE / "metrics.py"),  str(triples_csv), "--llm", display])
            run([PY, str(CODE / "topology.py"), str(triples_csv), "--llm", display])
        else:
            print(f"[skip] no triples CSV at {triples_csv}")
            continue

        # 3. QA
        if not args.skip_qa:
            run([PY, str(CODE / "qa_eval.py"),
                 "--llm", display, "--provider", provider, "--model", model_id])

    # 4. Auto-grade QA (if QA was run)
    if not args.skip_qa:
        run([PY, str(CODE / "auto_grade_qa.py")])

    # 5. Cross-model summaries
    run([PY, str(CODE / "jaccard.py")])
    run([PY, str(CODE / "visualize.py")])
    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
