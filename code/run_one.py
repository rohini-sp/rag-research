"""
Run the full pipeline for a single model. Useful when you want to iterate
one model at a time — e.g. when GPT-5 takes 3 minutes per chunk and you
don't want to waste time if something breaks mid-run.

Usage:
    # Gold slice only (cheap sanity check, one chunk)
    python code/run_one.py --model "Llama 3.3 70B" --gold-only --skip-qa

    # Full corpus
    python code/run_one.py --model "GPT-5 mini"

    # Full corpus + QA (uses same model to answer questions)
    python code/run_one.py --model "GPT-5 mini" --with-qa

    # Different QA model (e.g. use a cheap model to grade an expensive one)
    python code/run_one.py --model "GPT-5" --with-qa \\
        --qa-provider groq --qa-model llama-3.3-70b-versatile
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


def find_model(name: str) -> tuple[str, str, str]:
    """Look up a model by display name (case-insensitive partial match)."""
    needle = name.lower().strip()
    matches = [m for m in MODELS if needle in m[0].lower()]
    if not matches:
        raise ValueError(f"No model matches '{name}'. "
                         f"Available: {[m[0] for m in MODELS]}")
    if len(matches) > 1:
        raise ValueError(f"'{name}' matches multiple: "
                         f"{[m[0] for m in matches]}")
    return matches[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="display name (case-insensitive partial match)")
    ap.add_argument("--gold-only", action="store_true",
                    help="extract only chunk 112 (cheap sanity check)")
    ap.add_argument("--with-qa", action="store_true",
                    help="also run QA evaluation after extraction")
    ap.add_argument("--skip-qa", action="store_true",
                    help="skip QA even on full runs")
    ap.add_argument("--qa-provider", default=None,
                    help="override QA provider (default: same as extraction)")
    ap.add_argument("--qa-model", default=None,
                    help="override QA model id")
    ap.add_argument("--prompt-variant", default="default",
                    choices=["default", "schema_hint", "cot"])
    args = ap.parse_args()

    display, provider, model_id = find_model(args.model)
    print(f"\n{'=' * 70}")
    print(f"  {display}  ({provider}/{model_id})")
    print(f"  prompt-variant={args.prompt_variant}, "
          f"gold_only={args.gold_only}")
    print(f"{'=' * 70}")

    # 0. Manifest
    run([PY, str(CODE / "run_manifest.py"), "--append"])

    # 1. Extraction
    cmd = [PY, str(CODE / "run_extraction.py"),
           "--provider", provider, "--model", model_id,
           "--display-name", display,
           "--prompt-variant", args.prompt_variant]
    if args.gold_only:
        cmd.append("--gold-only")
    if run(cmd) != 0:
        print(f"\n[ABORT] extraction failed for {display}")
        return 1

    # 2. Locate the produced triples CSV using the SAME slugify() that
    #    run_extraction.py uses (not a hand-rolled approximation — the old
    #    glob grabbed other models' CSVs when display names shared a prefix).
    import re as _re
    slug = _re.sub(r"[^a-z0-9]+", "_", display.lower()).strip("_")
    if args.prompt_variant != "default":
        slug += f"__{args.prompt_variant}"
    triples_csv = RESULTS / f"extracted_triples_{slug}.csv"
    if not triples_csv.exists():
        print(f"\n[ABORT] no triples CSV produced at {triples_csv}")
        return 1
    print(f"\n  Triples CSV: {triples_csv.name}")

    # 3. Metrics + topology
    run([PY, str(CODE / "metrics.py"),  str(triples_csv), "--llm", display])
    run([PY, str(CODE / "topology.py"), str(triples_csv), "--llm", display])

    # 4. QA (only if requested and not --gold-only)
    run_qa = args.with_qa and not args.gold_only and not args.skip_qa
    if run_qa:
        qa_provider = args.qa_provider or provider
        qa_model    = args.qa_model    or model_id
        run([PY, str(CODE / "qa_eval.py"),
             "--llm", display,
             "--provider", qa_provider, "--model", qa_model])
        run([PY, str(CODE / "auto_grade_qa.py"), "--llm", display])

    # 5. Regenerate cross-model summaries (cheap, always)
    run([PY, str(CODE / "jaccard.py")])
    run([PY, str(CODE / "visualize.py")])

    print(f"\n{'=' * 70}")
    print(f"  Done: {display}")
    print(f"{'=' * 70}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
