"""
Generate a reproducibility manifest for the current run environment.

Captures: Python version, OS, package versions, git hash (if available),
config.MODELS, timestamp, and hardware info.

Usage:
    python code/run_manifest.py              # prints + saves to results/manifest.json
    python code/run_manifest.py --append     # appends a run entry with timestamp
"""
from __future__ import annotations
import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from config import MODELS, CHUNK_TOKENS, CHUNK_OVERLAP_TOKENS, CORPUS_ARTICLES, RESULTS


def get_git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "not-a-git-repo"


def get_package_versions() -> dict[str, str]:
    packages = [
        "openai", "anthropic", "tiktoken", "networkx",
        "pandas", "numpy", "matplotlib", "scikit-learn",
        "google-genai", "python-dotenv",
    ]
    versions = {}
    for pkg in packages:
        try:
            mod = __import__(pkg.replace("-", "_"))
            versions[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[pkg] = "not-installed"
    return versions


def build_manifest() -> dict:
    return {
        "timestamp":        datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python_version":   platform.python_version(),
        "platform":         platform.platform(),
        "architecture":     platform.machine(),
        "git_hash":         get_git_hash(),
        "models":           [{"display_name": d, "provider": p, "model_id": m}
                             for d, p, m in MODELS],
        "corpus_articles":  [{"title": t, "type": ty} for t, ty in CORPUS_ARTICLES],
        "chunk_tokens":     CHUNK_TOKENS,
        "chunk_overlap":    CHUNK_OVERLAP_TOKENS,
        "extraction_temperature": 0.0,
        "extraction_prompt_variant": "default",
        "packages":         get_package_versions(),
        "env_keys_present": {
            "OPENAI_API_KEY":    bool(os.environ.get("OPENAI_API_KEY")),
            "ANTHROPIC_API_KEY": bool(os.environ.get("ANTHROPIC_API_KEY")),
            "GEMINI_API_KEY":    bool(os.environ.get("GEMINI_API_KEY")),
            "GROQ_API_KEY":      bool(os.environ.get("GROQ_API_KEY")),
            "MISTRAL_API_KEY":   bool(os.environ.get("MISTRAL_API_KEY")),
        },
        "tooling": {
            "claude_cli_on_path": bool(shutil.which("claude")),
        },
    }


def main() -> int:
    from dotenv import load_dotenv
    load_dotenv()

    ap = argparse.ArgumentParser()
    ap.add_argument("--append", action="store_true",
                    help="append to existing manifest as a new run entry")
    args = ap.parse_args()

    manifest = build_manifest()
    out = RESULTS / "manifest.json"

    if args.append and out.exists():
        existing = json.loads(out.read_text(encoding="utf-8"))
        if isinstance(existing, list):
            existing.append(manifest)
        else:
            existing = [existing, manifest]
        out.write_text(json.dumps(existing, indent=2, ensure_ascii=False),
                       encoding="utf-8")
    else:
        out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                       encoding="utf-8")

    print(json.dumps(manifest, indent=2))
    print(f"\nWritten to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
