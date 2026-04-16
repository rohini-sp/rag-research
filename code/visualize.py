"""
Generate the figures that go into Chapter 10 of the report.

Produces (when the relevant CSV exists in results/):
    plots/precision_recall_bar.png
    plots/cost_vs_f1_scatter.png
    plots/topology_heatmap.png      (n_nodes / n_edges per LLM)
    plots/jaccard_heatmap.png

Usage:
    python code/visualize.py         # generate everything available
"""
from __future__ import annotations
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")              # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import RESULTS, PLOTS

# Consistent colour palette + simple styling — clean, paper-friendly
plt.rcParams.update({
    "figure.dpi":        140,
    "savefig.dpi":       180,
    "savefig.bbox":      "tight",
    "font.family":       "serif",
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.labelsize":    10,
    "axes.grid":         True,
    "grid.alpha":        0.25,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})


def _plot_path(name: str) -> Path:
    return PLOTS / name


# ----------------------------------------------------------------------
def precision_recall_bar() -> Path | None:
    p = RESULTS / "extraction_quality.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p).sort_values("f1", ascending=True)
    fig, ax = plt.subplots(figsize=(7, 0.5 + 0.45 * len(df)))
    y = np.arange(len(df))
    w = 0.4
    ax.barh(y - w / 2, df["precision"], w, label="Precision", color="#3b82f6")
    ax.barh(y + w / 2, df["recall"],    w, label="Recall",    color="#f59e0b")
    ax.set_yticks(y)
    ax.set_yticklabels(df["llm"])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Score")
    ax.set_title("Extraction quality vs.\\ gold reference")
    ax.legend(loc="lower right", frameon=False)
    out = _plot_path("precision_recall_bar.png")
    fig.savefig(out)
    plt.close(fig)
    return out


# ----------------------------------------------------------------------
def cost_vs_f1_scatter() -> Path | None:
    eq = RESULTS / "extraction_quality.csv"
    ca = RESULTS / "cost_accuracy.csv"
    if not (eq.exists() and ca.exists()):
        return None
    a = pd.read_csv(eq)[["llm", "f1"]]
    b = pd.read_csv(ca)[["llm", "cost_usd"]]
    df = a.merge(b, on="llm", how="inner")
    if df.empty:
        return None
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(df["cost_usd"], df["f1"], s=80, color="#0ea5e9", edgecolor="black")
    for r in df.itertuples():
        ax.annotate(r.llm, (r.cost_usd, r.f1), xytext=(6, 4),
                    textcoords="offset points", fontsize=9)
    ax.set_xlabel("API cost (USD)")
    ax.set_ylabel("Strict F1 vs.\\ gold")
    ax.set_title("Precision--cost trade-off across LLMs")
    out = _plot_path("cost_vs_f1_scatter.png")
    fig.savefig(out)
    plt.close(fig)
    return out


# ----------------------------------------------------------------------
def topology_bars() -> Path | None:
    p = RESULTS / "topology.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p).sort_values("n_nodes", ascending=True)
    fig, ax = plt.subplots(figsize=(7, 0.5 + 0.45 * len(df)))
    y = np.arange(len(df))
    w = 0.4
    ax.barh(y - w / 2, df["n_nodes"], w, label="Nodes", color="#10b981")
    ax.barh(y + w / 2, df["n_edges"], w, label="Edges", color="#6366f1")
    ax.set_yticks(y)
    ax.set_yticklabels(df["llm"])
    ax.set_xlabel("Count")
    ax.set_title("Graph topology by LLM")
    ax.legend(loc="lower right", frameon=False)
    out = _plot_path("topology_bars.png")
    fig.savefig(out)
    plt.close(fig)
    return out


# ----------------------------------------------------------------------
def jaccard_heatmap() -> Path | None:
    p = RESULTS / "jaccard_overlap.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    llms = sorted(set(df["llm_a"]) | set(df["llm_b"]))
    matrix = pd.DataFrame(index=llms, columns=llms, dtype=float)
    for llm in llms:
        matrix.loc[llm, llm] = 1.0
    for r in df.itertuples():
        matrix.loc[r.llm_a, r.llm_b] = r.jaccard
        matrix.loc[r.llm_b, r.llm_a] = r.jaccard

    fig, ax = plt.subplots(figsize=(0.6 + 0.7 * len(llms), 0.6 + 0.7 * len(llms)))
    im = ax.imshow(matrix.values, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(llms)))
    ax.set_yticks(range(len(llms)))
    ax.set_xticklabels(llms, rotation=45, ha="right")
    ax.set_yticklabels(llms)
    for i in range(len(llms)):
        for j in range(len(llms)):
            v = matrix.values[i, j]
            colour = "white" if v > 0.55 else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color=colour, fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Jaccard")
    ax.set_title("Inter-LLM entity overlap")
    out = _plot_path("jaccard_heatmap.png")
    fig.savefig(out)
    plt.close(fig)
    return out


# ----------------------------------------------------------------------
def main() -> int:
    generated = []
    for fn in (precision_recall_bar, cost_vs_f1_scatter, topology_bars, jaccard_heatmap):
        out = fn()
        if out:
            generated.append(out)
            print(f"  wrote {out}")
        else:
            print(f"  skipped {fn.__name__} (input CSV not present yet)")
    if not generated:
        print("\nNo plots generated — run the experiments first.")
        return 0
    print(f"\nGenerated {len(generated)} plot(s) in {PLOTS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
