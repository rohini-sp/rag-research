"""
End-to-end QA evaluation against a built knowledge graph.

For each question in code/qa_questions.json:
    1. Identify candidate entity nodes in the graph that are mentioned in the
       question (string-overlap heuristic — deliberately simple so the eval
       isolates GRAPH QUALITY rather than a sophisticated retriever).
    2. Retrieve all triples within `--hops` of any candidate node.
    3. Pass (question, retrieved triples) to the LLM with QA_SYSTEM prompt.
    4. Save the answer.

Usage:
    python code/qa_eval.py --llm "GPT-5 mini" --provider openai --model gpt-5-mini
    python code/qa_eval.py --llm "Claude Sonnet 4.6" --provider anthropic \\
        --model claude-sonnet-4-6 --hops 2

Inputs:
    results/extracted_triples_<slug>.csv   (must already exist — built by
                                            run_extraction.py)
Outputs:
    results/qa_<slug>.csv                  (one row per question + LLM answer)
    Updates qa_correct_out_of_10 in results/cost_accuracy.csv after you grade
    answers manually (see --grade-template).
"""
from __future__ import annotations
import argparse
import json
import re
import sys
from pathlib import Path

import networkx as nx
import pandas as pd

from config import RESULTS
from llm_clients import chat
from prompts import QA_SYSTEM, QA_USER_TEMPLATE
from topology import build_graph
from metrics import normalise


def slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def find_seed_nodes(question: str, graph: nx.MultiDiGraph) -> list[str]:
    """Return graph nodes whose normalised name appears in the question."""
    q = " " + normalise(question) + " "
    seeds = []
    for node in graph.nodes:
        nn = normalise(node)
        if not nn or len(nn) < 3:
            continue
        if f" {nn} " in q:
            seeds.append(node)
    return seeds


def retrieve_subgraph(graph: nx.MultiDiGraph, seeds: list[str], hops: int) -> nx.MultiDiGraph:
    """Subgraph induced by all nodes within `hops` of any seed."""
    if not seeds:
        return nx.MultiDiGraph()
    visited = set(seeds)
    frontier = set(seeds)
    for _ in range(hops):
        new_frontier = set()
        for n in frontier:
            new_frontier.update(graph.successors(n))
            new_frontier.update(graph.predecessors(n))
        new_frontier -= visited
        visited |= new_frontier
        frontier = new_frontier
        if not frontier:
            break
    return graph.subgraph(visited).copy()


def serialise_subgraph(sub: nx.MultiDiGraph, max_triples: int = 60) -> str:
    """Linearise the subgraph as a numbered list of triples."""
    triples = []
    for s, o, data in sub.edges(data=True):
        triples.append((s, data.get("predicate", "related_to"), o))
    triples = triples[:max_triples]
    return "\n".join(f"  T{i+1}: ({s} | {p} | {o})"
                     for i, (s, p, o) in enumerate(triples)) or "  (no triples retrieved)"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm",      required=True,
                    help="display name; must match the one used in run_extraction.py")
    ap.add_argument("--provider", required=True,
                    choices=["openai", "anthropic", "gemini", "groq", "kimi", "openrouter"])
    ap.add_argument("--model",    required=True)
    ap.add_argument("--hops",     type=int, default=2)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens",  type=int, default=600)
    ap.add_argument("--max-evidence-triples", type=int, default=60)
    ap.add_argument("--grade-template", action="store_true",
                    help="just emit a grading template and exit")
    args = ap.parse_args()

    questions = json.loads(
        (Path(__file__).parent / "qa_questions.json").read_text(encoding="utf-8")
    )["questions"]

    slug = slugify(args.llm)
    triples_csv = RESULTS / f"extracted_triples_{slug}.csv"
    if not triples_csv.exists():
        # Fall back to any prompt-variant build if the default isn't present
        candidates = sorted(RESULTS.glob(f"extracted_triples_{slug}*.csv"))
        if not candidates:
            print(f"No extracted-triples CSV found for '{args.llm}'.")
            print(f"  Expected: {triples_csv}")
            print(f"  Run code/run_extraction.py first.")
            return 1
        triples_csv = candidates[0]
        print(f"  Using {triples_csv.name}")

    df = pd.read_csv(triples_csv)
    g  = build_graph(df)
    print(f"Graph for {args.llm}: {g.number_of_nodes()} nodes, "
          f"{g.number_of_edges()} edges\n")

    if args.grade_template:
        # Emit a CSV the user can fill in: question, expected answer, blank score
        rows = [{
            "question_id":     q["id"],
            "type":            q["type"],
            "question":        q["question"],
            "expected_answer": q["answer"],
            "score (0/1)":     "",
            "notes":           "",
        } for q in questions]
        out = RESULTS / f"qa_{slug}_grading.csv"
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"Grading template written to {out}")
        return 0

    rows = []
    for q in questions:
        seeds = find_seed_nodes(q["question"], g)
        sub   = retrieve_subgraph(g, seeds, args.hops)
        evidence = serialise_subgraph(sub, max_triples=args.max_evidence_triples)
        user = QA_USER_TEMPLATE.format(question=q["question"], evidence_block=evidence)

        try:
            resp = chat(provider=args.provider, model_id=args.model,
                        system=QA_SYSTEM, user=user,
                        max_tokens=args.max_tokens, temperature=args.temperature)
            answer = resp.text.strip()
            cost   = resp.cost_usd
        except Exception as e:
            answer = f"[ERROR] {e}"
            cost   = 0.0

        print(f"  Q{q['id']} ({q['type']:<11}): seeds={len(seeds)}, "
              f"sub={sub.number_of_nodes()}n/{sub.number_of_edges()}e -> "
              f"{answer[:90]}...")

        rows.append({
            "question_id":     q["id"],
            "type":            q["type"],
            "question":        q["question"],
            "expected_answer": q["answer"],
            "n_seeds":         len(seeds),
            "subgraph_nodes":  sub.number_of_nodes(),
            "subgraph_edges":  sub.number_of_edges(),
            "answer":          answer,
            "cost_usd":        round(cost, 4),
            "score":           "",      # fill in manually (0 or 1)
            "notes":           "",
        })

    out = RESULTS / f"qa_{slug}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nQA results written to {out}")
    print(f"  -> grade the 'score' column manually (0 or 1) then run "
          f"code/finalise_qa.py to roll the totals into cost_accuracy.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
