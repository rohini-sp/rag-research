# LLM Choice in Graph-Based RAG Construction — empirical study

Companion code for **Chapter 10** of the project report.

## What this does

Picks one graph-based RAG framework (LightRAG-style indexing) and asks: how
does the LLM used for entity / relation extraction affect the resulting
knowledge graph and the downstream QA accuracy?

Pipeline:

```
Wikipedia ──► chunker ──► extraction (LLM A) ──► triples_A.csv
                          extraction (LLM B) ──► triples_B.csv
                          ...
                                        │
              ┌─── metrics.py    ◄──────┤
              │                         │
              ├─── topology.py   ◄──────┤
              │                         │
              ├─── jaccard.py    ◄──────┤
              │                         │
              └─── qa_eval.py    ◄──────┘
                            │
                            ▼
                     visualize.py ──► plots/
```

## Layout

```
rag_study/
├── README.md
├── .env                      # OPENAI_API_KEY=..., ANTHROPIC_API_KEY=...
├── corpus/
│   ├── source.txt            # combined corpus
│   ├── articles/             # one .txt per Wikipedia article
│   ├── manifest.json         # article metadata
│   ├── chunks.json           # token-bounded chunks
│   └── chunks_summary.csv
├── gold/
│   ├── SLICE_README.md       # annotation guide
│   ├── slice_text.txt        # the chunk to annotate (chunk 112, EZH2)
│   └── gold_triples.csv      # ★ human-validated reference graph
├── code/
│   ├── config.py             # paths + corpus list + model registry
│   ├── load_corpus.py        # Wikipedia downloader
│   ├── chunker.py            # token-bounded chunking
│   ├── validate_gold.py      # check gold_triples.csv for schema errors
│   ├── llm_clients.py        # provider-agnostic chat() wrapper
│   ├── prompts.py            # extraction + QA prompts (3 variants)
│   ├── run_extraction.py     # ★ run extraction on one LLM
│   ├── metrics.py            # precision / recall / F1 vs gold
│   ├── topology.py           # graph metrics (nodes, edges, degree, ...)
│   ├── jaccard.py            # pairwise entity overlap between LLMs
│   ├── qa_questions.json     # 10 fixed QA questions
│   ├── qa_eval.py            # answer the 10 questions using each graph
│   ├── finalise_qa.py        # roll graded scores into cost_accuracy.csv
│   ├── visualize.py          # generate plots from results CSVs
│   └── run_all.py            # full pipeline for every model in MODELS
├── results/
│   ├── extracted_triples_<slug>.csv
│   ├── extraction_quality.csv
│   ├── topology.csv
│   ├── jaccard_overlap.csv
│   ├── cost_accuracy.csv
│   ├── qa_<slug>.csv
│   └── prompt_sensitivity.csv
├── graphs/<slug>.graphml
├── logs/extraction_<slug>.jsonl
└── plots/*.png
```

## Setup

```bash
cd ~/rag_study
source .venv/bin/activate
pip install -r requirements.txt   # (or: see code/config.py imports)

# Create .env with whatever keys you have:
cat > .env <<EOF
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=...
GROQ_API_KEY=...
OPENROUTER_API_KEY=...
EOF
```

## Workflow

### 1.  Build the corpus (already done, but re-runnable)
```bash
python code/load_corpus.py        # downloads 10 Wikipedia articles
python code/chunker.py            # 148 chunks of ~500 tokens each
```

### 2.  Curate the gold graph (manual, do once)
Open `gold/gold_triples.csv`, review the 33 pre-filled triples against
`gold/slice_text.txt`, and extend to 40-60 high-confidence triples.
Then validate:
```bash
python code/validate_gold.py
```

### 3.  Register the LLMs you want to run
Edit `code/config.py` and uncomment / add entries in `MODELS`:
```python
MODELS = [
    ("GPT-5 mini",        "openai",    "gpt-5-mini"),
    ("Claude Sonnet 4.6", "anthropic", "claude-sonnet-4-6"),
    ("Claude Haiku 4.5",  "anthropic", "claude-haiku-4-5-20251001"),
    ("Gemini 2.5 Flash",  "gemini",    "gemini-2.5-flash"),
]
```

### 4.  Cheap dry-run on the gold slice (one chunk, every model)
```bash
python code/run_all.py --gold-only --skip-qa
```
This costs cents and exercises the full extract → metric → topology
pipeline. Inspect `results/extraction_quality.csv` before committing budget.

### 5.  Full corpus + QA
```bash
python code/run_all.py
```

### 6.  Grade the QA answers
For each `results/qa_<slug>.csv`, fill in the `score` column (0 or 1) by
hand. Then:
```bash
python code/finalise_qa.py
```

### 7.  Generate plots
```bash
python code/visualize.py
```
Plots land in `plots/` and can be embedded into the report.

### 8.  Prompt-sensitivity sub-experiment (Phase 5)
Pick one cheap LLM and rerun extraction with each prompt variant:
```bash
python code/run_extraction.py --provider openai --model gpt-5-mini \
    --display-name "GPT-5 mini (default)"     --prompt-variant default --gold-only
python code/run_extraction.py --provider openai --model gpt-5-mini \
    --display-name "GPT-5 mini (schema_hint)" --prompt-variant schema_hint --gold-only
python code/run_extraction.py --provider openai --model gpt-5-mini \
    --display-name "GPT-5 mini (cot)"         --prompt-variant cot --gold-only

# Then run metrics on each
for v in default schema_hint cot; do
  python code/metrics.py "results/extracted_triples_gpt_5_mini____${v}.csv" \
      --llm "GPT-5 mini ($v)"
done
```

## Cost expectations (one full corpus pass)

| Model               | Approx. cost |
|---------------------|--------------|
| GPT-5               | ~$2.50       |
| GPT-5 mini          | ~$0.20       |
| Claude Opus 4.6     | ~$10         |
| Claude Sonnet 4.6   | ~$2          |
| Claude Haiku 4.5    | ~$0.50       |
| Gemini 2.5 Flash    | ~$0.10       |
| Llama 3.1 8B (Groq) | free         |

Use `--gold-only` (one chunk) for a feasibility test before each full run.
# rag-research
