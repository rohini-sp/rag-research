# LLM Choice in Graph-Based RAG Construction — empirical study

Companion code for **Chapter 10** of the project report.

## What this does

Tests how the choice of LLM used for entity/relation extraction affects the
resulting knowledge graph and downstream QA accuracy. Uses a minimal,
framework-independent pipeline (no LangChain / LlamaIndex / LightRAG) to
isolate LLM choice as the sole independent variable.

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
                  auto_grade_qa.py ──► visualize.py ──► plots/
```

## Model lineup

| Model | Provider | Type | Cost |
|---|---|---|---|
| GPT-5 | OpenAI | Flagship reasoning | ~$34/full run |
| GPT-5 mini | OpenAI | Efficient reasoning | ~$1.50/full run |
| o4-mini | OpenAI | Dedicated reasoning | ~$6/full run |
| Gemini 2.5 Flash | Google AI Studio | Efficient | Free |
| Llama 3.3 70B | Groq | Open-source large | Free |
| Llama 3.1 8B | Groq | Open-source small | Free |

## Layout

```
rag_study/
├── README.md
├── .env                      # API keys (git-ignored)
├── .env.template             # template for .env
├── .gitignore
├── requirements.txt          # pinned dependencies
├── corpus/
│   ├── source.txt            # combined corpus (48K words)
│   ├── articles/             # one .txt per Wikipedia article
│   ├── manifest.json         # article metadata
│   ├── chunks.json           # 148 token-bounded chunks
│   └── chunks_summary.csv
├── gold/
│   ├── SLICE_README.md       # annotation guide
│   ├── slice_text.txt        # chunk 112 (EZH2 inhibitors)
│   └── gold_triples.csv      # 60 human-validated triples
├── code/
│   ├── config.py             # paths + model registry
│   ├── load_corpus.py        # Wikipedia downloader
│   ├── chunker.py            # token-bounded chunking
│   ├── validate_gold.py      # gold CSV schema validation
│   ├── llm_clients.py        # provider-agnostic chat() wrapper
│   ├── prompts.py            # extraction + QA prompts (3 variants)
│   ├── run_extraction.py     # extract triples with one LLM
│   ├── metrics.py            # precision / recall / F1 vs gold
│   ├── topology.py           # graph metrics + GraphML export
│   ├── jaccard.py            # pairwise entity overlap
│   ├── qa_questions.json     # 10 fixed QA questions
│   ├── qa_eval.py            # end-to-end QA evaluation
│   ├── auto_grade_qa.py      # semi-automated QA scoring
│   ├── finalise_qa.py        # roll scores into cost_accuracy.csv
│   ├── visualize.py          # generate publication plots
│   ├── run_manifest.py       # reproducibility manifest
│   └── run_all.py            # orchestrate full pipeline
├── results/                  # all output CSVs
├── graphs/                   # per-model .graphml files
├── logs/                     # per-model extraction logs
└── plots/                    # generated figures
```

## Reproducibility

### Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

### API keys / credentials
```bash
cp .env.template .env
# fill keys you need in .env (openai/gemini/groq/etc)
```

for claude without anthropic api key:
- install claude cli
- run `claude auth login` once
- set model provider to `claude_cli` in `code/config.py`

### Run manifest
Every `run_all.py` invocation automatically saves a manifest to
`results/manifest.json` capturing: Python version, OS, package versions,
git hash, model list, corpus config, and which API keys are present.

### Deterministic controls
- Temperature is set to 0.0 for all models that support it.
- GPT-5 and o-series models don't support custom temperature (fixed at 1.0
  by OpenAI) — this is noted as a limitation in the report.
- The same extraction prompt is used for all models (variants available for
  the prompt-sensitivity sub-experiment).

## Workflow

### 1. Build corpus (already done)
```bash
python code/load_corpus.py
python code/chunker.py
```

### 2. Validate gold graph
```bash
python code/validate_gold.py
```

### 3. Dry-run on gold slice (~$0.30, tests all models)
```bash
python code/run_all.py --gold-only --skip-qa
```
Inspect `results/extraction_quality.csv` before committing full budget.

### 4. Full run
```bash
python code/run_all.py
```

### 5. Grade QA answers
Auto-grading runs automatically. Review `results/qa_*.csv`, fix the
`score` column where `suggested_score` is wrong, then:
```bash
python code/finalise_qa.py
```

### 6. Generate plots
```bash
python code/visualize.py
```

### 7. Prompt-sensitivity sub-experiment
```bash
for v in default schema_hint cot; do
  python code/run_extraction.py --provider groq --model llama-3.3-70b-versatile \
      --display-name "Llama 3.3 70B ($v)" --prompt-variant $v --gold-only
  python code/metrics.py "results/extracted_triples_llama_3_3_70b__${v}.csv" \
      --llm "Llama 3.3 70B ($v)"
done
```

## Baseline artifacts (commit these)

These files represent the fixed experimental baseline and should be
committed to git:
- `corpus/manifest.json`
- `corpus/chunks_summary.csv`
- `gold/gold_triples.csv`
- `gold/slice_text.txt`
- `code/qa_questions.json`
- `code/prompts.py`
- `requirements.txt`
- `.env.template`
