"""Central configuration for the LLM-graph-construction study."""
from pathlib import Path

# ---- Paths ----
ROOT      = Path(__file__).resolve().parent.parent
CORPUS    = ROOT / "corpus"
GOLD      = ROOT / "gold"
RESULTS   = ROOT / "results"
GRAPHS    = ROOT / "graphs"
LOGS      = ROOT / "logs"
PLOTS     = ROOT / "plots"

for d in (CORPUS, GOLD, RESULTS, GRAPHS, LOGS, PLOTS):
    d.mkdir(parents=True, exist_ok=True)

# ---- Corpus definition ----
# 5 Wikipedia articles spanning four entity types in the cancer drug-target
# domain. Shrunk from 10 to keep full-corpus runs tractable within the
# free-tier token budgets of Groq and Gemini.
CORPUS_ARTICLES = [
    ("Breast cancer",              "disease"),   # motivating disease
    ("Trastuzumab",                "drug"),      # breast-cancer drug (multi-hop)
    ("Imatinib",                   "drug"),      # CML drug (covers CML in body)
    ("EZH2",                       "gene"),      # gold-graph source (inhibitors section)
    ("P53",                        "pathway"),   # pathway + tumour suppressor
]

# ---- Gold annotation anchor ----
# Chunk ID for the manually-annotated gold slice. Recompute whenever the
# corpus/chunker changes (see scripts/find_gold_chunk.py). In the 5-article
# corpus with 500-token chunks this resolves to chunk 54 — the "Inhibitors"
# section of the EZH2 article.
GOLD_CHUNK_ID = 54

# ---- Chunking ----
# Token budget per chunk for downstream LLM extraction.
# 500 tokens roughly = 350-400 words = 2-3 paragraphs.
CHUNK_TOKENS = 500
CHUNK_OVERLAP_TOKENS = 50

# ---- Model registry ----
# Each entry: (display_name, provider, model_id)
# provider must be one of: openai, anthropic, gemini, groq, openrouter
#
# To add/remove models, comment/uncomment lines below.
# Use the exact model_id each provider lists in their docs.
MODELS = [
    # --- OpenAI (paid, you have a key) ---
    ("GPT-5",              "openai",  "gpt-5"),
    ("GPT-5 mini",         "openai",  "gpt-5-mini"),
    ("o4-mini",            "openai",  "o4-mini"),       # reasoning model

    # --- Google AI Studio (free, but 5 req/min rate limit makes it
    #     impractical for 148-chunk corpus — disabled) ---
    # ("Gemini 2.5 Flash",   "gemini",  "gemini-2.5-flash"),


    # --- Groq (free) ---
    ("Llama 3.3 70B",      "groq",    "llama-3.3-70b-versatile"),
    ("Llama 3.1 8B",       "groq",    "llama-3.1-8b-instant"),
]
