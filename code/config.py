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
# 10 Wikipedia articles spanning four entity types in the cancer drug-target domain.
# Article titles are exact Wikipedia page titles (case- and punctuation-sensitive).
CORPUS_ARTICLES = [
    # Cancers (diseases)
    ("Breast cancer",              "disease"),
    ("Chronic myeloid leukemia",   "disease"),
    ("Melanoma",                   "disease"),
    # Drugs
    ("Trastuzumab",                "drug"),
    ("Imatinib",                   "drug"),
    ("Pembrolizumab",              "drug"),
    # Genes / targets
    ("BRCA1",                      "gene"),
    ("EZH2",                       "gene"),
    # Pathways
    ("Apoptosis",                  "pathway"),
    ("P53",                        "pathway"),
]

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

    # --- Google AI Studio (free) ---
    ("Gemini 2.5 Flash",   "gemini",  "gemini-2.5-flash"),


    # --- Groq (free) ---
    ("Llama 3.3 70B",      "groq",    "llama-3.3-70b-versatile"),
    ("Llama 3.1 8B",       "groq",    "llama-3.1-8b-instant"),
]
