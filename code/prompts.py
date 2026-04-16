"""
Centralised prompts for the entity / relation extraction step and the
QA-evaluation step. Keeping them here so the prompt-sensitivity
sub-experiment (Phase 5) can swap variants cleanly.
"""

# ============================================================
#  EXTRACTION PROMPTS
# ============================================================

EXTRACTION_SYSTEM_DEFAULT = """\
You are an expert biomedical knowledge-extraction system. From the user's
text, extract all entities and the relations between them.

Output a JSON object with one key "triples" whose value is a list of objects.
Each triple object has the keys:
    "subject"        — the source entity, canonical name (no abbreviations
                       on first mention)
    "subject_type"   — one of: gene, drug, disease, pathway, organisation,
                       chemical, other
    "predicate"      — verb-phrase relation in lowercase snake_case
                       (e.g. inhibits, treats, developed_by, part_of)
    "object"         — the target entity, canonical name
    "object_type"    — same vocabulary as subject_type
    "evidence_span"  — verbatim quote (≤30 words) from the text supporting
                       this triple

Rules:
    * Stay strictly within the supplied text — do not bring in outside
      knowledge.
    * Skip negations.
    * Reuse predicates across triples wherever possible — a small,
      consistent vocabulary is more valuable than novel phrasing.
    * Output JSON only, with no surrounding prose or markdown fences.
"""

EXTRACTION_SYSTEM_SCHEMA_HINT = EXTRACTION_SYSTEM_DEFAULT + """\

Recommended predicate vocabulary (use these where applicable; introduce new
ones only when no listed predicate fits): inhibits, treats, developed_by,
approved_for, mutated_in, overexpressed_in, targets, binds_to, competes_with,
combined_with, selective_over, is_a, part_of, tradename_of.
"""

EXTRACTION_SYSTEM_COT = EXTRACTION_SYSTEM_DEFAULT + """\

Before producing the final JSON, briefly think through which entities appear
and how they relate (a few sentences of reasoning). Then output ONLY the JSON
object — no reasoning, no markdown — as your final response.
"""


EXTRACTION_USER_TEMPLATE = """\
Text to extract from:

{text}
"""

# ============================================================
#  QA PROMPTS
# ============================================================

QA_SYSTEM = """\
You are a careful biomedical question-answering system. You will be given:
    1. A user question
    2. A knowledge sub-graph as a list of (subject, predicate, object)
       triples that you may use as evidence

Answer the question concisely. Cite the supporting triple(s) inline using
[Sx,Py,Oz] format.

If the answer cannot be supported by the supplied triples, reply: "Answer
not supported by retrieved evidence." Do NOT use parametric knowledge to
fill in.
"""

QA_USER_TEMPLATE = """\
Question:
{question}

Retrieved evidence:
{evidence_block}
"""
