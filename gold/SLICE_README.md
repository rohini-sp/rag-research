# Gold Reference Slice — annotation guide

## What is this?

To evaluate the LLMs we'll later run for entity/relation extraction, we need a
**gold-standard knowledge graph** for at least one chunk of the corpus. This is
the human-validated ground truth that everything else gets compared against.

## The chosen slice

- **Source article:** EZH2 (gene)
- **Chunk ID:** 112 (`corpus/chunks.json`)
- **Tokens:** ~647
- **Why this chunk:** it contains all four entity types we care about (gene,
  drug, disease, pathway/mechanism), plus organisations, plus the
  *tazemetostat → epithelioid sarcoma* relation that is the motivating example
  in the GraphRAG literature (Han et al., 2025).

The full text of the slice is in `gold/slice_text.txt`.

## What you need to do

Open `gold/gold_triples.csv` and fill in one row per (subject, predicate,
object) triple you can extract from the slice text.

### Schema

| Column          | Required | Description                                                        |
|-----------------|----------|--------------------------------------------------------------------|
| `triple_id`     | yes      | Sequential integer starting at 1                                   |
| `subject`       | yes      | Canonical name of the source entity (e.g., `EZH2`, `tazemetostat`) |
| `subject_type`  | yes      | One of: `gene`, `drug`, `disease`, `pathway`, `organisation`, `chemical`, `other` |
| `predicate`     | yes      | Verb-phrase relation (lowercase, snake_case). See predicate list.  |
| `object`        | yes      | Canonical name of the target entity                                |
| `object_type`   | yes      | Same vocabulary as `subject_type`                                  |
| `evidence_span` | yes      | Verbatim quote from the text (≤30 words) supporting the triple     |
| `confidence`    | optional | `high` / `medium` / `low` — your annotation confidence             |
| `notes`         | optional | Free text                                                          |

### Recommended predicate vocabulary

Keep predicates short and consistent. Reusing the same predicate across
triples is more important than being maximally descriptive.

| Predicate              | Example                                              |
|------------------------|------------------------------------------------------|
| `inhibits`             | `tazemetostat — inhibits — EZH2`                     |
| `treats`               | `tazemetostat — treats — epithelioid sarcoma`        |
| `developed_by`         | `tazemetostat — developed_by — Epizyme`              |
| `approved_for`         | `tazemetostat — approved_for — follicular lymphoma`  |
| `mutated_in`           | `EZH2 — mutated_in — Weaver syndrome`                |
| `overexpressed_in`     | `EZH2 — overexpressed_in — breast cancer`            |
| `targets`              | `GSK126 — targets — EZH2`                            |
| `binds_to`             | `EPZ005687 — binds_to — SET domain`                  |
| `competes_with`        | `EPZ005687 — competes_with — SAM`                    |
| `combined_with`        | `etoposide — combined_with — EZH2 inhibitor`         |
| `selective_over`       | `GSK126 — selective_over — EZH1`                     |
| `is_a`                 | `etoposide — is_a — topoisomerase inhibitor`         |
| `part_of`              | `EZH2 — part_of — PRC2 complex`                      |
| `tradename_of`         | `Tazverik — tradename_of — tazemetostat`             |
| `mutation_in_gene`     | `Y641 mutation — mutation_in_gene — EZH2`            |
| `effective_against`    | `EZH2 inhibitor — effective_against — non-small cell lung cancer` |

If you need a predicate that isn't in this list, add it — but try to reuse
existing ones first.

## Tips

1. **One row per triple.** Don't combine multiple facts in one row.
2. **Canonicalise names.** Use `EZH2` everywhere, not `Ezh2` or `enhancer of
   zeste homolog 2`. Same for drugs (use the generic name, not tradename, in
   the `subject` field — keep tradenames as separate `tradename_of` triples).
3. **Stay close to the text.** Don't bring in outside knowledge. If the chunk
   doesn't say "X inhibits Y", don't write that triple.
4. **Skip negations.** "EZH2 does NOT inhibit X" is harder to evaluate; leave
   these out unless trivially important.
5. **Aim for 40–60 triples.** Less than ~30 makes the metrics noisy; more than
   ~80 means you're probably extracting too aggressively.
6. **Save often.**

## When you're done

Run the validator:

```bash
cd ~/rag_study
source .venv/bin/activate
python code/validate_gold.py
```

It will check schema correctness, predicate consistency, and entity-name
duplicates (e.g., `EZH2` vs `Ezh2`).
