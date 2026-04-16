"""
Split the corpus into token-bounded chunks for LLM extraction.

Strategy:
    - Read each per-article file from corpus/articles/.
    - Split on paragraph boundaries first (preserves semantic units).
    - Pack paragraphs greedily up to CHUNK_TOKENS (with overlap).
    - Token count is computed via tiktoken (cl100k_base — used by GPT-4/4o).

Output:
    corpus/chunks.json     — list of {chunk_id, source_article, source_slug,
                                       entity_type, text, n_tokens, paragraph_idx}
    corpus/chunks_summary.csv — quick stats per article
"""
from __future__ import annotations
import json
import re
import sys

import pandas as pd
import tiktoken

from config import CORPUS, CHUNK_TOKENS, CHUNK_OVERLAP_TOKENS

ENC = tiktoken.get_encoding("cl100k_base")


def n_tokens(text: str) -> int:
    return len(ENC.encode(text))


def split_paragraphs(text: str) -> list[str]:
    """Split on blank lines, drop empty fragments and lone headings."""
    parts = re.split(r"\n\s*\n+", text)
    paras = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        # Skip stand-alone Wikipedia section headings ("== History ==")
        if re.fullmatch(r"=+\s*[^=]+\s*=+", p):
            continue
        paras.append(p)
    return paras


def chunk_article(text: str, target_tokens: int, overlap_tokens: int) -> list[tuple[str, int]]:
    """
    Greedily pack paragraphs into chunks of ~target_tokens.
    Returns list of (chunk_text, paragraph_start_idx).
    Adds overlap by repeating the tail of the previous chunk.
    """
    paragraphs = split_paragraphs(text)
    chunks = []
    buffer: list[str] = []
    buffer_tokens = 0
    para_start = 0

    for i, para in enumerate(paragraphs):
        pt = n_tokens(para)
        if buffer and buffer_tokens + pt > target_tokens:
            chunks.append(("\n\n".join(buffer), para_start))
            # Build overlap: take trailing paragraphs whose total <= overlap_tokens
            overlap_paras: list[str] = []
            overlap_so_far = 0
            for back in reversed(buffer):
                back_t = n_tokens(back)
                if overlap_so_far + back_t > overlap_tokens:
                    break
                overlap_paras.insert(0, back)
                overlap_so_far += back_t
            buffer = list(overlap_paras)
            buffer_tokens = overlap_so_far
            para_start = i  # next chunk's first new paragraph
        buffer.append(para)
        buffer_tokens += pt

    if buffer:
        chunks.append(("\n\n".join(buffer), para_start))
    return chunks


def main() -> int:
    manifest_path = CORPUS / "manifest.json"
    if not manifest_path.exists():
        print("manifest.json not found. Run load_corpus.py first.")
        return 1

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    all_chunks = []
    summary_rows = []
    chunk_id = 0

    print(f"Chunking {len(manifest)} articles "
          f"(target {CHUNK_TOKENS} tokens, overlap {CHUNK_OVERLAP_TOKENS} tokens)\n")

    for art in manifest:
        text = (CORPUS / art["file"]).read_text(encoding="utf-8")
        chunks = chunk_article(text, CHUNK_TOKENS, CHUNK_OVERLAP_TOKENS)
        chunk_token_counts = []
        for ch_text, para_start in chunks:
            tok = n_tokens(ch_text)
            chunk_token_counts.append(tok)
            all_chunks.append({
                "chunk_id":       chunk_id,
                "source_article": art["title"],
                "source_slug":    art["slug"],
                "entity_type":    art["entity_type"],
                "paragraph_idx":  para_start,
                "n_tokens":       tok,
                "text":           ch_text,
            })
            chunk_id += 1
        summary_rows.append({
            "title":         art["title"],
            "entity_type":   art["entity_type"],
            "n_chunks":      len(chunks),
            "min_tokens":    min(chunk_token_counts) if chunk_token_counts else 0,
            "max_tokens":    max(chunk_token_counts) if chunk_token_counts else 0,
            "mean_tokens":   round(sum(chunk_token_counts) / len(chunk_token_counts), 1) if chunk_token_counts else 0,
            "total_tokens":  sum(chunk_token_counts),
        })
        print(f"  {art['title']:<35} -> {len(chunks):>3} chunks "
              f"({sum(chunk_token_counts):,} tokens)")

    # Persist
    (CORPUS / "chunks.json").write_text(
        json.dumps(all_chunks, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    df = pd.DataFrame(summary_rows)
    df.to_csv(CORPUS / "chunks_summary.csv", index=False)

    print(f"\nTotal: {len(all_chunks)} chunks, "
          f"{sum(c['n_tokens'] for c in all_chunks):,} tokens.")
    print(f"  chunks.json:        {CORPUS / 'chunks.json'}")
    print(f"  chunks_summary.csv: {CORPUS / 'chunks_summary.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
