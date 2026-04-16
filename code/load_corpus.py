"""
Download the cancer drug-target corpus from Wikipedia.

Output:
    corpus/source.txt           — concatenated plain-text corpus
    corpus/articles/<slug>.txt  — one file per article
    corpus/manifest.json        — metadata (title, type, word count, URL)
"""
from __future__ import annotations
import json
import re
import sys
import time
from pathlib import Path

import wikipedia

from config import CORPUS, CORPUS_ARTICLES


def slugify(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")


def fetch_article(title: str, retries: int = 3) -> str:
    """Fetch the plain-text content of a Wikipedia page, with retry on rate limit."""
    for attempt in range(retries):
        try:
            page = wikipedia.page(title, auto_suggest=False, redirect=True)
            return page.content, page.url
        except wikipedia.exceptions.DisambiguationError as e:
            # Take the first disambiguation option as a fallback
            print(f"  [warn] '{title}' is a disambiguation page; trying '{e.options[0]}'")
            return fetch_article(e.options[0], retries=retries - 1)
        except wikipedia.exceptions.PageError:
            print(f"  [error] page not found: {title}")
            return None, None
        except Exception as e:
            print(f"  [warn] attempt {attempt + 1} failed for '{title}': {e}")
            time.sleep(2)
    return None, None


def main() -> int:
    articles_dir = CORPUS / "articles"
    articles_dir.mkdir(exist_ok=True)

    manifest = []
    full_text_parts = []

    print(f"Downloading {len(CORPUS_ARTICLES)} Wikipedia articles...\n")

    for title, entity_type in CORPUS_ARTICLES:
        print(f"  [{entity_type:<8}] {title}")
        content, url = fetch_article(title)
        if content is None:
            print(f"  [SKIP] could not fetch '{title}'")
            continue

        slug = slugify(title)
        article_path = articles_dir / f"{slug}.txt"
        article_path.write_text(content, encoding="utf-8")

        word_count = len(content.split())
        manifest.append({
            "title":        title,
            "entity_type":  entity_type,
            "slug":         slug,
            "url":          url,
            "word_count":   word_count,
            "file":         f"articles/{slug}.txt",
        })

        # Add to combined corpus with a clear delimiter
        full_text_parts.append(
            f"\n\n{'=' * 80}\n# {title} ({entity_type})\n# Source: {url}\n{'=' * 80}\n\n{content}"
        )
        print(f"           {word_count:,} words")

    # Write combined corpus and manifest
    (CORPUS / "source.txt").write_text("".join(full_text_parts), encoding="utf-8")
    (CORPUS / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    total_words = sum(a["word_count"] for a in manifest)
    print(f"\nDone. {len(manifest)} articles, {total_words:,} total words.")
    print(f"  Combined corpus: {CORPUS / 'source.txt'}")
    print(f"  Per-article files: {articles_dir}")
    print(f"  Manifest: {CORPUS / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
