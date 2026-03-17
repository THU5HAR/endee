"""
Stage 5 — Post-Processing & Hybrid Ranking
Combine Endee vector similarity scores with BM25 keyword scores for improved
result relevance. Deduplicates results and returns a final ranked list.
"""

import logging
import math
import re
from typing import Optional

from rank_bm25 import BM25Okapi

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

VECTOR_WEIGHT = 0.7
BM25_WEIGHT = 0.3

LOG_LEVEL = "INFO"
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("rank")


# ---------------------------------------------------------------------------
# Tokenisation helpers
# ---------------------------------------------------------------------------

_CAMEL_RE = re.compile(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_NON_ALPHA = re.compile(r"[^a-z0-9\s]")


def tokenize(text: str) -> list[str]:
    """
    Tokenize code/text for BM25:
    - Split camelCase / snake_case
    - Lowercase
    - Remove punctuation
    - Return list of non-empty tokens
    """
    text = _CAMEL_RE.sub(" ", text)
    text = text.replace("_", " ").replace("-", " ")
    text = _NON_ALPHA.sub(" ", text.lower())
    return [t for t in text.split() if t]


def build_corpus_text(hit: dict) -> str:
    """Build the BM25 search corpus string for a single hit."""
    parts = [
        hit.get("function_name", ""),
        hit.get("docstring", ""),
        hit.get("filepath", ""),
        hit.get("code", ""),
    ]
    return " ".join(p for p in parts if p)


# ---------------------------------------------------------------------------
# BM25 scoring
# ---------------------------------------------------------------------------

def compute_bm25_scores(hits: list[dict], query: str) -> list[float]:
    """
    Compute BM25 scores for all hits given the query.
    Returns a list of raw BM25 scores (same order as hits).
    """
    if not hits:
        return []

    corpus_texts = [build_corpus_text(h) for h in hits]
    tokenized_corpus = [tokenize(text) for text in corpus_texts]
    query_tokens = tokenize(query)

    if not query_tokens:
        return [0.0] * len(hits)

    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(query_tokens)
    return scores.tolist()


def normalize_scores(scores: list[float]) -> list[float]:
    """
    Min-max normalize a list of scores to [0, 1].
    Handles edge case where all scores are equal.
    """
    if not scores:
        return []
    min_s = min(scores)
    max_s = max(scores)
    if max_s - min_s < 1e-9:
        return [1.0 if s > 0 else 0.0 for s in scores]
    return [(s - min_s) / (max_s - min_s) for s in scores]


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def deduplicate(hits: list[dict]) -> list[dict]:
    """
    Remove duplicate hits with identical (repo, filepath, start_line).
    Keeps the highest-scored occurrence.
    """
    seen: set[tuple] = set()
    deduped = []
    for hit in hits:
        key = (
            hit.get("repo", ""),
            hit.get("filepath", ""),
            hit.get("start_line", 0),
        )
        if key not in seen:
            seen.add(key)
            deduped.append(hit)
    removed = len(hits) - len(deduped)
    if removed > 0:
        log.debug("Deduplication removed %d duplicate hits.", removed)
    return deduped


# ---------------------------------------------------------------------------
# Hybrid ranking
# ---------------------------------------------------------------------------

def hybrid_rank(
    vector_results: list[dict],
    query: str,
    vector_weight: float = VECTOR_WEIGHT,
    bm25_weight: float = BM25_WEIGHT,
    top_n: Optional[int] = None,
) -> list[dict]:
    """
    Re-rank vector search results using a hybrid BM25 + vector score.

    Args:
        vector_results: Raw hits from query.py (each has 'score' / 'vector_score').
        query:          The original user query string.
        vector_weight:  Weight for the vector similarity component (default 0.7).
        bm25_weight:    Weight for the BM25 keyword component (default 0.3).
        top_n:          Optional limit on number of final results.

    Returns:
        Re-ranked list of hit dicts with 'rank', 'score', 'vector_score', 'bm25_score'.
    """
    if not vector_results:
        return []

    hits = deduplicate(vector_results)

    # Normalize vector scores to [0, 1]
    raw_vec_scores = [float(h.get("vector_score", h.get("score", 0.0))) for h in hits]
    norm_vec_scores = normalize_scores(raw_vec_scores)

    # Compute and normalize BM25 scores
    raw_bm25_scores = compute_bm25_scores(hits, query)
    norm_bm25_scores = normalize_scores(raw_bm25_scores)

    # Compute hybrid scores
    ranked = []
    for i, hit in enumerate(hits):
        vec_s = norm_vec_scores[i]
        bm25_s = norm_bm25_scores[i]
        final_score = vector_weight * vec_s + bm25_weight * bm25_s

        result = dict(hit)
        result["vector_score"] = round(vec_s, 4)
        result["bm25_score"] = round(bm25_s, 4)
        result["score"] = round(final_score, 4)
        ranked.append(result)

    # Sort descending by hybrid score
    ranked.sort(key=lambda x: x["score"], reverse=True)

    # Assign final ranks
    for i, r in enumerate(ranked, 1):
        r["rank"] = i

    if top_n:
        ranked = ranked[:top_n]

    log.info("Hybrid ranking complete: %d results (query: '%s')", len(ranked), query[:60])
    return ranked


# ---------------------------------------------------------------------------
# Keyword-only fallback search (for BM25-only mode)
# ---------------------------------------------------------------------------

def keyword_search(snippets: list[dict], query: str, top_k: int = 10) -> list[dict]:
    """
    Pure BM25 keyword search over a list of snippets (no vector store needed).
    Useful for fallback when Endee is unavailable.
    """
    if not snippets:
        return []

    corpus_texts = [build_corpus_text(s) for s in snippets]
    tokenized_corpus = [tokenize(t) for t in corpus_texts]
    query_tokens = tokenize(query)

    if not query_tokens:
        return snippets[:top_k]

    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(query_tokens)

    indexed = [(score, i) for i, score in enumerate(scores)]
    indexed.sort(key=lambda x: x[0], reverse=True)

    results = []
    for rank, (score, idx) in enumerate(indexed[:top_k], 1):
        result = dict(snippets[idx])
        result["rank"] = rank
        result["score"] = round(float(score), 4)
        result["bm25_score"] = round(float(score), 4)
        result["vector_score"] = 0.0
        results.append(result)

    return results


# ---------------------------------------------------------------------------
# Main entry point (demo)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    from query import semantic_search, health_check

    demo_query = "sort a list of numbers"

    if not health_check():
        print("Endee not reachable — running BM25-only demo with sample data.")
        sample = [
            {"id": "1", "function_name": "sort_numbers", "code": "def sort_numbers(lst):\n    return sorted(lst)", "docstring": "Sort a list of numbers.", "repo": "demo", "filepath": "demo.py", "start_line": 1, "end_line": 2, "language": "python", "score": 0.9},
            {"id": "2", "function_name": "bubble_sort", "code": "def bubble_sort(arr):\n    for i in range(len(arr)):\n        for j in range(len(arr)-i-1):\n            if arr[j] > arr[j+1]:\n                arr[j], arr[j+1] = arr[j+1], arr[j]", "docstring": "Bubble sort algorithm.", "repo": "demo", "filepath": "sort.py", "start_line": 1, "end_line": 5, "language": "python", "score": 0.8},
        ]
        results = keyword_search(sample, demo_query)
    else:
        raw_hits = semantic_search(demo_query)
        results = hybrid_rank(raw_hits, demo_query)

    print(f"\n=== Hybrid ranked results for: '{demo_query}' ===\n")
    for r in results:
        print(f"[{r['rank']}] score={r['score']:.4f} (vec={r.get('vector_score',0):.3f}, bm25={r.get('bm25_score',0):.3f})")
        print(f"    {r.get('repo','')}:{r.get('filepath','')}:{r.get('start_line','')}")
        print(f"    {r.get('function_name','')} — {r.get('docstring','')[:80]}\n")
