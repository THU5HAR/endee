"""
Stage 4 — Query Retrieval
Embed a natural language query and retrieve the top-K semantically similar
code snippets from Endee using approximate nearest-neighbor search.
"""

import os
import json
import logging
import time
from functools import lru_cache
from typing import Optional

import numpy as np
import requests
import msgpack
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ENDEE_HOST = os.getenv("ENDEE_HOST", "localhost")
ENDEE_PORT = int(os.getenv("ENDEE_PORT", "8080"))
ENDEE_BASE = f"http://{ENDEE_HOST}:{ENDEE_PORT}"
ENDEE_TOKEN = os.getenv("ENDEE_TOKEN", "")
ENDEE_INDEX = os.getenv("ENDEE_INDEX", "code_search")

EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
DEFAULT_TOP_K = int(os.getenv("TOP_K", "20"))

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("query")


# ---------------------------------------------------------------------------
# Model singleton (lazy-loaded)
# ---------------------------------------------------------------------------

_model: Optional[SentenceTransformer] = None

def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        log.info("Loading embedding model '%s' ...", EMBED_MODEL)
        t0 = time.time()
        _model = SentenceTransformer(EMBED_MODEL)
        log.info("Model loaded in %.1fs.", time.time() - t0)
    return _model


# ---------------------------------------------------------------------------
# Endee client helpers
# ---------------------------------------------------------------------------

def _headers_json() -> dict:
    h = {"Content-Type": "application/json", "Accept": "application/msgpack"}
    if ENDEE_TOKEN:
        h["Authorization"] = ENDEE_TOKEN
    return h


def _decode_msgpack_response(resp: requests.Response) -> dict:
    """Decode a msgpack-encoded Endee search response."""
    content_type = resp.headers.get("Content-Type", "")
    if "msgpack" in content_type:
        return msgpack.unpackb(resp.content, raw=False)
    # Fallback: try JSON
    try:
        return resp.json()
    except Exception:
        return {"results": [], "raw": resp.text}


def health_check() -> bool:
    """Quick check that Endee server is up."""
    try:
        resp = requests.get(f"{ENDEE_BASE}/api/v1/health", timeout=5)
        return resp.status_code == 200
    except requests.RequestException:
        return False


# ---------------------------------------------------------------------------
# Query embedding (cached)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=128)
def embed_query(query: str) -> tuple:
    """
    Embed a query string and return as a tuple (for hashability / LRU cache).
    Uses the same model and normalization as embed.py.
    """
    model = get_model()
    vec = model.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]
    return tuple(vec.astype(np.float32).tolist())


# ---------------------------------------------------------------------------
# Core retrieval
# ---------------------------------------------------------------------------

def semantic_search(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    lang_filter: Optional[str] = None,
    repo_filter: Optional[list[str]] = None,
) -> list[dict]:
    """
    Embed the query, run ANN search on Endee, return a list of raw hit dicts.

    Args:
        query:       Natural language query string.
        top_k:       Number of results to retrieve from Endee.
        lang_filter: Optional language filter ("python", "javascript").
        repo_filter: Optional list of repo names to restrict results to.

    Returns:
        List of dicts with keys: id, score, meta (parsed dict).
    """
    if not query.strip():
        return []

    t0 = time.time()

    # Embed the query
    query_vec = list(embed_query(query))
    embed_ms = (time.time() - t0) * 1000

    # Request extra candidates when filters are applied so client-side filter still yields enough
    k_request = top_k
    if (lang_filter and lang_filter != "all") or (repo_filter and len(repo_filter) > 0):
        k_request = min(top_k * 5, 100)  # ask for more, then filter and slice to top_k

    payload: dict = {
        "vector": query_vec,
        "k": k_request,
    }

    # Build filter expression if language filter is specified
    if lang_filter and lang_filter != "all":
        payload["filter"] = json.dumps([{"language": {"$eq": lang_filter}}])

    t1 = time.time()
    try:
        resp = requests.post(
            f"{ENDEE_BASE}/api/v1/index/{ENDEE_INDEX}/search",
            headers=_headers_json(),
            json=payload,
            timeout=10,
        )
    except requests.RequestException as exc:
        log.error("Endee search request failed: %s", exc)
        return []

    search_ms = (time.time() - t1) * 1000
    total_ms = (time.time() - t0) * 1000

    if resp.status_code != 200:
        log.error("Endee search returned HTTP %d: %s", resp.status_code, resp.text[:200])
        return []

    raw = _decode_msgpack_response(resp)
    hits = _parse_hits(raw)

    # Apply client-side filters (repo, lang) in case Endee filter not supported
    hits = _apply_filters(hits, lang_filter=lang_filter, repo_filter=repo_filter)
    hits = hits[:top_k]  # keep only top_k after filtering

    log.info(
        "Query '%s': %d hits | embed=%.1fms | search=%.1fms | total=%.1fms",
        query[:60], len(hits), embed_ms, search_ms, total_ms,
    )
    return hits


def _parse_hits(raw) -> list[dict]:
    """
    Parse the Endee msgpack ResultSet into a flat list of hit dicts.

    Endee returns a list of lists, each item:
      [score, id, meta_bytes, filter_str, norm, sparse_data]
    """
    # Normalise: handle both list-of-lists (msgpack) and list/dict (JSON fallback)
    if isinstance(raw, dict):
        results = raw.get("results", [])
    elif isinstance(raw, list):
        results = raw
    else:
        return []

    hits = []
    for item in results:
        try:
            if isinstance(item, (list, tuple)):
                # Endee msgpack format: [score, id, meta, filter, norm, sparse]
                score = float(item[0]) if len(item) > 0 else 0.0
                hit_id = item[1] if len(item) > 1 else ""
                meta_raw = item[2] if len(item) > 2 else b"{}"
            elif isinstance(item, dict):
                # JSON fallback format
                score = float(item.get("score", 0.0))
                hit_id = item.get("id", "")
                meta_raw = item.get("meta", "{}")
            else:
                continue

            # Decode meta bytes → dict
            if isinstance(meta_raw, (bytes, bytearray)):
                meta_raw = meta_raw.decode("utf-8", errors="replace")
            try:
                meta = json.loads(meta_raw) if isinstance(meta_raw, str) else {}
            except json.JSONDecodeError:
                meta = {}

            hits.append({
                "id": str(hit_id),
                "score": score,
                "vector_score": score,
                **meta,
            })
        except Exception as exc:
            log.debug("Skipping malformed hit: %s — %s", item, exc)
            continue

    return hits


def _apply_filters(
    hits: list[dict],
    lang_filter: Optional[str] = None,
    repo_filter: Optional[list[str]] = None,
) -> list[dict]:
    """Apply client-side language and repo filters to retrieved hits."""
    if lang_filter and lang_filter != "all":
        hits = [h for h in hits if h.get("language", "") == lang_filter]
    if repo_filter:
        repo_set = {r.lower() for r in repo_filter}
        hits = [h for h in hits if h.get("repo", "").lower() in repo_set]
    return hits


# ---------------------------------------------------------------------------
# Utility: get index stats
# ---------------------------------------------------------------------------

def get_index_stats() -> dict:
    """Fetch live stats from the Endee index."""
    try:
        h = {"Content-Type": "application/json"}
        if ENDEE_TOKEN:
            h["Authorization"] = ENDEE_TOKEN
        resp = requests.get(f"{ENDEE_BASE}/api/v1/index/{ENDEE_INDEX}/info", headers=h, timeout=5)
        if resp.status_code == 200:
            return resp.json()
    except requests.RequestException:
        pass
    return {}


# ---------------------------------------------------------------------------
# Main entry point (CLI demo)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Semantic code search via Endee.")
    parser.add_argument("query", nargs="?", default="parse HTTP headers", help="Search query.")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--lang", default=None, help="Filter by language: python or javascript.")
    parser.add_argument("--repo", nargs="*", default=None, help="Filter by repo name(s).")
    args = parser.parse_args()

    if not health_check():
        log.error("Endee not reachable at %s. Start it first.", ENDEE_BASE)
        raise SystemExit(1)

    hits = semantic_search(args.query, top_k=args.top_k, lang_filter=args.lang, repo_filter=args.repo)

    print(f"\n=== Results for: '{args.query}' ===\n")
    for i, h in enumerate(hits, 1):
        print(f"[{i}] score={h['score']:.4f} | {h.get('repo','')}:{h.get('filepath','')}:{h.get('start_line','')}")
        print(f"    {h.get('function_name','')} — {h.get('docstring','')[:80]}")
        print(f"    {h.get('code','')[:120].replace(chr(10), ' ')}\n")
