"""
Stage 3 — Endee Indexing
Load embeddings from disk, create an Endee collection, upsert all vectors with metadata.
Uses the Endee HTTP API directly (no external SDK required).
"""

import os
import json
import logging
import time
from pathlib import Path

import numpy as np
import requests
import msgpack

from embed import load_embeddings

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ENDEE_HOST = os.getenv("ENDEE_HOST", "localhost")
ENDEE_PORT = int(os.getenv("ENDEE_PORT", "8080"))
ENDEE_BASE = f"http://{ENDEE_HOST}:{ENDEE_PORT}"
ENDEE_TOKEN = os.getenv("ENDEE_TOKEN", "")
ENDEE_INDEX = os.getenv("ENDEE_INDEX", "code_search")

INSERT_BATCH = int(os.getenv("INSERT_BATCH", "100"))
MAX_RETRIES = 3
RETRY_BASE_DELAY = 1.0  # seconds

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("index")


# ---------------------------------------------------------------------------
# Endee HTTP client helpers
# ---------------------------------------------------------------------------

def _headers() -> dict:
    h = {"Content-Type": "application/json"}
    if ENDEE_TOKEN:
        h["Authorization"] = ENDEE_TOKEN
    return h


def _request_with_retry(method: str, url: str, **kwargs) -> requests.Response:
    """HTTP request with exponential backoff on 5xx errors."""
    delay = RETRY_BASE_DELAY
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.request(method, url, timeout=30, **kwargs)
            if resp.status_code < 500:
                return resp
            log.warning("Attempt %d/%d: %s %s → %d", attempt, MAX_RETRIES, method, url, resp.status_code)
        except requests.RequestException as exc:
            log.warning("Attempt %d/%d: request failed — %s", attempt, MAX_RETRIES, exc)
        if attempt < MAX_RETRIES:
            time.sleep(delay)
            delay *= 2
    raise RuntimeError(f"All {MAX_RETRIES} attempts failed for {method} {url}")


def health_check() -> bool:
    """Verify Endee server is reachable."""
    try:
        resp = requests.get(f"{ENDEE_BASE}/api/v1/health", timeout=5)
        ok = resp.status_code == 200
        if ok:
            log.info("Endee health check: OK (port %d)", ENDEE_PORT)
        else:
            log.error("Endee health check failed: HTTP %d", resp.status_code)
        return ok
    except requests.RequestException as exc:
        log.error("Endee not reachable at %s: %s", ENDEE_BASE, exc)
        return False


def create_index(dim: int = 384) -> bool:
    """Create the code_search index in Endee. Idempotent — skips if already exists."""
    payload = {
        "index_name": ENDEE_INDEX,
        "dim": dim,
        "space_type": "cosine",
        "M": 16,
        "ef_con": 200,
        "precision": "float32",
    }
    log.info("Creating Endee index '%s' (dim=%d, space=cosine) ...", ENDEE_INDEX, dim)
    resp = _request_with_retry("POST", f"{ENDEE_BASE}/api/v1/index/create",
                                headers=_headers(), json=payload)
    if resp.status_code == 200:
        log.info("Index '%s' created successfully.", ENDEE_INDEX)
        return True
    if resp.status_code == 409:
        log.info("Index '%s' already exists — skipping creation.", ENDEE_INDEX)
        return True
    log.error("Failed to create index: HTTP %d — %s", resp.status_code, resp.text)
    return False


def delete_index() -> bool:
    """Drop the code_search index (used to force full re-index)."""
    log.warning("Deleting Endee index '%s' ...", ENDEE_INDEX)
    resp = _request_with_retry("DELETE", f"{ENDEE_BASE}/api/v1/index/{ENDEE_INDEX}/delete",
                                headers=_headers())
    if resp.status_code in (200, 404):
        log.info("Index '%s' deleted (or did not exist).", ENDEE_INDEX)
        return True
    log.error("Failed to delete index: HTTP %d — %s", resp.status_code, resp.text)
    return False


def get_index_info() -> dict:
    """Fetch index metadata (vector count, dim, etc.)."""
    resp = _request_with_retry("GET", f"{ENDEE_BASE}/api/v1/index/{ENDEE_INDEX}/info",
                                headers=_headers())
    if resp.status_code == 200:
        return resp.json()
    return {}


def insert_batch(batch: list[dict]) -> bool:
    """Insert a batch of vector objects into Endee via JSON."""
    resp = _request_with_retry(
        "POST",
        f"{ENDEE_BASE}/api/v1/index/{ENDEE_INDEX}/vector/insert",
        headers=_headers(),
        json=batch,
    )
    if resp.status_code == 200:
        return True
    log.error("Insert batch failed: HTTP %d — %s", resp.status_code, resp.text[:300])
    return False


# ---------------------------------------------------------------------------
# Core indexing logic
# ---------------------------------------------------------------------------

def build_insert_payload(vectors: np.ndarray, metadata: list[dict]) -> list[dict]:
    """Convert numpy vectors + metadata into Endee insert format."""
    payload = []
    for vec, meta in zip(vectors, metadata):
        payload.append({
            "id": meta["id"],
            "vector": vec.tolist(),
            "meta": json.dumps({
                "repo": meta.get("repo", ""),
                "filepath": meta.get("filepath", ""),
                "function_name": meta.get("function_name", ""),
                "language": meta.get("language", ""),
                "start_line": meta.get("start_line", 0),
                "end_line": meta.get("end_line", 0),
                "docstring": meta.get("docstring", ""),
                "code": meta.get("code", "")[:2000],
            }, ensure_ascii=False),
        })
    return payload


def upsert_all(
    vectors: np.ndarray,
    metadata: list[dict],
    batch_size: int = INSERT_BATCH,
    force_recreate: bool = False,
) -> int:
    """
    Upsert all vectors into Endee in batches.
    Returns the number of successfully inserted vectors.
    """
    if force_recreate:
        delete_index()

    dim = vectors.shape[1]
    if not create_index(dim=dim):
        raise RuntimeError("Could not create Endee index. Is the server running?")

    total = len(vectors)
    inserted = 0
    failed = 0

    log.info("Starting upsert of %d vectors in batches of %d ...", total, batch_size)
    t_start = time.time()

    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch_vecs = vectors[batch_start:batch_end]
        batch_meta = metadata[batch_start:batch_end]

        payload = build_insert_payload(batch_vecs, batch_meta)

        success = insert_batch(payload)
        batch_count = len(payload)

        if success:
            inserted += batch_count
        else:
            failed += batch_count
            log.warning("Batch [%d:%d] failed — %d vectors lost.", batch_start, batch_end, batch_count)

        # Progress logging every 10 batches
        if (batch_start // batch_size + 1) % 10 == 0 or batch_end == total:
            elapsed = time.time() - t_start
            rate = inserted / elapsed if elapsed > 0 else 0
            log.info("Progress: %d/%d vectors | %.0f vec/s | failed: %d",
                     inserted, total, rate, failed)

    elapsed = time.time() - t_start
    log.info("Upsert complete: %d inserted, %d failed in %.1fs.", inserted, failed, elapsed)
    return inserted


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Index embeddings into Endee vector DB.")
    parser.add_argument("--data-dir", default="data", help="Directory with embeddings.npz and metadata.json.")
    parser.add_argument("--batch-size", type=int, default=INSERT_BATCH, help="Insert batch size.")
    parser.add_argument("--force-recreate", action="store_true",
                        help="Delete and recreate the index before inserting.")
    parser.add_argument("--endee-host", default=ENDEE_HOST)
    parser.add_argument("--endee-port", type=int, default=ENDEE_PORT)
    args = parser.parse_args()

    # Override globals from CLI args
    ENDEE_HOST = args.endee_host
    ENDEE_PORT = args.endee_port
    ENDEE_BASE = f"http://{ENDEE_HOST}:{ENDEE_PORT}"

    if not health_check():
        log.error("Endee server not available. Start it with: docker-compose up endee")
        raise SystemExit(1)

    data_dir = Path(args.data_dir)
    vectors, metadata = load_embeddings(data_dir=data_dir)

    count = upsert_all(
        vectors=vectors,
        metadata=metadata,
        batch_size=args.batch_size,
        force_recreate=args.force_recreate,
    )

    # Print index stats
    info = get_index_info()
    log.info("Endee index info: %s", json.dumps(info, indent=2))
    log.info("Stage 3 complete. %d vectors indexed. Run app.py to search.", count)
