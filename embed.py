"""
Stage 2 — Embedding
Load parsed JSONL snippets, encode with sentence-transformers/all-MiniLM-L6-v2,
save dense vectors + metadata to data/embeddings.npz and data/metadata.json.
"""

import os
import json
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR = Path("data")
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
EMBED_BATCH = int(os.getenv("EMBED_BATCH", "32"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("embed")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_all_snippets(data_dir: Path) -> list[dict]:
    """Load all JSONL files from data_dir into a flat list of records."""
    records: list[dict] = []
    jsonl_files = sorted(data_dir.glob("*.jsonl"))

    if not jsonl_files:
        log.error("No .jsonl files found in %s. Run ingest.py first.", data_dir)
        return records

    for path in jsonl_files:
        count = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    records.append(rec)
                    count += 1
                except json.JSONDecodeError as exc:
                    log.warning("Skipping malformed JSON line in %s: %s", path, exc)
        log.info("Loaded %d snippets from %s", count, path.name)

    log.info("Total snippets loaded: %d", len(records))
    return records


def build_embed_texts(records: list[dict]) -> list[str]:
    """
    Construct the text string to embed for each record.
    Format: <function_name>\n<docstring>\n<code>
    Combining name + docstring + code improves recall for natural language queries.
    """
    texts = []
    for rec in records:
        parts = [rec.get("function_name", "")]
        if rec.get("docstring"):
            parts.append(rec["docstring"])
        parts.append(rec.get("code", ""))
        texts.append("\n".join(p for p in parts if p))
    return texts


# ---------------------------------------------------------------------------
# Main embedding routine
# ---------------------------------------------------------------------------

def generate_embeddings(
    records: Optional[list[dict]] = None,
    data_dir: Path = DATA_DIR,
    model_name: str = EMBED_MODEL,
    batch_size: int = EMBED_BATCH,
) -> tuple[np.ndarray, list[dict]]:
    """
    Embed all code snippets.
    Returns (vectors: np.ndarray shape (N, 384), metadata: list of dicts).
    """
    if records is None:
        records = load_all_snippets(data_dir)

    if not records:
        raise ValueError("No records to embed. Run ingest.py first.")

    log.info("Loading model '%s' ...", model_name)
    t0 = time.time()
    model = SentenceTransformer(model_name)
    log.info("Model loaded in %.1fs. Embedding dimension: %d", time.time() - t0, model.get_sentence_embedding_dimension())

    texts = build_embed_texts(records)
    log.info("Encoding %d snippets with batch_size=%d ...", len(texts), batch_size)

    t1 = time.time()
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,   # unit-norm for cosine similarity
        convert_to_numpy=True,
    )
    elapsed = time.time() - t1

    log.info(
        "Embedding complete: %d vectors, shape %s, dtype %s — %.1fs (%.0f snippets/s)",
        len(vectors), vectors.shape, vectors.dtype, elapsed, len(texts) / elapsed,
    )
    return vectors.astype(np.float32), records


def save_embeddings(
    vectors: np.ndarray,
    metadata: list[dict],
    data_dir: Path = DATA_DIR,
) -> None:
    """Persist vectors and metadata to disk."""
    data_dir.mkdir(parents=True, exist_ok=True)

    npz_path = data_dir / "embeddings.npz"
    meta_path = data_dir / "metadata.json"

    log.info("Saving %d vectors to %s ...", len(vectors), npz_path)
    np.savez_compressed(str(npz_path), vectors=vectors)

    log.info("Saving metadata (%d records) to %s ...", len(metadata), meta_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False)

    # Human-readable summary
    size_mb = npz_path.stat().st_size / (1024 * 1024)
    log.info("embeddings.npz: %.2f MB | metadata.json: %d records", size_mb, len(metadata))


def load_embeddings(data_dir: Path = DATA_DIR) -> tuple[np.ndarray, list[dict]]:
    """Load previously saved embeddings and metadata from disk."""
    npz_path = data_dir / "embeddings.npz"
    meta_path = data_dir / "metadata.json"

    if not npz_path.exists():
        raise FileNotFoundError(f"embeddings.npz not found at {npz_path}. Run embed.py first.")
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.json not found at {meta_path}. Run embed.py first.")

    log.info("Loading embeddings from %s ...", npz_path)
    data = np.load(str(npz_path))
    vectors = data["vectors"]

    log.info("Loading metadata from %s ...", meta_path)
    with open(meta_path, encoding="utf-8") as f:
        metadata = json.load(f)

    log.info("Loaded %d vectors (shape: %s) and %d metadata records.", len(vectors), vectors.shape, len(metadata))
    return vectors, metadata


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Embed code snippets with sentence-transformers.")
    parser.add_argument("--data-dir", default="data", help="Directory containing JSONL files.")
    parser.add_argument("--model", default=EMBED_MODEL, help="Sentence-transformers model name.")
    parser.add_argument("--batch-size", type=int, default=EMBED_BATCH, help="Encoding batch size.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    vectors, metadata = generate_embeddings(
        data_dir=data_dir,
        model_name=args.model,
        batch_size=args.batch_size,
    )
    save_embeddings(vectors, metadata, data_dir=data_dir)
    log.info("Stage 2 complete. Run index.py to upload to Endee.")
