# Technical Requirements Document
## Endee Code Search — System Design & Implementation Spec

**Version:** 1.0  
**Date:** March 17, 2026  
**Project:** Semantic Code Search Engine using Endee Vector DB

---

## 1. Tech Stack

```
Backend:        Python 3.11
UI:             Streamlit 1.32+
Vector DB:      Endee (self-hosted, HTTP API on :8080)
Embeddings:     sentence-transformers/all-MiniLM-L6-v2 (384-dim)
Code Parser:    tree-sitter 0.21+ (Python + JavaScript grammars)
BM25 Engine:    rank-bm25 0.2.2
HTTP Client:    requests + msgpack
Serialization:  numpy (.npz), jsonlines (.jsonl)
Containerization: Docker + Docker Compose
```

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    OFFLINE PIPELINE                          │
│                                                              │
│  GitHub Repos ──► ingest.py ──► embed.py ──► index.py       │
│  (git clone)     (tree-sitter  (MiniLM-L6   (Endee HTTP     │
│                   AST parse)    384-dim)      upsert)        │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                     ONLINE PIPELINE                          │
│                                                              │
│  User Query ──► query.py ──► rank.py ──► app.py             │
│  (natural lang)  (embed +     (BM25 +     (Streamlit        │
│                  Endee ANN)   vector       results UI)       │
│                               fusion)                        │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Pipeline Stage Specifications

### Stage 1: Data Ingestion (`ingest.py`)

**Purpose:** Clone target repositories, parse all top-level functions and methods via tree-sitter AST, save as JSONL.

**Input:** List of GitHub repository URLs  
**Output:** `data/<repo_name>.jsonl` — one JSON object per function

**Schema per record:**
```json
{
  "id": "uuid4-string",
  "repo": "scikit-learn",
  "filepath": "sklearn/linear_model/_base.py",
  "function_name": "fit",
  "language": "python",
  "start_line": 142,
  "end_line": 187,
  "code": "def fit(self, X, y):\n    ...",
  "docstring": "Fit linear model."
}
```

**Target repos (v1):**
- `scikit-learn/scikit-learn`
- `tiangolo/fastapi`
- `psf/requests`
- `pallets/flask`
- `numpy/numpy`
- `pandas-dev/pandas`
- `django/django`
- `sqlalchemy/sqlalchemy`
- `pydantic/pydantic`
- `encode/httpx`

**Parser:** tree-sitter with `tree-sitter-python` and `tree-sitter-javascript` grammars. Extracts:
- `function_definition` (Python)
- `function_declaration`, `method_definition`, `arrow_function` (JavaScript)

**Error handling:** Skip files with parse errors; log to `data/ingest_errors.log`

---

### Stage 2: Embedding (`embed.py`)

**Purpose:** Load all JSONL snippets, encode code text into 384-dim dense vectors.

**Input:** `data/*.jsonl`  
**Output:** `data/embeddings.npz` (vectors), `data/metadata.json` (all records)

**Model:** `sentence-transformers/all-MiniLM-L6-v2`
- Dimension: 384
- Max token length: 256 (code is truncated at `encode` time)
- Batch size: 32 (memory-safe for 16GB RAM)
- Device: CPU (GPU optional via `CUDA_VISIBLE_DEVICES`)

**Text format sent to encoder:**
```
<function_name>\n<docstring>\n<code>
```
Combining name + docstring + code improves recall for natural language queries.

**Output file format:**
```python
np.savez_compressed(
    'data/embeddings.npz',
    vectors=np.array(embeddings, dtype=np.float32),  # shape: (N, 384)
)
# metadata.json: list of dicts matching JSONL records
```

---

### Stage 3: Endee Indexing (`index.py`)

**Purpose:** Create an Endee collection and upsert all embeddings with metadata.

**Input:** `data/embeddings.npz`, `data/metadata.json`  
**Output:** Populated Endee index `code_search` (accessible at `localhost:8080`)

**Endee API calls:**

```
POST /api/v1/index/create
{
  "index_name": "code_search",
  "dim": 384,
  "space_type": "cosine",
  "M": 16,
  "ef_con": 200,
  "precision": "float32"
}
```

```
POST /api/v1/index/code_search/vector/insert
Content-Type: application/json
[
  {
    "id": "<uuid>",
    "vector": [0.12, -0.34, ...],   // 384 floats
    "meta": "<json-encoded metadata string>"
  }
]
```

**Batch size:** 100 vectors per HTTP request (balance throughput vs. latency)  
**Retry logic:** Exponential backoff (3 retries, 1s/2s/4s delays) on 5xx errors

---

### Stage 4: Query Retrieval (`query.py`)

**Purpose:** Accept a natural language query, embed it, run ANN search on Endee.

**Input:** Query string, `top_k` (default 20)  
**Output:** List of raw hit dicts from Endee

**Endee search call:**
```
POST /api/v1/index/code_search/search
{
  "vector": [0.11, -0.22, ...],  // 384-dim query embedding
  "k": 20
}
```

**Response:** msgpack-encoded ResultSet (deserialized via `msgpack` library)

**Caching:** LRU cache (128 entries) for repeated identical queries within session.

---

### Stage 5: Post-Processing / Hybrid Ranking (`rank.py`)

**Purpose:** Combine Endee vector similarity scores with BM25 keyword scores for final ranked results.

**Input:** Endee raw hits + original query string  
**Output:** Re-ranked, deduplicated list of results

**Hybrid score formula:**
```
final_score = 0.7 × vector_score + 0.3 × bm25_score
```

- `vector_score`: Endee cosine similarity (0–1, normalized)
- `bm25_score`: BM25 score over `code + docstring` corpus (normalized 0–1)

**Deduplication:** Remove results with identical `(repo, filepath, start_line)` tuples.

**Output schema per result:**
```json
{
  "rank": 1,
  "score": 0.847,
  "vector_score": 0.91,
  "bm25_score": 0.62,
  "id": "uuid",
  "repo": "fastapi",
  "filepath": "fastapi/routing.py",
  "function_name": "add_api_route",
  "language": "python",
  "start_line": 312,
  "end_line": 367,
  "code": "def add_api_route(...):\n    ...",
  "docstring": "Add an API route handler."
}
```

---

## 4. Streamlit UI (`app.py`)

**Components:**
1. Search bar (full-width text input)
2. Sidebar filters: language (py/js/all), repo (multi-select)
3. Results table with columns: Rank | Score | Repo | File:Line | Preview
4. Expandable code viewer with `streamlit-code-editor` or `st.code` syntax highlighting
5. Copy button per result (clipboard via `pyperclip`)
6. Stats footer: "Indexed: N vectors | Query: Xms | Model: all-MiniLM-L6-v2"

---

## 5. Endee HTTP API Reference (used in this project)

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/api/v1/index/create` | Create `code_search` index |
| GET | `/api/v1/index/list` | List all indexes |
| POST | `/api/v1/index/{name}/vector/insert` | Batch upsert vectors |
| POST | `/api/v1/index/{name}/search` | ANN search (returns msgpack) |
| GET | `/api/v1/index/{name}/info` | Index stats (vector count, dim) |
| DELETE | `/api/v1/index/{name}/delete` | Drop and recreate index |
| GET | `/api/v1/health` | Server health check |

**Base URL:** `http://localhost:8080`  
**Auth:** Set `NDD_AUTH_TOKEN` env var if server started with auth enabled (default: disabled)

---

## 6. File & Directory Structure

```
endee-code-search/
├── PRD.md                  Product requirements
├── TRD.md                  This document
├── README.md               Setup, architecture, quickstart
├── requirements.txt        Python dependencies with pinned versions
├── ingest.py               Stage 1: Clone + AST parse → JSONL
├── embed.py                Stage 2: JSONL → embeddings.npz
├── index.py                Stage 3: embeddings.npz → Endee index
├── query.py                Stage 4: natural language → Endee ANN hits
├── rank.py                 Stage 5: hybrid BM25+vector ranking
├── app.py                  Streamlit search UI
├── docker-compose.yml      Endee + Streamlit services
├── data/
│   ├── *.jsonl             Per-repo parsed functions
│   ├── embeddings.npz      Dense vectors (float32, shape N×384)
│   └── metadata.json       All snippet metadata
└── tests/
    └── test_pipeline.py    End-to-end integration tests
```

---

## 7. Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ENDEE_HOST` | `localhost` | Endee server hostname |
| `ENDEE_PORT` | `8080` | Endee server port |
| `ENDEE_TOKEN` | `` | Auth token (leave empty if auth disabled) |
| `ENDEE_INDEX` | `code_search` | Endee index/collection name |
| `EMBED_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformer model |
| `EMBED_BATCH` | `32` | Embedding batch size |
| `TOP_K` | `20` | Default ANN top-k |
| `REPOS_DIR` | `data/repos` | Local clone directory |
| `LOG_LEVEL` | `INFO` | Python logging level |

---

## 8. Performance Benchmarks (Target)

| Operation | Target p50 | Target p95 |
|-----------|-----------|-----------|
| Ingest 10 repos | — | < 10 min |
| Embed 10k functions | — | < 5 min (CPU) |
| Endee upsert 10k vectors | — | < 2 min |
| Query embed + ANN search | 30ms | 50ms |
| Full UI render (query → display) | 40ms | 80ms |

---

## 9. Dependency Versions

See `requirements.txt` for exact pinned versions. Key constraints:
- `sentence-transformers >= 2.6.0` (all-MiniLM-L6-v2 support)
- `tree-sitter == 0.21.3` (stable grammar binding API)
- `tree-sitter-python == 0.21.0`
- `tree-sitter-javascript == 0.21.4`
- `rank-bm25 >= 0.2.2`
- `msgpack >= 1.0.7`
- `streamlit >= 1.32.0`
