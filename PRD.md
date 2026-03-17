# Product Requirements Document
## Semantic Code Search Engine using Endee Vector DB

**Version:** 1.0  
**Date:** March 17, 2026  
**Author:** Tap Academy Assignment  
**Deadline:** March 18, 2026 5PM IST

---

## 1. Problem Statement

Developers working with large, unfamiliar codebases spend significant time searching for relevant code snippets, utility functions, or implementation patterns. Traditional keyword-based search tools (grep, GitHub search) require knowing the exact function name or terminology used in the code, making discoverability extremely low.

**Core pain:** Developers cannot express what they are looking for in natural language and receive semantically relevant code as a result. A developer asking "how to parse HTTP headers" must know a function is called `parse_headers` or `extract_header_value` — there is no way to bridge natural language intent to code without semantic understanding.

---

## 2. Solution

Build a **Semantic Code Search Engine** that:

1. **Ingests** open-source GitHub repositories and parses all functions/methods using tree-sitter AST parsing.
2. **Embeds** each code snippet into a 384-dimensional dense vector using `sentence-transformers/all-MiniLM-L6-v2`.
3. **Indexes** all vectors into an Endee vector database collection for fast approximate nearest-neighbor (ANN) retrieval.
4. **Retrieves** the top-K semantically similar code snippets when a developer submits a natural language query.
5. **Re-ranks** results using hybrid BM25 keyword scoring combined with vector similarity for improved precision.
6. **Presents** results in a clean Streamlit UI with syntax highlighting, repo context, and copy-to-clipboard.

---

## 3. Target Users

### Primary
- **Bengaluru developers** participating in Tap Academy bootcamp/evaluation programs.
- **Software engineers** exploring new open-source libraries such as scikit-learn, FastAPI, requests.
- **Junior developers** who know what a function should do but not what it's named.

### Secondary
- **Assignment evaluators** at Tap Academy assessing semantic search pipeline quality.
- **Technical leads** benchmarking Endee vector DB against Qdrant, Milvus, Pinecone for code search workloads.

---

## 4. Success Metrics

| Metric | Target | Measurement Method |
|--------|--------|--------------------|
| Query latency (p95) | < 50ms | Streamlit timer, end-to-end |
| Result relevance | ≥ 90% Recall@10 | Manual spot-check on 50 queries |
| Indexed vectors | ≥ 10,000 functions | `data/embeddings.npz` size |
| Pipeline completeness | All 5 stages | ingest → embed → index → query → rank |
| UI usability | Functional with filters | Streamlit demo |

---

## 5. Scope

### In Scope
- Python and JavaScript code parsing via tree-sitter
- Embedding with `all-MiniLM-L6-v2` (384-dim, CPU-friendly)
- Endee HTTP API integration for vector storage and ANN search
- Hybrid ranking with BM25 + vector score fusion
- Streamlit UI with syntax highlighting and metadata filters
- Docker Compose for reproducible local deployment
- Integration tests for end-to-end pipeline validation

### Out of Scope
- Real-time incremental indexing (webhooks, push-based)
- Authentication / multi-user support
- Paid cloud Endee deployment
- Support for compiled languages (C++, Java, Rust) in v1

---

## 6. Constraints

- **Deadline:** March 18, 2026 5PM IST — no extensions.
- **Endee version:** Must use the open-source repo at `https://github.com/THU5HAR/endee` (forked from `endee-io/endee`).
- **Embeddings:** Must use `sentence-transformers/all-MiniLM-L6-v2` (no OpenAI API dependency).
- **Pipeline order:** Strictly `ingest → embed → index → query → rank` — no shortcuts.
- **Language:** Python 3.11; backend via FastAPI or Streamlit only.

---

## 7. User Stories

| ID | As a... | I want to... | So that... |
|----|---------|-------------|------------|
| US-01 | Developer | Type "merge two sorted lists" | I find relevant merge functions in indexed repos |
| US-02 | Developer | Filter results by repo (sklearn) | I narrow results to a trusted library |
| US-03 | Developer | See the file path and line number | I can jump directly to the source |
| US-04 | Developer | Copy a snippet with one click | I can paste it without re-typing |
| US-05 | Evaluator | See query latency and index stats | I can benchmark the system objectively |
| US-06 | Admin | Run `python ingest.py` | I can refresh the index with new repos |

---

## 8. Non-Functional Requirements

- **Performance:** ANN search < 30ms (Endee internal), total UI response < 50ms.
- **Scalability:** Index must support 100,000+ vectors without architecture changes.
- **Reliability:** Pipeline must handle malformed code files gracefully (log and skip).
- **Observability:** Structured logging at each pipeline stage with timestamps and vector counts.
- **Reproducibility:** `docker-compose up` must produce a working system from scratch.
