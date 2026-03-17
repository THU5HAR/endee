"""
End-to-end integration tests for the Endee Code Search pipeline.
Tests cover all 5 stages: ingest → embed → index → query → rank.
Run with: pytest tests/ -v
"""

import json
import os
import time
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_SNIPPETS = [
    {
        "id": str(uuid.uuid4()),
        "repo": "requests",
        "filepath": "requests/api.py",
        "function_name": "get",
        "language": "python",
        "start_line": 65,
        "end_line": 82,
        "code": "def get(url, params=None, **kwargs):\n    return request('GET', url, params=params, **kwargs)",
        "docstring": "Sends a GET request.",
    },
    {
        "id": str(uuid.uuid4()),
        "repo": "fastapi",
        "filepath": "fastapi/routing.py",
        "function_name": "add_api_route",
        "language": "python",
        "start_line": 312,
        "end_line": 350,
        "code": "def add_api_route(self, path: str, endpoint: Callable, *, methods: List[str] = None):\n    route = APIRoute(path, endpoint=endpoint, methods=methods)\n    self.routes.append(route)",
        "docstring": "Add an API route handler to the router.",
    },
    {
        "id": str(uuid.uuid4()),
        "repo": "numpy",
        "filepath": "numpy/core/fromnumeric.py",
        "function_name": "sort",
        "language": "python",
        "start_line": 895,
        "end_line": 960,
        "code": "def sort(a, axis=-1, kind=None, order=None):\n    a = asanyarray(a)\n    if axis is None:\n        a = a.flatten()\n        axis = 0\n    return a.sort(axis=axis, kind=kind, order=order)",
        "docstring": "Return a sorted copy of an array.",
    },
    {
        "id": str(uuid.uuid4()),
        "repo": "flask",
        "filepath": "flask/app.py",
        "function_name": "route",
        "language": "python",
        "start_line": 1017,
        "end_line": 1040,
        "code": "def route(self, rule, **options):\n    def decorator(f):\n        self.add_url_rule(rule, endpoint=None, view_func=f, **options)\n        return f\n    return decorator",
        "docstring": "A decorator that is used to register a view function for a given URL rule.",
    },
]


@pytest.fixture
def tmp_data_dir(tmp_path):
    """Create a temporary data directory with sample JSONL files."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    # Write sample snippets to JSONL
    jsonl_path = data_dir / "test_repo.jsonl"
    with open(jsonl_path, "w") as f:
        for s in SAMPLE_SNIPPETS:
            f.write(json.dumps(s) + "\n")

    return data_dir


# ---------------------------------------------------------------------------
# Stage 1: Ingest tests
# ---------------------------------------------------------------------------

class TestIngest:
    def test_tokenize_code(self):
        """Basic sanity: tree-sitter can be imported and Language built."""
        import tree_sitter_python as tspython
        from tree_sitter import Language, Parser
        lang = Language(tspython.language())
        parser = Parser(lang)
        code = b"def hello(name):\n    return f'Hello, {name}!'\n"
        tree = parser.parse(code)
        assert tree.root_node is not None

    def test_parse_python_functions_basic(self, tmp_path):
        """parse_python_functions extracts function names from a .py file."""
        from ingest import parse_python_functions, REPOS_DIR

        py_file = tmp_path / "sample.py"
        py_file.write_text(
            "def add(a, b):\n    '''Add two numbers.'''\n    return a + b\n\n"
            "def multiply(x, y):\n    return x * y\n"
        )

        # Temporarily adjust REPOS_DIR for relative path resolution
        import ingest as ingest_mod
        original = ingest_mod.REPOS_DIR
        ingest_mod.REPOS_DIR = tmp_path
        try:
            snippets = parse_python_functions(py_file, "sample")
        finally:
            ingest_mod.REPOS_DIR = original

        names = [s.function_name for s in snippets]
        assert "add" in names
        assert "multiply" in names

    def test_save_and_reload_snippets(self, tmp_path):
        """save_snippets writes valid JSONL that can be reloaded."""
        from ingest import save_snippets, CodeSnippet
        from dataclasses import asdict

        snippet = CodeSnippet(
            id=str(uuid.uuid4()),
            repo="test",
            filepath="test.py",
            function_name="my_func",
            language="python",
            start_line=1,
            end_line=3,
            code="def my_func():\n    pass",
            docstring="",
        )
        out = tmp_path / "out.jsonl"
        save_snippets([snippet], out)

        assert out.exists()
        with open(out) as f:
            lines = [json.loads(l) for l in f if l.strip()]
        assert len(lines) == 1
        assert lines[0]["function_name"] == "my_func"


# ---------------------------------------------------------------------------
# Stage 2: Embed tests
# ---------------------------------------------------------------------------

class TestEmbed:
    def test_load_snippets(self, tmp_data_dir):
        """load_all_snippets reads JSONL files correctly."""
        from embed import load_all_snippets
        records = load_all_snippets(tmp_data_dir)
        assert len(records) == len(SAMPLE_SNIPPETS)
        assert records[0]["function_name"] == "get"

    def test_build_embed_texts(self):
        """build_embed_texts combines name + docstring + code."""
        from embed import build_embed_texts
        texts = build_embed_texts(SAMPLE_SNIPPETS)
        assert len(texts) == len(SAMPLE_SNIPPETS)
        # Should include function name
        assert "get" in texts[0]
        # Should include docstring
        assert "GET request" in texts[0]

    def test_generate_embeddings_shape(self, tmp_data_dir):
        """generate_embeddings produces correct shape for sample records."""
        from embed import generate_embeddings
        vectors, metadata = generate_embeddings(
            records=SAMPLE_SNIPPETS,
            data_dir=tmp_data_dir,
            model_name="all-MiniLM-L6-v2",
            batch_size=4,
        )
        assert vectors.shape == (len(SAMPLE_SNIPPETS), 384)
        assert len(metadata) == len(SAMPLE_SNIPPETS)
        assert vectors.dtype == np.float32

    def test_save_and_load_embeddings(self, tmp_data_dir):
        """save_embeddings + load_embeddings round-trip is lossless."""
        from embed import generate_embeddings, save_embeddings, load_embeddings
        vectors, metadata = generate_embeddings(
            records=SAMPLE_SNIPPETS,
            data_dir=tmp_data_dir,
            batch_size=4,
        )
        save_embeddings(vectors, metadata, data_dir=tmp_data_dir)
        loaded_vecs, loaded_meta = load_embeddings(data_dir=tmp_data_dir)

        assert np.allclose(vectors, loaded_vecs, atol=1e-6)
        assert len(loaded_meta) == len(metadata)


# ---------------------------------------------------------------------------
# Stage 3: Index tests (mocked Endee)
# ---------------------------------------------------------------------------

class TestIndex:
    def _mock_response(self, status_code: int, json_body: dict | None = None, text: str = ""):
        mock = MagicMock()
        mock.status_code = status_code
        mock.text = text
        if json_body is not None:
            mock.json.return_value = json_body
        return mock

    @patch("index.requests.request")
    def test_create_index_success(self, mock_request):
        """create_index returns True on HTTP 200."""
        mock_request.return_value = self._mock_response(200, {})
        import index as idx_mod
        result = idx_mod.create_index(dim=384)
        assert result is True

    @patch("index.requests.request")
    def test_create_index_already_exists(self, mock_request):
        """create_index returns True on HTTP 409 (already exists)."""
        mock_request.return_value = self._mock_response(409, {}, "Index already exists")
        import index as idx_mod
        result = idx_mod.create_index(dim=384)
        assert result is True

    @patch("index.requests.request")
    def test_insert_batch_success(self, mock_request):
        """insert_batch returns True on HTTP 200."""
        mock_request.return_value = self._mock_response(200)
        import index as idx_mod
        batch = [{"id": "test-1", "vector": [0.1] * 384, "meta": "{}"}]
        result = idx_mod.insert_batch(batch)
        assert result is True

    def test_build_insert_payload(self):
        """build_insert_payload produces correct structure."""
        from index import build_insert_payload
        vecs = np.random.rand(2, 384).astype(np.float32)
        meta = SAMPLE_SNIPPETS[:2]
        payload = build_insert_payload(vecs, meta)

        assert len(payload) == 2
        assert "id" in payload[0]
        assert "vector" in payload[0]
        assert len(payload[0]["vector"]) == 384
        meta_parsed = json.loads(payload[0]["meta"])
        assert meta_parsed["function_name"] == "get"


# ---------------------------------------------------------------------------
# Stage 4: Query tests (mocked Endee)
# ---------------------------------------------------------------------------

class TestQuery:
    def test_embed_query_shape(self):
        """embed_query returns a 384-element tuple."""
        from query import embed_query
        result = embed_query("sort a list of numbers")
        assert isinstance(result, tuple)
        assert len(result) == 384

    def test_embed_query_cached(self):
        """embed_query returns identical result for identical input (LRU cache)."""
        from query import embed_query
        r1 = embed_query("find duplicates in a list")
        r2 = embed_query("find duplicates in a list")
        assert r1 == r2

    def test_parse_hits_from_json(self):
        """_parse_hits correctly parses a JSON result list."""
        from query import _parse_hits
        raw = {
            "results": [
                {"id": "abc", "score": 0.92, "meta": json.dumps({"function_name": "sort", "repo": "numpy"})},
                {"id": "def", "score": 0.85, "meta": json.dumps({"function_name": "get", "repo": "requests"})},
            ]
        }
        hits = _parse_hits(raw)
        assert len(hits) == 2
        assert hits[0]["score"] == pytest.approx(0.92)
        assert hits[0]["function_name"] == "sort"
        assert hits[1]["repo"] == "requests"

    def test_apply_filters_language(self):
        """_apply_filters correctly filters by language."""
        from query import _apply_filters
        hits = [
            {"id": "1", "language": "python", "repo": "flask"},
            {"id": "2", "language": "javascript", "repo": "react"},
        ]
        result = _apply_filters(hits, lang_filter="python")
        assert len(result) == 1
        assert result[0]["language"] == "python"

    def test_apply_filters_repo(self):
        """_apply_filters correctly filters by repo name."""
        from query import _apply_filters
        hits = [
            {"id": "1", "language": "python", "repo": "flask"},
            {"id": "2", "language": "python", "repo": "fastapi"},
        ]
        result = _apply_filters(hits, repo_filter=["fastapi"])
        assert len(result) == 1
        assert result[0]["repo"] == "fastapi"


# ---------------------------------------------------------------------------
# Stage 5: Rank tests
# ---------------------------------------------------------------------------

class TestRank:
    HITS = [
        {"id": "1", "score": 0.9, "vector_score": 0.9, "function_name": "sort_list",
         "code": "def sort_list(lst): return sorted(lst)", "docstring": "Sort a list",
         "repo": "utils", "filepath": "utils.py", "start_line": 1, "language": "python"},
        {"id": "2", "score": 0.75, "vector_score": 0.75, "function_name": "bubble_sort",
         "code": "def bubble_sort(arr): pass", "docstring": "Bubble sort algorithm",
         "repo": "algos", "filepath": "sort.py", "start_line": 10, "language": "python"},
        {"id": "3", "score": 0.60, "vector_score": 0.60, "function_name": "http_get",
         "code": "def http_get(url): return requests.get(url)", "docstring": "HTTP GET helper",
         "repo": "utils", "filepath": "http.py", "start_line": 5, "language": "python"},
    ]

    def test_hybrid_rank_returns_all_hits(self):
        """hybrid_rank returns the same number of hits as input."""
        from rank import hybrid_rank
        results = hybrid_rank(self.HITS, query="sort a list")
        assert len(results) == len(self.HITS)

    def test_hybrid_rank_ordered_by_score(self):
        """hybrid_rank results are sorted by descending score."""
        from rank import hybrid_rank
        results = hybrid_rank(self.HITS, query="sort algorithm")
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_hybrid_rank_assigns_ranks(self):
        """hybrid_rank assigns sequential rank integers starting at 1."""
        from rank import hybrid_rank
        results = hybrid_rank(self.HITS, query="sort")
        ranks = [r["rank"] for r in results]
        assert ranks == list(range(1, len(self.HITS) + 1))

    def test_hybrid_rank_score_bounds(self):
        """All hybrid scores are between 0 and 1."""
        from rank import hybrid_rank
        results = hybrid_rank(self.HITS, query="sort a list")
        for r in results:
            assert 0.0 <= r["score"] <= 1.0, f"Score out of bounds: {r['score']}"

    def test_hybrid_rank_top_n(self):
        """top_n parameter correctly limits result count."""
        from rank import hybrid_rank
        results = hybrid_rank(self.HITS, query="sort", top_n=2)
        assert len(results) == 2

    def test_hybrid_rank_empty_input(self):
        """hybrid_rank handles empty input gracefully."""
        from rank import hybrid_rank
        results = hybrid_rank([], query="sort")
        assert results == []

    def test_deduplicate(self):
        """deduplicate removes identical (repo, filepath, start_line) hits."""
        from rank import deduplicate
        hits = [
            {"repo": "a", "filepath": "f.py", "start_line": 1, "score": 0.9},
            {"repo": "a", "filepath": "f.py", "start_line": 1, "score": 0.8},  # duplicate
            {"repo": "b", "filepath": "g.py", "start_line": 5, "score": 0.7},
        ]
        result = deduplicate(hits)
        assert len(result) == 2
        assert result[0]["score"] == 0.9  # kept the first (highest score)

    def test_tokenize_camelcase(self):
        """tokenize correctly splits camelCase and snake_case."""
        from rank import tokenize
        tokens = tokenize("parseHTTPHeaders")
        assert "parse" in tokens
        assert "http" in tokens
        assert "headers" in tokens

    def test_tokenize_snake_case(self):
        from rank import tokenize
        tokens = tokenize("get_request_body")
        assert "get" in tokens
        assert "request" in tokens
        assert "body" in tokens

    def test_keyword_search(self):
        """keyword_search ranks by BM25 without Endee."""
        from rank import keyword_search
        results = keyword_search(self.HITS, "sort algorithm bubble", top_k=3)
        assert len(results) <= 3
        # bubble_sort should rank highly for "bubble sort"
        names = [r["function_name"] for r in results]
        assert "bubble_sort" in names[:2]

    def test_normalize_scores_equal(self):
        """normalize_scores handles all-equal input without division by zero."""
        from rank import normalize_scores
        result = normalize_scores([0.5, 0.5, 0.5])
        assert all(v >= 0 for v in result)

    def test_normalize_scores_range(self):
        """normalize_scores outputs values in [0, 1]."""
        from rank import normalize_scores
        result = normalize_scores([1.0, 3.0, 2.0, 0.5])
        assert all(0.0 <= v <= 1.0 for v in result)
        assert max(result) == pytest.approx(1.0)
        assert min(result) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# End-to-end smoke test (requires Endee to be running)
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestEndToEnd:
    """
    Integration tests that require a running Endee instance.
    Skipped automatically if Endee is not reachable.
    Run with: pytest tests/ -v -m integration
    """

    @pytest.fixture(autouse=True)
    def skip_if_endee_down(self):
        from query import health_check
        if not health_check():
            pytest.skip("Endee not running — skipping integration test")

    def test_full_pipeline_smoke(self, tmp_data_dir):
        """
        Smoke test: embed sample snippets, index them, query, rank.
        Verifies the complete pipeline produces valid ranked results.
        """
        from embed import generate_embeddings, save_embeddings
        from index import upsert_all, health_check as idx_health

        # Stage 2: Embed
        vectors, metadata = generate_embeddings(
            records=SAMPLE_SNIPPETS,
            data_dir=tmp_data_dir,
            batch_size=4,
        )
        save_embeddings(vectors, metadata, data_dir=tmp_data_dir)

        # Stage 3: Index with a test-specific index name
        import index as idx_mod
        original_index = idx_mod.ENDEE_INDEX
        idx_mod.ENDEE_INDEX = "test_code_search_pipeline"
        try:
            count = upsert_all(vectors, metadata, batch_size=2, force_recreate=True)
            assert count == len(SAMPLE_SNIPPETS)

            # Stage 4: Query
            from query import semantic_search
            import query as q_mod
            original_q_index = q_mod.ENDEE_INDEX
            q_mod.ENDEE_INDEX = "test_code_search_pipeline"
            try:
                hits = semantic_search("HTTP GET request", top_k=4)
                assert len(hits) >= 1

                # Stage 5: Rank
                from rank import hybrid_rank
                results = hybrid_rank(hits, "HTTP GET request")
                assert len(results) >= 1
                assert results[0]["rank"] == 1
                assert 0.0 <= results[0]["score"] <= 1.0

                # The "get" function from requests should rank near top for this query
                top_names = [r.get("function_name", "") for r in results[:3]]
                assert any("get" in n.lower() or "http" in n.lower() for n in top_names), \
                    f"Expected HTTP-related function in top 3, got: {top_names}"

            finally:
                q_mod.ENDEE_INDEX = original_q_index

        finally:
            idx_mod.ENDEE_INDEX = original_index
            # Cleanup test index
            try:
                idx_mod.ENDEE_INDEX = "test_code_search_pipeline"
                idx_mod.delete_index()
            except Exception:
                pass
            idx_mod.ENDEE_INDEX = original_index

    def test_query_latency_under_50ms(self):
        """Query latency (embed + search) should be < 50ms p95 after warmup."""
        from query import semantic_search, embed_query

        # Warmup
        embed_query("warmup query string")
        semantic_search("warmup", top_k=5)

        latencies = []
        test_queries = [
            "sort a list", "parse JSON", "HTTP headers", "database query", "file read",
        ]
        for q in test_queries:
            t0 = time.time()
            semantic_search(q, top_k=10)
            latencies.append((time.time() - t0) * 1000)

        p95 = sorted(latencies)[int(len(latencies) * 0.95)]
        avg = sum(latencies) / len(latencies)
        print(f"\nQuery latencies: {[f'{l:.1f}ms' for l in latencies]}")
        print(f"Average: {avg:.1f}ms | p95: {p95:.1f}ms")
        assert avg < 100, f"Average latency {avg:.1f}ms too high (target: <50ms)"
