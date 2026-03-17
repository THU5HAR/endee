"""
Streamlit UI — Endee Code Search
Full-featured search interface with syntax highlighting, filters, and stats.
"""

import json
import os
import time
from pathlib import Path
from typing import Optional

import streamlit as st

from query import semantic_search, get_index_stats, health_check
from rank import hybrid_rank

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Endee Code Search",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1f1f2e;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #6b7280;
        margin-bottom: 1.5rem;
    }
    .score-badge {
        background: #10b981;
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .repo-badge {
        background: #6366f1;
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.8rem;
    }
    .lang-badge {
        background: #f59e0b;
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.8rem;
    }
    .stats-bar {
        background: #f3f4f6;
        padding: 8px 16px;
        border-radius: 6px;
        font-size: 0.85rem;
        color: #374151;
        margin-bottom: 1rem;
    }
    .no-results {
        text-align: center;
        padding: 3rem;
        color: #9ca3af;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Language map constant
# ---------------------------------------------------------------------------

LANG_MAP = {"python": "python", "javascript": "javascript", "typescript": "typescript"}


# ---------------------------------------------------------------------------
# Result card renderer
# ---------------------------------------------------------------------------

def render_result_card(result: dict, show_scores: bool = False, show_full_code: bool = True) -> None:
    rank = result.get("rank", "?")
    score = result.get("score", 0)
    repo = result.get("repo", "unknown")
    filepath = result.get("filepath", "")
    start_line = result.get("start_line", "?")
    end_line = result.get("end_line", "?")
    func_name = result.get("function_name", "unknown")
    language = result.get("language", "python")
    docstring = result.get("docstring", "")
    code = result.get("code", "")

    with st.expander(
        f"#{rank}  {func_name}  |  {repo}  |  score: {score:.3f}",
        expanded=(rank <= 3),
    ):
        col_meta, col_copy = st.columns([5, 1])
        with col_meta:
            st.markdown(
                f'<span class="score-badge">score: {score:.3f}</span>&nbsp;'
                f'<span class="repo-badge">{repo}</span>&nbsp;'
                f'<span class="lang-badge">{language}</span>',
                unsafe_allow_html=True,
            )
            st.caption(f"`{filepath}` — lines {start_line}–{end_line}")
            if docstring:
                st.markdown(f"*{docstring[:200]}*")

        with col_copy:
            st.button("📋 Copy", key=f"copy_{result.get('id', rank)}_{rank}",
                      help="Copy code snippet to clipboard")

        if show_scores:
            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("Hybrid Score", f"{result.get('score', 0):.3f}")
            sc2.metric("Vector Score", f"{result.get('vector_score', 0):.3f}")
            sc3.metric("BM25 Score", f"{result.get('bm25_score', 0):.3f}")

        if code:
            code_lang = LANG_MAP.get(language, "python")
            if show_full_code:
                st.code(code[:3000], language=code_lang)
            else:
                preview = code[:300] + ("..." if len(code) > 300 else "")
                st.code(preview, language=code_lang)


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

if "search_history" not in st.session_state:
    st.session_state.search_history = []
if "last_latency_ms" not in st.session_state:
    st.session_state.last_latency_ms = None
if "index_stats" not in st.session_state:
    st.session_state.index_stats = {}


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## Endee Code Search")
    st.markdown("*Semantic search over indexed GitHub repos*")
    st.divider()

    # Connection status
    endee_ok = health_check()
    if endee_ok:
        st.success("Endee Connected")
    else:
        st.error("Endee Offline")
        host = os.getenv("ENDEE_HOST", "localhost")
        port = os.getenv("ENDEE_PORT", "8080")
        st.caption(f"Expected: http://{host}:{port}")

    # Filters
    st.markdown("### Filters")
    lang_filter = st.selectbox(
        "Language",
        options=["all", "python", "javascript"],
        index=0,
        help="Filter results by programming language",
    )

    repo_options = [
        "scikit-learn", "fastapi", "requests", "flask", "numpy",
        "pandas", "django", "sqlalchemy", "pydantic", "httpx",
    ]
    repo_filter = st.multiselect(
        "Repositories",
        options=repo_options,
        default=[],
        help="Leave empty to search all repos",
    )

    top_k = st.slider("Results per query", min_value=5, max_value=50, value=20, step=5)

    # Advanced options
    with st.expander("Advanced"):
        vector_weight = st.slider("Vector score weight", 0.0, 1.0, 0.7, 0.05)
        bm25_weight = round(1.0 - vector_weight, 2)
        st.caption(f"BM25 weight: {bm25_weight}")
        show_scores = st.checkbox("Show score breakdown", value=False)
        show_full_code = st.checkbox("Show full function code", value=True)

    st.divider()

    # Index stats
    st.markdown("### Index Stats")
    if st.button("Refresh Stats"):
        st.session_state.index_stats = get_index_stats()

    stats = st.session_state.index_stats
    if not stats and endee_ok:
        try:
            st.session_state.index_stats = get_index_stats()
            stats = st.session_state.index_stats
        except Exception:
            pass

    if stats:
        vec_count = stats.get("count", stats.get("num_vectors", "—"))
        st.metric("Vectors Indexed", vec_count)
        st.metric("Dimension", stats.get("dim", "384"))
    else:
        st.caption("Stats unavailable — index may be empty")

    if st.session_state.last_latency_ms is not None:
        st.metric("Last Query Latency", f"{st.session_state.last_latency_ms:.0f}ms")

    st.divider()

    # Search history
    if st.session_state.search_history:
        st.markdown("### Recent Searches")
        for past_q in reversed(st.session_state.search_history[-5:]):
            if st.button(past_q[:40], key=f"hist_{past_q[:40]}"):
                st.session_state["prefill_query"] = past_q
                st.rerun()


# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------

st.markdown('<div class="main-header">🔍 Endee Code Search</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Search 10,000+ functions from top Python & JavaScript repos using natural language — powered by Endee Vector DB</div>',
    unsafe_allow_html=True,
)

# Search bar
prefill = st.session_state.pop("prefill_query", "")
query = st.text_input(
    label="Search query",
    value=prefill,
    placeholder='Try: "sort a list", "flask route decorator", "parse JSON response"',
    label_visibility="collapsed",
)

col_search, col_clear, col_spacer = st.columns([1, 1, 8])
with col_search:
    search_clicked = st.button("Search", type="primary", use_container_width=True)
with col_clear:
    if st.button("Clear", use_container_width=True):
        st.rerun()

st.markdown(
    "**Examples:** `merge two sorted lists` · `parse HTTP headers` · "
    "`flask endpoint decorator` · `read csv with pandas` · `binary search tree insert`"
)

st.divider()

# ---------------------------------------------------------------------------
# Search execution
# ---------------------------------------------------------------------------

if search_clicked and query.strip():
    if not endee_ok:
        st.error(
            "Endee server is not running. "
            "Start it with `docker-compose up endee` then re-run the pipeline."
        )
        st.stop()

    # Update search history
    if query not in st.session_state.search_history:
        st.session_state.search_history.append(query)

    with st.spinner(f"Searching: *{query}*"):
        t_start = time.time()

        # Stage 4: Retrieve from Endee
        raw_hits = semantic_search(
            query=query,
            top_k=top_k,
            lang_filter=lang_filter if lang_filter != "all" else None,
            repo_filter=repo_filter if repo_filter else None,
        )

        # Stage 5: Hybrid ranking
        results = hybrid_rank(
            vector_results=raw_hits,
            query=query,
            vector_weight=vector_weight,
            bm25_weight=bm25_weight,
            top_n=top_k,
        )

        latency_ms = (time.time() - t_start) * 1000
        st.session_state.last_latency_ms = latency_ms

    # Stats bar
    vec_count_display = stats.get("count", stats.get("num_vectors", "—")) if stats else "—"
    st.markdown(
        f'<div class="stats-bar">'
        f'Found <strong>{len(results)}</strong> results &nbsp;·&nbsp; '
        f'Latency: <strong>{latency_ms:.0f}ms</strong> &nbsp;·&nbsp; '
        f'Indexed: <strong>{vec_count_display}</strong> vectors &nbsp;·&nbsp; '
        f'Model: <strong>all-MiniLM-L6-v2</strong>'
        f'</div>',
        unsafe_allow_html=True,
    )

    if not results:
        st.markdown(
            '<div class="no-results">'
            '<p style="font-size:1.5rem">No results found</p>'
            '<p>Try rephrasing your query, or check that the index has been populated '
            '(run <code>python ingest.py && python embed.py && python index.py</code>)</p>'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        for result in results:
            render_result_card(result, show_scores=show_scores, show_full_code=show_full_code)

elif not search_clicked:
    # Landing / idle state
    st.markdown("""
    ### How it works
    1. **Type** a natural language description of the code you need
    2. **Filter** by language (Python / JavaScript) or specific repositories
    3. **Click Search** — the engine embeds your query and finds semantically similar functions

    ### Pipeline
    ```
    Your Query → all-MiniLM-L6-v2 embed → Endee ANN search (top-20) → Hybrid BM25+Vector re-rank → Results
    ```

    ### First-time setup
    ```bash
    # Start Endee vector DB
    docker-compose up -d endee

    # Build the search index (~5-10 min, ~10k functions)
    python ingest.py      # Clone & parse 10 GitHub repos
    python embed.py       # Generate 384-dim embeddings
    python index.py       # Upload to Endee

    # Launch this UI
    streamlit run app.py
    ```

    ### Benchmark targets
    | Query | Latency | Recall@10 |
    |-------|---------|-----------|
    | "merge lists" | ~38ms | 92% |
    | "flask endpoint" | ~45ms | 89% |
    | "parse JSON" | ~41ms | 91% |
    """)
