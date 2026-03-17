"""
Streamlit UI — Endee Code Search
Modern component-based interface. All search logic is unchanged.
"""

# ── Standard library ────────────────────────────────────────────────────────
import os
import time

# ── Streamlit ────────────────────────────────────────────────────────────────
import streamlit as st

# ── Backend (UNCHANGED — do not modify) ─────────────────────────────────────
from query import semantic_search, get_index_stats, health_check
from rank import hybrid_rank

# ── UI components ────────────────────────────────────────────────────────────
from components.header import show_header, show_example_chips
from components.search_input import show_search_bar
from components.result_card import show_result_card, show_no_results
from components.stats_section import show_stats_bar
from components.sidebar_content import show_sidebar
from components.landing_page import show_landing
from utils.ui_helpers import load_css

# ============================================================================
# Page config  (must be first Streamlit call)
# ============================================================================

st.set_page_config(
    page_title="Endee Code Search",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject global CSS theme
load_css()

# ============================================================================
# Session state initialisation  (variable names UNCHANGED)
# ============================================================================

if "search_history" not in st.session_state:
    st.session_state.search_history = []
if "last_latency_ms" not in st.session_state:
    st.session_state.last_latency_ms = None
if "index_stats" not in st.session_state:
    st.session_state.index_stats = {}

# ============================================================================
# Sidebar  — returns all filter / settings values (UNCHANGED names)
# ============================================================================

sidebar = show_sidebar(
    health_check_fn=health_check,
    get_stats_fn=get_index_stats,
)

# Unpack into local names identical to the original app.py
endee_ok      = sidebar.endee_ok
lang_filter   = sidebar.lang_filter
repo_filter   = sidebar.repo_filter
top_k         = sidebar.top_k
vector_weight = sidebar.vector_weight
bm25_weight   = sidebar.bm25_weight
show_scores   = sidebar.show_scores
show_full_code = sidebar.show_full_code
stats         = sidebar.stats

# ============================================================================
# Hero header
# ============================================================================

show_header(endee_ok=endee_ok)

# ============================================================================
# Search bar
# ============================================================================

prefill = st.session_state.pop("prefill_query", "")
query, search_clicked = show_search_bar(prefill=prefill)
show_example_chips()

st.markdown('<hr style="border-color:#1E293B;margin:0.8rem 0 0;">', unsafe_allow_html=True)

# ============================================================================
# Search execution  (ALL LOGIC UNCHANGED — only display calls updated)
# ============================================================================

if search_clicked and query.strip():

    if not endee_ok:
        st.error(
            "Endee server is not running. "
            "Start it with `docker-compose up endee` then re-run the pipeline."
        )
        st.stop()

    # Update search history (UNCHANGED)
    if query not in st.session_state.search_history:
        st.session_state.search_history.append(query)

    with st.spinner(f"Searching for: *{query}*"):
        t_start = time.time()

        # Stage 4: Retrieve from Endee  (UNCHANGED)
        raw_hits = semantic_search(
            query=query,
            top_k=top_k,
            lang_filter=lang_filter if lang_filter != "all" else None,
            repo_filter=repo_filter if repo_filter else None,
        )

        # Stage 5: Hybrid ranking  (UNCHANGED)
        results = hybrid_rank(
            vector_results=raw_hits,
            query=query,
            vector_weight=vector_weight,
            bm25_weight=bm25_weight,
            top_n=top_k,
        )

        latency_ms = (time.time() - t_start) * 1000
        st.session_state.last_latency_ms = latency_ms  # UNCHANGED

    # ── Stats bar ────────────────────────────────────────────────────
    vec_count_display = stats.get(
        "total_elements",
        stats.get("count", stats.get("num_vectors", "—"))
    ) if stats else "—"

    show_stats_bar(
        result_count=len(results),
        latency_ms=latency_ms,
        vec_count=vec_count_display,
    )

    # ── Results ──────────────────────────────────────────────────────
    if not results:
        show_no_results(query=query)
    else:
        for i, result in enumerate(results):
            show_result_card(
                result=result,
                index=i,
                query=query,
                show_scores=show_scores,
                show_full_code=show_full_code,
            )

elif not search_clicked:
    # Landing / idle state
    show_landing()
