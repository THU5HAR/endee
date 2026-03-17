"""
components/sidebar_content.py
Full sidebar: branding, connection status, filters, index stats, search history.

Returns a SidebarState dataclass so app.py can unpack all values cleanly.
All widget keys, defaults, and variable names are identical to the original app.py.
"""

from dataclasses import dataclass, field
from typing import Optional
import os

import streamlit as st

from components.advanced_filters import show_advanced_filters
from utils.ui_helpers import section_title
from styles.colors import PRIMARY, SUCCESS, DANGER, TEXT_SECONDARY


@dataclass
class SidebarState:
    """All values produced by the sidebar — mirrors original app.py locals."""
    endee_ok:       bool
    lang_filter:    str
    repo_filter:    list
    top_k:          int
    vector_weight:  float
    bm25_weight:    float
    show_scores:    bool
    show_full_code: bool
    stats:          dict


def show_sidebar(
    health_check_fn,
    get_stats_fn,
) -> SidebarState:
    """
    Render the full sidebar and return all filter/settings values.

    Parameters
    ----------
    health_check_fn : callable — from query.health_check
    get_stats_fn    : callable — from query.get_index_stats
    """
    with st.sidebar:
        # ── Branding ────────────────────────────────────────────────────
        st.markdown(
            """
            <div style="text-align:center;padding:0.5rem 0 0.8rem;">
                <div style="font-size:1.6rem;font-weight:800;
                            background:linear-gradient(135deg,#0D9488,#F97316);
                            -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                            background-clip:text;">
                    ⟨ Endee ⟩
                </div>
                <p style="font-size:0.72rem;color:#64748B;margin:0.1rem 0 0;">
                    Semantic Code Search Engine
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<hr style="border-color:#334155;margin:0 0 0.8rem;">', unsafe_allow_html=True)

        # ── Connection status ────────────────────────────────────────────
        endee_ok = health_check_fn()
        if endee_ok:
            st.markdown(
                '<div style="background:rgba(16,185,129,0.1);border:1px solid rgba(16,185,129,0.3);'
                'border-radius:8px;padding:6px 12px;font-size:0.82rem;color:#10B981;">'
                '● &nbsp;Endee Connected</div>',
                unsafe_allow_html=True,
            )
        else:
            host = os.getenv("ENDEE_HOST", "localhost")
            port = os.getenv("ENDEE_PORT", "8080")
            st.markdown(
                f'<div style="background:rgba(239,68,68,0.1);border:1px solid rgba(239,68,68,0.3);'
                f'border-radius:8px;padding:6px 12px;font-size:0.82rem;color:#EF4444;">'
                f'● &nbsp;Endee Offline'
                f'<br><span style="color:#64748B;font-size:0.72rem;">'
                f'http://{host}:{port}</span></div>',
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Filters ─────────────────────────────────────────────────────
        section_title("Filters", "🔎")

        lang_filter = st.selectbox(
            "Language",
            options=["all", "python"],
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

        top_k = st.slider(
            "Results per query",
            min_value=5,
            max_value=50,
            value=20,
            step=5,
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Advanced settings ────────────────────────────────────────────
        vector_weight, bm25_weight, show_scores, show_full_code = show_advanced_filters()

        st.markdown('<hr style="border-color:#334155;margin:0.8rem 0;">', unsafe_allow_html=True)

        # ── Index stats ──────────────────────────────────────────────────
        section_title("Index Stats", "📊")

        col_refresh, _ = st.columns([1, 1])
        with col_refresh:
            if st.button("↻ Refresh", use_container_width=True):
                st.session_state.index_stats = get_stats_fn()

        stats = st.session_state.index_stats
        if not stats and endee_ok:
            try:
                st.session_state.index_stats = get_stats_fn()
                stats = st.session_state.index_stats
            except Exception:
                pass

        if stats:
            vec_count = stats.get("total_elements", stats.get("count", stats.get("num_vectors", "—")))
            m1, m2 = st.columns(2)
            m1.metric("Vectors", vec_count)
            m2.metric("Dim", stats.get("dimension", stats.get("dim", 384)))
        else:
            st.caption("Index empty — run the pipeline first")

        if st.session_state.last_latency_ms is not None:
            lat = st.session_state.last_latency_ms
            color = "#10B981" if lat < 50 else ("#F59E0B" if lat < 150 else "#EF4444")
            st.markdown(
                f'<div style="background:rgba(13,148,136,0.08);border-radius:8px;'
                f'padding:6px 12px;font-size:0.82rem;color:#94A3B8;margin-top:0.5rem;">'
                f'Last query: <strong style="color:{color};">{lat:.0f}ms</strong></div>',
                unsafe_allow_html=True,
            )

        st.markdown('<hr style="border-color:#334155;margin:0.8rem 0;">', unsafe_allow_html=True)

        # ── Search history ────────────────────────────────────────────────
        if st.session_state.search_history:
            section_title("Recent Searches", "🕐")
            for past_q in reversed(st.session_state.search_history[-5:]):
                if st.button(
                    f"↩ {past_q[:38]}",
                    key=f"hist_{past_q[:40]}",
                    use_container_width=True,
                ):
                    st.session_state["prefill_query"] = past_q
                    st.rerun()

        # ── Footer ────────────────────────────────────────────────────────
        st.markdown(
            '<p style="font-size:0.68rem;color:#334155;text-align:center;margin-top:2rem;">'
            'Tap Academy Assignment · March 2026</p>',
            unsafe_allow_html=True,
        )

    return SidebarState(
        endee_ok=endee_ok,
        lang_filter=lang_filter,
        repo_filter=repo_filter,
        top_k=top_k,
        vector_weight=vector_weight,
        bm25_weight=bm25_weight,
        show_scores=show_scores,
        show_full_code=show_full_code,
        stats=stats,
    )
