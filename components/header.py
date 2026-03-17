"""
components/header.py
Gradient hero header with title, subtitle, and status pill.
"""

import streamlit as st
from styles.colors import PRIMARY, SECONDARY, ACCENT


def show_header(endee_ok: bool = True) -> None:
    """Render the full gradient hero header."""
    status_color = "#10B981" if endee_ok else "#EF4444"
    status_dot   = "●" if endee_ok else "●"
    status_label = "Connected" if endee_ok else "Offline"

    st.markdown(
        f"""
        <div class="ecs-hero">
            <div class="ecs-hero-badge">{status_dot}&nbsp;Endee&nbsp;{status_label}</div>
            <div class="ecs-hero-title">🔍 Endee Code Search</div>
            <p class="ecs-hero-sub">
                Semantic search over 10,000+ functions from top Python
                repositories &mdash; powered by&nbsp;
                <span style="color:{ACCENT};font-weight:600;">Endee Vector DB</span>
                &nbsp;+&nbsp;
                <span style="color:{SECONDARY};font-weight:600;">Hybrid BM25 Ranking</span>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_example_chips() -> None:
    """Render clickable example query chips (same look as before, now real buttons)."""
    examples = [
        "merge two sorted lists",
        "parse HTTP headers",
        "flask route decorator",
        "read csv with pandas",
        "binary search tree insert",
        "database query with filter",
    ]

    # One row: hidden marker for CSS, "Try:" label, then chip buttons (same layout as before)
    st.markdown(
        '<div id="ecs-chip-row" style="display:none" aria-hidden="true"></div>',
        unsafe_allow_html=True,
    )
    cols = st.columns([1] + [2] * len(examples))  # narrow col for "Try:", rest for chips
    with cols[0]:
        st.markdown(
            '<span style="color:#64748B;font-size:0.78rem;line-height:2.4;">Try:</span>',
            unsafe_allow_html=True,
        )
    for col, ex in zip(cols[1:], examples):
        with col:
            if st.button(ex, key=f"chip_{ex}", use_container_width=True):
                st.session_state["prefill_query"] = ex
                st.session_state["trigger_search"] = True
                st.rerun()
