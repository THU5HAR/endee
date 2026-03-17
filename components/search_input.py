"""
components/search_input.py
Enhanced search bar with Search + Clear buttons.
Returns (query_text, search_clicked) — identical to what app.py currently produces.
"""

import streamlit as st
from styles.colors import PRIMARY


def show_search_bar(prefill: str = "") -> tuple[str, bool]:
    """
    Render the search text input and action buttons.

    Returns
    -------
    (query, search_clicked)
        query         — current text in the input field
        search_clicked — True if the Search button was pressed this run
    """
    query = st.text_input(
        label="Search query",
        value=prefill,
        placeholder='Describe the code you need, e.g. "sort a list of numbers"',
        label_visibility="collapsed",
        key="main_search_input",
    )

    col_search, col_clear, col_spacer = st.columns([1, 1, 8])
    with col_search:
        search_clicked = st.button(
            "Search",
            type="primary",
            use_container_width=True,
            help="Run semantic search",
        )

    # Allow other components (e.g. example chips) to trigger a search
    if "trigger_search" in st.session_state:
        search_clicked = search_clicked or bool(st.session_state.pop("trigger_search"))
    with col_clear:
        if st.button("Clear", use_container_width=True, help="Clear search and reset"):
            st.rerun()

    return query, search_clicked


def show_keyboard_hint() -> None:
    """Small tip shown below the search bar."""
    st.markdown(
        '<p style="font-size:0.75rem;color:#475569;margin-top:0.2rem;">'
        'Press <kbd style="background:#1E293B;border:1px solid #334155;border-radius:4px;'
        'padding:1px 5px;font-size:0.7rem;color:#94A3B8;">Enter</kbd> or click '
        '<strong style="color:#0D9488;">Search</strong> to find code</p>',
        unsafe_allow_html=True,
    )
