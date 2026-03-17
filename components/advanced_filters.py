"""
components/advanced_filters.py
Advanced settings expander rendered inside the sidebar.
Returns (vector_weight, bm25_weight, show_scores, show_full_code).
All widget keys and default values match the original app.py exactly.
"""

import streamlit as st


def show_advanced_filters() -> tuple[float, float, bool, bool]:
    """
    Render the Advanced settings expander.

    Returns
    -------
    (vector_weight, bm25_weight, show_scores, show_full_code)
    """
    with st.expander("⚙️ Advanced Settings"):
        st.markdown(
            '<p style="font-size:0.75rem;color:#64748B;margin-bottom:0.6rem;">'
            'Tune ranking and display options</p>',
            unsafe_allow_html=True,
        )

        vector_weight = st.slider(
            "Vector score weight",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.05,
            help="Higher = more weight on semantic similarity",
        )
        bm25_weight = round(1.0 - vector_weight, 2)
        st.markdown(
            f'<p style="font-size:0.75rem;color:#64748B;margin-top:-0.5rem;">'
            f'BM25 keyword weight: <strong style="color:#0D9488;">{bm25_weight}</strong></p>',
            unsafe_allow_html=True,
        )

        st.markdown('<hr style="border-color:#334155;margin:0.6rem 0;">', unsafe_allow_html=True)

        show_scores = st.checkbox(
            "Show score breakdown",
            value=False,
            help="Display individual vector and BM25 scores per result",
        )
        show_full_code = st.checkbox(
            "Show full function code",
            value=True,
            help="Uncheck for preview-only (first 300 chars)",
        )

    return vector_weight, bm25_weight, show_scores, show_full_code
