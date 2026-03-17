"""
components/landing_page.py
Idle / landing state shown when no search has been run yet.
"""

import streamlit as st
from components.stats_section import show_landing_stats
from styles.colors import PRIMARY, SECONDARY, ACCENT


def show_landing() -> None:
    """Render the landing page content."""

    # ── How it works ─────────────────────────────────────────────────
    st.markdown(
        '<p style="font-size:0.78rem;font-weight:700;color:#64748B;'
        'text-transform:uppercase;letter-spacing:1px;margin-bottom:0.8rem;">'
        '⚡ How it works</p>',
        unsafe_allow_html=True,
    )
    steps = [
        ("Type", "Describe what code you're looking for in plain English"),
        ("Filter", "Narrow results by language or a specific repository"),
        ("Search", "Your query is matched against 10,000+ indexed functions"),
        ("Rank", "Results are sorted by combining meaning similarity and keyword match"),
    ]
    for i, (title, desc) in enumerate(steps, 1):
        st.markdown(
            f'<div class="ecs-step">'
            f'<div class="ecs-step-num">{i}</div>'
            f'<div class="ecs-step-text"><strong>{title}</strong> — {desc}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Pipeline diagram ─────────────────────────────────────────────
    st.markdown(
        '<p style="font-size:0.78rem;font-weight:700;color:#64748B;'
        'text-transform:uppercase;letter-spacing:1px;margin-bottom:0.5rem;">'
        '🔗 Pipeline</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="ecs-pipeline">
            <span class="ecs-pipeline-node">Your Query</span>
            <span class="ecs-pipeline-arrow"> → </span>
            <span style="color:#94A3B8;">Converted to a number vector</span>
            <span class="ecs-pipeline-arrow"> → </span>
            <span style="color:#94A3B8;">Closest matches found in Endee</span>
            <span class="ecs-pipeline-arrow"> → </span>
            <span style="color:#94A3B8;">Re-sorted by keyword relevance</span>
            <span class="ecs-pipeline-arrow"> → </span>
            <span class="ecs-pipeline-node">Results</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Benchmark cards ───────────────────────────────────────────────
    st.markdown(
        '<p style="font-size:0.78rem;font-weight:700;color:#64748B;'
        'text-transform:uppercase;letter-spacing:1px;margin-bottom:0.8rem;">'
        '📊 Benchmark targets</p>',
        unsafe_allow_html=True,
    )
    show_landing_stats()

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Setup snippet ─────────────────────────────────────────────────
    st.markdown(
        '<p style="font-size:0.78rem;font-weight:700;color:#64748B;'
        'text-transform:uppercase;letter-spacing:1px;margin-bottom:0.5rem;">'
        '🚀 First-time setup</p>',
        unsafe_allow_html=True,
    )
    st.code(
        "docker-compose up -d endee\n"
        "python ingest.py      # Clone & parse 10 GitHub repos\n"
        "python embed.py       # Generate 384-dim embeddings\n"
        "python index.py       # Upload to Endee\n"
        "streamlit run app.py  # Launch UI",
        language="bash",
    )
