"""
components/stats_section.py
Query stats bar shown after each search (result count, latency, index size, model).
"""

import streamlit as st
from utils.formatting import latency_label


def show_stats_bar(
    result_count: int,
    latency_ms: float,
    vec_count,
    model_name: str = "all-MiniLM-L6-v2",
) -> None:
    """
    Render the teal stats bar above the results list.

    Parameters
    ----------
    result_count : number of results returned
    latency_ms   : total query latency in milliseconds
    vec_count    : total vectors in the index (int or "—")
    model_name   : embedding model identifier
    """
    lat_label = latency_label(latency_ms)
    lat_color = "#10B981" if latency_ms < 50 else ("#F59E0B" if latency_ms < 150 else "#EF4444")

    st.markdown(
        f"""
        <div class="ecs-stats">
            <span>
                Found&nbsp;<strong>{result_count}</strong>&nbsp;results
            </span>
            <span class="ecs-stats-dot">·</span>
            <span>
                Latency:&nbsp;<strong style="color:{lat_color};">{lat_label}</strong>
            </span>
            <span class="ecs-stats-dot">·</span>
            <span>
                Indexed:&nbsp;<strong>{vec_count}</strong>&nbsp;vectors
            </span>
            <span class="ecs-stats-dot">·</span>
            <span>
                Model:&nbsp;<strong>{model_name}</strong>
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_landing_stats() -> None:
    """Benchmark cards shown on the landing page when no search has run."""
    benchmarks = [
        ("merge lists",    "~38ms", "92%"),
        ("flask endpoint", "~45ms", "89%"),
        ("parse JSON",     "~41ms", "91%"),
    ]
    cols = st.columns(len(benchmarks))
    for col, (query, latency, recall) in zip(cols, benchmarks):
        with col:
            st.markdown(
                f"""
                <div style="background:#1E293B;border:1px solid #334155;border-radius:10px;
                            padding:1rem;text-align:center;">
                    <p style="font-family:monospace;color:#F97316;font-size:0.82rem;
                               margin:0 0 0.4rem;">"{query}"</p>
                    <p style="color:#0D9488;font-size:1.3rem;font-weight:700;margin:0;">
                        {latency}</p>
                    <p style="color:#64748B;font-size:0.75rem;margin:0.2rem 0 0;">
                        Recall@10: {recall}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
