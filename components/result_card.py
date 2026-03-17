"""
components/result_card.py
Renders a single search result as a modern dark card with badges,
syntax-highlighted code, and copy/expand actions.

All result field access is identical to the original render_result_card().
"""

import streamlit as st
from styles.colors import PRIMARY, SECONDARY, score_color, lang_color
from utils.ui_helpers import score_badge, repo_badge, lang_badge, rank_badge, inline_badges
from utils.formatting import (
    syntax_lang, truncate_code, preview_code,
    clean_docstring, filepath_display, score_to_pct,
    highlight_query_in_text, repo_display,
)


def show_result_card(
    result: dict,
    index: int,
    query: str = "",
    show_scores: bool = False,
    show_full_code: bool = True,
) -> None:
    """
    Display one result as a styled card inside a Streamlit expander.

    Parameters match the original render_result_card() call signature
    (plus query for optional term highlighting).
    """
    # ── Extract fields (same as original) ──────────────────────────────
    rank       = result.get("rank", index + 1)
    score      = float(result.get("score", 0))
    repo       = result.get("repo", "unknown")
    filepath   = result.get("filepath", "")
    start_line = result.get("start_line", 0)
    end_line   = result.get("end_line", 0)
    func_name  = result.get("function_name", "unknown")
    language   = result.get("language", "python")
    docstring  = result.get("docstring", "")
    code       = result.get("code", "")

    # ── Formatted values ────────────────────────────────────────────────
    display_repo  = repo_display(repo)
    display_path  = filepath_display(filepath, start_line, end_line)
    display_doc   = clean_docstring(docstring, max_len=180)
    code_lang     = syntax_lang(language)
    border_color  = score_color(score)

    # ── Expander label ──────────────────────────────────────────────────
    label = f"#{rank}  {func_name}  ·  {display_repo}  ·  {score_to_pct(score)}"

    with st.expander(label, expanded=(rank <= 3)):

        # ── Card header: wrapper + badges + filepath (single HTML block) ─
        badges_html = inline_badges(
            rank_badge(rank),
            score_badge(score),
            repo_badge(display_repo),
            lang_badge(language),
        )
        card_header_html = (
            f'<div class="ecs-card" style="border-left-color:{border_color};">'
            f'{badges_html}'
            f'<p class="ecs-filepath" style="margin:0.35rem 0 0;">📄 {display_path}</p>'
        )
        st.markdown(card_header_html, unsafe_allow_html=True)

        # ── Docstring ───────────────────────────────────────────────────
        if display_doc:
            highlighted_doc = highlight_query_in_text(display_doc, query) if query else display_doc
            st.markdown(
                f'<p class="ecs-docstring">{highlighted_doc}</p>',
                unsafe_allow_html=True,
            )

        # ── Optional copy button aligned at bottom of card ──────────────
        col_copy, _ = st.columns([1, 6])
        with col_copy:
            copy_key = f"copy_{result.get('id', rank)}_{index}"
            if st.button("📋 Copy", key=copy_key, help="Copy code to clipboard"):
                st.session_state[f"copied_{copy_key}"] = True

        if st.session_state.get(f"copied_copy_{result.get('id', rank)}_{index}"):
            st.markdown(
                '<span class="ecs-copy-success">✓ Copied!</span>',
                unsafe_allow_html=True,
            )

        # Close the card wrapper after all header/body controls
        st.markdown('</div>', unsafe_allow_html=True)

        # ── Score breakdown (optional) ──────────────────────────────────
        if show_scores:
            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("Hybrid", f"{result.get('score', 0):.3f}")
            sc2.metric("Vector", f"{result.get('vector_score', 0):.3f}")
            sc3.metric("BM25",   f"{result.get('bm25_score', 0):.3f}")

        # ── Code block ──────────────────────────────────────────────────
        if code:
            if show_full_code:
                display_code = truncate_code(code, max_chars=3000)
            else:
                display_code = preview_code(code, max_chars=300)
            st.code(display_code, language=code_lang)


def show_no_results(query: str = "") -> None:
    """Display the empty-state when search returns nothing."""
    st.markdown(
        f"""
        <div class="ecs-no-results">
            <div class="ecs-no-results-icon">🔍</div>
            <h3>No results found</h3>
            <p>
                No functions matched <strong style="color:#F97316;">"{query}"</strong>.<br>
                Try rephrasing your query, broadening the filters,
                or check that the index is populated:
            </p>
            <code style="background:#1E293B;border:1px solid #334155;border-radius:6px;
                         padding:6px 14px;color:#94A3B8;font-size:0.82rem;">
                python ingest.py &amp;&amp; python embed.py &amp;&amp; python index.py
            </code>
        </div>
        """,
        unsafe_allow_html=True,
    )
