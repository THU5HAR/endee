"""
UI helper utilities — loading CSS, injecting HTML, badges, etc.
Pure display helpers; no search logic.
"""

import streamlit as st
from styles.colors import (
    PRIMARY, SECONDARY, SUCCESS, WARNING, DANGER,
    TEXT_SECONDARY, BORDER, BG_CARD,
    score_color, lang_color,
)


def load_css() -> None:
    """
    Inject the global dark theme CSS.
    CSS is inlined as a string (most reliable method across Streamlit versions).
    """
    st.markdown("""
<style>
/* ── Reset & Base ── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0F172A !important;
    color: #F1F5F9 !important;
}
/* ── Hide Streamlit chrome (keep sidebar toggle visible) ── */
#MainMenu, footer { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }
/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #1E293B !important;
    border-right: 1px solid #334155 !important;
}
[data-testid="stSidebar"] * { color: #F1F5F9 !important; }
/* ── Main container ── */
[data-testid="stMainBlockContainer"] {
    padding-top: 1.5rem !important;
    max-width: 1200px !important;
}
/* ── Gradient hero header ── */
.ecs-hero {
    background: linear-gradient(135deg, #0D9488 0%, #0F172A 50%, #F97316 100%);
    border-radius: 16px;
    padding: 2.5rem 2rem 2rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.ecs-hero::before {
    content: '';
    position: absolute;
    top: -40%; right: -10%;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(6,182,212,0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.ecs-hero-title {
    font-size: 2.4rem; font-weight: 700; color: #FFFFFF;
    margin: 0 0 0.4rem; letter-spacing: -0.5px;
}
.ecs-hero-sub { font-size: 1rem; color: rgba(241,245,249,0.75); margin: 0; }
.ecs-hero-badge {
    display: inline-block;
    background: rgba(13,148,136,0.25); border: 1px solid rgba(13,148,136,0.5);
    color: #5EEAD4; font-size: 0.72rem; font-weight: 600;
    padding: 2px 10px; border-radius: 20px; margin-bottom: 0.8rem;
    letter-spacing: 0.5px; text-transform: uppercase;
}
/* ── Search bar ── */
[data-testid="stTextInput"] input {
    background: #1E293B !important; border: 2px solid #334155 !important;
    border-radius: 12px !important; color: #F1F5F9 !important;
    font-size: 1.05rem !important; padding: 0.75rem 1rem !important;
    transition: border-color 0.2s ease !important;
}
[data-testid="stTextInput"] input:focus {
    border-color: #0D9488 !important;
    box-shadow: 0 0 0 3px rgba(13,148,136,0.2) !important;
}
[data-testid="stTextInput"] input::placeholder { color: #64748B !important; }
/* ── Buttons ── */
[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(135deg, #0D9488, #0891B2) !important;
    border: none !important; border-radius: 10px !important;
    color: white !important; font-weight: 600 !important;
    transition: all 0.2s ease !important;
}
[data-testid="stButton"] > button[kind="primary"]:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(13,148,136,0.35) !important;
}
[data-testid="stButton"] > button[kind="secondary"] {
    background: #1E293B !important; border: 1px solid #334155 !important;
    border-radius: 10px !important; color: #94A3B8 !important;
}
/* ── Example chips: same pill look as before, clickable ── */
[data-testid="stVerticalBlock"]:has(#ecs-chip-row) > [data-testid="stHorizontalBlock"] button {
    background: rgba(13,148,136,0.12) !important;
    border: 1px solid rgba(13,148,136,0.3) !important;
    color: #5EEAD4 !important;
    border-radius: 20px !important;
    padding: 3px 12px !important;
    font-size: 0.78rem !important;
    white-space: nowrap !important;
}
[data-testid="stVerticalBlock"]:has(#ecs-chip-row) > [data-testid="stHorizontalBlock"] button:hover {
    background: rgba(13,148,136,0.22) !important;
    border-color: rgba(13,148,136,0.5) !important;
}
/* ── Result card ── */
.ecs-card {
    background: #1E293B; border: 1px solid #334155;
    border-left: 4px solid #F97316; border-radius: 12px;
    padding: 1.2rem 1.4rem; margin-bottom: 0.85rem;
    transition: border-color 0.2s, box-shadow 0.2s;
}
.ecs-card:hover {
    border-left-color: #0D9488;
    box-shadow: 0 4px 20px rgba(13,148,136,0.15);
}
.ecs-func-name {
    font-family: monospace; font-size: 1rem; font-weight: 600; color: #F97316;
}
.ecs-filepath { font-family: monospace; font-size: 0.75rem; color: #64748B; }
.ecs-docstring {
    font-size: 0.88rem; color: #94A3B8; font-style: italic;
    margin: 0.35rem 0 0.6rem; line-height: 1.5;
}
/* ── Badges ── */
.ecs-badge {
    display: inline-block; font-size: 0.72rem; font-weight: 600;
    padding: 2px 9px; border-radius: 20px; white-space: nowrap;
}
/* ── Stats bar ── */
.ecs-stats {
    display: flex; gap: 1.5rem; flex-wrap: wrap;
    background: rgba(13,148,136,0.08); border: 1px solid rgba(13,148,136,0.2);
    border-radius: 10px; padding: 0.7rem 1.2rem; margin-bottom: 1.2rem;
    font-size: 0.85rem; color: #94A3B8;
}
.ecs-stats strong { color: #0D9488; font-weight: 600; }
.ecs-stats-dot { color: #334155; }
/* ── No results ── */
.ecs-no-results { text-align: center; padding: 4rem 2rem; }
.ecs-no-results-icon { font-size: 3rem; margin-bottom: 1rem; }
.ecs-no-results h3 { color: #F1F5F9; font-size: 1.3rem; margin-bottom: 0.5rem; }
.ecs-no-results p { color: #64748B; font-size: 0.9rem; }
/* ── Metrics ── */
[data-testid="stMetricValue"] { color: #0D9488 !important; font-weight: 700 !important; }
[data-testid="stMetricLabel"] { color: #94A3B8 !important; }
/* ── Pipeline diagram ── */
.ecs-pipeline {
    background: #1E293B; border: 1px solid #334155; border-radius: 12px;
    padding: 1.2rem 1.5rem; font-family: monospace; font-size: 0.82rem;
    color: #94A3B8; margin: 0.8rem 0; overflow-x: auto; white-space: nowrap;
}
.ecs-pipeline-arrow { color: #0D9488; font-weight: 700; }
.ecs-pipeline-node  { color: #F97316; }
/* ── How-it-works steps ── */
.ecs-step { display: flex; align-items: flex-start; gap: 1rem; margin-bottom: 0.9rem; }
.ecs-step-num {
    background: linear-gradient(135deg, #0D9488, #06B6D4); color: white;
    width: 28px; height: 28px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.8rem; font-weight: 700; flex-shrink: 0;
}
.ecs-step-text { color: #94A3B8; font-size: 0.9rem; padding-top: 4px; }
.ecs-step-text strong { color: #F1F5F9; }
/* ── Copy toast ── */
.ecs-copy-success {
    background: rgba(16,185,129,0.15); border: 1px solid rgba(16,185,129,0.3);
    color: #10B981; border-radius: 6px; padding: 4px 12px; font-size: 0.78rem;
}
/* ── Expander tweaks ── */
[data-testid="stExpander"] {
    background: #1E293B !important; border: 1px solid #334155 !important;
    border-radius: 10px !important;
}
/* ── Dividers ── */
hr { border-color: #334155 !important; }
[data-testid="stSidebar"] [data-testid="stMetricValue"] { color: #0D9488 !important; }
</style>
""", unsafe_allow_html=True)


def badge(text: str, color: str, bg_alpha: float = 0.15) -> str:
    """Return an HTML inline badge span."""
    return (
        f'<span style="display:inline-block;font-size:0.72rem;font-weight:600;'
        f'padding:2px 9px;border-radius:20px;letter-spacing:0.2px;white-space:nowrap;'
        f'background:rgba({_hex_to_rgb(color)},{bg_alpha});'
        f'color:{color};border:1px solid rgba({_hex_to_rgb(color)},0.3);">'
        f'{text}</span>'
    )


def score_badge(score: float) -> str:
    color = score_color(score)
    pct = int(score * 100)
    return badge(f"★ {pct}%", color)


def repo_badge(repo: str) -> str:
    return badge(f"◈ {repo}", "#A5B4FC")


def lang_badge(lang: str) -> str:
    color = lang_color(lang)
    icons = {"python": "🐍", "javascript": "⚡", "typescript": "📘"}
    icon = icons.get(lang.lower() if lang else "", "◆")
    return badge(f"{icon} {lang}", color)


def rank_badge(rank: int) -> str:
    colors = {1: SECONDARY, 2: "#94A3B8", 3: WARNING}
    color = colors.get(rank, "#475569")
    return badge(f"#{rank}", color)


def inline_badges(*badges_html: str) -> str:
    """Join multiple badge HTML strings with a small gap."""
    return "&nbsp;".join(badges_html)


def card_html(content: str, border_color: str = SECONDARY) -> str:
    """Wrap content in a styled card div."""
    return (
        f'<div class="ecs-card" style="border-left-color:{border_color};">'
        f'{content}'
        f'</div>'
    )


def _hex_to_rgb(hex_color: str) -> str:
    """Convert '#RRGGBB' to 'R,G,B' string for rgba() usage."""
    h = hex_color.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"{r},{g},{b}"


def horizontal_rule() -> None:
    st.markdown('<hr style="border-color:#334155;margin:1rem 0;">', unsafe_allow_html=True)


def section_title(title: str, icon: str = "") -> None:
    prefix = f"{icon}&nbsp;" if icon else ""
    st.markdown(
        f'<p style="font-size:0.78rem;font-weight:700;color:#64748B;'
        f'text-transform:uppercase;letter-spacing:1px;margin-bottom:0.5rem;">'
        f'{prefix}{title}</p>',
        unsafe_allow_html=True,
    )
