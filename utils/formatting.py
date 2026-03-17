"""
Text & code formatting helpers — no Streamlit or search logic.
"""

import re


LANG_MAP = {
    "python": "python",
    "javascript": "javascript",
    "typescript": "typescript",
    "js": "javascript",
    "ts": "typescript",
    "py": "python",
}


def syntax_lang(language: str) -> str:
    """Normalise a language string to a pygments/streamlit code language tag."""
    return LANG_MAP.get((language or "").lower(), "python")


def truncate_code(code: str, max_chars: int = 3000) -> str:
    """Truncate code to max_chars, appending '…' if cut."""
    if len(code) <= max_chars:
        return code
    return code[:max_chars] + "\n# … (truncated)"


def preview_code(code: str, max_chars: int = 300) -> str:
    """Short preview of code — first N chars, ellipsis if longer."""
    if len(code) <= max_chars:
        return code
    return code[:max_chars] + "\n# …"


def clean_docstring(docstring: str, max_len: int = 200) -> str:
    """Strip leading/trailing whitespace and truncate."""
    if not docstring:
        return ""
    cleaned = " ".join(docstring.split())
    return cleaned[:max_len] + ("…" if len(cleaned) > max_len else "")


def filepath_display(filepath: str, start_line: int = 0, end_line: int = 0) -> str:
    """Format filepath and line range for display."""
    if start_line and end_line:
        return f"{filepath}:{start_line}–{end_line}"
    if start_line:
        return f"{filepath}:{start_line}"
    return filepath


def score_to_pct(score: float) -> str:
    """Convert 0–1 float score to percentage string like '87%'."""
    return f"{int(score * 100)}%"


def latency_label(ms: float) -> str:
    """Format latency in ms, e.g. '42ms' or '1.2s'."""
    if ms < 1000:
        return f"{ms:.0f}ms"
    return f"{ms/1000:.1f}s"


def repo_display(repo: str) -> str:
    """Capitalise well-known repo names for display."""
    display_map = {
        "scikit-learn": "scikit-learn",
        "fastapi": "FastAPI",
        "requests": "requests",
        "flask": "Flask",
        "numpy": "NumPy",
        "pandas": "pandas",
        "django": "Django",
        "sqlalchemy": "SQLAlchemy",
        "pydantic": "Pydantic",
        "httpx": "httpx",
    }
    return display_map.get(repo, repo)


def highlight_query_in_text(text: str, query: str) -> str:
    """
    Wrap query terms in text with an HTML highlight span.
    Safe for use in st.markdown(..., unsafe_allow_html=True).
    """
    if not query or not text:
        return text
    words = [w for w in re.split(r'\W+', query) if len(w) > 2]
    result = text
    for word in words:
        pattern = re.compile(re.escape(word), re.IGNORECASE)
        result = pattern.sub(
            lambda m: f'<mark style="background:rgba(249,115,22,0.25);'
                      f'color:#F97316;border-radius:2px;padding:0 2px;">'
                      f'{m.group()}</mark>',
            result,
        )
    return result
