# ============================================================
# Color Palette — Endee Code Search
# ============================================================

PRIMARY          = "#0D9488"   # Teal  — main buttons / actions
SECONDARY        = "#F97316"   # Orange — highlights / badges
ACCENT           = "#06B6D4"   # Cyan — secondary elements
BG_DARK          = "#0F172A"   # Dark background (page)
BG_CARD          = "#1E293B"   # Card / sidebar background
BG_CARD_HOVER    = "#263248"   # Card hover state
BG_LIGHT         = "#F8FAFC"   # Light background (unused in dark mode)
SUCCESS          = "#10B981"   # Green — success / score high
WARNING          = "#F59E0B"   # Amber — warning / score mid
DANGER           = "#EF4444"   # Red — error / score low
TEXT_PRIMARY     = "#F1F5F9"   # Main text (light on dark)
TEXT_SECONDARY   = "#94A3B8"   # Muted text
TEXT_MUTED       = "#64748B"   # Very muted text
BORDER           = "#334155"   # Card / divider borders
BORDER_ACCENT    = "#0D9488"   # Highlighted border (teal)

# Language tag colours
LANG_PY          = "#3B82F6"   # Blue  — Python
LANG_JS          = "#EAB308"   # Yellow — JavaScript
LANG_TS          = "#06B6D4"   # Cyan — TypeScript
LANG_DEFAULT     = "#8B5CF6"   # Purple — other

# Score thresholds → color mapping
def score_color(score: float) -> str:
    """Return a hex color based on relevance score (0–1)."""
    if score >= 0.75:
        return SUCCESS
    if score >= 0.50:
        return WARNING
    return DANGER

def lang_color(lang: str) -> str:
    """Return badge color for a programming language string."""
    return {
        "python": LANG_PY,
        "javascript": LANG_JS,
        "typescript": LANG_TS,
    }.get(lang.lower() if lang else "", LANG_DEFAULT)
