"""Facade module for the modularized Streamlit UI."""

from __future__ import annotations

from app.components.chrome import footer, render_header, render_i18n_metrics, soft_note
from app.lib.i18n import get_T, load_ui_text
from app.lib.media import load_and_apply_css
from app.pages.assistant import render_page as assistant_tab
from app.pages.analytics import render_page as search_tab
from app.pages.storage import render_page as gallery_tab

__all__ = [
    "assistant_tab",
    "footer",
    "gallery_tab",
    "get_T",
    "load_and_apply_css",
    "load_ui_text",
    "render_header",
    "render_i18n_metrics",
    "search_tab",
    "soft_note",
]
