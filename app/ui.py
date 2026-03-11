# app/ui.py
"""
UI facade for SmartCampus V2T Streamlit app.

Purpose:
- Re-export the modularized page and shell render entrypoints.
- Preserve one stable UI module surface for the Streamlit launcher.
"""

from __future__ import annotations

from app.components.chrome import footer, render_header, render_i18n_metrics, soft_note
from app.lib.i18n import get_T, load_ui_text
from app.lib.media import load_and_apply_css
from app.pages.home import render_page as home_tab
from app.pages.analytics import render_page as search_tab
from app.pages.processing import render_page as processing_tab
from app.pages.storage import render_page as gallery_tab

__all__ = [
    "footer",
    "gallery_tab",
    "get_T",
    "home_tab",
    "load_and_apply_css",
    "load_ui_text",
    "processing_tab",
    "render_header",
    "render_i18n_metrics",
    "search_tab",
    "soft_note",
]
