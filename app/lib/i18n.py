# app/lib/i18n.py
"""
UI translation helpers for SmartCampus V2T.

Purpose:
- Load localized UI text resources for the Streamlit frontend.
- Provide lightweight access helpers for the active interface language.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def load_ui_text(path_str: str, mtime: float, langs_key: str) -> Dict[str, Dict[str, Any]]:
    """Load i18n text and validate the tab labels contract."""

    _ = mtime
    data = json.loads(Path(path_str).read_text(encoding="utf-8"))
    langs = [item.strip().lower() for item in (langs_key or "").split(",") if item.strip()]
    for lang in langs:
        bucket = data.get(lang)
        if not isinstance(bucket, dict):
            raise ValueError(f"Missing UI language: {lang}")
        tabs = bucket.get("tabs")
        if not isinstance(tabs, list) or len(tabs) != 4:
            raise ValueError(f"Invalid tabs for language: {lang}")
    return data


def get_T(ui_text: Dict[str, Dict[str, Any]], lang: str) -> Dict[str, Any]:
    """Return the text dictionary for the selected language."""

    key = (lang or "en").strip().lower()
    return ui_text.get(key) or ui_text.get("en") or {}


def Tget(T: Dict[str, Any], key: str, fallback: str) -> str:
    """Lookup translated text with fallback."""

    value = T.get(key)
    return fallback if value is None else str(value)
