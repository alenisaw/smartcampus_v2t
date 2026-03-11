# app/components/chrome.py
"""
Chrome-level UI components for SmartCampus V2T.

Purpose:
- Render shared page shell elements such as header, footer, and notices.
- Keep top-level layout framing consistent across Streamlit pages.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import streamlit as st

from app.lib.formatters import E
from app.lib.i18n import Tget
from app.lib.media import img_to_data_uri


def soft_note(text: str, kind: str = "info") -> None:
    """Render a compact inline message."""

    css = {"info": "soft-note", "warn": "soft-warn", "ok": "soft-ok"}.get(kind, "soft-note")
    st.markdown(f"<div class='{css}'>{E(text)}</div>", unsafe_allow_html=True)


def render_header(T: Dict[str, Any], labels: List[str], ids: List[str], current_tab: str, logo_path: Path, cfg: Any) -> str:
    """Render the top brand shell, global language control, and tab navigation."""

    logo_html = ""
    logo_uri = img_to_data_uri(logo_path)
    if logo_uri:
        logo_html = f"<img class='brand-logo' src='{logo_uri}' alt='logo' />"

    langs = list(cfg.ui.langs or ["ru", "kz", "en"])
    current_lang = str(st.session_state.get("ui_lang") or cfg.ui.default_lang or langs[0])
    if current_lang not in langs:
        current_lang = langs[0]
        st.session_state["ui_lang"] = current_lang

    selector_key = "header_ui_lang_selector"
    if st.session_state.get(selector_key) not in langs:
        st.session_state[selector_key] = current_lang

    links: List[str] = []
    for label, tab_id in zip(labels, ids):
        css = "nav-pill active" if tab_id == current_tab else "nav-pill"
        links.append(f"<a class='{css}' href='?tab={E(tab_id)}' target='_self'>{E(label)}</a>")

    top = st.columns([5.2, 1.15], gap="small")
    with top[0]:
        st.markdown(
            f"""
            <div class="hero-shell">
                <div class="brand-row">
                    <div class="brand-badge">{logo_html}</div>
                    <div>
                        <div class="brand-title">{E(Tget(T, "app_title", "SmartCampus V2T"))}</div>
                        <div class="brand-subtitle">{E(Tget(T, "app_subtitle", ""))}</div>
                    </div>
                </div>
                <div class="nav-strip">{''.join(links)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with top[1]:
        st.markdown("<div class='hero-lang-label'>UI</div>", unsafe_allow_html=True)
        selected_lang = st.selectbox(
            Tget(T, "ui_lang", "UI language"),
            options=langs,
            index=langs.index(current_lang),
            key=selector_key,
            label_visibility="visible",
        )
        if selected_lang != st.session_state.get("ui_lang"):
            st.session_state["ui_lang"] = selected_lang

    try:
        st.query_params["tab"] = current_tab
    except Exception:
        pass
    return str(current_tab)


def footer(T: Dict[str, Any]) -> None:
    """Render the footer strip."""

    st.markdown(
        f"""
        <div class="footer-shell">
            <div>{E(Tget(T, 'footer_left', 'SmartCampus V2T'))}</div>
            <div>{E(Tget(T, 'footer_right', 'Operator Console'))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_language_switcher(T: Dict[str, Any], cfg: Any) -> None:
    """Legacy compatibility no-op."""

    _ = T
    _ = cfg
    return


def render_i18n_metrics() -> None:
    """No-op compatibility hook for the legacy entrypoint."""

    return
