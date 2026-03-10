"""Chrome-level UI components such as header, footer, and notices."""

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


def render_header(T: Dict[str, Any], labels: List[str], ids: List[str], current_tab: str, logo_path: Path) -> None:
    """Render the top hero block and tab navigation."""

    links: List[str] = []
    for label, tab_id in zip(labels, ids):
        css = "nav-pill active" if tab_id == current_tab else "nav-pill"
        links.append(f"<a class='{css}' href='?tab={E(tab_id)}' target='_self'>{E(label)}</a>")

    logo_html = ""
    logo_uri = img_to_data_uri(logo_path)
    if logo_uri:
        logo_html = f"<img class='brand-logo' src='{logo_uri}' alt='logo' />"

    st.markdown(
        f"""
        <div class="hero-shell">
            <div class="hero-grid">
                <div>
                    <div class="hero-brand">
                        <div class="brand-badge">{logo_html}</div>
                        <div>
                            <div class="brand-title">{E(Tget(T, "app_title", "SmartCampus V2T"))}</div>
                            <div class="brand-subtitle">{E(Tget(T, "app_subtitle", ""))}</div>
                        </div>
                    </div>
                    <div class="hero-copy">{E(Tget(T, "hero_copy", ""))}</div>
                </div>
                <div class="hero-panel">
                    <div class="hero-panel-label">Pipeline Surface</div>
                    <div class="hero-panel-value">Process, Search, Report, QA, RAG</div>
                    <div class="hero-panel-copy">Operator console over FastAPI, workers, translation, and hybrid retrieval.</div>
                </div>
            </div>
            <div class="nav-strip">{''.join(links)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


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
    """Render the global UI language selector."""

    langs = list(cfg.ui.langs or ["ru", "kz", "en"])
    current = str(st.session_state.get("ui_lang") or cfg.ui.default_lang or langs[0])
    if current not in langs:
        current = langs[0]
        st.session_state["ui_lang"] = current

    left, right = st.columns([5, 2], gap="small")
    with left:
        st.markdown(f"<div class='page-kicker'>{E(Tget(T, 'page_note', 'Operator analytics console'))}</div>", unsafe_allow_html=True)
    with right:
        st.selectbox(
            Tget(T, "ui_lang", "UI language"),
            options=langs,
            index=langs.index(current),
            key="ui_lang",
        )


def render_i18n_metrics() -> None:
    """No-op compatibility hook for the legacy entrypoint."""

    return
