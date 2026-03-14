# app/view/shared.py
"""
Shared Streamlit UI helpers for the Streamlit frontend.

Purpose:
- Hold localization, layout, and reusable rendering helpers shared across page modules.
- Keep common UI primitives out of the page entry modules.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

from app.api_client import BackendClient
from app.lib.formatters import E, collect_available_languages, variant_from_token, video_variant_tokens
from app.lib.i18n import Tget
from app.lib.media import img_to_data_uri

PAGE_SIZE = 8

ICON_CLEAR = "\u232b"
ICON_REFRESH = "\u21bb"
ICON_START = "\u25b6"
ICON_PAUSE = "\u23f8"
ICON_RESUME = "\u25b7"
ICON_OPEN = "\u2197"
ICON_DELETE = "\u2715"
ICON_CONFIRM = "\u2713"
ICON_UP = "\u2191"
ICON_DOWN = "\u2193"
ICON_PREV = "\u25c0"
ICON_NEXT = "\u25b6"


def _lang(default: str = "en") -> str:
    """Return the active UI language."""

    return str(st.session_state.get("ui_lang") or default).strip().lower()


def _loc(ru: str, kz: str, en: str, *, lang: Optional[str] = None) -> str:
    """Return localized UI copy."""

    key = str(lang or _lang()).strip().lower()
    return {"ru": ru, "kz": kz, "en": en}.get(key, en)


def _footer_text(lang: str) -> str:
    """Return the localized footer attribution."""

    return _loc(
        "Дипломный проект Issayev Alen, BDA-2302",
        "Issayev Alen дипломдық жобасы, BDA-2302",
        "Thesis project by Issayev Alen, BDA-2302",
        lang=lang,
    )


def _error_prefix(lang: str) -> str:
    """Return a localized generic error prefix."""

    return _loc("Ошибка", "Қате", "Error", lang=lang)


def _mark(*classes: str) -> None:
    """Render a hidden CSS marker for a native Streamlit container."""

    clean = " ".join(part.strip() for part in classes if str(part).strip())
    st.markdown(f"<div class='{clean}'></div>", unsafe_allow_html=True)


def _section(title: str) -> None:
    """Render a consistent section heading."""

    st.markdown(
        f"""
        <div class="section-heading">
            <div class="section-title">{E(title)}</div>
            <div class="section-divider"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _caption(text: str) -> None:
    """Render compact muted copy."""

    st.markdown(f"<div class='section-caption'>{E(text)}</div>", unsafe_allow_html=True)


def _ui_lang(cfg: Any, default: str = "en") -> str:
    """Return the active UI language with config fallback."""

    return str(st.session_state.get("ui_lang") or getattr(getattr(cfg, "ui", None), "default_lang", None) or default)


def _page_title(title: str) -> None:
    """Render a consistent page title."""

    st.markdown(f"<div class='page-title'>{E(title)}</div>", unsafe_allow_html=True)


def _video_items(client: BackendClient) -> List[Dict[str, Any]]:
    """Return clean video items from the backend client."""

    return [item for item in client.list_videos() if isinstance(item, dict)]


def _video_ids(videos: List[Dict[str, Any]]) -> List[str]:
    """Return normalized video ids for UI selectors."""

    return [str(item.get("video_id") or "") for item in videos if str(item.get("video_id") or "").strip()]


def _session_choice(key: str, options: List[str], *, default: Optional[str] = None) -> str:
    """Normalize one session-backed choice against the available options."""

    if not options:
        st.session_state[key] = ""
        return ""
    fallback = default if default in options else options[0]
    value = str(st.session_state.get(key) or fallback)
    if value not in options:
        value = fallback
    st.session_state[key] = value
    return value


def _resolve_video_context(
    videos: List[Dict[str, Any]],
    *,
    video_key: str,
    variant_key: str,
    lang_key: str,
    base_lang: str,
    allow_empty_video: bool = False,
) -> Dict[str, Any]:
    """Resolve video, variant, and language session state for one page."""

    video_ids = _video_ids(videos)
    video_options = [""] + video_ids if allow_empty_video else video_ids
    selected_video_id = _session_choice(video_key, video_options, default="" if allow_empty_video else None)
    selected_item = next((item for item in videos if str(item.get("video_id") or "") == selected_video_id), None) or {}

    variant_tokens = video_variant_tokens(selected_item)
    current_variant = _session_choice(variant_key, variant_tokens)

    available_languages = collect_available_languages(selected_item, variant_from_token(current_variant))
    current_lang = _session_choice(lang_key, available_languages or [base_lang], default=base_lang)

    return {
        "video_ids": video_ids,
        "video_options": video_options,
        "selected_video_id": selected_video_id,
        "selected_item": selected_item,
        "variant_tokens": variant_tokens,
        "current_variant": current_variant,
        "available_languages": available_languages or [base_lang],
        "current_lang": current_lang,
    }


def soft_note(text: str, kind: str = "info") -> None:
    """Render a compact inline notice."""

    css = {"info": "notice notice--info", "warn": "notice notice--warn", "ok": "notice notice--ok"}.get(kind, "notice notice--info")
    st.markdown(f"<div class='{css}'>{E(text)}</div>", unsafe_allow_html=True)


def render_i18n_metrics() -> None:
    """Compatibility no-op for the entrypoint."""

    return


def render_header(T: Dict[str, Any], labels: List[str], ids: List[str], current_tab: str, logo_path: Path, cfg: Any) -> str:
    """Render the shared header block with logo, tabs, and language selector."""

    available_langs = [code for code in ("ru", "en", "kz") if code in list(cfg.ui.langs or ["ru", "en", "kz"])]
    if not available_langs:
        available_langs = ["ru", "en", "kz"]

    current_lang = str(st.session_state.get("ui_lang") or getattr(cfg.ui, "default_lang", None) or available_langs[0]).strip().lower()
    if current_lang not in available_langs:
        current_lang = available_langs[0]
        st.session_state["ui_lang"] = current_lang

    selector_key = "header_lang_code"
    if st.session_state.get(selector_key) not in available_langs:
        st.session_state[selector_key] = current_lang

    logo_uri = img_to_data_uri(logo_path)
    logo_html = f"<img class='brand-logo' src='{logo_uri}' alt='logo' />" if logo_uri else "<div class='brand-logo-fallback'>SC</div>"
    title = Tget(T, "app_title", "SmartCampus V2T")
    links = []
    for tab_id, label in zip(ids, labels):
        css = "nav-pill active" if tab_id == current_tab else "nav-pill"
        links.append(f"<a class='{css}' href='?tab={E(tab_id)}' target='_self'>{E(label)}</a>")

    with st.container():
        _mark("header-marker")
        left, right = st.columns([5.2, 0.9], gap="small")
        with left:
            st.markdown(
                f"""
                <div class="brand-row">
                    <div class="brand-mark">{logo_html}</div>
                    <div class="brand-copy">
                        <div class="brand-title">{E(title)}</div>
                        <div class="brand-subtitle">{E(_product_subtitle(current_lang))}</div>
                    </div>
                </div>
                <div class="nav-strip">{''.join(links)}</div>
                """,
                unsafe_allow_html=True,
            )
        with right:
            st.selectbox(
                "Language",
                options=available_langs,
                index=available_langs.index(current_lang),
                key=selector_key,
                label_visibility="collapsed",
                format_func=lambda code: str(code or "").upper(),
            )
            selected_lang = str(st.session_state.get(selector_key) or current_lang).strip().lower()
            if selected_lang != st.session_state.get("ui_lang"):
                st.session_state["ui_lang"] = selected_lang
                st.rerun()

    try:
        st.query_params["tab"] = current_tab
    except Exception:
        pass
    return str(current_tab)


def _product_subtitle(lang: str) -> str:
    """Return the localized product subtitle."""

    return _loc(
        "Многофункциональная аналитическая система событий и видео",
        "Оқиғалар мен бейнеге арналған көпфункционалды аналитикалық жүйе",
        "Multifunctional analytics system for events and video",
        lang=lang,
    )


def footer(T: Dict[str, Any], lang: str) -> None:
    """Render the footer strip."""

    _ = T
    with st.container():
        st.markdown(
            f"""
            <div class='footer-shell'>
                <div class='footer-divider'></div>
                <div class='footer-strip'>{E(_footer_text(lang))}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


__all__ = [
    "PAGE_SIZE",
    "ICON_CLEAR",
    "ICON_REFRESH",
    "ICON_START",
    "ICON_PAUSE",
    "ICON_RESUME",
    "ICON_OPEN",
    "ICON_DELETE",
    "ICON_CONFIRM",
    "ICON_UP",
    "ICON_DOWN",
    "ICON_PREV",
    "ICON_NEXT",
    "_lang",
    "_loc",
    "_footer_text",
    "_error_prefix",
    "_mark",
    "_section",
    "_caption",
    "_ui_lang",
    "_page_title",
    "_video_items",
    "_video_ids",
    "_session_choice",
    "_resolve_video_context",
    "soft_note",
    "render_i18n_metrics",
    "render_header",
    "_product_subtitle",
    "footer",
]
