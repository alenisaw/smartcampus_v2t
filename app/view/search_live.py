# app/view/search_live.py
"""
Search page overrides with isolated result reruns and card-style hits.

Purpose:
- Keep the legacy search helpers intact while overriding the active page entrypoint.
- Render search results inside storage-style cards and isolate left-panel reruns.
"""

from __future__ import annotations

from typing import Any, Dict, List

import streamlit as st

from app.api_client import BackendClient
from app.lib.formatters import E, clip_text, humanize_token, mmss
from app.view import search as _search
from app.view.shared import (
    ICON_DELETE,
    _loc,
    _mark,
    _page_title,
    _resolve_video_context,
    _run_fragment,
    _section,
    _ui_lang,
    _video_items,
    soft_note,
)


def _render_search_results(hits: List[Dict[str, Any]], lang: str) -> None:
    """Render search hits as bordered result cards."""

    _section(_loc("Результаты", "Нәтижелер", "Results", lang=lang))
    visible_hits = [hit for hit in hits[:20] if isinstance(hit, dict)]
    total_hits = len([hit for hit in hits if isinstance(hit, dict)])
    st.markdown(
        f"<div class='status-line search-results-count'>{E(_loc(f'Найдено: {total_hits}', f'Табылды: {total_hits}', f'Found: {total_hits}', lang=lang))}</div>",
        unsafe_allow_html=True,
    )
    if not visible_hits:
        _search._caption(_loc("Результаты появятся после поиска.", "Нәтижелер іздеуден кейін көрінеді.", "Results appear after search.", lang=lang))
        return

    for idx, hit in enumerate(visible_hits, start=1):
        anomaly = bool(hit.get("anomaly_flag"))
        with st.container():
            _mark("row-marker")
            info_col, action_col = st.columns([4.8, 0.8], gap="small")
            with info_col:
                headline = (
                    f"#{idx:02d} | {str(hit.get('video_id') or '-')}"
                    f" | {mmss(float(hit.get('start_sec', 0.0) or 0.0))}"
                    f" - {mmss(float(hit.get('end_sec', 0.0) or 0.0))}"
                )
                score_line = f"{_loc('Счёт', 'Ұпай', 'Score', lang=lang)}: {float(hit.get('score', 0.0) or 0.0):.3f}"
                meta_bits = [score_line]
                for key in ("event_type", "risk_level", "motion_type", "people_count_bucket"):
                    value = str(hit.get(key) or "").strip()
                    if value:
                        meta_bits.append(humanize_token(value))
                st.markdown(f"<div class='row-title'>{E(headline)}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='row-meta'>{E(' | '.join(meta_bits))}</div>", unsafe_allow_html=True)
                st.markdown(
                    f"<div class='row-copy{' alert' if anomaly else ''}'>{E(clip_text(str(hit.get('description') or ''), 220))}</div>",
                    unsafe_allow_html=True,
                )
            with action_col:
                if st.button(
                    _search.ICON_OPEN,
                    key=f"search_open_card_{idx}",
                    use_container_width=True,
                    help=_loc("Открыть в аналитике", "Аналитикада ашу", "Open in analytics", lang=lang),
                ):
                    st.session_state["selected_video_id"] = str(hit.get("video_id") or "")
                    st.session_state["video_seek_sec"] = int(float(hit.get("start_sec", 0.0) or 0.0))
                    st.query_params["tab"] = "video"
                    st.rerun()


def search_tab(client: BackendClient, cfg: Any, ui_text: Dict[str, Dict[str, Any]]) -> None:
    """Render the search page with isolated left-panel reruns."""

    _ = ui_text
    _search._bind_search_state()
    lang = _ui_lang(cfg)
    videos = _video_items(client)
    if not videos:
        soft_note(_loc("Библиотека пуста. Загрузите видео, чтобы начать.", "Қойма бос. Бастау үшін бейне жүктеңіз.", "The library is empty. Upload a video to begin.", lang=lang))
        return

    context = _resolve_video_context(
        videos,
        video_key="search_video_id",
        variant_key="search_variant",
        lang_key="search_lang",
        base_lang=lang,
        allow_empty_video=True,
    )
    video_ids = context["video_options"]
    variant_tokens = context["variant_tokens"]
    available_languages = context["available_languages"]
    current_lang = context["current_lang"]

    _page_title(_loc("Поиск", "Іздеу", "Search", lang=lang))

    _search._render_search_results = _render_search_results
    left, right = st.columns([1.68, 1.08], gap="large")
    with left:
        def _left_panel() -> None:
            _search._render_search_panel_fragment(
                client,
                lang,
                current_lang,
                video_ids,
                available_languages,
                variant_tokens,
            )

        _run_fragment(_left_panel)

    with right:
        _section(_loc("Умный ассистент", "Ақылды ассистент", "Smart assistant", lang=lang))
        _search._render_chat(list(st.session_state.get("search_rag_messages") or []), lang)

        input_row = st.columns([4.28, 0.56, 0.56], gap="small")
        with input_row[0]:
            _mark("chat-input-marker")
            st.text_input(
                _loc("Сообщение", "Хабарлама", "Message", lang=lang),
                key="search_rag_input",
                placeholder=_loc(
                    "Были ли аномалии с людьми в чёрном?",
                    "Қара киімдегі адамдармен аномалиялар болды ма?",
                    "Were there anomalies with people in black?",
                    lang=lang,
                ),
                label_visibility="collapsed",
                on_change=_search._submit_chat_from_enter,
                args=(client, lang, current_lang),
            )
        with input_row[1]:
            _mark("chat-new-marker")
            if st.button(_search.CHAT_NEW_ICON, key="search_new_chat_btn", use_container_width=True, help=_loc("Новый чат", "Жаңа чат", "New chat", lang=lang)):
                _search._start_new_chat()
                st.rerun()
        with input_row[2]:
            _mark("chat-clear-marker")
            if st.button(ICON_DELETE, key="search_clear_chat_btn", use_container_width=True, help=_loc("Очистить чат", "Чатты тазарту", "Clear chat", lang=lang)):
                _search._clear_chat_history()
                st.rerun()
