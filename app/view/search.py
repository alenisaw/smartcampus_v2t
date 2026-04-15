# app/view/search.py
"""
Search and assistant Streamlit page logic.

Purpose:
- Render retrieval filters, results, and the grounded assistant workflow.
- Keep search-specific state, chat handling, and result rendering in one module.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import streamlit as st

from app.api_client import BackendClient
from app.lib.formatters import E, clip_text, humanize_token, mmss, variant_from_token
from app.view.shared import (
    ICON_DELETE,
    ICON_OPEN,
    ICON_REFRESH,
    _caption,
    _error_prefix,
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

CHAT_NEW_ICON = "\u271A"


def _bind_search_state() -> None:
    """Initialize search page state."""

    defaults = {
        "search_query_box": "",
        "search_video_id": "",
        "search_lang": "",
        "search_variant": "__base__",
        "search_topk": 5,
        "search_event_type": "",
        "search_risk_level": "",
        "search_motion_type": "",
        "search_people_count_bucket": "",
        "search_anomaly_only": False,
        "search_dedupe": True,
        "search_hits": [],
        "search_rebuild_status": "",
        "search_rag_input": "",
        "search_rag_messages": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _search_status_line(lang: str, *, hits_count: Optional[int] = None, rebuild_status: str = "") -> None:
    """Render compact search status copy."""

    _ = hits_count
    parts: List[str] = []
    if rebuild_status.strip():
        parts.append(rebuild_status.strip())
    if not parts:
        return
    st.markdown(f"<div class='status-line'>{E(' · '.join(parts))}</div>", unsafe_allow_html=True)


def _chat_message(role: str, content: str, *, evidence: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Build one assistant message payload."""

    return {"role": role, "content": str(content or "").strip(), "evidence": list(evidence or [])}


def _render_chat(messages: List[Dict[str, Any]], lang: str) -> None:
    """Render the assistant chat history."""

    if not messages:
        st.markdown(
            f"""
            <div class="chat-empty">
                <div class="chat-empty-title">{E(_loc("Спросите о найденных эпизодах", "Табылған эпизодтар туралы сұраңыз", "Ask about the retrieved episodes", lang=lang))}</div>
                <div class="chat-empty-copy">{E(_loc("Ассистент использует активные фильтры и результаты поиска, чтобы отвечать по событиям.", "Ассистент белсенді сүзгілер мен іздеу нәтижелерін пайдаланып, оқиғалар бойынша жауап береді.", "The assistant uses the active filters and search results to answer questions about the events.", lang=lang))}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    parts: List[str] = ["<div class='chat-thread'>"]
    for message in messages:
        role = str(message.get("role") or "assistant")
        bubble_role = "user" if role == "user" else "assistant"
        meta = _loc("Вы", "Сіз", "You", lang=lang) if role == "user" else _loc("Ассистент", "Ассистент", "Assistant", lang=lang)
        parts.append(
            f"""
            <div class="chat-row {bubble_role}">
                <div class="chat-bubble {bubble_role}">
                    <div class="chat-meta">{E(meta)}</div>
                    <div class="chat-text">{E(str(message.get("content") or ""))}</div>
            """
        )

        if bubble_role == "assistant":
            evidence = message.get("evidence") if isinstance(message.get("evidence"), list) else []
            if evidence:
                chips = []
                for hit in evidence[:4]:
                    if not isinstance(hit, dict):
                        continue
                    chips.append(
                        f"<span class='evidence-chip'>{E(str(hit.get('video_id') or '-'))} · {E(mmss(float(hit.get('start_sec', 0.0) or 0.0)))} - {E(mmss(float(hit.get('end_sec', 0.0) or 0.0)))}</span>"
                    )
                if chips:
                    parts.append(f"<div class='evidence-row'>{''.join(chips)}</div>")

        parts.append("</div></div>")

    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def _run_search(client: BackendClient, base_lang: str) -> None:
    """Execute search using the visible controls."""

    hits = client.search(
        query=str(st.session_state.get("search_query_box") or "").strip(),
        top_k=int(st.session_state.get("search_topk") or 5),
        video_id=str(st.session_state.get("search_video_id") or "") or None,
        language=str(st.session_state.get("search_lang") or base_lang),
        variant=variant_from_token(st.session_state.get("search_variant")),
        dedupe=bool(st.session_state.get("search_dedupe", True)),
        anomaly_only=bool(st.session_state.get("search_anomaly_only")),
    )
    st.session_state["search_hits"] = hits


def _start_new_chat() -> None:
    """Reset chat history and current draft."""

    st.session_state["search_rag_messages"] = []
    st.session_state["search_rag_input"] = ""


def _clear_chat_history() -> None:
    """Reset chat history but keep the current draft message."""

    st.session_state["search_rag_messages"] = []


def _run_search_from_enter(client: BackendClient, base_lang: str) -> None:
    """Run search when the user confirms the query input."""

    try:
        _run_search(client, base_lang)
    except Exception as exc:
        st.session_state["search_hits"] = []
        soft_note(f"{_error_prefix(base_lang)}: {exc}", kind="warn")


def _submit_chat_from_enter(client: BackendClient, lang: str, base_lang: str) -> None:
    """Submit one chat prompt from the Enter key."""

    prompt = str(st.session_state.get("search_rag_input") or "").strip()
    if not prompt:
        return

    messages = list(st.session_state.get("search_rag_messages") or [])
    messages.append(_chat_message("user", prompt))
    st.session_state["search_rag_messages"] = messages

    try:
        rag = client.ask_rag(
            query=prompt,
            language=str(st.session_state.get("search_lang") or base_lang),
            variant=variant_from_token(st.session_state.get("search_variant")),
            video_id=str(st.session_state.get("search_video_id") or "") or None,
            top_k=int(st.session_state.get("search_topk") or 5),
        )
        answer = str(rag.get("answer") or "").strip() or _loc(
            "Не удалось подготовить ответ. Попробуйте уточнить запрос.",
            "Жауапты дайындау мүмкін болмады. Сұрауды нақтылап көріңіз.",
            "The assistant could not prepare an answer. Try a more specific prompt.",
            lang=lang,
        )
        evidence = list(rag.get("supporting_hits") or [])
    except Exception as exc:
        answer = _loc(
            f"Не удалось получить ответ: {exc}",
            f"Жауап алу мүмкін болмады: {exc}",
            f"Failed to get an answer: {exc}",
            lang=lang,
        )
        evidence = []

    messages.append(_chat_message("assistant", answer, evidence=evidence))
    st.session_state["search_rag_messages"] = messages
    st.session_state["search_rag_input"] = ""


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
        _caption(_loc("Результаты появятся после поиска.", "Нәтижелер іздеуден кейін көрінеді.", "Results appear after search.", lang=lang))
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
                if st.button(ICON_OPEN, key=f"search_open_card_{idx}", use_container_width=True, help=_loc("Открыть в аналитике", "Аналитикада ашу", "Open in analytics", lang=lang)):
                    st.session_state["selected_video_id"] = str(hit.get("video_id") or "")
                    st.session_state["video_seek_sec"] = int(float(hit.get("start_sec", 0.0) or 0.0))
                    st.query_params["tab"] = "video"
                    st.rerun()


def _render_search_panel_fragment(
    client: BackendClient,
    lang: str,
    current_lang: str,
    video_ids: List[str],
    available_languages: List[str],
    variant_tokens: List[str],
) -> None:
    """Render the interactive search controls and results."""

    _ = (video_ids, variant_tokens)
    _section(_loc("Запрос и фильтры", "Сұрау және сүзгілер", "Query and filters", lang=lang))

    query_row = st.columns([1.0], gap="small")
    with query_row[0]:
        st.text_input(
            _loc("Запрос", "Сұрау", "Query", lang=lang),
            key="search_query_box",
            placeholder=_loc(
                "Например: человек у входа, падение, толпа",
                "Мысалы: кіреберісте адам, құлау, топ",
                "For example: person near the entrance, fall, crowd",
                lang=lang,
            ),
            on_change=_run_search_from_enter,
            args=(client, current_lang),
        )

    rebuild_row = st.columns([1.0], gap="small")
    with rebuild_row[0]:
        if st.button(ICON_REFRESH, key="search_rebuild_btn", use_container_width=True, help=_loc("Пересобрать индекс", "Индексті жаңарту", "Rebuild index", lang=lang)):
            try:
                result = client.index_rebuild()
                message = str(result.get("message") or _loc("Индекс обновлён.", "Индекс жаңартылды.", "Index rebuilt.", lang=lang))
                st.session_state["search_rebuild_status"] = message
            except Exception as exc:
                soft_note(f"{_error_prefix(lang)}: {exc}", kind="warn")

    row1 = st.columns([1.0, 0.8], gap="small")
    with row1[0]:
        st.selectbox(_loc("Язык", "Тіл", "Language", lang=lang), options=available_languages or [lang], key="search_lang")
    with row1[1]:
        st.number_input("Top-K", min_value=1, max_value=20, key="search_topk")

    row2 = st.columns([1.0, 1.0], gap="small")
    with row2[0]:
        st.checkbox(_loc("Только аномалии", "Тек ауытқулар", "Anomalies only", lang=lang), key="search_anomaly_only")
    with row2[1]:
        st.checkbox(_loc("Скрывать повторы", "Қайталануды жасыру", "Hide duplicates", lang=lang), key="search_dedupe")

    _search_status_line(
        lang,
        hits_count=len([hit for hit in list(st.session_state.get("search_hits") or []) if isinstance(hit, dict)]),
        rebuild_status=str(st.session_state.get("search_rebuild_status") or ""),
    )
    _render_search_results(list(st.session_state.get("search_hits") or []), lang)


def search_tab(client: BackendClient, cfg: Any, ui_text: Dict[str, Dict[str, Any]]) -> None:
    """Render the search page."""

    _ = ui_text
    _bind_search_state()
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

    left, right = st.columns([1.68, 1.08], gap="large")
    with left:
        def _left_panel() -> None:
            _render_search_panel_fragment(
                client,
                lang,
                current_lang,
                video_ids,
                available_languages,
                variant_tokens,
            )

        _run_fragment(_left_panel)

    with right:
        def _right_panel() -> None:
            _section(_loc("Умный ассистент", "Ақылды ассистент", "Smart assistant", lang=lang))
            _render_chat(list(st.session_state.get("search_rag_messages") or []), lang)

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
                    on_change=_submit_chat_from_enter,
                    args=(client, lang, current_lang),
                )
            with input_row[1]:
                _mark("chat-new-marker")
                st.button(
                    CHAT_NEW_ICON,
                    key="search_new_chat_btn",
                    use_container_width=True,
                    help=_loc("Новый чат", "Жаңа чат", "New chat", lang=lang),
                    on_click=_start_new_chat,
                )
            with input_row[2]:
                _mark("chat-clear-marker")
                st.button(
                    ICON_DELETE,
                    key="search_clear_chat_btn",
                    use_container_width=True,
                    help=_loc("Очистить чат", "Чатты тазарту", "Clear chat", lang=lang),
                    on_click=_clear_chat_history,
                )

        _run_fragment(_right_panel)
