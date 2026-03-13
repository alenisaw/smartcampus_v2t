# app/ui.py
"""
Unified UI layer for SmartCampus V2T Streamlit app.

Purpose:
- Keep the Streamlit presentation layer in one maintainable module.
- Rebuild the demo UI without changing backend contracts or page routing.
"""

from __future__ import annotations

import math
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import streamlit as st

from app.api_client import BackendClient
from app.lib.formatters import (
    E,
    available_variant_ids,
    clip_text,
    collect_available_languages,
    first_sentence,
    fmt_bytes,
    hms,
    humanize_token,
    mmss,
    variant_from_token,
    variant_label,
    video_variant_tokens,
)
from app.lib.i18n import Tget, get_T, load_ui_text
from app.lib.media import ensure_browser_video, get_video_meta, img_to_data_uri, load_and_apply_css, mtime

PAGE_SIZE = 8

ICON_CLEAR = "⌫"
ICON_REFRESH = "↻"
ICON_START = "▶"
ICON_PAUSE = "⏸"
ICON_RESUME = "▷"
ICON_OPEN = "↗"
ICON_DELETE = "✕"
ICON_CONFIRM = "✓"
ICON_UP = "↑"
ICON_DOWN = "↓"
ICON_PREV = "◀"
ICON_NEXT = "▶"


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


def _bind_storage_state() -> None:
    """Initialize storage page state."""

    defaults = {
        "storage_query": "",
        "storage_sort": "newest",
        "storage_status": "all",
        "storage_tag": "all",
        "storage_window": "all",
        "storage_page": 1,
        "storage_filter_sig": "",
        "storage_upload_sig": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _storage_status(item: Dict[str, Any]) -> str:
    """Return a simple readiness token for one library item."""

    languages = item.get("languages") if isinstance(item.get("languages"), list) else []
    return "ready" if languages else "raw"


def _storage_tags(item: Dict[str, Any]) -> List[str]:
    """Return clean storage-facing tags."""

    tags: List[str] = []
    languages = [str(lang).strip().lower() for lang in (item.get("languages") or []) if str(lang).strip()]
    if str(item.get("summary") or "").strip():
        tags.append("summary")
    if len(languages) > 1 or any(lang != "en" for lang in languages):
        tags.append("translated")
    if isinstance(item.get("variants"), dict) and item.get("variants"):
        tags.append("variants")
    if languages:
        tags.append("outputs")
    deduped: List[str] = []
    for tag in tags:
        if tag not in deduped:
            deduped.append(tag)
    return deduped


def _storage_tag_label(tag: str, lang: str) -> str:
    """Return a localized storage tag label."""

    return {
        "summary": _loc("Сводка", "Қысқаша мазмұн", "Summary", lang=lang),
        "translated": _loc("Перевод", "Аударма", "Translation", lang=lang),
        "variants": _loc("Варианты", "Нұсқалар", "Variants", lang=lang),
        "outputs": _loc("Результаты", "Нәтижелер", "Outputs", lang=lang),
    }.get(str(tag or ""), str(tag or ""))


def _prepare_storage_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """Enrich a video item with UI metadata."""

    prepared = dict(item)
    path = Path(str(item.get("path") or ""))
    created_ts = 0.0
    updated_ts = 0.0
    meta: Dict[str, Any] = {}
    try:
        stat = path.stat()
        created_ts = float(getattr(stat, "st_ctime", 0.0) or 0.0)
        updated_ts = float(stat.st_mtime or 0.0)
    except Exception:
        pass
    if path.exists():
        meta = get_video_meta(str(path), mtime(path))
    prepared["_ui_path"] = path
    prepared["_ui_meta"] = meta
    prepared["_ui_created_ts"] = created_ts or updated_ts
    prepared["_ui_updated_ts"] = updated_ts
    prepared["_ui_status"] = _storage_status(item)
    prepared["_ui_tags"] = _storage_tags(item)
    return prepared


def _format_time(ts: float) -> str:
    """Format a timestamp for compact UI display."""

    try:
        value = float(ts or 0.0)
    except Exception:
        return "-"
    if value <= 0:
        return "-"
    return datetime.fromtimestamp(value).strftime("%d.%m.%Y %H:%M")


def _window_match(ts: float, window: str) -> bool:
    """Return whether the timestamp matches the selected window."""

    if window == "all" or not ts:
        return True
    now = datetime.now()
    if window == "today":
        return datetime.fromtimestamp(ts).date() == now.date()
    days = {"7d": 7, "30d": 30, "90d": 90}.get(window)
    if days is None:
        return True
    return ts >= (now - timedelta(days=days)).timestamp()


def _filter_storage_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Apply compact storage filters."""

    query = str(st.session_state.get("storage_query") or "").strip().lower()
    status = str(st.session_state.get("storage_status") or "all")
    tag = str(st.session_state.get("storage_tag") or "all")
    window = str(st.session_state.get("storage_window") or "all")
    sort_key = str(st.session_state.get("storage_sort") or "newest")

    filtered: List[Dict[str, Any]] = []
    for item in items:
        if status != "all" and str(item.get("_ui_status") or "raw") != status:
            continue
        if tag != "all" and tag not in list(item.get("_ui_tags") or []):
            continue
        if not _window_match(float(item.get("_ui_created_ts") or 0.0), window):
            continue
        if query:
            haystack = " ".join(
                [
                    str(item.get("video_id") or "").lower(),
                    str(item.get("summary") or "").lower(),
                    " ".join(str(code).lower() for code in (item.get("languages") or [])),
                    " ".join(str(code).lower() for code in (item.get("_ui_tags") or [])),
                    str(item.get("path") or "").lower(),
                ]
            )
            if query not in haystack:
                continue
        filtered.append(item)

    if sort_key == "name":
        filtered.sort(key=lambda row: str(row.get("video_id") or "").lower())
    elif sort_key == "oldest":
        filtered.sort(key=lambda row: (float(row.get("_ui_created_ts") or 0.0), str(row.get("video_id") or "").lower()))
    else:
        filtered.sort(key=lambda row: (float(row.get("_ui_created_ts") or 0.0), str(row.get("video_id") or "").lower()), reverse=True)
    return filtered


def _storage_filter_signature() -> str:
    """Build a small signature for pagination reset."""

    return "|".join(
        [
            str(st.session_state.get("storage_query") or ""),
            str(st.session_state.get("storage_status") or ""),
            str(st.session_state.get("storage_tag") or ""),
            str(st.session_state.get("storage_window") or ""),
            str(st.session_state.get("storage_sort") or ""),
        ]
    )


def _profile_label(profile: str, lang: str) -> str:
    """Return a localized processing profile label."""

    return {
        "main": _loc("Основной", "Негізгі", "Main", lang=lang),
        "experimental": _loc("Экспериментальный", "Эксперименттік", "Experimental", lang=lang),
    }.get(str(profile or "").strip().lower(), str(profile or ""))


def _queue_job_status(item: Dict[str, Any], lang: str) -> str:
    """Return a localized job status."""

    raw = str(item.get("status") or item.get("state") or "").strip().lower()
    return {
        "queued": _loc("В очереди", "Кезекте", "Queued", lang=lang),
        "pending": _loc("Ожидание", "Күту", "Pending", lang=lang),
        "running": _loc("В работе", "Жұмыс үстінде", "Running", lang=lang),
        "paused": _loc("Пауза", "Аялда", "Paused", lang=lang),
        "failed": _loc("Ошибка", "Қате", "Failed", lang=lang),
        "done": _loc("Готово", "Дайын", "Done", lang=lang),
    }.get(raw, _loc("В очереди", "Кезекте", "Queued", lang=lang))


def _variant_option_label(token: str, lang: str) -> str:
    """Return a localized variant label for the queue controls."""

    if token == "__base__":
        return _loc("Базовый", "Базалық", "Base", lang=lang)
    if token == "__fanout__":
        return _loc("Все варианты", "Барлық нұсқа", "All variants", lang=lang)
    return variant_label(token)


def _storage_pagination(total_items: int, lang: str) -> None:
    """Render compact storage pagination."""

    total_pages = max(1, math.ceil(total_items / PAGE_SIZE))
    page = max(1, min(int(st.session_state.get("storage_page") or 1), total_pages))
    st.session_state["storage_page"] = page

    nav = st.columns([0.8, 1.8, 0.8], gap="small")
    with nav[0]:
        if st.button(ICON_PREV, key="storage_prev_page", use_container_width=True, help=_loc("Назад", "Артқа", "Previous", lang=lang), disabled=page <= 1):
            st.session_state["storage_page"] = page - 1
            st.rerun()
    with nav[1]:
        selected_page = st.selectbox(
            _loc("Страница", "Бет", "Page", lang=lang),
            options=list(range(1, total_pages + 1)),
            index=page - 1,
            label_visibility="collapsed",
            format_func=lambda value: _loc(f"Стр. {value} / {total_pages}", f"Бет {value} / {total_pages}", f"Page {value} / {total_pages}", lang=lang),
        )
        if int(selected_page) != page:
            st.session_state["storage_page"] = int(selected_page)
            st.rerun()
    with nav[2]:
        if st.button(ICON_NEXT, key="storage_next_page", use_container_width=True, help=_loc("Далее", "Келесі", "Next", lang=lang), disabled=page >= total_pages):
            st.session_state["storage_page"] = page + 1
            st.rerun()


def _analytics_status_items(outputs: Dict[str, Any], raw_exists: bool, lang: str) -> List[tuple[str, bool]]:
    """Return the localized status list for analytics."""

    metrics = outputs.get("metrics") if isinstance(outputs.get("metrics"), dict) else {}
    annotations = outputs.get("annotations") if isinstance(outputs.get("annotations"), list) else []
    indexing = metrics.get("indexing") if isinstance(metrics.get("indexing"), dict) else {}
    return [
        (_loc("Исходный файл", "Бастапқы файл", "Source file", lang=lang), raw_exists),
        (_loc("Результаты обработки", "Өңдеу нәтижелері", "Processed outputs", lang=lang), bool(annotations or outputs.get("global_summary"))),
        (_loc("Поисковый индекс", "Іздеу индексі", "Search index", lang=lang), bool(indexing)),
    ]


def _stage_timing_line(outputs: Dict[str, Any]) -> str:
    """Build a compact stage timing summary."""

    metrics = outputs.get("metrics") if isinstance(outputs.get("metrics"), dict) else {}
    stage_stats = metrics.get("stage_stats_sec") if isinstance(metrics.get("stage_stats_sec"), dict) else {}
    if not stage_stats:
        return "-"
    parts: List[str] = []
    for stage_name, payload in stage_stats.items():
        if not isinstance(payload, dict):
            continue
        mean_sec = payload.get("mean_sec")
        if mean_sec is None:
            continue
        label = humanize_token(stage_name).replace("Video", "").strip() or humanize_token(stage_name)
        parts.append(f"{label} {float(mean_sec):.1f}s")
    return " · ".join(parts[:4]) if parts else "-"


def _render_info_rows(rows: Iterable[tuple[str, str]]) -> None:
    """Render key-value rows."""

    html = []
    for label, value in rows:
        html.append(
            f"""
            <div class="info-row">
                <div class="info-label">{E(label)}</div>
                <div class="info-value">{E(value)}</div>
            </div>
            """
        )
    st.markdown("".join(html), unsafe_allow_html=True)


def _render_status_rows(rows: Iterable[tuple[str, bool]], lang: str) -> None:
    """Render status rows for the analytics side panel."""

    html = []
    for label, ready in rows:
        state = _loc("Готово", "Дайын", "Ready", lang=lang) if ready else _loc("Ожидание", "Күту", "Pending", lang=lang)
        css = "status-chip" if ready else "status-chip muted"
        html.append(
            f"""
            <div class="info-row">
                <div class="info-label">{E(label)}</div>
                <div class="info-value"><span class="{css}">{E(state)}</span></div>
            </div>
            """
        )
    st.markdown("".join(html), unsafe_allow_html=True)


def _render_timeline(annotations: List[Dict[str, Any]], lang: str) -> None:
    """Render the clean analytics timeline."""

    _section(_loc("Таймлайн", "Таймлайн", "Timeline", lang=lang))
    if not annotations:
        _caption(_loc("Сегменты ещё не готовы.", "Сегменттер әлі дайын емес.", "Segments are not available yet.", lang=lang))
        return

    visible = annotations[:60]
    for idx, ann in enumerate(visible):
        if not isinstance(ann, dict):
            continue
        start_sec = float(ann.get("start_sec", 0.0) or 0.0)
        end_sec = float(ann.get("end_sec", 0.0) or 0.0)
        description = str(ann.get("normalized_caption") or ann.get("description") or "").strip()
        notes = [str(note).strip() for note in (ann.get("anomaly_notes") or []) if str(note).strip()]
        risk = str(ann.get("risk_level") or "").strip().lower()
        is_alert = bool(ann.get("anomaly_flag")) or risk in {"attention", "warning", "critical"} or bool(notes)

        time_col, body_col = st.columns([1.0, 4.6], gap="medium")
        with time_col:
            if st.button(f"{mmss(start_sec)} - {mmss(end_sec)}", key=f"timeline_seek_{idx}", use_container_width=True):
                st.session_state["video_seek_sec"] = int(start_sec)
                st.rerun()
        with body_col:
            st.markdown(
                f"<div class='timeline-text{' alert' if is_alert else ''}'>{E(description or _loc('Нет описания', 'Сипаттама жоқ', 'No description', lang=lang))}</div>",
                unsafe_allow_html=True,
            )
            if notes:
                note = first_sentence(" ".join(notes))
                st.markdown(
                    f"<div class='timeline-note'>{E(_loc('Внимание', 'Назар', 'Attention', lang=lang))}: {E(note)}</div>",
                    unsafe_allow_html=True,
                )
        if idx < len(visible) - 1:
            st.markdown("<div class='timeline-divider'></div>", unsafe_allow_html=True)


def _bind_search_state() -> None:
    """Initialize search page state."""

    defaults = {
        "search_query_box": "",
        "search_video_id": "",
        "search_lang": "",
        "search_variant": "__base__",
        "search_topk": 8,
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


def _risk_label(value: str, lang: str) -> str:
    """Return a localized risk label."""

    return {
        "": _loc("Любой", "Кез келген", "Any", lang=lang),
        "normal": _loc("Нормальный", "Қалыпты", "Normal", lang=lang),
        "attention": _loc("Внимание", "Назар", "Attention", lang=lang),
        "warning": _loc("Предупреждение", "Ескерту", "Warning", lang=lang),
        "critical": _loc("Критический", "Сындарлы", "Critical", lang=lang),
    }.get(str(value or ""), str(value or ""))


def _search_status_line(lang: str, *, hits_count: Optional[int] = None, rebuild_status: str = "") -> None:
    """Render compact search status copy."""

    parts: List[str] = []
    if hits_count is not None:
        parts.append(_loc(f"Результатов: {hits_count}", f"Нәтижелер: {hits_count}", f"Results: {hits_count}", lang=lang))
    if rebuild_status.strip():
        parts.append(rebuild_status.strip())
    if not parts:
        return
    st.markdown(f"<div class='status-line'>{E(' · '.join(parts))}</div>", unsafe_allow_html=True)


def _chat_message(role: str, content: str, *, evidence: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Build one assistant message payload."""

    return {"role": role, "content": str(content or "").strip(), "evidence": list(evidence or [])}


def _render_chat(messages: List[Dict[str, Any]], lang: str, *, typing_text: str = "") -> None:
    """Render the assistant chat history."""

    if not messages and not typing_text:
        st.markdown(
            f"""
            <div class="chat-empty">
                <div class="chat-empty-title">{E(_loc('Спросите о найденных эпизодах', 'Табылған эпизодтар туралы сұраңыз', 'Ask about the retrieved episodes', lang=lang))}</div>
                <div class="chat-empty-copy">{E(_loc('Ассистент использует активные фильтры и найденные результаты, чтобы дать краткий ответ по событиям.', 'Көмекші белсенді сүзгілер мен табылған нәтижелерді пайдаланып, оқиғалар бойынша қысқа жауап береді.', 'The assistant uses the active filters and results to answer questions about the events.', lang=lang))}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    parts: List[str] = ["<div class='chat-thread'>"]
    for message in messages:
        role = str(message.get("role") or "assistant")
        bubble_role = "user" if role == "user" else "assistant"
        meta = _loc("Вы", "Сіз", "You", lang=lang) if role == "user" else _loc("Ассистент", "Көмекші", "Assistant", lang=lang)
        parts.append(
            f"""
            <div class="chat-row {bubble_role}">
                <div class="chat-bubble {bubble_role}">
                    <div class="chat-meta">{E(meta)}</div>
                    <div class="chat-text">{E(str(message.get('content') or ''))}</div>
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

    if typing_text:
        parts.append(
            f"""
            <div class="chat-row assistant">
                <div class="chat-bubble assistant">
                    <div class="chat-meta">{E(_loc('Ассистент', 'Көмекші', 'Assistant', lang=lang))}</div>
                    <div class="chat-text">{E(typing_text)}</div>
                    <div class="typing-dots"><span></span><span></span><span></span></div>
                </div>
            </div>
            """
        )
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def _run_search(client: BackendClient, base_lang: str) -> None:
    """Execute search using the visible controls."""

    hits = client.search(
        query=str(st.session_state.get("search_query_box") or "").strip(),
        top_k=int(st.session_state.get("search_topk") or 8),
        video_id=str(st.session_state.get("search_video_id") or "") or None,
        language=str(st.session_state.get("search_lang") or base_lang),
        variant=variant_from_token(st.session_state.get("search_variant")),
        dedupe=bool(st.session_state.get("search_dedupe", True)),
        event_type=str(st.session_state.get("search_event_type") or "").strip() or None,
        risk_level=str(st.session_state.get("search_risk_level") or "").strip() or None,
        people_count_bucket=str(st.session_state.get("search_people_count_bucket") or "").strip() or None,
        motion_type=str(st.session_state.get("search_motion_type") or "").strip() or None,
        anomaly_only=bool(st.session_state.get("search_anomaly_only")),
    )
    st.session_state["search_hits"] = hits


def _clear_search_state() -> None:
    """Reset only the search controls, not the chat history."""

    st.session_state["search_query_box"] = ""
    st.session_state["search_event_type"] = ""
    st.session_state["search_risk_level"] = ""
    st.session_state["search_motion_type"] = ""
    st.session_state["search_people_count_bucket"] = ""
    st.session_state["search_anomaly_only"] = False
    st.session_state["search_dedupe"] = True
    st.session_state["search_hits"] = []
    st.session_state["search_rebuild_status"] = ""


def _process_chat_prompt(client: BackendClient, lang: str, base_lang: str, history_placeholder: Any) -> None:
    """Submit one RAG prompt and animate the assistant response."""

    prompt = str(st.session_state.get("search_rag_input") or "").strip()
    if not prompt:
        return

    messages = list(st.session_state.get("search_rag_messages") or [])
    messages.append(_chat_message("user", prompt))
    st.session_state["search_rag_messages"] = messages

    with history_placeholder.container():
        _render_chat(messages, lang, typing_text=_loc("Подбираю эпизоды и формирую ответ...", "Эпизодтарды іріктеп, жауап дайындап жатырмын...", "Finding evidence and preparing the answer...", lang=lang))

    try:
        rag = client.ask_rag(
            query=prompt,
            language=str(st.session_state.get("search_lang") or base_lang),
            variant=variant_from_token(st.session_state.get("search_variant")),
            video_id=str(st.session_state.get("search_video_id") or "") or None,
            top_k=int(st.session_state.get("search_topk") or 8),
        )
        answer = str(rag.get("answer") or "").strip() or _loc(
            "Не удалось подготовить ответ. Попробуйте уточнить запрос.",
            "Жауап дайындау мүмкін болмады. Сұрауды нақтылап көріңіз.",
            "The assistant could not prepare an answer. Try a more specific prompt.",
            lang=lang,
        )
        evidence = list(rag.get("supporting_hits") or [])
    except Exception as exc:
        answer = _loc(f"Не удалось получить ответ: {exc}", f"Жауап алу мүмкін болмады: {exc}", f"Failed to get an answer: {exc}", lang=lang)
        evidence = []

    partial = ""
    for token in answer.split():
        partial = (partial + " " + token).strip()
        with history_placeholder.container():
            _render_chat(messages, lang, typing_text=partial)
        time.sleep(0.02)

    messages.append(_chat_message("assistant", answer, evidence=evidence))
    st.session_state["search_rag_messages"] = messages
    st.session_state["search_rag_input"] = ""
    st.rerun()


def _bind_reports_state() -> None:
    """Initialize reports page state."""

    defaults = {
        "reports_video_id": "",
        "reports_lang": "",
        "reports_variant": "__base__",
        "reports_focus": "",
        "reports_payload": {},
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _report_overview_rows(video_id: str, lang_code: str, variant_token: str, outputs: Dict[str, Any], lang: str) -> List[tuple[str, str]]:
    """Build the reports snapshot rows."""

    metrics = outputs.get("metrics") if isinstance(outputs.get("metrics"), dict) else {}
    annotations = outputs.get("annotations") if isinstance(outputs.get("annotations"), list) else []
    return [
        (_loc("Видео", "Бейне", "Video", lang=lang), video_id or "-"),
        (_loc("Язык", "Тіл", "Language", lang=lang), str(lang_code or "-").upper()),
        (_loc("Вариант", "Нұсқа", "Variant", lang=lang), variant_label(variant_token)),
        (_loc("Сегменты", "Сегменттер", "Segments", lang=lang), str(len(annotations))),
        (_loc("Общее время", "Жалпы уақыт", "Total time", lang=lang), f"{float(metrics.get('total_time_sec') or 0.0):.1f}s" if metrics.get("total_time_sec") is not None else "-"),
    ]


def _storage_upload_panel(client: BackendClient, lang: str) -> None:
    """Render one-shot upload inside the storage side rail."""

    _section(_loc("Загрузка", "Жүктеу", "Upload", lang=lang))
    upload = st.file_uploader(
        _loc("Выберите видео", "Бейне таңдаңыз", "Choose video", lang=lang),
        type=["mp4", "mov", "mkv", "avi"],
        key="storage_upload",
        label_visibility="collapsed",
    )
    if upload is None:
        st.session_state["storage_upload_sig"] = ""
        return

    payload = upload.getvalue()
    upload_sig = f"{upload.name}:{len(payload)}"
    if str(st.session_state.get("storage_upload_sig") or "") == upload_sig:
        return

    st.session_state["storage_upload_sig"] = upload_sig
    try:
        result = client.upload_video(upload.name, payload)
        st.session_state["selected_video_id"] = str(result.get("video_id") or Path(upload.name).stem)
        soft_note(_loc("Видео загружено", "Бейне жүктелді", "Video uploaded", lang=lang), kind="ok")
        st.rerun()
    except Exception as exc:
        soft_note(f"{_error_prefix(lang)}: {exc}", kind="warn")


def _queue_state_line(queue: Dict[str, Any], selected_video_id: str, lang: str) -> None:
    """Render one compact queue status line."""

    _ = selected_video_id
    status = queue.get("status") if isinstance(queue, dict) else {}
    running = queue.get("running") if isinstance(queue, dict) else {}
    queued = queue.get("queued") if isinstance(queue, dict) else []

    is_paused = bool((status or {}).get("paused"))
    state_text = _loc("Пауза", "Аялда", "Paused", lang=lang) if is_paused else _loc("Активна", "Белсенді", "Active", lang=lang)
    running_text = str((running or {}).get("video_id") or "").strip() or _loc("нет", "жоқ", "none", lang=lang)
    queue_label = _loc("В очереди", "Кезекте", "Queued", lang=lang)
    current_label = _loc("Сейчас", "Қазір", "Now", lang=lang)

    st.markdown(
        f"""
        <div class="queue-status">
            <span class="status-chip">{E(state_text)}</span>
            <span class="queue-copy">{E(f'{queue_label}: {len(queued)} · {current_label}: {running_text}')}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_queue_panel(client: BackendClient, cfg: Any, queue: Dict[str, Any], selected_video_id: str, lang: str) -> None:
    """Render the storage queue controls."""

    _section(_loc("Очередь обработки", "Өңдеу кезегі", "Processing queue", lang=lang))

    profile_options = ["main", "experimental"]
    if "run_profile" not in st.session_state:
        st.session_state["run_profile"] = "main"
    if st.session_state.get("run_profile") not in profile_options:
        st.session_state["run_profile"] = "main"

    st.selectbox(
        _loc("Профиль", "Профиль", "Profile", lang=lang),
        options=profile_options,
        key="run_profile",
        format_func=lambda value: _profile_label(str(value), lang),
    )

    variants = available_variant_ids(cfg)
    variant_options = ["__base__"] if st.session_state.get("run_profile") == "main" else ["__fanout__"] + variants
    default_variant = "__base__" if st.session_state.get("run_profile") == "main" else "__fanout__"
    current_variant = str(st.session_state.get("run_variant") or default_variant)
    if current_variant not in variant_options:
        current_variant = default_variant
        st.session_state["run_variant"] = current_variant

    st.selectbox(
        _loc("Вариант", "Нұсқа", "Variant", lang=lang),
        options=variant_options,
        index=variant_options.index(current_variant),
        key="run_variant",
        format_func=lambda token: _variant_option_label(str(token), lang),
    )
    st.checkbox(_loc("Перезапись", "Қайта жазу", "Overwrite", lang=lang), key="run_force_overwrite")

    action_row = st.columns(3, gap="small")
    with action_row[0]:
        if st.button(ICON_START, key="queue_start_btn", use_container_width=True, help=_loc("Запустить обработку", "Өңдеуді бастау", "Start processing", lang=lang), disabled=not bool(selected_video_id)):
            try:
                selected_variant = None
                if st.session_state.get("run_profile") == "experimental" and st.session_state.get("run_variant") not in {"__fanout__", "__base__"}:
                    selected_variant = variant_from_token(st.session_state.get("run_variant"))
                payload = client.create_job(
                    str(selected_video_id or ""),
                    extra={"force_overwrite": bool(st.session_state.get("run_force_overwrite", False))},
                    profile=str(st.session_state.get("run_profile") or "main"),
                    variant=selected_variant,
                )
                soft_note(f"{_loc('Задача добавлена', 'Тапсырма қосылды', 'Job queued', lang=lang)}: {payload.get('job_id')}", kind="ok")
            except Exception as exc:
                soft_note(f"{_error_prefix(lang)}: {exc}", kind="warn")
    with action_row[1]:
        paused = bool((queue.get("status") or {}).get("paused")) if isinstance(queue, dict) else False
        if paused:
            if st.button(ICON_RESUME, key="queue_resume_btn", use_container_width=True, help=_loc("Продолжить", "Жалғастыру", "Resume", lang=lang)):
                try:
                    client.queue_resume()
                    st.rerun()
                except Exception as exc:
                    soft_note(f"{_error_prefix(lang)}: {exc}", kind="warn")
        else:
            if st.button(ICON_PAUSE, key="queue_pause_btn", use_container_width=True, help=_loc("Пауза", "Аялда", "Pause", lang=lang)):
                try:
                    client.queue_pause()
                    st.rerun()
                except Exception as exc:
                    soft_note(f"{_error_prefix(lang)}: {exc}", kind="warn")
    with action_row[2]:
        if st.button(ICON_REFRESH, key="queue_refresh_btn", use_container_width=True, help=_loc("Обновить", "Жаңарту", "Refresh", lang=lang)):
            st.rerun()

    _queue_state_line(queue, selected_video_id, lang)

    queued = queue.get("queued") if isinstance(queue, dict) else []
    if not isinstance(queued, list) or not queued:
        _caption(_loc("Очередь пока пуста.", "Кезек әзірге бос.", "Queue is empty.", lang=lang))
        return

    for idx, item in enumerate(queued[:12]):
        if not isinstance(item, dict):
            continue
        job_id = str(item.get("job_id") or "")
        variant_token = "__base__" if not item.get("variant") else str(item.get("variant"))
        with st.container():
            _mark("row-marker")
            head, status_col = st.columns([3.1, 1.0], gap="small")
            with head:
                title = f"{idx + 1}. {str(item.get('video_id') or '-')}"
                meta = " · ".join(
                    [
                        _profile_label(str(item.get("profile") or "main"), lang),
                        _variant_option_label(variant_token, lang),
                        humanize_token(str(item.get("job_type") or "process")),
                    ]
                )
                st.markdown(f"<div class='row-title'>{E(title)}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='row-meta'>{E(meta)}</div>", unsafe_allow_html=True)
            with status_col:
                st.markdown(f"<div class='row-status'>{E(_queue_job_status(item, lang))}</div>", unsafe_allow_html=True)

            actions = st.columns(3, gap="small")
            with actions[0]:
                if st.button(ICON_UP, key=f"queue_up_{job_id}_{idx}", use_container_width=True, help=_loc("Выше", "Жоғары", "Move up", lang=lang)):
                    try:
                        client.queue_move(job_id, "up")
                        st.rerun()
                    except Exception as exc:
                        soft_note(f"{_error_prefix(lang)}: {exc}", kind="warn")
            with actions[1]:
                if st.button(ICON_DOWN, key=f"queue_down_{job_id}_{idx}", use_container_width=True, help=_loc("Ниже", "Төмен", "Move down", lang=lang)):
                    try:
                        client.queue_move(job_id, "down")
                        st.rerun()
                    except Exception as exc:
                        soft_note(f"{_error_prefix(lang)}: {exc}", kind="warn")
            with actions[2]:
                if st.button(ICON_DELETE, key=f"queue_delete_{job_id}_{idx}", use_container_width=True, help=_loc("Убрать", "Алып тастау", "Remove", lang=lang)):
                    try:
                        client.queue_remove(job_id)
                        st.rerun()
                    except Exception as exc:
                        soft_note(f"{_error_prefix(lang)}: {exc}", kind="warn")


def video_analytics_tab(client: BackendClient, cfg: Any, ui_text: Dict[str, Dict[str, Any]]) -> None:
    """Render the analytics page without nested panel wrappers."""

    _ = ui_text
    lang = _ui_lang(cfg)
    videos = _video_items(client)
    if not videos:
        soft_note(_loc("Библиотека пуста. Загрузите видео, чтобы начать.", "Қойма бос. Бастау үшін бейне жүктеңіз.", "The library is empty. Upload a video to begin.", lang=lang))
        return

    context = _resolve_video_context(
        videos,
        video_key="selected_video_id",
        variant_key="library_variant",
        lang_key="library_lang",
        base_lang=lang,
    )
    video_ids = context["video_ids"]
    variant_tokens = context["variant_tokens"]
    available_languages = context["available_languages"]

    _page_title(_loc("Видеоаналитика", "Бейне аналитикасы", "Video Analytics", lang=lang))

    controls = st.columns([2.2, 1.0, 1.0], gap="small")
    with controls[0]:
        st.selectbox(_loc("Видео", "Бейне", "Video", lang=lang), options=video_ids, key="selected_video_id")
    with controls[1]:
        st.selectbox(_loc("Вариант", "Нұсқа", "Variant", lang=lang), options=variant_tokens, key="library_variant", format_func=variant_label)
    with controls[2]:
        st.selectbox(_loc("Язык", "Тіл", "Language", lang=lang), options=available_languages, key="library_lang")

    selected_video = next((item for item in videos if str(item.get("video_id") or "") == st.session_state.get("selected_video_id")), None) or {}
    raw_path = Path(str(selected_video.get("path") or ""))
    playable_path = ensure_browser_video(raw_path) if raw_path.exists() else raw_path
    raw_exists = raw_path.exists()
    meta = get_video_meta(str(playable_path), mtime(playable_path)) if playable_path.exists() else {}

    try:
        outputs = client.get_video_outputs(
            str(st.session_state.get("selected_video_id") or ""),
            str(st.session_state.get("library_lang") or "en"),
            variant=variant_from_token(st.session_state.get("library_variant")),
        )
    except Exception as exc:
        outputs = {}
        soft_note(f"{_error_prefix(lang)}: {exc}", kind="warn")

    top = st.columns([1.88, 1.12], gap="large")
    with top[0]:
        if playable_path.exists():
            try:
                st.video(str(playable_path), start_time=int(st.session_state.get("video_seek_sec") or 0))
            except TypeError:
                st.video(str(playable_path))
        else:
            soft_note(_loc("Исходное видео не найдено.", "Бастапқы бейне табылмады.", "Source video not found.", lang=lang), kind="warn")

    with top[1]:
        summary_text = str(outputs.get("global_summary") or "").strip()
        headline = first_sentence(summary_text) or _loc("Сводка появится после обработки.", "Қысқаша мазмұн өңдеуден кейін пайда болады.", "The summary appears after processing.", lang=lang)
        _section(_loc("Описание видео", "Бейне сипаттамасы", "Video description", lang=lang))
        st.markdown(f"<div class='analytics-summary'>{E(headline)}</div>", unsafe_allow_html=True)
        if summary_text and summary_text.strip() != headline.strip():
            st.markdown(f"<div class='analytics-copy'>{E(clip_text(summary_text, 320))}</div>", unsafe_allow_html=True)

        _section(_loc("Статус", "Күйі", "Status", lang=lang))
        _render_status_rows(_analytics_status_items(outputs, raw_exists, lang), lang)

        _section(_loc("Метрики", "Метрикалар", "Metrics", lang=lang))
        _render_info_rows(
            [
                (_loc("Общее время обработки", "Жалпы өңдеу уақыты", "Total processing time", lang=lang), f"{float(((outputs.get('metrics') or {}).get('total_time_sec') or 0.0)):.1f}s" if ((outputs.get("metrics") or {}).get("total_time_sec") is not None) else "-"),
                (_loc("Время по этапам", "Кезеңдер уақыты", "Stage processing time", lang=lang), _stage_timing_line(outputs)),
                (_loc("Длительность видео", "Бейне ұзақтығы", "Video duration", lang=lang), hms(float(meta.get("duration_sec") or 0.0)) if meta.get("duration_sec") else "-"),
                (_loc("Разрешение", "Ажыратылымдылығы", "Video resolution", lang=lang), f"{int(meta.get('width'))}x{int(meta.get('height'))}" if meta.get("width") and meta.get("height") else "-"),
            ]
        )

    annotations = outputs.get("annotations") if isinstance(outputs.get("annotations"), list) else []
    _render_timeline(annotations, lang)


def _render_supporting_evidence(hits: List[Dict[str, Any]], lang: str) -> None:
    """Render supporting evidence as clean divider rows."""

    _section(_loc("Опорные эпизоды", "Тірек эпизодтар", "Supporting evidence", lang=lang))
    if not hits:
        _caption(_loc("Эпизоды появятся после построения отчёта.", "Эпизодтар есеп құрылғаннан кейін көрінеді.", "Supporting evidence appears after the report is built.", lang=lang))
        return

    visible_hits = [hit for hit in hits[:8] if isinstance(hit, dict)]
    for idx, hit in enumerate(visible_hits):
        headline = f"{str(hit.get('video_id') or '-')} · {mmss(float(hit.get('start_sec', 0.0) or 0.0))} - {mmss(float(hit.get('end_sec', 0.0) or 0.0))}"
        st.markdown(f"<div class='row-title'>{E(headline)}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='row-copy'>{E(clip_text(str(hit.get('description') or ''), 220))}</div>", unsafe_allow_html=True)
        if idx < len(visible_hits) - 1:
            st.markdown("<div class='timeline-divider'></div>", unsafe_allow_html=True)


def _render_storage_card(client: BackendClient, cfg: Any, item: Dict[str, Any], thumbs_dir: Path, selected_video_id: str, lang: str) -> None:
    """Render one storage video row with queue action."""

    _ = selected_video_id
    video_id = str(item.get("video_id") or "")
    summary = clip_text(first_sentence(str(item.get("summary") or "")), 150)
    meta = item.get("_ui_meta") if isinstance(item.get("_ui_meta"), dict) else {}
    languages = [str(code).upper() for code in (item.get("languages") or []) if str(code).strip()]
    tags = [_storage_tag_label(tag, lang) for tag in list(item.get("_ui_tags") or [])[:2]]
    facts = [
        fmt_bytes(meta.get("size_bytes")),
        hms(float(meta.get("duration_sec") or 0.0)) if meta.get("duration_sec") else "-",
        _loc("Готово", "Дайын", "Ready", lang=lang) if str(item.get("_ui_status")) == "ready" else _loc("Сырой файл", "Шикі файл", "Raw file", lang=lang),
        _format_time(float(item.get("_ui_created_ts") or 0.0)),
    ]
    meta_line = " · ".join(part for part in facts if part and part != "-") or "-"
    extra_line = " · ".join(languages + tags)
    thumb_path = thumbs_dir / f"{video_id}.jpg"

    with st.container():
        _mark("row-marker")
        preview_col, copy_col, actions_col = st.columns([1.05, 3.4, 0.92], gap="medium")
        with preview_col:
            if thumb_path.exists():
                st.image(str(thumb_path), use_container_width=True)
            else:
                st.markdown(f"<div class='thumb-fallback'>{E(video_id)}</div>", unsafe_allow_html=True)
        with copy_col:
            st.markdown(f"<div class='row-title'>{E(video_id)}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='row-meta'>{E(meta_line)}</div>", unsafe_allow_html=True)
            if extra_line:
                st.markdown(f"<div class='row-meta subtle'>{E(extra_line)}</div>", unsafe_allow_html=True)
            if summary:
                st.markdown(f"<div class='row-copy'>{E(summary)}</div>", unsafe_allow_html=True)
        with actions_col:
            if st.button(ICON_START, key=f"queue_video_{video_id}", use_container_width=True, help=_loc("Добавить в очередь", "Кезекке қосу", "Add to queue", lang=lang)):
                try:
                    profile_options = ["main", "experimental"]
                    profile = str(st.session_state.get("run_profile") or "main")
                    if profile not in profile_options:
                        profile = "main"
                        st.session_state["run_profile"] = profile

                    variants = available_variant_ids(cfg)
                    variant_options = ["__base__"] if profile == "main" else ["__fanout__"] + variants
                    default_variant = "__base__" if profile == "main" else "__fanout__"
                    current_variant = str(st.session_state.get("run_variant") or default_variant)
                    if current_variant not in variant_options:
                        current_variant = default_variant
                        st.session_state["run_variant"] = current_variant

                    selected_variant = None
                    if profile == "experimental" and current_variant not in {"__fanout__", "__base__"}:
                        selected_variant = variant_from_token(current_variant)

                    client.create_job(
                        video_id,
                        extra={"force_overwrite": bool(st.session_state.get("run_force_overwrite", False))},
                        profile=profile,
                        variant=selected_variant,
                    )
                    st.rerun()
                except Exception as exc:
                    soft_note(f"{_error_prefix(lang)}: {exc}", kind="warn")
            if st.button(ICON_OPEN, key=f"open_video_{video_id}", use_container_width=True, help=_loc("Открыть аналитику", "Аналитиканы ашу", "Open analytics", lang=lang)):
                st.session_state["selected_video_id"] = video_id
                st.session_state["video_seek_sec"] = 0
                st.query_params["tab"] = "video"
                st.rerun()
            if st.button(ICON_DELETE, key=f"delete_video_{video_id}", use_container_width=True, help=_loc("Удалить", "Жою", "Delete", lang=lang)):
                st.session_state["confirm_delete_video_id"] = video_id

    if st.session_state.get("confirm_delete_video_id") == video_id:
        confirm_cols = st.columns([2.7, 0.8, 0.8], gap="small")
        with confirm_cols[0]:
            st.markdown(
                f"<div class='confirm-copy'>{E(_loc('Удалить выбранное видео из библиотеки?', 'Таңдалған бейнені кітапханадан өшіру керек пе?', 'Delete the selected video from the library?', lang=lang))}</div>",
                unsafe_allow_html=True,
            )
        with confirm_cols[1]:
            if st.button(ICON_CONFIRM, key=f"confirm_delete_{video_id}", use_container_width=True, help=_loc("Подтвердить", "Растау", "Confirm", lang=lang)):
                try:
                    client.delete_video(video_id)
                    if st.session_state.get("selected_video_id") == video_id:
                        st.session_state["selected_video_id"] = ""
                    st.session_state["confirm_delete_video_id"] = ""
                    soft_note(_loc("Видео удалено", "Бейне жойылды", "Video deleted", lang=lang), kind="ok")
                    st.rerun()
                except Exception as exc:
                    soft_note(f"{_error_prefix(lang)}: {exc}", kind="warn")
        with confirm_cols[2]:
            if st.button(ICON_DELETE, key=f"cancel_delete_{video_id}", use_container_width=True, help=_loc("Отмена", "Болдырмау", "Cancel", lang=lang)):
                st.session_state["confirm_delete_video_id"] = ""
                st.rerun()


def storage_tab(client: BackendClient, cfg: Any, ui_text: Dict[str, Dict[str, Any]]) -> None:
    """Render the storage page with a flat side rail."""

    _ = ui_text
    _bind_storage_state()
    lang = _ui_lang(cfg)

    videos = _video_items(client)
    prepared = [_prepare_storage_item(item) for item in videos]
    video_ids = _video_ids(prepared)
    if video_ids:
        selected = _session_choice("selected_video_id", video_ids)
    else:
        st.session_state["selected_video_id"] = ""
        selected = ""

    try:
        queue = client.queue_list()
    except Exception:
        queue = {}

    filter_sig = _storage_filter_signature()
    if filter_sig != str(st.session_state.get("storage_filter_sig") or ""):
        st.session_state["storage_filter_sig"] = filter_sig
        st.session_state["storage_page"] = 1

    _page_title(_loc("Хранилище", "Қойма", "Storage", lang=lang))

    layout = st.columns([1.9, 0.96], gap="large")
    with layout[0]:
        _section(_loc("Видеотека", "Бейне кітапханасы", "Video library", lang=lang))
        tag_options = ["all"] + sorted({tag for item in prepared for tag in list(item.get("_ui_tags") or [])}, key=lambda value: _storage_tag_label(value, lang))
        filters = st.columns([2.25, 1.0, 1.0, 1.05, 0.75], gap="small")
        with filters[0]:
            st.text_input(
                _loc("Поиск", "Іздеу", "Search", lang=lang),
                key="storage_query",
                placeholder=_loc("ID, путь, сводка, язык", "ID, жол, қысқаша мазмұн, тіл", "ID, path, summary, language", lang=lang),
            )
        with filters[1]:
            st.selectbox(
                _loc("Сортировка", "Сұрыптау", "Sort", lang=lang),
                options=["newest", "oldest", "name"],
                key="storage_sort",
                format_func=lambda value: {
                    "newest": _loc("Новые", "Жаңалары", "Newest", lang=lang),
                    "oldest": _loc("Старые", "Ескілері", "Oldest", lang=lang),
                    "name": _loc("Имя / ID", "Аты / ID", "Name / ID", lang=lang),
                }.get(value, value),
            )
        with filters[2]:
            st.selectbox(
                _loc("Статус", "Күйі", "Status", lang=lang),
                options=["all", "ready", "raw"],
                key="storage_status",
                format_func=lambda value: {
                    "all": _loc("Все", "Барлығы", "All", lang=lang),
                    "ready": _loc("Готово", "Дайын", "Ready", lang=lang),
                    "raw": _loc("Сырой файл", "Шикі файл", "Raw file", lang=lang),
                }.get(value, value),
            )
        with filters[3]:
            st.selectbox(
                _loc("Тег", "Тег", "Tag", lang=lang),
                options=tag_options,
                key="storage_tag",
                format_func=lambda value: _loc("Любой", "Кез келген", "Any", lang=lang) if value == "all" else _storage_tag_label(str(value), lang),
            )
        with filters[4]:
            st.markdown("<div class='filter-button-offset'></div>", unsafe_allow_html=True)
            if st.button(ICON_CLEAR, key="storage_clear_filters", use_container_width=True, help=_loc("Сбросить фильтры", "Сүзгілерді тазарту", "Clear filters", lang=lang)):
                st.session_state["storage_query"] = ""
                st.session_state["storage_sort"] = "newest"
                st.session_state["storage_status"] = "all"
                st.session_state["storage_tag"] = "all"
                st.rerun()

        filtered = _filter_storage_items(prepared)
        if not filtered and prepared:
            _caption(_loc("По текущим фильтрам ничего не найдено.", "Ағымдағы сүзгілер бойынша ештеңе табылмады.", "No videos matched the current filters.", lang=lang))
        elif not filtered:
            _caption(_loc("Библиотека пуста. Загрузите первое видео.", "Қойма бос. Алғашқы бейнені жүктеңіз.", "The library is empty. Upload the first video.", lang=lang))
        else:
            page = max(1, int(st.session_state.get("storage_page") or 1))
            total_pages = max(1, math.ceil(len(filtered) / PAGE_SIZE))
            page = min(page, total_pages)
            st.session_state["storage_page"] = page
            start = (page - 1) * PAGE_SIZE
            end = start + PAGE_SIZE
            thumbs_dir = Path(cfg.paths.thumbs_dir)
            for item in filtered[start:end]:
                _render_storage_card(client, cfg, item, thumbs_dir, selected, lang)
            _storage_pagination(len(filtered), lang)

    with layout[1]:
        _mark("storage-side-marker")
        _storage_upload_panel(client, lang)
        _render_queue_panel(client, cfg, queue, selected, lang)


def _run_search_from_enter(client: BackendClient, base_lang: str) -> None:
    """Run search when the user confirms the query input."""

    try:
        _run_search(client, base_lang)
    except Exception as exc:
        st.session_state["search_hits"] = []
        soft_note(f"{_error_prefix(base_lang)}: {exc}", kind="warn")


def _submit_chat_from_enter(client: BackendClient, lang: str, base_lang: str) -> None:
    """Submit one chat prompt from the Enter key without extra UI wrappers."""

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
            top_k=int(st.session_state.get("search_topk") or 8),
        )
        answer = str(rag.get("answer") or "").strip() or _loc(
            "Не удалось подготовить ответ. Попробуйте уточнить запрос.",
            "Жауап дайындау мүмкін болмады. Сұрауды нақтылап көріңіз.",
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
    """Render search results as flat rows inside the search page."""

    _section(_loc("Результаты", "Нәтижелер", "Results", lang=lang))
    visible_hits = [hit for hit in hits[:20] if isinstance(hit, dict)]
    total_hits = len([hit for hit in hits if isinstance(hit, dict)])
    st.markdown(
        f"<div class='status-line'>{E(_loc(f'Найдено: {total_hits}', f'Табылды: {total_hits}', f'Found: {total_hits}', lang=lang))}</div>",
        unsafe_allow_html=True,
    )
    if not visible_hits:
        _caption(_loc("Результаты появятся после запуска поиска.", "Нәтижелер іздеуден кейін көрінеді.", "Results appear after search.", lang=lang))
        return

    for idx, hit in enumerate(visible_hits, start=1):
        anomaly = bool(hit.get("anomaly_flag"))
        info_col, action_col = st.columns([4.8, 0.8], gap="small")
        with info_col:
            headline = f"#{idx:02d} · {str(hit.get('video_id') or '-')} · {mmss(float(hit.get('start_sec', 0.0) or 0.0))} - {mmss(float(hit.get('end_sec', 0.0) or 0.0))}"
            st.markdown(f"<div class='row-title'>{E(headline)}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='row-copy{' alert' if anomaly else ''}'>{E(clip_text(str(hit.get('description') or ''), 220))}</div>", unsafe_allow_html=True)
            score_line = f"{_loc('Счет', 'Ұпай', 'Score', lang=lang)}: {float(hit.get('score', 0.0) or 0.0):.3f}"
            meta_bits = [score_line]
            for key in ("event_type", "risk_level", "motion_type", "people_count_bucket"):
                value = str(hit.get(key) or "").strip()
                if value:
                    meta_bits.append(humanize_token(value))
            st.markdown(f"<div class='row-meta'>{E(' · '.join(meta_bits))}</div>", unsafe_allow_html=True)
        with action_col:
            if st.button(ICON_OPEN, key=f"search_open_{idx}", use_container_width=True, help=_loc("Открыть в аналитике", "Аналитикада ашу", "Open in analytics", lang=lang)):
                st.session_state["selected_video_id"] = str(hit.get("video_id") or "")
                st.session_state["video_seek_sec"] = int(float(hit.get("start_sec", 0.0) or 0.0))
                st.query_params["tab"] = "video"
                st.rerun()
        if idx < len(visible_hits):
            st.markdown("<div class='timeline-divider'></div>", unsafe_allow_html=True)


@st.fragment
def _render_search_panel_fragment(
    client: BackendClient,
    lang: str,
    current_lang: str,
    video_ids: List[str],
    available_languages: List[str],
    variant_tokens: List[str],
) -> None:
    """Render the interactive search controls and results as an isolated fragment."""

    _section(_loc("Запрос и фильтры", "Сұрау және сүзгілер", "Query and filters", lang=lang))
    st.text_input(
        _loc("Запрос", "Сұрау", "Query", lang=lang),
        key="search_query_box",
        placeholder=_loc("Например: человек у входа, падение, толпа", "Мысалы: кіреберістегі адам, құлау, топ", "For example: person near the entrance, fall, crowd", lang=lang),
        on_change=_run_search_from_enter,
        args=(client, lang),
    )

    row1 = st.columns([1.55, 0.85, 0.9, 0.7], gap="small")
    with row1[0]:
        st.selectbox(
            _loc("Видео", "Бейне", "Video", lang=lang),
            options=video_ids,
            key="search_video_id",
            format_func=lambda value: value or _loc("Все видео", "Барлық бейне", "All videos", lang=lang),
        )
    with row1[1]:
        st.selectbox(_loc("Язык", "Тіл", "Language", lang=lang), options=available_languages or [lang], key="search_lang")
    with row1[2]:
        st.selectbox(_loc("Вариант", "Нұсқа", "Variant", lang=lang), options=variant_tokens, key="search_variant", format_func=variant_label)
    with row1[3]:
        st.number_input(_loc("Top", "Top", "Top", lang=lang), min_value=1, max_value=20, key="search_topk")

    row2 = st.columns([1.15, 1.0, 1.0, 1.05], gap="small")
    with row2[0]:
        st.text_input(
            _loc("Тип события", "Оқиға түрі", "Event type", lang=lang),
            key="search_event_type",
            placeholder=_loc("вход, падение, бег", "кіру, құлау, жүгіру", "entry, fall, running", lang=lang),
        )
    with row2[1]:
        st.selectbox(
            _loc("Риск", "Тәуекел", "Risk", lang=lang),
            options=["", "normal", "attention", "warning", "critical"],
            key="search_risk_level",
            format_func=lambda value: _risk_label(str(value), lang),
        )
    with row2[2]:
        st.text_input(
            _loc("Движение", "Қозғалыс", "Motion", lang=lang),
            key="search_motion_type",
            placeholder=_loc("стоит, идёт, бежит", "тұр, жүр, жүгіріп келеді", "standing, walking, running", lang=lang),
        )
    with row2[3]:
        st.text_input(
            _loc("Люди в кадре", "Кадрдағы адамдар", "People count", lang=lang),
            key="search_people_count_bucket",
            placeholder=_loc("1, 2-3, группа", "1, 2-3, топ", "1, 2-3, group", lang=lang),
        )

    controls_row = st.columns([1.08, 1.08, 0.95, 1.15], gap="small")
    with controls_row[0]:
        st.checkbox(_loc("Только аномалии", "Тек ауытқулар", "Anomalies only", lang=lang), key="search_anomaly_only")
    with controls_row[1]:
        st.checkbox(_loc("Скрывать повторы", "Қайталануды жасыру", "Hide duplicates", lang=lang), key="search_dedupe")
    with controls_row[2]:
        if st.button(ICON_CLEAR, key="search_clear_btn", use_container_width=True, help=_loc("Очистить фильтры", "Сүзгілерді тазарту", "Clear filters", lang=lang)):
            _clear_search_state()
    with controls_row[3]:
        if st.button(ICON_REFRESH, key="search_rebuild_btn", use_container_width=True, help=_loc("Пересобрать индекс", "Индексті жаңарту", "Rebuild index", lang=lang)):
            try:
                result = client.index_rebuild()
                message = str(result.get("message") or _loc("Индекс обновлён.", "Индекс жаңартылды.", "Index rebuilt.", lang=lang))
                st.session_state["search_rebuild_status"] = message
            except Exception as exc:
                soft_note(f"{_error_prefix(lang)}: {exc}", kind="warn")

    _search_status_line(lang, rebuild_status=str(st.session_state.get("search_rebuild_status") or ""))
    _render_search_results(list(st.session_state.get("search_hits") or []), lang)


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


def reports_metrics_tab(client: BackendClient, cfg: Any, ui_text: Dict[str, Dict[str, Any]]) -> None:
    """Render the reports page without nested panel wrappers."""

    _ = ui_text
    _bind_reports_state()
    lang = _ui_lang(cfg)
    videos = _video_items(client)
    if not videos:
        soft_note(_loc("Библиотека пуста. Сначала загрузите видео.", "Қойма бос. Алдымен бейне жүктеңіз.", "The library is empty. Upload a video to begin.", lang=lang))
        return

    context = _resolve_video_context(
        videos,
        video_key="reports_video_id",
        variant_key="reports_variant",
        lang_key="reports_lang",
        base_lang=lang,
    )
    video_ids = context["video_ids"]
    variant_tokens = context["variant_tokens"]
    selected_video_id = context["selected_video_id"]
    current_variant = context["current_variant"]
    available_languages = context["available_languages"]
    current_lang = context["current_lang"]

    try:
        outputs = client.get_video_outputs(
            selected_video_id,
            str(st.session_state.get("reports_lang") or lang),
            variant=variant_from_token(st.session_state.get("reports_variant")),
        )
    except Exception:
        outputs = {}

    _page_title(_loc("Отчёты", "Есептер", "Reports", lang=lang))

    controls = st.columns([1.8, 0.95, 0.95], gap="small")
    with controls[0]:
        st.selectbox(_loc("Видео", "Бейне", "Video", lang=lang), options=video_ids, key="reports_video_id")
    with controls[1]:
        st.selectbox(_loc("Вариант", "Нұсқа", "Variant", lang=lang), options=variant_tokens, key="reports_variant", format_func=variant_label)
    with controls[2]:
        st.selectbox(_loc("Язык", "Тіл", "Language", lang=lang), options=available_languages or [lang], key="reports_lang")

    top = st.columns([1.52, 1.08], gap="large")
    with top[0]:
        _section(_loc("Демо-отчёт", "Демо-есеп", "Demo report", lang=lang))
        st.text_area(
            _loc("Фокус отчёта", "Есеп бағыты", "Report focus", lang=lang),
            key="reports_focus",
            height=120,
            placeholder=_loc("Например: что происходило у входа и были ли сигналы риска?", "Мысалы: кіреберісте не болды және тәуекел белгілері болды ма?", "For example: what happened near the entrance and were there risk signals?", lang=lang),
        )
        action_cols = st.columns([0.85, 3.15], gap="small")
        with action_cols[0]:
            if st.button(ICON_START, key="reports_build_btn", use_container_width=True, help=_loc("Построить отчёт", "Есеп құрастыру", "Build report", lang=lang)):
                try:
                    st.session_state["reports_payload"] = client.build_report(
                        video_id=selected_video_id or None,
                        language=str(st.session_state.get("reports_lang") or lang),
                        variant=variant_from_token(st.session_state.get("reports_variant")),
                        query=str(st.session_state.get("reports_focus") or "").strip() or None,
                        top_k=8,
                    )
                except Exception as exc:
                    soft_note(f"{_error_prefix(lang)}: {exc}", kind="warn")
        with action_cols[1]:
            st.markdown(
                f"<div class='status-line'>{E(_loc('Страница подготовлена как компактный демо-отчёт с опорой на найденные эпизоды.', 'Бет табылған эпизодтарға сүйенген ықшам демо-есеп ретінде жасалған.', 'This page is designed as a compact demo report grounded in retrieved evidence.', lang=lang))}</div>",
                unsafe_allow_html=True,
            )

        report_payload = st.session_state.get("reports_payload") or {}
        report_text = str(report_payload.get("report") or "").strip()
        if report_text:
            st.markdown(f"<div class='report-body'>{E(report_text)}</div>", unsafe_allow_html=True)
        else:
            st.markdown(
                f"<div class='report-placeholder'>{E(_loc('Сформируйте отчёт, чтобы получить краткий аналитический вывод по выбранному видео.', 'Таңдалған бейне бойынша қысқа аналитикалық қорытынды алу үшін есеп құрастырыңыз.', 'Build a report to generate a short analytical summary for the selected video.', lang=lang))}</div>",
                unsafe_allow_html=True,
            )

    with top[1]:
        _section(_loc("Снимок запуска", "Іске қосу көрінісі", "Run snapshot", lang=lang))
        _render_info_rows(_report_overview_rows(selected_video_id, current_lang, current_variant, outputs, lang))
        if outputs:
            summary_text = str(outputs.get("global_summary") or "").strip()
            if summary_text:
                st.markdown(f"<div class='analytics-copy'>{E(clip_text(summary_text, 260))}</div>", unsafe_allow_html=True)
        else:
            _caption(_loc("Готовые результаты для выбранного видео пока недоступны.", "Таңдалған бейне үшін дайын нәтижелер әзірге жоқ.", "Processed outputs for the selected video are not available yet.", lang=lang))

    _render_supporting_evidence(list((st.session_state.get("reports_payload") or {}).get("supporting_hits") or []), lang)


def search_tab(client: BackendClient, cfg: Any, ui_text: Dict[str, Dict[str, Any]]) -> None:
    """Render the search page without nested panel wrappers."""

    _ = ui_text
    _bind_search_state()
    lang = _ui_lang(cfg)
    videos = _video_items(client)
    if not videos:
        soft_note(_loc("Библиотека пуста. Сначала загрузите видео.", "Қойма бос. Алдымен бейне жүктеңіз.", "The library is empty. Upload a video to begin.", lang=lang))
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
        _render_search_panel_fragment(
            client,
            lang,
            current_lang,
            video_ids,
            available_languages,
            variant_tokens,
        )

    with right:
        _section(_loc("Умный асситент", "Ақылды ассистент", "Smart assistant", lang=lang))
        clear_row = st.columns([4.24, 0.56], gap="small")
        with clear_row[1]:
            _mark("chat-clear-marker")
            if st.button("\U0001F5D1", key="search_clear_chat_btn", help=_loc("Очистить чат", "Чатты тазарту", "Clear chat", lang=lang)):
                st.session_state["search_rag_messages"] = []
                st.session_state["search_rag_input"] = ""
                st.rerun()
        _render_chat(list(st.session_state.get("search_rag_messages") or []), lang)

        st.text_input(
            _loc("Сообщение", "Хабарлама", "Message", lang=lang),
            key="search_rag_input",
            placeholder=_loc(
                "Были ли аномалии с людьми в черном?",
                "Қара киімдегі адамдармен аномалиялар болды ма?",
                "Were there anomalies with people in black?",
                lang=lang,
            ),
            label_visibility="collapsed",
            on_change=_submit_chat_from_enter,
            args=(client, lang, current_lang),
        )

        st.markdown(
            f"""
            <div class='assistant-info'>
                <div class='assistant-info-icon'>ℹ</div>
                <div class='assistant-info-copy'>{E(_loc('Ассистент использует текущие фильтры и найденные эпизоды.', 'Көмекші ағымдағы сүзгілер мен табылған эпизодтарды пайдаланады.', 'The assistant uses the current filters and retrieved episodes.', lang=lang))}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

def _overview_features(lang: str) -> List[Dict[str, str]]:
    """Return overview cards with clearer but more compact copy."""

    return [
        {
            "tone": "feature-card tone-a",
            "label": _loc("Обзор системы", "Жүйе шолуы", "System overview", lang=lang),
            "title": _loc("Понятный сценарий работы", "Түсінікті жұмыс сценариі", "A clear workflow", lang=lang),
            "copy": _loc(
                "Раздел помогает быстро понять, как видео проходит путь от загрузки до поиска событий и итогового отчета.",
                "Бұл бөлім бейненің жүктеуден бастап оқиғаны іздеуге және қорытынды есепке дейінгі жолын тез түсінуге көмектеседі.",
                "This section quickly explains how video moves from upload to event retrieval and reporting.",
                lang=lang,
            ),
            "detail": _loc(
                "Пользователь сразу видит структуру системы и понимает, куда идти на каждом этапе работы.",
                "Пайдаланушы жүйе құрылымын бірден көріп, әр кезеңде қайда өту керегін түсінеді.",
                "Users immediately see the system structure and understand where to go at each step.",
                lang=lang,
            ),
        },
        {
            "tone": "feature-card tone-b",
            "label": _loc("Хранилище", "Қойма", "Storage", lang=lang),
            "title": _loc("Видеотека и очередь", "Бейнеқор және кезек", "Library and queue", lang=lang),
            "copy": _loc(
                "Все ролики собраны в одном месте. Это упрощает загрузку, фильтрацию, запуск обработки и контроль очереди.",
                "Барлық ролик бір жерде жиналған. Бұл жүктеуді, сүзуді, өңдеуді іске қосуды және кезекті бақылауды жеңілдетеді.",
                "All videos are kept in one place, which simplifies upload, filtering, queueing, and processing control.",
                lang=lang,
            ),
            "detail": _loc(
                "Даже если файлов становится больше, оператору остается удобно находить нужный материал и работать с ним дальше.",
                "Файл көбейсе де операторға керекті материалды табу және онымен әрі қарай жұмыс істеу ыңғайлы болып қалады.",
                "Even as the number of files grows, operators can still find the right material quickly.",
                lang=lang,
            ),
        },
        {
            "tone": "feature-card tone-c",
            "label": _loc("Видеоаналитика", "Бейне аналитикасы", "Video analytics", lang=lang),
            "title": _loc("Проверка результата", "Нәтижені тексеру", "Result verification", lang=lang),
            "copy": _loc(
                "После обработки система показывает плеер, краткое описание и таймлайн найденных фрагментов, привязанных ко времени.",
                "Өңдеуден кейін жүйе плеерді, қысқаша сипаттаманы және уақытқа байланыстырылған табылған фрагменттер таймлайнын көрсетеді.",
                "After processing, the system shows a player, a short summary, and a timeline of detected fragments.",
                lang=lang,
            ),
            "detail": _loc(
                "Это делает демонстрацию понятной: можно быстро показать, где именно произошло событие и на чем основан результат.",
                "Бұл демонстрацияны түсінікті етеді: оқиға нақты қай жерде болғанын және нәтиже неге сүйенетінін тез көрсетуге болады.",
                "This keeps the demo clear: you can quickly show where an event happened and what evidence supports it.",
                lang=lang,
            ),
        },
        {
            "tone": "feature-card tone-d",
            "label": _loc("Поиск и отчёты", "Іздеу және есептер", "Search and reports", lang=lang),
            "title": _loc("Осмысленный поиск", "Мағыналы іздеу", "Meaning-based search", lang=lang),
            "copy": _loc(
                "Поиск находит нужные эпизоды по смыслу запроса, а ассистент и отчеты используют те же найденные фрагменты.",
                "Іздеу сұраудың мағынасы бойынша керекті эпизодтарды табады, ал ассистент пен есептер сол табылған фрагменттерді пайдаланады.",
                "Search retrieves relevant episodes from the meaning of the query, and the assistant and reports use the same evidence.",
                lang=lang,
            ),
            "detail": _loc(
                "Это позволяет получить понятный, проверяемый и удобный для презентации итоговый вывод.",
                "Бұл түсінікті, тексерілетін және презентацияға ыңғайлы қорытынды алуға мүмкіндік береді.",
                "This leads to a clear, verifiable, presentation-ready conclusion.",
                lang=lang,
            ),
        },
    ]


def _overview_stages(lang: str) -> List[Dict[str, str]]:
    """Return more detailed overview stages in plain language."""

    return [
        {
            "emoji": "📥",
            "title": _loc("Загрузка видео", "Бейнені жүктеу", "Video upload", lang=lang),
            "short": _loc("Видео попадает в систему и становится доступным для дальнейшей работы.", "Бейне жүйеге түсіп, кейінгі жұмысқа қолжетімді болады.", "The video enters the system and becomes available for further work.", lang=lang),
            "copy": _loc(
                "Сначала оператор добавляет ролик в библиотеку. Система принимает исходный файл, сохраняет базовые параметры и создает для него понятную рабочую запись. С этого момента видео уже можно отправлять в обработку, отслеживать в очереди и использовать в других разделах интерфейса.",
                "Алдымен оператор роликті кітапханаға қосады. Жүйе бастапқы файлды қабылдап, негізгі параметрлерін сақтап, оған түсінікті жұмыс жазбасын жасайды. Осы сәттен бастап бейнені өңдеуге жіберуге, кезекте бақылауға және интерфейстің басқа бөлімдерінде пайдалануға болады.",
                "First, the operator adds the video to the library. The system stores the source file, captures its basic metadata, and creates a clear working record that can be processed and tracked.",
                lang=lang,
            ),
            "impact": _loc("Это стартовая точка для всего дальнейшего сценария.", "Бұл бүкіл келесі сценарийдің бастапқы нүктесі.", "This is the starting point for the rest of the workflow.", lang=lang),
        },
        {
            "emoji": "⚙️",
            "title": _loc("Подготовка материала", "Материалды дайындау", "Preparation", lang=lang),
            "short": _loc("Система приводит ролик к стабильному рабочему виду перед анализом.", "Жүйе талдау алдында роликті тұрақты жұмыс форматына келтіреді.", "The system normalizes the video before analysis.", lang=lang),
            "copy": _loc(
                "Дальше ролик подготавливается к устойчивой аналитике. Система выравнивает технические свойства, чтобы разные файлы обрабатывались по одному сценарию. Пользователю не нужно думать о форматах и параметрах: эта работа скрыта внутри пайплайна.",
                "Одан кейін ролик тұрақты аналитикаға дайындалады. Жүйе әртүрлі файл бір сценариймен өңделуі үшін техникалық қасиеттерін теңестіреді. Пайдаланушыға формат пен параметр туралы ойлаудың қажеті жоқ: бұл жұмыс пайплайн ішінде жасалады.",
                "Next, the video is normalized for stable analytics. The system aligns technical properties so different files can be processed in a consistent way.",
                lang=lang,
            ),
            "impact": _loc("Этот этап снижает риск технических ошибок в следующих шагах.", "Бұл кезең келесі қадамдардағы техникалық қателер қаупін азайтады.", "This stage reduces the risk of technical issues later in the flow.", lang=lang),
        },
        {
            "emoji": "🔎",
            "title": _loc("Анализ сцен и событий", "Сахна мен оқиғаны талдау", "Scene and event analysis", lang=lang),
            "short": _loc("Кадры превращаются в наблюдения, события и понятные описания.", "Кадрлар бақылауларға, оқиғаларға және түсінікті сипаттамаларға айналады.", "Frames turn into observations, events, and readable descriptions.", lang=lang),
            "copy": _loc(
                "На этом этапе система уже работает со смыслом происходящего. Она рассматривает временные фрагменты ролика, выделяет важные моменты, формирует описания и готовит данные для таймлайна. Именно здесь видео перестает быть просто набором кадров и становится понятной последовательностью событий.",
                "Бұл кезеңде жүйе болып жатқанның мағынасымен жұмыс істейді. Ол роликтің уақыт фрагменттерін қарайды, маңызды сәттерді бөледі, сипаттамалар жасайды және таймлайнға дерек дайындайды. Дәл осы жерде бейне жай кадр жиыны емес, түсінікті оқиғалар тізбегіне айналады.",
                "At this stage, the system starts working with the meaning of what is happening. It reviews time segments, highlights important moments, creates descriptions, and prepares the timeline.",
                lang=lang,
            ),
            "impact": _loc("Именно здесь появляется основа для просмотра, поиска и объяснений.", "Дәл осы жерде қарау, іздеу және түсіндіру үшін негіз пайда болады.", "This is where the foundation for review, search, and explanation is created.", lang=lang),
        },
        {
            "emoji": "🗂️",
            "title": _loc("Индексация результатов", "Нәтижелерді индекстеу", "Indexing", lang=lang),
            "short": _loc("Найденные эпизоды попадают в поисковую основу системы.", "Табылған эпизодтар жүйенің іздеу негізіне түседі.", "Detected episodes are written into the system index.", lang=lang),
            "copy": _loc(
                "После анализа результаты нужно не просто сохранить, а сделать пригодными для быстрого поиска. Система складывает эпизоды и описания в индекс, на который потом опираются поиск, ассистент и отчеты. За счет этого нужный момент можно находить не вручную, а по смыслу запроса.",
                "Талдаудан кейін нәтижені жай сақтау жеткіліксіз, оны жылдам іздеуге ыңғайлы ету керек. Жүйе эпизодтар мен сипаттамаларды кейін іздеу, ассистент және есептер сүйенетін индекске салады. Соның арқасында керекті сәтті қолмен емес, сұраудың мағынасы бойынша табуға болады.",
                "After analysis, the results are stored in a retrieval-ready index. This is the base later used by search, the assistant, and reports.",
                lang=lang,
            ),
            "impact": _loc("Этот шаг превращает обработанные данные в рабочий поисковый инструмент.", "Бұл қадам өңделген деректі жұмыс істейтін іздеу құралына айналдырады.", "This step turns processed output into a usable retrieval tool.", lang=lang),
        },
        {
            "emoji": "🧾",
            "title": _loc("Поиск, ответы и отчёты", "Іздеу, жауап және есеп", "Search, answers, and reports", lang=lang),
            "short": _loc("Пользователь находит нужный эпизод и получает понятный итоговый вывод.", "Пайдаланушы керекті эпизодты тауып, түсінікті қорытынды алады.", "Users retrieve the right episode and get a clear final conclusion.", lang=lang),
            "copy": _loc(
                "Финальный этап объединяет все предыдущие результаты. Пользователь задает запрос, находит нужный эпизод, открывает его в аналитике, уточняет вопрос через ассистента и при необходимости формирует короткий отчет. Важно, что весь итоговый ответ остается связанным с конкретными фрагментами видео, поэтому его легко показать и проверить.",
                "Соңғы кезең алдыңғы нәтижелердің бәрін біріктіреді. Пайдаланушы сұрау береді, керекті эпизодты табады, оны аналитикада ашады, ассистент арқылы сұрағын нақтылайды және қажет болса қысқа есеп жасайды. Маңыздысы, соңғы жауап нақты бейне фрагменттерімен байланыста қалады, сондықтан оны көрсету де, тексеру де оңай.",
                "The final stage brings everything together. Users search, open the relevant episode, refine their question through the assistant, and build a short report while staying grounded in concrete video evidence.",
                lang=lang,
            ),
            "impact": _loc("Так система становится не просто пайплайном, а понятным инструментом для анализа и демонстрации.", "Осылайша жүйе жай пайплайн емес, талдау мен демонстрацияға арналған түсінікті құралға айналады.", "This is where the pipeline becomes a practical tool for analysis and demonstration.", lang=lang),
        },
    ]


def _stage_nav_button_label(stage: Dict[str, str], label: str) -> str:
    """Build a compact multiline label for one adjacent stage button-card."""

    return f"{stage['emoji']}  {label}\n{stage['title']}\n{clip_text(stage['short'], 112)}"


def _render_overview_carousel(stages: List[Dict[str, str]], session_key: str, lang: str) -> None:
    """Render overview stages with one main card and two side card-buttons."""

    if not stages:
        return

    idx = int(st.session_state.get(session_key) or 0)
    idx = max(0, min(idx, len(stages) - 1))
    st.session_state[session_key] = idx
    current = stages[idx]

    progress = "".join(
        f"<span class='stage-dot{' active' if i == idx else ''}'>{i + 1}</span>"
        for i in range(len(stages))
    )
    st.markdown(f"<div class='stage-progress'>{progress}</div>", unsafe_allow_html=True)

    adjacent: List[tuple[int, str]] = []
    if idx > 0:
        adjacent.append((idx - 1, _loc("Предыдущий этап", "Алдыңғы кезең", "Previous stage", lang=lang)))
    if idx + 1 < len(stages):
        adjacent.append((idx + 1, _loc("Следующий этап", "Келесі кезең", "Next stage", lang=lang)))

    main_col, side_col = st.columns([2.1, 1.0], gap="large")
    with main_col:
        st.markdown(
            f"""
            <div class="stage-card">
                <div class="stage-card-head">
                    <div class="stage-badge">{E(current['emoji'])}</div>
                    <div>
                        <div class="stage-kicker">{E(_loc('Этап', 'Кезең', 'Stage', lang=lang))} {idx + 1}</div>
                        <div class="stage-title">{E(current['title'])}</div>
                    </div>
                </div>
                <div class="stage-short">{E(current['short'])}</div>
                <div class="stage-copy">{E(current['copy'])}</div>
                <div class="stage-impact">{E(current['impact'])}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with side_col:
        for target_idx, label in adjacent:
            stage = stages[target_idx]
            with st.container():
                _mark("stage-nav-marker")
                if st.button(
                    _stage_nav_button_label(stage, label),
                    key=f"{session_key}_{target_idx}",
                    use_container_width=True,
                ):
                    st.session_state[session_key] = target_idx
                    st.rerun()
        if len(adjacent) < 2:
            st.markdown("<div class='stage-nav-spacer'></div>", unsafe_allow_html=True)


def overview_tab(client: BackendClient, cfg: Any, ui_text: Dict[str, Dict[str, Any]]) -> None:
    """Render the overview page with cleaner spacing."""

    _ = client
    _ = ui_text
    lang = _ui_lang(cfg)

    st.markdown(
        f"""
        <div class="intro-card">
            <div class="intro-title">{E(_loc('SmartCampus V2T для поиска событий и понятного анализа видео', 'SmartCampus V2T бейне оқиғаларын іздеуге және түсінікті талдауға арналған', 'SmartCampus V2T for event retrieval and clear video review', lang=lang))}</div>
            <div class="intro-copy">{E(_loc('Демо-интерфейс показывает полный путь: от загрузки видео и запуска обработки до просмотра результата, поиска нужного эпизода и подготовки итогового отчёта.', 'Демо-интерфейс толық жолды көрсетеді: бейнені жүктеу мен өңдеуді іске қосудан бастап нәтижені қарауға, керекті эпизодты іздеуге және қорытынды есеп дайындауға дейін.', 'This demo interface shows the full path from upload and processing to review, event retrieval, and final reporting.', lang=lang))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='overview-section-gap'></div>", unsafe_allow_html=True)
    _section(_loc("Основные функции системы", "Жүйенің негізгі функциялары", "Main system functions", lang=lang))
    feature_rows = _overview_features(lang)
    for start in range(0, len(feature_rows), 2):
        columns = st.columns(2, gap="large")
        for col, item in zip(columns, feature_rows[start : start + 2]):
            with col:
                st.markdown(
                    f"""
                    <div class="{E(item['tone'])}">
                        <div class="feature-label">{E(item['label'])}</div>
                        <div class="feature-title">{E(item['title'])}</div>
                        <div class="feature-copy">{E(item['copy'])}</div>
                        <div class="feature-detail">{E(item['detail'])}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        if start + 2 < len(feature_rows):
            st.markdown("<div class='overview-grid-gap'></div>", unsafe_allow_html=True)

    st.markdown("<div class='overview-section-gap'></div>", unsafe_allow_html=True)
    _section(_loc("Этапы работы системы", "Жүйе жұмысының кезеңдері", "System stages", lang=lang))
    _render_overview_carousel(_overview_stages(lang), "overview_stage_index", lang)


__all__ = [
    "footer",
    "get_T",
    "load_and_apply_css",
    "load_ui_text",
    "overview_tab",
    "render_header",
    "render_i18n_metrics",
    "reports_metrics_tab",
    "search_tab",
    "soft_note",
    "storage_tab",
    "video_analytics_tab",
]
