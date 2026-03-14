# app/view/analytics.py
"""
Video analytics Streamlit page logic.

Purpose:
- Render playback, summaries, metrics, and the timeline inspection surface.
- Keep analytics-specific helpers close to the analytics page entrypoint.
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
from app.lib.media import ensure_browser_video, get_video_meta, img_to_data_uri, mtime
from app.view.shared import (
    PAGE_SIZE,
    ICON_CLEAR,
    ICON_CONFIRM,
    ICON_DELETE,
    ICON_DOWN,
    ICON_NEXT,
    ICON_OPEN,
    ICON_PAUSE,
    ICON_PREV,
    ICON_REFRESH,
    ICON_RESUME,
    ICON_START,
    ICON_UP,
    _caption,
    _error_prefix,
    _loc,
    _mark,
    _page_title,
    _resolve_video_context,
    _section,
    _session_choice,
    _ui_lang,
    _video_items,
    soft_note,
)

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
