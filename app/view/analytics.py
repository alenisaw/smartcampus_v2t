# app/view/analytics.py
"""
Video analytics Streamlit page logic.

Purpose:
- Render playback, summaries, metrics, and the timeline inspection surface.
- Keep analytics-specific helpers close to the analytics page entrypoint.
"""

from __future__ import annotations

import math
import re
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
    _run_live_fragment,
    _section,
    _session_choice,
    _ui_lang,
    _video_items,
    soft_note,
)

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_STAGE_ORDER = (
    "preprocess_video",
    "build_clips",
    "run_vlm_pipeline",
    "write_clip_observations",
    "build_segment_schema",
    "structuring_segments",
    "write_segments",
    "build_summary",
    "guard_summary",
    "write_summary",
    "write_run_manifest",
    "update_outputs_manifest",
    "index_build",
    "enqueue_translation_jobs",
    "load_source_segments",
    "translate_segments_mt",
    "post_edit_selected_segments",
    "load_source_summary",
    "translate_summary_mt",
    "post_edit_summary",
    "translate_total",
    "process_total",
    "write_metrics",
)

def _analytics_status_items(outputs: Dict[str, Any], raw_exists: bool, lang: str) -> List[tuple[str, bool]]:
    """Return the localized status list for analytics."""

    metrics = outputs.get("metrics") if isinstance(outputs.get("metrics"), dict) else {}
    annotations = outputs.get("annotations") if isinstance(outputs.get("annotations"), list) else []
    observations = outputs.get("clip_observations") if isinstance(outputs.get("clip_observations"), list) else []
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


def _current_manifest_entry(outputs: Dict[str, Any]) -> Dict[str, Any]:
    """Return the most relevant manifest entry for the current language."""

    manifest = outputs.get("manifest") if isinstance(outputs.get("manifest"), dict) else {}
    languages = manifest.get("languages") if isinstance(manifest.get("languages"), dict) else {}
    current_lang = str(outputs.get("language") or "").strip().lower()
    current = languages.get(current_lang) if isinstance(languages, dict) else None
    if isinstance(current, dict):
        return current

    latest: Dict[str, Any] = {}
    latest_ts = -1.0
    for payload in languages.values():
        if not isinstance(payload, dict):
            continue
        try:
            updated_at = float(payload.get("updated_at") or 0.0)
        except Exception:
            updated_at = 0.0
        if updated_at >= latest_ts:
            latest_ts = updated_at
            latest = payload
    return latest


def _analytics_status_items(outputs: Dict[str, Any], raw_exists: bool, lang: str) -> List[tuple[str, bool]]:
    """Return the localized status list for analytics."""

    metrics = outputs.get("metrics") if isinstance(outputs.get("metrics"), dict) else {}
    annotations = outputs.get("annotations") if isinstance(outputs.get("annotations"), list) else []
    observations = outputs.get("clip_observations") if isinstance(outputs.get("clip_observations"), list) else []
    indexing = metrics.get("indexing") if isinstance(metrics.get("indexing"), dict) else {}
    return [
        (_loc("Исходный файл", "Бастапқы файл", "Source file", lang=lang), raw_exists),
        (_loc("Наблюдения по клипам", "Клип бақылаулары", "Clip observations", lang=lang), bool(observations)),
        (_loc("Структурированные сегменты", "Құрылымдалған сегменттер", "Structured segments", lang=lang), bool(annotations)),
        (_loc("Краткая сводка", "Қысқаша мазмұн", "Summary", lang=lang), bool(str(outputs.get("global_summary") or "").strip())),
        (_loc("Поисковый индекс", "Іздеу индексі", "Search index", lang=lang), bool(indexing)),
    ]


def _metric_seconds_value(value: Any) -> str:
    """Format one metric value as seconds when available."""

    try:
        if value is None:
            return "-"
        return f"{float(value):.1f}s"
    except Exception:
        return "-"


def _stage_metric_mean(outputs: Dict[str, Any], stage_name: str) -> Optional[float]:
    """Read one stage mean from the aggregated metrics payload."""

    metrics = outputs.get("metrics") if isinstance(outputs.get("metrics"), dict) else {}
    stage_stats = metrics.get("stage_stats_sec") if isinstance(metrics.get("stage_stats_sec"), dict) else {}
    payload = stage_stats.get(stage_name) if isinstance(stage_stats.get(stage_name), dict) else {}
    try:
        value = payload.get("mean_sec")
        return float(value) if value is not None else None
    except Exception:
        return None


def _summary_brief_text(summary_text: str, lang: str) -> str:
    """Return a concise one-two sentence video description."""

    text = " ".join(str(summary_text or "").split()).strip()
    if not text:
        return _loc(
            "Краткая выжимка появится после завершения базовой обработки.",
            "Қысқа мазмұн негізгі өңдеу аяқталғаннан кейін пайда болады.",
            "A short summary will appear after the base processing is complete.",
            lang=lang,
        )

    sentences = [item.strip() for item in _SENTENCE_SPLIT_RE.split(text) if item.strip()]
    if len(sentences) >= 2:
        return " ".join(sentences[:2]).strip()
    if len(sentences) == 1:
        return clip_text(sentences[0], 220)
    return clip_text(text, 220)


def _stage_label(stage_name: str, lang: str) -> str:
    """Return a localized label for one runtime stage token."""

    token = str(stage_name or "").strip().lower()
    mapping = {
        "preprocess_video": _loc("Препроцессинг", "Алдын ала өңдеу", "Preprocess", lang=lang),
        "build_clips": _loc("Нарезка клипов", "Клиптерге бөлу", "Build clips", lang=lang),
        "run_vlm_pipeline": _loc("VLM-инференс", "VLM-инференс", "VLM inference", lang=lang),
        "write_clip_observations": _loc("Сохранение наблюдений", "Бақылауларды сақтау", "Save observations", lang=lang),
        "build_segment_schema": _loc("Сборка сегментов", "Сегменттерді құрастыру", "Build segments", lang=lang),
        "structuring_segments": _loc("Структурирование", "Құрылымдау", "Structuring", lang=lang),
        "write_segments": _loc("Сохранение сегментов", "Сегменттерді сақтау", "Save segments", lang=lang),
        "build_summary": _loc("Сборка summary", "Summary құрастыру", "Build summary", lang=lang),
        "guard_summary": _loc("Guard summary", "Guard summary", "Guard summary", lang=lang),
        "write_summary": _loc("Сохранение summary", "Summary сақтау", "Save summary", lang=lang),
        "write_run_manifest": _loc("Обновление manifest", "Manifest жаңарту", "Update manifest", lang=lang),
        "update_outputs_manifest": _loc("Статус outputs", "Outputs күйі", "Outputs status", lang=lang),
        "index_build": _loc("Построение индекса", "Индексті құру", "Index build", lang=lang),
        "enqueue_translation_jobs": _loc("Постановка переводов", "Аудармаларды кезекке қою", "Queue translations", lang=lang),
        "process_total": _loc("Полный process", "Толық process", "Process total", lang=lang),
        "write_metrics": _loc("Запись метрик", "Метрикаларды жазу", "Write metrics", lang=lang),
        "load_source_segments": _loc("Чтение исходных сегментов", "Бастапқы сегменттерді оқу", "Load source segments", lang=lang),
        "translate_segments_mt": _loc("MT сегментов", "Сегменттер MT", "Translate segments", lang=lang),
        "post_edit_selected_segments": _loc("Полировка выбранных сегментов", "Таңдалған сегменттерді түзету", "Polish selected segments", lang=lang),
        "load_source_summary": _loc("Чтение исходной summary", "Бастапқы summary оқу", "Load source summary", lang=lang),
        "translate_summary_mt": _loc("MT summary", "Summary MT", "Translate summary", lang=lang),
        "post_edit_summary": _loc("Полировка summary", "Summary түзету", "Polish summary", lang=lang),
        "translate_total": _loc("Полный перевод", "Толық аударма", "Translation total", lang=lang),
    }
    return mapping.get(token, humanize_token(token))


def _stage_metric_rows(stage_stats: Dict[str, Any], lang: str) -> List[tuple[str, str]]:
    """Return one metrics row per stage."""

    rows: List[tuple[str, str]] = []
    for stage_name, payload in (stage_stats or {}).items():
        if not isinstance(payload, dict):
            continue
        mean_sec = payload.get("mean_sec")
        if mean_sec is None:
            continue
        rows.append((_stage_label(stage_name, lang), _metric_seconds_value(mean_sec)))
    return rows


def _analytics_metric_rows(outputs: Dict[str, Any], meta: Dict[str, Any], language: str, lang: str) -> List[tuple[str, str]]:
    """Build the compact metrics rows shown in the analytics side panel."""

    metrics = outputs.get("metrics") if isinstance(outputs.get("metrics"), dict) else {}
    indexing = metrics.get("indexing") if isinstance(metrics.get("indexing"), dict) else {}
    translations = metrics.get("translations") if isinstance(metrics.get("translations"), dict) else {}
    summary_polish = metrics.get("summary_polish") if isinstance(metrics.get("summary_polish"), dict) else {}
    language_key = str(language or "").strip().lower()
    lang_index_metrics = indexing.get(language_key) if isinstance(indexing.get(language_key), dict) else {}
    lang_translation_metrics = translations.get(language_key) if isinstance(translations.get(language_key), dict) else None
    stage_stats = metrics.get("stage_stats_sec") if isinstance(metrics.get("stage_stats_sec"), dict) else {}
    if isinstance(lang_translation_metrics, dict) and lang_translation_metrics:
        translation_stage_stats = lang_translation_metrics.get("stage_stats_sec")
        if isinstance(translation_stage_stats, dict):
            stage_stats = translation_stage_stats

    rows = [
        (
            _loc("Базовый пайплайн", "Негізгі пайплайн", "Base pipeline", lang=lang),
            _metric_seconds_value((metrics.get("total_time_sec") if isinstance(metrics, dict) else None)),
        ),
        (
            _loc("Полная задача", "Толық тапсырма", "Queue total", lang=lang),
            _metric_seconds_value(_stage_metric_mean(outputs, "process_total") or _stage_metric_mean(outputs, "translate_total")),
        ),
        (
            _loc("Длительность видео", "Бейне ұзақтығы", "Video duration", lang=lang),
            hms(float(meta.get("duration_sec") or 0.0)) if meta.get("duration_sec") else "-",
        ),
        (
            _loc("Разрешение", "Ажыратылымдылығы", "Video resolution", lang=lang),
            f"{int(meta.get('width'))}x{int(meta.get('height'))}" if meta.get("width") and meta.get("height") else "-",
        ),
    ]

    index_time = lang_index_metrics.get("time_sec") if isinstance(lang_index_metrics, dict) else None
    if index_time is None:
        index_time = _stage_metric_mean(outputs, "index_build")
    if index_time is not None:
        rows.insert(
            3,
            (_loc("Индекс", "Индекс", "Index update", lang=lang), _metric_seconds_value(index_time)),
        )

    if isinstance(summary_polish, dict) and summary_polish.get("applied"):
        rows.insert(
            4,
            (_loc("Полировка summary", "Summary жылтырату", "Summary polish", lang=lang), _metric_seconds_value(summary_polish.get("time_sec"))),
        )

    if isinstance(lang_translation_metrics, dict):
        rows.insert(
            2,
            (_loc("Перевод", "Аударма", "Translation", lang=lang), _metric_seconds_value(lang_translation_metrics.get("time_sec"))),
        )

    rows.extend(_stage_metric_rows(stage_stats, lang))
    return rows


def _analytics_live_stage(outputs: Dict[str, Any], queue: Dict[str, Any], video_id: str) -> str:
    """Return the best live processing stage string for the selected video."""

    running = queue.get("running") if isinstance(queue, dict) else None
    if isinstance(running, dict) and str(running.get("video_id") or "") == str(video_id):
        stage = humanize_token(str(running.get("stage") or running.get("job_type") or "running"))
        message = str(running.get("message") or "").strip()
        try:
            progress = max(0.0, min(1.0, float(running.get("progress") or 0.0)))
        except Exception:
            progress = 0.0
        parts = [stage]
        if progress > 0:
            parts.append(f"{int(progress * 100)}%")
        if message and message.lower() != stage.lower():
            parts.append(message)
        return " · ".join(part for part in parts if part)

    entry = _current_manifest_entry(outputs)
    status = humanize_token(str(entry.get("status") or ""))
    note = humanize_token(str(entry.get("note") or ""))
    if status and status.lower() != "ready":
        return " · ".join(part for part in [status, note] if part)
    return ""


def _render_timeline(outputs: Dict[str, Any], lang: str) -> None:
    """Render structured segments or raw observations as a timeline."""

    annotations = outputs.get("annotations") if isinstance(outputs.get("annotations"), list) else []
    observations = outputs.get("clip_observations") if isinstance(outputs.get("clip_observations"), list) else []
    rows = annotations if annotations else observations

    _section(_loc("Таймлайн", "Таймлайн", "Timeline", lang=lang))
    if not rows:
        _caption(_loc("Сегменты ещё не готовы.", "Сегменттер әлі дайын емес.", "Segments are not available yet.", lang=lang))
        return
    if not annotations and observations:
        _caption(
            _loc(
                "Пока structuring не завершён, показываются сырые наблюдения по клипам.",
                "Structuring аяқталғанша клип бойынша бастапқы бақылаулар көрсетіледі.",
                "Showing raw clip observations while structuring is still running.",
                lang=lang,
            )
        )

    visible = rows[:60]
    for idx, ann in enumerate(visible):
        if not isinstance(ann, dict):
            continue
        start_sec = float(ann.get("start_sec", 0.0) or 0.0)
        end_sec = float(ann.get("end_sec", 0.0) or 0.0)
        description = str(ann.get("normalized_caption") or ann.get("description") or "").strip()
        notes = [str(note).strip() for note in (ann.get("anomaly_notes") or []) if str(note).strip()]
        event_type = str(ann.get("event_type") or "").strip()
        risk = str(ann.get("risk_level") or "").strip().lower()
        is_alert = bool(ann.get("anomaly_flag")) or risk in {"attention", "warning", "critical"} or bool(notes)

        time_col, body_col = st.columns([1.0, 4.6], gap="medium")
        with time_col:
            if st.button(f"{mmss(start_sec)} - {mmss(end_sec)}", key=f"timeline_seek_{idx}", use_container_width=True):
                st.session_state["video_seek_sec"] = int(start_sec)
                st.rerun()
        with body_col:
            if event_type and event_type.lower() not in {"unknown", "none", "normal"}:
                event_label = _loc("Событие", "Оқиға", "Event", lang=lang)
                event_text = humanize_token(event_type)
                st.markdown(
                    f"<div class='timeline-event{' alert' if is_alert else ''}'>{E(f'{event_label}: {event_text}')}</div>",
                    unsafe_allow_html=True,
                )
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

    variant_id = variant_from_token(st.session_state.get("library_variant"))
    language = str(st.session_state.get("library_lang") or "en")
    video_id = str(st.session_state.get("selected_video_id") or "")

    try:
        initial_outputs = client.get_video_outputs(video_id, language, variant=variant_id)
    except Exception:
        initial_outputs = {}
    try:
        initial_queue = client.queue_list()
    except Exception:
        initial_queue = {}

    def _analytics_body() -> None:
        try:
            outputs = client.get_video_outputs(video_id, language, variant=variant_id)
        except Exception as exc:
            outputs = {}
            soft_note(f"{_error_prefix(lang)}: {exc}", kind="warn")
        try:
            queue = client.queue_list()
        except Exception:
            queue = initial_queue

        top = st.columns([1.88, 1.12], gap="large")
        with top[0]:
            if playable_path.exists():
                try:
                    st.video(str(playable_path), start_time=int(st.session_state.get("video_seek_sec") or 0))
                except TypeError:
                    st.video(str(playable_path))
            else:
                soft_note(_loc("Исходное видео не найдено.", "Бастапқы бейне табылмады.", "Source video not found.", lang=lang), kind="warn")
            st.markdown("<div class='analytics-timeline-offset'></div>", unsafe_allow_html=True)
            _render_timeline(outputs, lang)

        with top[1]:
            summary_text = str(outputs.get("global_summary") or "").strip()
            stage_copy = _analytics_live_stage(outputs, queue, video_id)
            _section(_loc("Описание видео", "Бейне сипаттамасы", "Video description", lang=lang))
            if summary_text:
                st.markdown(f"<div class='analytics-summary'>{E(_summary_brief_text(summary_text, lang))}</div>", unsafe_allow_html=True)
            else:
                placeholder = _loc(
                    "Сводка обновится автоматически, когда базовые результаты будут готовы.",
                    "Негізгі нәтижелер дайын болғанда мазмұн автоматты түрде жаңарады.",
                    "The summary will update automatically after the base outputs are ready.",
                    lang=lang,
                )
                st.markdown(f"<div class='analytics-summary'>{E(placeholder)}</div>", unsafe_allow_html=True)
            if stage_copy:
                _caption(stage_copy)

            _section(_loc("Статус", "Күйі", "Status", lang=lang))
            _render_status_rows(_analytics_status_items(outputs, raw_exists, lang), lang)

            _section(_loc("Метрики", "Метрикалар", "Metrics", lang=lang))
            _render_info_rows(_analytics_metric_rows(outputs, meta, language, lang))

    refresh_sec = max(2, int(getattr(getattr(cfg, "ui", None), "cache_ttl_sec", 2) or 2))
    manifest_entry = _current_manifest_entry(initial_outputs)
    manifest_status = str(manifest_entry.get("status") or "").strip().lower()
    active_queue = isinstance(initial_queue.get("running") if isinstance(initial_queue, dict) else None, dict) and str((initial_queue.get("running") or {}).get("video_id") or "") == video_id
    needs_poll = active_queue or manifest_status in {"processing", "observed", "segments_ready", "summary_ready", "translating"}
    _run_live_fragment(_analytics_body, run_every_sec=refresh_sec if needs_poll else None)


# Final layout override: keep the timeline directly under the player.
def video_analytics_tab(client: BackendClient, cfg: Any, ui_text: Dict[str, Dict[str, Any]]) -> None:
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

    variant_id = variant_from_token(st.session_state.get("library_variant"))
    language = str(st.session_state.get("library_lang") or "en")
    video_id = str(st.session_state.get("selected_video_id") or "")

    try:
        initial_outputs = client.get_video_outputs(video_id, language, variant=variant_id)
    except Exception:
        initial_outputs = {}
    try:
        initial_queue = client.queue_list()
    except Exception:
        initial_queue = {}

    def _analytics_body() -> None:
        try:
            outputs = client.get_video_outputs(video_id, language, variant=variant_id)
        except Exception as exc:
            outputs = {}
            soft_note(f"{_error_prefix(lang)}: {exc}", kind="warn")
        try:
            queue = client.queue_list()
        except Exception:
            queue = initial_queue

        top = st.columns([1.88, 1.12], gap="large")
        with top[0]:
            if playable_path.exists():
                try:
                    st.video(str(playable_path), start_time=int(st.session_state.get("video_seek_sec") or 0))
                except TypeError:
                    st.video(str(playable_path))
            else:
                soft_note(_loc("Исходное видео не найдено.", "Бастапқы бейне табылмады.", "Source video not found.", lang=lang), kind="warn")
            st.markdown("<div class='analytics-timeline-offset'></div>", unsafe_allow_html=True)
            _render_timeline(outputs, lang)

        with top[1]:
            summary_text = str(outputs.get("global_summary") or "").strip()
            stage_copy = _analytics_live_stage(outputs, queue, video_id)
            brief_text = _summary_brief_text(summary_text, lang) if summary_text else _brief_from_outputs(outputs, lang)

            _section(_loc("Описание видео", "Бейне сипаттамасы", "Video description", lang=lang))
            if brief_text:
                st.markdown(f"<div class='analytics-summary'>{E(brief_text)}</div>", unsafe_allow_html=True)
            else:
                placeholder = _loc(
                    "Краткая сводка появится автоматически, когда базовые результаты будут готовы.",
                    "Негізгі нәтижелер дайын болғанда қысқаша мазмұн автоматты түрде пайда болады.",
                    "A short summary will appear automatically when the base outputs are ready.",
                    lang=lang,
                )
                st.markdown(f"<div class='analytics-summary'>{E(placeholder)}</div>", unsafe_allow_html=True)
            if stage_copy:
                _caption(stage_copy)

            _section(_loc("Статус", "Күйі", "Status", lang=lang))
            _render_status_rows(_analytics_status_items(outputs, raw_exists, lang), lang)

            _section(_loc("Метрики", "Метрикалар", "Metrics", lang=lang))
            _render_info_rows(_analytics_metric_rows(outputs, meta, language, lang))

    refresh_sec = max(2, int(getattr(getattr(cfg, "ui", None), "cache_ttl_sec", 2) or 2))
    manifest_entry = _current_manifest_entry(initial_outputs)
    manifest_status = str(manifest_entry.get("status") or "").strip().lower()
    active_queue = isinstance(initial_queue.get("running") if isinstance(initial_queue, dict) else None, dict) and str((initial_queue.get("running") or {}).get("video_id") or "") == video_id
    needs_poll = active_queue or manifest_status in {"processing", "observed", "segments_ready", "summary_ready", "translating"}
    _run_live_fragment(_analytics_body, run_every_sec=refresh_sec if needs_poll else None)


def video_analytics_tab(client: BackendClient, cfg: Any, ui_text: Dict[str, Dict[str, Any]]) -> None:
    """Render the analytics page with the timeline directly under the video."""

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

    variant_id = variant_from_token(st.session_state.get("library_variant"))
    language = str(st.session_state.get("library_lang") or "en")
    video_id = str(st.session_state.get("selected_video_id") or "")

    try:
        initial_outputs = client.get_video_outputs(video_id, language, variant=variant_id)
    except Exception:
        initial_outputs = {}
    try:
        initial_queue = client.queue_list()
    except Exception:
        initial_queue = {}

    def _analytics_body() -> None:
        try:
            outputs = client.get_video_outputs(video_id, language, variant=variant_id)
        except Exception as exc:
            outputs = {}
            soft_note(f"{_error_prefix(lang)}: {exc}", kind="warn")
        try:
            queue = client.queue_list()
        except Exception:
            queue = initial_queue

        top = st.columns([1.88, 1.12], gap="large")
        with top[0]:
            if playable_path.exists():
                try:
                    st.video(str(playable_path), start_time=int(st.session_state.get("video_seek_sec") or 0))
                except TypeError:
                    st.video(str(playable_path))
            else:
                soft_note(_loc("Исходное видео не найдено.", "Бастапқы бейне табылмады.", "Source video not found.", lang=lang), kind="warn")
            st.markdown("<div class='analytics-timeline-offset'></div>", unsafe_allow_html=True)
            _render_timeline(outputs, lang)

        with top[1]:
            summary_text = str(outputs.get("global_summary") or "").strip()
            stage_copy = _analytics_live_stage(outputs, queue, video_id)
            brief_text = _summary_brief_text(summary_text, lang) if summary_text else _brief_from_outputs(outputs, lang)

            _section(_loc("Описание видео", "Бейне сипаттамасы", "Video description", lang=lang))
            if brief_text:
                st.markdown(f"<div class='analytics-summary'>{E(brief_text)}</div>", unsafe_allow_html=True)
            else:
                placeholder = _loc(
                    "Краткая сводка появится автоматически, когда базовые результаты будут готовы.",
                    "Негізгі нәтижелер дайын болғанда қысқаша мазмұн автоматты түрде пайда болады.",
                    "A short summary will appear automatically when the base outputs are ready.",
                    lang=lang,
                )
                st.markdown(f"<div class='analytics-summary'>{E(placeholder)}</div>", unsafe_allow_html=True)
            if stage_copy:
                _caption(stage_copy)

            _section(_loc("Статус", "Күйі", "Status", lang=lang))
            _render_status_rows(_analytics_status_items(outputs, raw_exists, lang), lang)

            _section(_loc("Метрики", "Метрикалар", "Metrics", lang=lang))
            _render_info_rows(_analytics_metric_rows(outputs, meta, language, lang))

    refresh_sec = max(2, int(getattr(getattr(cfg, "ui", None), "cache_ttl_sec", 2) or 2))
    manifest_entry = _current_manifest_entry(initial_outputs)
    manifest_status = str(manifest_entry.get("status") or "").strip().lower()
    active_queue = isinstance(initial_queue.get("running") if isinstance(initial_queue, dict) else None, dict) and str((initial_queue.get("running") or {}).get("video_id") or "") == video_id
    needs_poll = active_queue or manifest_status in {"processing", "observed", "segments_ready", "summary_ready", "translating"}
    _run_live_fragment(_analytics_body, run_every_sec=refresh_sec if needs_poll else None)


def video_analytics_tab(client: BackendClient, cfg: Any, ui_text: Dict[str, Dict[str, Any]]) -> None:
    """Render the analytics page with the timeline directly under the video."""

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

    variant_id = variant_from_token(st.session_state.get("library_variant"))
    language = str(st.session_state.get("library_lang") or "en")
    video_id = str(st.session_state.get("selected_video_id") or "")

    try:
        initial_outputs = client.get_video_outputs(video_id, language, variant=variant_id)
    except Exception:
        initial_outputs = {}
    try:
        initial_queue = client.queue_list()
    except Exception:
        initial_queue = {}

    def _analytics_body() -> None:
        try:
            outputs = client.get_video_outputs(video_id, language, variant=variant_id)
        except Exception as exc:
            outputs = {}
            soft_note(f"{_error_prefix(lang)}: {exc}", kind="warn")
        try:
            queue = client.queue_list()
        except Exception:
            queue = initial_queue

        top = st.columns([1.88, 1.12], gap="large")
        with top[0]:
            if playable_path.exists():
                try:
                    st.video(str(playable_path), start_time=int(st.session_state.get("video_seek_sec") or 0))
                except TypeError:
                    st.video(str(playable_path))
            else:
                soft_note(_loc("Исходное видео не найдено.", "Бастапқы бейне табылмады.", "Source video not found.", lang=lang), kind="warn")
            st.markdown("<div class='analytics-timeline-offset'></div>", unsafe_allow_html=True)
            _render_timeline(outputs, lang)

        with top[1]:
            summary_text = str(outputs.get("global_summary") or "").strip()
            stage_copy = _analytics_live_stage(outputs, queue, video_id)
            brief_text = _summary_brief_text(summary_text, lang) if summary_text else _brief_from_outputs(outputs, lang)

            _section(_loc("Описание видео", "Бейне сипаттамасы", "Video description", lang=lang))
            if brief_text:
                st.markdown(f"<div class='analytics-summary'>{E(brief_text)}</div>", unsafe_allow_html=True)
            else:
                placeholder = _loc(
                    "Краткая сводка появится автоматически, когда базовые результаты будут готовы.",
                    "Негізгі нәтижелер дайын болғанда қысқаша мазмұн автоматты түрде пайда болады.",
                    "A short summary will appear automatically when the base outputs are ready.",
                    lang=lang,
                )
                st.markdown(f"<div class='analytics-summary'>{E(placeholder)}</div>", unsafe_allow_html=True)
            if stage_copy:
                _caption(stage_copy)

            _section(_loc("Статус", "Күйі", "Status", lang=lang))
            _render_status_rows(_analytics_status_items(outputs, raw_exists, lang), lang)

            _section(_loc("Метрики", "Метрикалар", "Metrics", lang=lang))
            _render_info_rows(_analytics_metric_rows(outputs, meta, language, lang))

    refresh_sec = max(2, int(getattr(getattr(cfg, "ui", None), "cache_ttl_sec", 2) or 2))
    manifest_entry = _current_manifest_entry(initial_outputs)
    manifest_status = str(manifest_entry.get("status") or "").strip().lower()
    active_queue = isinstance(initial_queue.get("running") if isinstance(initial_queue, dict) else None, dict) and str((initial_queue.get("running") or {}).get("video_id") or "") == video_id
    needs_poll = active_queue or manifest_status in {"processing", "observed", "segments_ready", "summary_ready", "translating"}
    _run_live_fragment(_analytics_body, run_every_sec=refresh_sec if needs_poll else None)


def _analytics_status_items(outputs: Dict[str, Any], raw_exists: bool, lang: str) -> List[tuple[str, bool]]:
    """Return localized readiness rows for the active analytics panel."""

    metrics = outputs.get("metrics") if isinstance(outputs.get("metrics"), dict) else {}
    annotations = outputs.get("annotations") if isinstance(outputs.get("annotations"), list) else []
    observations = outputs.get("clip_observations") if isinstance(outputs.get("clip_observations"), list) else []
    indexing = metrics.get("indexing") if isinstance(metrics.get("indexing"), dict) else {}
    return [
        (_loc("Исходный файл", "Бастапқы файл", "Source file", lang=lang), raw_exists),
        (_loc("Результаты обработки", "Өңдеу нәтижелері", "Processed outputs", lang=lang), bool(annotations or observations or outputs.get("global_summary"))),
        (_loc("Поисковый индекс", "Іздеу индексі", "Search index", lang=lang), bool(indexing)),
    ]


def _summary_brief_text(summary_text: str, lang: str) -> str:
    """Return a concise one-two sentence description for the summary card."""

    text = " ".join(str(summary_text or "").split()).strip()
    if not text:
        return _loc(
            "Краткая выжимка появится после завершения базовой обработки.",
            "Қысқа мазмұн негізгі өңдеу аяқталғаннан кейін пайда болады.",
            "A short summary will appear after the base processing is complete.",
            lang=lang,
        )

    sentences = [item.strip() for item in _SENTENCE_SPLIT_RE.split(text) if item.strip()]
    if len(sentences) >= 2:
        return clip_text(" ".join(sentences[:2]).strip(), 220)
    if len(sentences) == 1:
        return clip_text(sentences[0], 220)
    return clip_text(text, 220)


def _event_type_label(event_type: str, lang: str) -> str:
    """Translate common event tokens used in structured segments."""

    token = str(event_type or "").strip().lower()
    mapping = {
        "scene_observation": _loc("обычная сцена", "қалыпты көрініс", "scene observation", lang=lang),
        "person_activity": _loc("активность человека", "адам әрекеті", "person activity", lang=lang),
        "entry_exit": _loc("вход или выход", "кіру немесе шығу", "entry or exit", lang=lang),
        "vehicle_activity": _loc("движение транспорта", "көлік қозғалысы", "vehicle activity", lang=lang),
        "suspicious_activity": _loc("аномальное событие", "аномал оқиға", "anomalous event", lang=lang),
        "unclassified": _loc("без классификации", "жіктелмеген", "unclassified", lang=lang),
    }
    return mapping.get(token, humanize_token(token))


def _brief_from_outputs(outputs: Dict[str, Any], lang: str) -> str:
    """Build a short fallback summary from ready segments or observations."""

    annotations = outputs.get("annotations") if isinstance(outputs.get("annotations"), list) else []
    observations = outputs.get("clip_observations") if isinstance(outputs.get("clip_observations"), list) else []
    rows = annotations if annotations else observations
    if not rows:
        return ""

    def _is_alert(row: Dict[str, Any]) -> bool:
        notes = row.get("anomaly_notes") if isinstance(row.get("anomaly_notes"), list) else []
        risk = str(row.get("risk_level") or "").strip().lower()
        return bool(row.get("anomaly_flag")) or risk in {"attention", "warning", "critical"} or bool(notes)

    first_row = next((row for row in rows if isinstance(row, dict) and str(row.get("normalized_caption") or row.get("description") or "").strip()), None)
    alert_row = next((row for row in rows if isinstance(row, dict) and _is_alert(row)), None)
    parts: List[str] = []

    if isinstance(first_row, dict):
        first_text = first_sentence(str(first_row.get("normalized_caption") or first_row.get("description") or "").strip())
        if first_text:
            parts.append(clip_text(first_text.rstrip(".") + ".", 170))

    if isinstance(alert_row, dict):
        event_type = str(alert_row.get("event_type") or "").strip()
        event_label = _event_type_label(event_type, lang) if event_type and event_type.lower() not in {"none", "normal", "unknown"} else _loc("аномальный сегмент", "аномал сегмент", "anomalous segment", lang=lang)
        note = first_sentence(" ".join(str(item).strip() for item in (alert_row.get("anomaly_notes") or []) if str(item).strip()))
        alert_text = _loc(
            f"Требует внимания: {event_label}.",
            f"Назар аударуды қажет етеді: {event_label}.",
            f"Needs attention: {event_label}.",
            lang=lang,
        )
        if note:
            alert_text = clip_text(
                _loc(
                    f"Требует внимания: {event_label}. {note}",
                    f"Назар аударуды қажет етеді: {event_label}. {note}",
                    f"Needs attention: {event_label}. {note}",
                    lang=lang,
                ),
                170,
            )
        parts.append(alert_text)

    return " ".join(part for part in parts[:2] if part).strip()


def _stage_label(stage_name: str, lang: str) -> str:
    """Return a localized title for one runtime stage token."""

    token = str(stage_name or "").strip().lower()
    mapping = {
        "preprocess_video": _loc("Подготовка видео", "Бейнені дайындау", "Preprocess", lang=lang),
        "build_clips": _loc("Нарезка клипов", "Клиптерге бөлу", "Build clips", lang=lang),
        "run_vlm_pipeline": _loc("VLM-инференс", "VLM-инференс", "VLM inference", lang=lang),
        "write_clip_observations": _loc("Сохранение наблюдений", "Бақылауларды сақтау", "Save observations", lang=lang),
        "build_segment_schema": _loc("Сборка сегментов", "Сегменттерді құрастыру", "Build segments", lang=lang),
        "structuring_segments": _loc("Структурирование", "Құрылымдау", "Structuring", lang=lang),
        "write_segments": _loc("Сохранение сегментов", "Сегменттерді сақтау", "Save segments", lang=lang),
        "build_summary": _loc("Сборка сводки", "Қысқаша мазмұн құрастыру", "Build summary", lang=lang),
        "guard_summary": _loc("Guard-проверка", "Guard тексеруі", "Guard summary", lang=lang),
        "write_summary": _loc("Сохранение сводки", "Қысқаша мазмұнды сақтау", "Save summary", lang=lang),
        "write_run_manifest": _loc("Обновление manifest", "Manifest жаңарту", "Update manifest", lang=lang),
        "update_outputs_manifest": _loc("Статус outputs", "Outputs күйі", "Outputs status", lang=lang),
        "index_build": _loc("Построение индекса", "Индексті құру", "Index build", lang=lang),
        "enqueue_translation_jobs": _loc("Постановка переводов", "Аудармаларды кезекке қою", "Queue translations", lang=lang),
        "load_source_segments": _loc("Чтение исходных сегментов", "Бастапқы сегменттерді оқу", "Load source segments", lang=lang),
        "translate_segments_mt": _loc("Перевод сегментов", "Сегменттерді аудару", "Translate segments", lang=lang),
        "post_edit_selected_segments": _loc("Полировка выбранных сегментов", "Таңдалған сегменттерді түзету", "Polish selected segments", lang=lang),
        "load_source_summary": _loc("Чтение исходной сводки", "Бастапқы мазмұнды оқу", "Load source summary", lang=lang),
        "translate_summary_mt": _loc("Перевод сводки", "Қысқаша мазмұнды аудару", "Translate summary", lang=lang),
        "post_edit_summary": _loc("Полировка сводки", "Қысқаша мазмұнды түзету", "Polish summary", lang=lang),
        "translate_total": _loc("Полный перевод", "Толық аударма", "Translation total", lang=lang),
        "process_total": _loc("Полный процесс", "Толық процесс", "Process total", lang=lang),
        "write_metrics": _loc("Запись метрик", "Метрикаларды жазу", "Write metrics", lang=lang),
    }
    return mapping.get(token, humanize_token(token))


def _stage_metric_rows(stage_stats: Dict[str, Any], lang: str) -> List[tuple[str, str]]:
    """Return one metrics row per stage in a stable order."""

    rows: List[tuple[str, str]] = []
    seen: set[str] = set()
    for stage_name in _STAGE_ORDER:
        payload = stage_stats.get(stage_name) if isinstance(stage_stats, dict) else None
        if not isinstance(payload, dict):
            continue
        mean_sec = payload.get("mean_sec")
        if mean_sec is None:
            continue
        rows.append((_stage_label(stage_name, lang), _metric_seconds_value(mean_sec)))
        seen.add(stage_name)
    for stage_name, payload in sorted((stage_stats or {}).items()):
        if not isinstance(payload, dict) or stage_name in seen:
            continue
        mean_sec = payload.get("mean_sec")
        if mean_sec is None:
            continue
        rows.append((_stage_label(stage_name, lang), _metric_seconds_value(mean_sec)))
    return rows


def _analytics_metric_rows(outputs: Dict[str, Any], meta: Dict[str, Any], language: str, lang: str) -> List[tuple[str, str]]:
    """Build metrics rows with each processing stage on its own line."""

    metrics = outputs.get("metrics") if isinstance(outputs.get("metrics"), dict) else {}
    indexing = metrics.get("indexing") if isinstance(metrics.get("indexing"), dict) else {}
    translations = metrics.get("translations") if isinstance(metrics.get("translations"), dict) else {}
    summary_polish = metrics.get("summary_polish") if isinstance(metrics.get("summary_polish"), dict) else {}
    language_key = str(language or "").strip().lower()
    lang_index_metrics = indexing.get(language_key) if isinstance(indexing.get(language_key), dict) else {}
    lang_translation_metrics = translations.get(language_key) if isinstance(translations.get(language_key), dict) else {}
    stage_stats = metrics.get("stage_stats_sec") if isinstance(metrics.get("stage_stats_sec"), dict) else {}
    if isinstance(lang_translation_metrics, dict):
        translation_stage_stats = lang_translation_metrics.get("stage_stats_sec")
        if isinstance(translation_stage_stats, dict):
            stage_stats = translation_stage_stats

    rows: List[tuple[str, str]] = [
        (_loc("Длительность видео", "Бейне ұзақтығы", "Video duration", lang=lang), hms(float(meta.get("duration_sec") or 0.0)) if meta.get("duration_sec") else "-"),
        (_loc("Разрешение", "Ажыратымдылығы", "Video resolution", lang=lang), f"{int(meta.get('width'))}x{int(meta.get('height'))}" if meta.get("width") and meta.get("height") else "-"),
        (_loc("Базовый пайплайн", "Негізгі пайплайн", "Base pipeline", lang=lang), _metric_seconds_value(metrics.get("total_time_sec"))),
        (_loc("Полная задача", "Толық тапсырма", "Queue total", lang=lang), _metric_seconds_value(_stage_metric_mean(outputs, "process_total") or _stage_metric_mean(outputs, "translate_total"))),
    ]

    index_time = lang_index_metrics.get("time_sec") if isinstance(lang_index_metrics, dict) else None
    if index_time is None:
        index_time = _stage_metric_mean(outputs, "index_build")
    if index_time is not None:
        rows.append((_loc("Индекс", "Индекс", "Index update", lang=lang), _metric_seconds_value(index_time)))

    if isinstance(lang_translation_metrics, dict) and lang_translation_metrics:
        rows.append((_loc("Перевод", "Аударма", "Translation", lang=lang), _metric_seconds_value(lang_translation_metrics.get("time_sec"))))

    if isinstance(summary_polish, dict) and summary_polish.get("applied"):
        rows.append((_loc("Полировка сводки", "Қысқаша мазмұнды жылтырату", "Summary polish", lang=lang), _metric_seconds_value(summary_polish.get("time_sec"))))

    rows.extend(_stage_metric_rows(stage_stats, lang))
    return rows


def _render_timeline(outputs: Dict[str, Any], lang: str) -> None:
    """Render structured segments or raw observations as a localized timeline."""

    annotations = outputs.get("annotations") if isinstance(outputs.get("annotations"), list) else []
    observations = outputs.get("clip_observations") if isinstance(outputs.get("clip_observations"), list) else []
    rows = annotations if annotations else observations

    _section(_loc("Таймлайн", "Таймлайн", "Timeline", lang=lang))
    if not rows:
        _caption(_loc("Сегменты еще не готовы.", "Сегменттер әлі дайын емес.", "Segments are not available yet.", lang=lang))
        return
    if not annotations and observations:
        _caption(
            _loc(
                "Пока структурирование не завершено, показываются сырые наблюдения по клипам.",
                "Құрылымдау аяқталғанша клиптер бойынша бастапқы бақылаулар көрсетіледі.",
                "Showing raw clip observations while structuring is still running.",
                lang=lang,
            )
        )

    visible = rows[:60]
    for idx, ann in enumerate(visible):
        if not isinstance(ann, dict):
            continue
        start_sec = float(ann.get("start_sec", 0.0) or 0.0)
        end_sec = float(ann.get("end_sec", 0.0) or 0.0)
        description = str(ann.get("normalized_caption") or ann.get("description") or "").strip()
        notes = [str(note).strip() for note in (ann.get("anomaly_notes") or []) if str(note).strip()]
        event_type = str(ann.get("event_type") or "").strip()
        risk = str(ann.get("risk_level") or "").strip().lower()
        is_alert = bool(ann.get("anomaly_flag")) or risk in {"attention", "warning", "critical"} or bool(notes)

        time_col, body_col = st.columns([1.0, 4.6], gap="medium")
        with time_col:
            if st.button(f"{mmss(start_sec)} - {mmss(end_sec)}", key=f"timeline_seek_{idx}", use_container_width=True):
                st.session_state["video_seek_sec"] = int(start_sec)
                st.rerun()
        with body_col:
            if event_type and event_type.lower() not in {"unknown", "none", "normal"}:
                event_label = _loc("Событие", "Оқиға", "Event", lang=lang)
                event_text = _event_type_label(event_type, lang)
                st.markdown(
                    f"<div class='timeline-event{' alert' if is_alert else ''}'>{E(f'{event_label}: {event_text}')}</div>",
                    unsafe_allow_html=True,
                )
            st.markdown(
                f"<div class='timeline-text{' alert' if is_alert else ''}'>{E(description or _loc('Нет описания', 'Сипаттама жоқ', 'No description', lang=lang))}</div>",
                unsafe_allow_html=True,
            )
            if notes:
                note = first_sentence(" ".join(notes))
                st.markdown(
                    f"<div class='timeline-note{' alert' if is_alert else ''}'>{E(_loc('Внимание', 'Назар', 'Attention', lang=lang))}: {E(note)}</div>",
                    unsafe_allow_html=True,
                )
        if idx < len(visible) - 1:
            st.markdown("<div class='timeline-divider'></div>", unsafe_allow_html=True)


def video_analytics_tab(client: BackendClient, cfg: Any, ui_text: Dict[str, Dict[str, Any]]) -> None:
    """Render the analytics page using the latest localized UI behavior."""

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

    variant_id = variant_from_token(st.session_state.get("library_variant"))
    language = str(st.session_state.get("library_lang") or "en")
    video_id = str(st.session_state.get("selected_video_id") or "")

    try:
        initial_outputs = client.get_video_outputs(video_id, language, variant=variant_id)
    except Exception:
        initial_outputs = {}
    try:
        initial_queue = client.queue_list()
    except Exception:
        initial_queue = {}

    def _analytics_body() -> None:
        try:
            outputs = client.get_video_outputs(video_id, language, variant=variant_id)
        except Exception as exc:
            outputs = {}
            soft_note(f"{_error_prefix(lang)}: {exc}", kind="warn")
        try:
            queue = client.queue_list()
        except Exception:
            queue = initial_queue

        top = st.columns([1.88, 1.12], gap="large")
        with top[0]:
            if playable_path.exists():
                try:
                    st.video(str(playable_path), start_time=int(st.session_state.get("video_seek_sec") or 0))
                except TypeError:
                    st.video(str(playable_path))
            else:
                soft_note(_loc("Исходное видео не найдено.", "Бастапқы бейне табылмады.", "Source video not found.", lang=lang), kind="warn")
            st.markdown("<div class='analytics-timeline-offset'></div>", unsafe_allow_html=True)
            _render_timeline(outputs, lang)

        with top[1]:
            summary_text = str(outputs.get("global_summary") or "").strip()
            stage_copy = _analytics_live_stage(outputs, queue, video_id)
            brief_text = _summary_brief_text(summary_text, lang) if summary_text else _brief_from_outputs(outputs, lang)

            _section(_loc("Описание видео", "Бейне сипаттамасы", "Video description", lang=lang))
            if brief_text:
                st.markdown(f"<div class='analytics-summary'>{E(brief_text)}</div>", unsafe_allow_html=True)
            else:
                placeholder = _loc(
                    "Краткая сводка появится автоматически, когда базовые результаты будут готовы.",
                    "Негізгі нәтижелер дайын болғанда қысқаша мазмұн автоматты түрде пайда болады.",
                    "A short summary will appear automatically when the base outputs are ready.",
                    lang=lang,
                )
                st.markdown(f"<div class='analytics-summary'>{E(placeholder)}</div>", unsafe_allow_html=True)
            if stage_copy:
                _caption(stage_copy)

            _section(_loc("Статус", "Күйі", "Status", lang=lang))
            _render_status_rows(_analytics_status_items(outputs, raw_exists, lang), lang)

            _section(_loc("Метрики", "Метрикалар", "Metrics", lang=lang))
            _render_info_rows(_analytics_metric_rows(outputs, meta, language, lang))

    refresh_sec = max(2, int(getattr(getattr(cfg, "ui", None), "cache_ttl_sec", 2) or 2))
    manifest_entry = _current_manifest_entry(initial_outputs)
    manifest_status = str(manifest_entry.get("status") or "").strip().lower()
    active_queue = isinstance(initial_queue.get("running") if isinstance(initial_queue, dict) else None, dict) and str((initial_queue.get("running") or {}).get("video_id") or "") == video_id
    needs_poll = active_queue or manifest_status in {"processing", "observed", "segments_ready", "summary_ready", "translating"}
    _run_live_fragment(_analytics_body, run_every_sec=refresh_sec if needs_poll else None)
