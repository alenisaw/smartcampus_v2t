# app/view/storage.py
"""
Storage and queue Streamlit page logic.

Purpose:
- Render the video library, upload controls, and processing queue workflow.
- Keep storage-specific UI state and helpers grouped in one module.
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
    available_profile_ids,
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
    _video_ids,
    _video_items,
    soft_note,
)

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

    status = str(item.get("status") or "").strip().lower()
    if status in {"processing", "observed", "segments_ready", "translating"}:
        return "processing"
    if status == "failed":
        return "failed"
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
        "main": _loc("Балансный", "Теңгерімді", "Balanced", lang=lang),
        "experimental": _loc("Экспериментальный", "Эксперименттік", "Experimental", lang=lang),
    }.get(str(profile or "").strip().lower(), str(profile or ""))


def _profile_supports_variants(profile: str) -> bool:
    """Return whether the profile exposes experimental variant controls."""

    return str(profile or "").strip().lower() == "experimental"


def _storage_state_label(status: str, lang: str) -> str:
    """Return a localized storage readiness label."""

    return {
        "processing": _loc("Обработка", "Өңделуде", "Processing", lang=lang),
        "failed": _loc("Ошибка", "Қате", "Failed", lang=lang),
        "ready": _loc("Готово", "Дайын", "Ready", lang=lang),
        "raw": _loc("Исходный файл", "Бастапқы файл", "Raw file", lang=lang),
    }.get(str(status or "").strip().lower(), _loc("Исходный файл", "Бастапқы файл", "Raw file", lang=lang))


def _queue_has_live_activity(queue: Dict[str, Any]) -> bool:
    """Return whether the queue has running or queued jobs."""

    if not isinstance(queue, dict):
        return False
    return bool(queue.get("running")) or bool(queue.get("queued"))


def _queue_stage_copy(item: Dict[str, Any], lang: str) -> str:
    """Return a compact stage/progress string for a queue item."""

    stage = humanize_token(str(item.get("stage") or item.get("job_type") or "queued")) or _queue_job_status(item, lang)
    message = str(item.get("message") or "").strip()
    try:
        progress = max(0.0, min(1.0, float(item.get("progress") or 0.0)))
    except Exception:
        progress = 0.0
    parts = [stage]
    if progress > 0:
        parts.append(f"{int(progress * 100)}%")
    if message and message.lower() != stage.lower():
        parts.append(message)
    return " · ".join(parts)


def _render_running_job(queue: Dict[str, Any], lang: str) -> None:
    """Render the currently running job with stage and progress details."""

    running = queue.get("running") if isinstance(queue, dict) else None
    if not isinstance(running, dict) or not running.get("job_id"):
        return

    variant_token = "__base__" if not running.get("variant") else str(running.get("variant"))
    title = str(running.get("video_id") or running.get("job_id") or "-")
    meta = " · ".join(
        [
            _profile_label(str(running.get("profile") or "main"), lang),
            _variant_option_label(variant_token, lang),
            humanize_token(str(running.get("job_type") or "process")),
        ]
    )
    updated_at = _format_time(float(running.get("updated_at") or 0.0))
    try:
        progress = max(0.0, min(1.0, float(running.get("progress") or 0.0)))
    except Exception:
        progress = 0.0
    updated_label = _loc("Обновлено", "Жаңартылды", "Updated", lang=lang)

    _caption(_loc("Текущая задача", "Ағымдағы тапсырма", "Current job", lang=lang))
    with st.container():
        _mark("row-marker", "row-marker--selected")
        st.markdown(f"<div class='row-title'>{E(title)}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='row-meta'>{E(meta)}</div>", unsafe_allow_html=True)
        st.progress(progress, text=_queue_stage_copy(running, lang))
        st.markdown(f"<div class='row-meta subtle'>{E(f'{updated_label}: {updated_at}')}</div>", unsafe_allow_html=True)


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
    variant = variant_from_token(token)
    variant_labels = {
        "fast": _loc("Быстрый", "Жылдам", "Fast", lang=lang),
        "throughput": _loc("Потоковый", "Өткізгіштік", "Throughput", lang=lang),
        "max_quality": _loc("Макс. качество", "Макс. сапа", "Max Quality", lang=lang),
    }
    if variant in variant_labels:
        return variant_labels[variant]
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
    state_text = _loc("Пауза", "Аялдаған", "Paused", lang=lang) if is_paused else _loc("Активна", "Белсенді", "Active", lang=lang)
    running_video_id = str((running or {}).get("video_id") or "").strip()
    running_stage = _queue_stage_copy(running, lang) if isinstance(running, dict) and running_video_id else ""
    running_text = f"{running_video_id} · {running_stage}".strip(" ·") if running_video_id else _loc("нет", "жоқ", "none", lang=lang)
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

    profile_options = available_profile_ids(cfg)
    default_profile = "main" if "main" in profile_options else (profile_options[0] if profile_options else "main")
    if "run_profile" not in st.session_state:
        st.session_state["run_profile"] = default_profile
    if st.session_state.get("run_profile") not in profile_options:
        st.session_state["run_profile"] = default_profile

    st.selectbox(
        _loc("Профиль", "Профиль", "Profile", lang=lang),
        options=profile_options,
        key="run_profile",
        format_func=lambda value: _profile_label(str(value), lang),
    )

    active_profile = str(st.session_state.get("run_profile") or default_profile)
    supports_variants = _profile_supports_variants(active_profile)
    variants = available_variant_ids(cfg)
    variant_options = ["__fanout__"] + variants if supports_variants else ["__base__"]
    default_variant = "__fanout__" if supports_variants else "__base__"
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
        disabled=not supports_variants,
    )
    st.checkbox(_loc("Перезаписать", "Қайта жазу", "Overwrite", lang=lang), key="run_force_overwrite")

    action_row = st.columns(3, gap="small")
    with action_row[0]:
        if st.button(ICON_START, key="queue_start_btn", use_container_width=True, help=_loc("Запустить обработку", "Өңдеуді бастау", "Start processing", lang=lang), disabled=not bool(selected_video_id)):
            try:
                selected_variant = None
                if supports_variants and st.session_state.get("run_variant") not in {"__fanout__", "__base__"}:
                    selected_variant = variant_from_token(st.session_state.get("run_variant"))
                payload = client.create_job(
                    str(selected_video_id or ""),
                    extra={"force_overwrite": bool(st.session_state.get("run_force_overwrite", False))},
                    profile=active_profile,
                    variant=selected_variant,
                )
                soft_note(f"{_loc('Задача поставлена в очередь', 'Тапсырма кезекке қойылды', 'Job queued', lang=lang)}: {payload.get('job_id')}", kind="ok")
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
            if st.button(ICON_PAUSE, key="queue_pause_btn", use_container_width=True, help=_loc("Пауза", "Аялдату", "Pause", lang=lang)):
                try:
                    client.queue_pause()
                    st.rerun()
                except Exception as exc:
                    soft_note(f"{_error_prefix(lang)}: {exc}", kind="warn")
    with action_row[2]:
        if st.button(ICON_REFRESH, key="queue_refresh_btn", use_container_width=True, help=_loc("Обновить", "Жаңарту", "Refresh", lang=lang)):
            st.rerun()

    _queue_state_line(queue, selected_video_id, lang)
    _render_running_job(queue, lang)

    queued = queue.get("queued") if isinstance(queue, dict) else []
    if not isinstance(queued, list) or not queued:
        if not isinstance(queue.get("running") if isinstance(queue, dict) else None, dict):
            _caption(_loc("Очередь пуста.", "Кезек бос.", "Queue is empty.", lang=lang))
        else:
            _caption(_loc("Ожидающих задач нет. Активная задача показана выше.", "Күтіп тұрған тапсырма жоқ. Белсенді тапсырма жоғарыда көрсетілген.", "No queued jobs. The active job is shown above.", lang=lang))
        return

    _caption(_loc("Задачи в очереди", "Кезектегі тапсырмалар", "Queued jobs", lang=lang))
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
                st.markdown(f"<div class='row-meta subtle'>{E(_queue_stage_copy(item, lang))}</div>", unsafe_allow_html=True)
            with status_col:
                st.markdown(f"<div class='row-status'>{E(_queue_job_status(item, lang))}</div>", unsafe_allow_html=True)

            actions = st.columns(3, gap="small")
            with actions[0]:
                if st.button(ICON_UP, key=f"queue_up_{job_id}_{idx}", use_container_width=True, help=_loc("Поднять", "Жоғары көтеру", "Move up", lang=lang)):
                    try:
                        client.queue_move(job_id, "up")
                        st.rerun()
                    except Exception as exc:
                        soft_note(f"{_error_prefix(lang)}: {exc}", kind="warn")
            with actions[1]:
                if st.button(ICON_DOWN, key=f"queue_down_{job_id}_{idx}", use_container_width=True, help=_loc("Опустить", "Төмен түсіру", "Move down", lang=lang)):
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
        _storage_state_label(str(item.get("_ui_status") or "raw"), lang),
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
            if st.button(ICON_START, key=f"queue_video_{video_id}", use_container_width=True, help=_loc("В очередь", "Кезекке", "Add to queue", lang=lang)):
                try:
                    profile_options = available_profile_ids(cfg)
                    default_profile = "main" if "main" in profile_options else (profile_options[0] if profile_options else "main")
                    profile = str(st.session_state.get("run_profile") or default_profile)
                    if profile not in profile_options:
                        profile = default_profile
                        st.session_state["run_profile"] = profile

                    supports_variants = _profile_supports_variants(profile)
                    variants = available_variant_ids(cfg)
                    variant_options = ["__fanout__"] + variants if supports_variants else ["__base__"]
                    default_variant = "__fanout__" if supports_variants else "__base__"
                    current_variant = str(st.session_state.get("run_variant") or default_variant)
                    if current_variant not in variant_options:
                        current_variant = default_variant
                        st.session_state["run_variant"] = current_variant

                    selected_variant = None
                    if supports_variants and current_variant not in {"__fanout__", "__base__"}:
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
                f"<div class='confirm-copy'>{E(_loc('Удалить выбранное видео из библиотеки?', 'Таңдалған бейнені кітапханадан жою керек пе?', 'Delete the selected video from the library?', lang=lang))}</div>",
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
            if st.button(ICON_DELETE, key=f"cancel_delete_{video_id}", use_container_width=True, help=_loc("Отмена", "Бас тарту", "Cancel", lang=lang)):
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
                placeholder=_loc("ID, путь, сводка, язык", "ID, жол, мазмұн, тіл", "ID, path, summary, language", lang=lang),
            )
        with filters[1]:
            st.selectbox(
                _loc("Сортировка", "Сұрыптау", "Sort", lang=lang),
                options=["newest", "oldest", "name"],
                key="storage_sort",
                format_func=lambda value: {
                    "newest": _loc("Сначала новые", "Алдымен жаңалары", "Newest", lang=lang),
                    "oldest": _loc("Сначала старые", "Алдымен ескілері", "Oldest", lang=lang),
                    "name": _loc("Имя / ID", "Атау / ID", "Name / ID", lang=lang),
                }.get(value, value),
            )
        with filters[2]:
            st.selectbox(
                _loc("Статус", "Күй", "Status", lang=lang),
                options=["all", "processing", "ready", "raw", "failed"],
                key="storage_status",
                format_func=lambda value: {
                    "all": _loc("Все", "Барлығы", "All", lang=lang),
                    "processing": _storage_state_label("processing", lang),
                    "ready": _storage_state_label("ready", lang),
                    "raw": _storage_state_label("raw", lang),
                    "failed": _storage_state_label("failed", lang),
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

        def _queue_panel_body() -> None:
            try:
                live_queue = client.queue_list()
            except Exception:
                live_queue = queue
            _render_queue_panel(client, cfg, live_queue, selected, lang)

        refresh_sec = max(2, int(getattr(getattr(cfg, "ui", None), "cache_ttl_sec", 2) or 2))
        _run_live_fragment(_queue_panel_body, run_every_sec=refresh_sec if _queue_has_live_activity(queue) else None)
