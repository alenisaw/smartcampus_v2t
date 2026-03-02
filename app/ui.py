# app/ui.py
"""
Compact custom UI for SmartCampus V2T.
"""

from __future__ import annotations

import base64
import html
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import streamlit as st

from app.api_client import BackendClient


def E(value: Any) -> str:
    """Escape text for safe HTML rendering."""

    return html.escape("" if value is None else str(value), quote=True)


def _mtime(path: Path) -> float:
    """Return file mtime or zero on failure."""

    try:
        return float(path.stat().st_mtime)
    except Exception:
        return 0.0


def mmss(sec: float) -> str:
    """Format seconds as M:SS."""

    total = max(0, int(round(float(sec or 0.0))))
    return f"{total // 60}:{total % 60:02d}"


def hms(sec: float) -> str:
    """Format seconds as H:MM:SS or M:SS."""

    total = max(0, int(round(float(sec or 0.0))))
    hours = total // 3600
    mins = (total % 3600) // 60
    secs = total % 60
    if hours > 0:
        return f"{hours}:{mins:02d}:{secs:02d}"
    return f"{mins}:{secs:02d}"


def fmt_bytes(num: Optional[float]) -> str:
    """Format byte counts for UI display."""

    if num is None:
        return "-"
    value = float(num)
    units = ["B", "KB", "MB", "GB", "TB"]
    idx = 0
    while value >= 1024.0 and idx < len(units) - 1:
        value /= 1024.0
        idx += 1
    if idx == 0:
        return f"{int(value)} {units[idx]}"
    return f"{value:.2f} {units[idx]}"


def soft_note(text: str, kind: str = "info") -> None:
    """Render a compact inline message."""

    css = {"info": "soft-note", "warn": "soft-warn", "ok": "soft-ok"}.get(kind, "soft-note")
    st.markdown(f"<div class='{css}'>{E(text)}</div>", unsafe_allow_html=True)


def load_and_apply_css(styles_path: Path) -> None:
    """Load and inject UI CSS."""

    if not styles_path.exists():
        return
    css = styles_path.read_text(encoding="utf-8")
    if css.strip():
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
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
        if not isinstance(tabs, list) or len(tabs) != 3:
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


def render_i18n_metrics() -> None:
    """No-op compatibility hook for app/main.py."""

    return


def _img_to_data_uri(path: Path) -> Optional[str]:
    """Convert an image file into a data URI."""

    if not path.exists():
        return None
    try:
        return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("utf-8")
    except Exception:
        return None


def render_header(T: Dict[str, Any], labels: List[str], ids: List[str], current_tab: str, logo_path: Path) -> None:
    """Render the top hero block and tab navigation."""

    links: List[str] = []
    for label, tab_id in zip(labels, ids):
        css = "nav-pill active" if tab_id == current_tab else "nav-pill"
        links.append(f"<a class='{css}' href='?tab={E(tab_id)}' target='_self'>{E(label)}</a>")

    logo_html = ""
    logo_uri = _img_to_data_uri(logo_path)
    if logo_uri:
        logo_html = f"<img class='brand-logo' src='{logo_uri}' alt='logo' />"

    st.markdown(
        f"""
        <div class="hero-shell">
            <div class="hero-brand">
                <div class="brand-badge">{logo_html}</div>
                <div>
                    <div class="brand-title">{E(Tget(T, "app_title", "SmartCampus V2T"))}</div>
                    <div class="brand-subtitle">{E(Tget(T, "app_subtitle", ""))}</div>
                </div>
            </div>
            <div class="hero-copy">{E(Tget(T, "hero_copy", ""))}</div>
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
            <div>{E(Tget(T, 'footer_left', 'Alen Issayev Diploma Project'))}</div>
            <div>{E(Tget(T, 'footer_right', 'SmartCampus V2T'))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _variant_token(value: Optional[str]) -> str:
    """Map optional variant ids to a stable UI token."""

    text = str(value or "").strip().lower()
    return text if text else "__base__"


def _variant_from_token(token: Optional[str]) -> Optional[str]:
    """Map a UI token back to an optional variant id."""

    text = str(token or "").strip().lower()
    return None if (not text or text == "__base__") else text


def _variant_label(token: Optional[str]) -> str:
    """Human-readable variant label."""

    variant = _variant_from_token(token)
    return "BASE" if variant is None else variant.upper()


def _video_variant_tokens(video_item: Optional[Dict[str, Any]]) -> List[str]:
    """Build available variant tokens for a video."""

    tokens = ["__base__"]
    if not isinstance(video_item, dict):
        return tokens
    variants = video_item.get("variants") or {}
    if not isinstance(variants, dict):
        return tokens
    for key in sorted(str(item).strip().lower() for item in variants.keys() if str(item).strip()):
        token = _variant_token(key)
        if token not in tokens:
            tokens.append(token)
    return tokens


@st.cache_data(show_spinner=False)
def get_video_meta(video_path_str: str, mtime: float) -> Dict[str, Any]:
    """Read basic video metadata through OpenCV."""

    _ = mtime
    path = Path(video_path_str)
    if not path.exists():
        return {}

    meta: Dict[str, Any] = {}
    try:
        stat = path.stat()
        meta["size_bytes"] = int(stat.st_size)
        meta["updated_at"] = float(stat.st_mtime)
    except Exception:
        pass

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return meta
    try:
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if width and height:
            meta["width"] = int(width)
            meta["height"] = int(height)
        if fps:
            meta["fps"] = float(fps)
        if frames:
            meta["frames"] = int(frames)
        if fps and frames and fps > 0:
            meta["duration_sec"] = float(frames) / float(fps)
    finally:
        cap.release()
    return meta


def _ensure_browser_video(video_path: Path) -> Path:
    """Ensure the selected video has a browser-friendly container for playback."""

    ext = video_path.suffix.lower()
    if ext == ".mp4":
        return video_path

    mp4_path = video_path.with_suffix(".mp4")
    if mp4_path.exists() and _mtime(mp4_path) >= _mtime(video_path):
        return mp4_path

    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        str(mp4_path),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode == 0 and mp4_path.exists():
            return mp4_path
    except Exception:
        pass
    return video_path


def _first_sentence(text: str) -> str:
    """Extract a short headline from a longer summary."""

    clean = " ".join(str(text or "").split()).strip()
    if not clean:
        return ""
    for sep in (". ", "! ", "? ", " - "):
        idx = clean.find(sep)
        if idx > 0:
            return clean[:idx].strip(" .")
    return clean[:160].strip()


def _clip_text(text: str, limit: int = 260) -> str:
    """Clip long text for compact card display."""

    clean = " ".join(str(text or "").split()).strip()
    if len(clean) <= limit:
        return clean
    return clean[: limit - 3].rstrip() + "..."


def _render_capsules(items: List[str], *, accent: bool = False) -> None:
    """Render capsule tags as a single responsive row."""

    filtered = [str(item).strip().upper() for item in items if str(item).strip()]
    if not filtered:
        return
    css = "capsule accent" if accent else "capsule"
    html_items = "".join([f"<span class='{css}'>{E(item)}</span>" for item in filtered])
    st.markdown(f"<div class='capsule-row'>{html_items}</div>", unsafe_allow_html=True)


def _hit_capsules(hit: Dict[str, Any]) -> List[str]:
    """Build compact search-result capsules from structured hit fields."""

    tags: List[str] = []
    for key in ("event_type", "risk_level", "people_count_bucket", "motion_type"):
        value = str(hit.get(key) or "").strip()
        if value:
            tags.append(value)
    if bool(hit.get("anomaly_flag", False)):
        tags.append("anomaly")
    variant = str(hit.get("variant") or "").strip()
    if variant:
        tags.append(variant)
    return tags[:6]


def _collect_scene_capsules(outputs: Dict[str, Any]) -> List[str]:
    """Derive scene-level capsules from structured outputs."""

    annotations = outputs.get("annotations") if isinstance(outputs.get("annotations"), list) else []
    event_counter: Counter[str] = Counter()
    risk_counter: Counter[str] = Counter()
    people_counter: Counter[str] = Counter()
    motion_counter: Counter[str] = Counter()
    anomaly_notes: List[str] = []
    anomaly_count = 0

    for ann in annotations:
        if not isinstance(ann, dict):
            continue
        for key, counter in (
            ("event_type", event_counter),
            ("risk_level", risk_counter),
            ("people_count_bucket", people_counter),
            ("motion_type", motion_counter),
        ):
            value = str(ann.get(key) or "").strip()
            if value:
                counter[value] += 1
        if bool(ann.get("anomaly_flag", False)):
            anomaly_count += 1
        notes = ann.get("anomaly_notes") or []
        if isinstance(notes, list):
            for note in notes:
                text = str(note).strip()
                if text:
                    anomaly_notes.append(text)

    tags: List[str] = []
    if event_counter:
        tags.extend([item for item, _ in event_counter.most_common(2)])
    if risk_counter:
        tags.append(next(iter(risk_counter.most_common(1)))[0])
    if people_counter:
        tags.append(next(iter(people_counter.most_common(1)))[0])
    if motion_counter:
        tags.append(next(iter(motion_counter.most_common(1)))[0])
    if anomaly_count > 0:
        if anomaly_notes:
            note_counts = Counter(anomaly_notes)
            tags.extend([item for item, _ in note_counts.most_common(2)])
        else:
            tags.append("anomaly detected")

    run_manifest = outputs.get("run_manifest") if isinstance(outputs.get("run_manifest"), dict) else {}
    profile = str(run_manifest.get("profile") or "").strip()
    variant = str(outputs.get("variant") or "").strip()
    if profile:
        tags.append(profile)
    if variant:
        tags.append(variant)

    deduped: List[str] = []
    seen = set()
    for item in tags:
        key = str(item).strip().lower()
        if key and key not in seen:
            seen.add(key)
            deduped.append(str(item))
    return deduped[:8]


def _collect_available_languages(video_item: Dict[str, Any], selected_variant: Optional[str]) -> List[str]:
    """Return available output languages for the current selection."""

    if selected_variant:
        variants = video_item.get("variants") or {}
        bucket = variants.get(selected_variant) if isinstance(variants, dict) else {}
        langs = bucket.get("languages") if isinstance(bucket, dict) else []
    else:
        langs = video_item.get("languages") or []
    clean = [str(item).strip().lower() for item in langs if str(item).strip()]
    return clean or ["en"]


def _render_language_switcher(T: Dict[str, Any], cfg: Any) -> None:
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


def _render_scene_panel(T: Dict[str, Any], outputs: Dict[str, Any], video_meta: Dict[str, Any]) -> None:
    """Render a scene overview with headline, summary, and top-level capsules."""

    summary_text = str(outputs.get("global_summary") or "").strip()
    headline = _first_sentence(summary_text) or Tget(T, "empty_summary", "No summary available yet.")

    st.markdown(f"<div class='section-title'>{E(Tget(T, 'scene_overview', 'Scene overview'))}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='scene-headline'>{E(headline)}</div>", unsafe_allow_html=True)
    _render_capsules(_collect_scene_capsules(outputs), accent=True)

    metrics = outputs.get("metrics") if isinstance(outputs.get("metrics"), dict) else {}
    info: List[str] = []
    if video_meta.get("duration_sec"):
        info.append(f"{Tget(T, 'meta_duration', 'Duration')}: {hms(video_meta.get('duration_sec', 0.0))}")
    if video_meta.get("width") and video_meta.get("height"):
        info.append(f"{Tget(T, 'meta_resolution', 'Resolution')}: {video_meta['width']}x{video_meta['height']}")
    if metrics.get("num_clips") is not None:
        info.append(f"{Tget(T, 'metric_clips', 'Clips')}: {metrics.get('num_clips')}")
    if metrics.get("total_time_sec") is not None:
        info.append(f"{Tget(T, 'metrics_total', 'Total')}: {float(metrics.get('total_time_sec') or 0.0):.2f}s")
    if info:
        st.markdown(
            "<div class='stat-row'>" + "".join([f"<span class='stat-chip'>{E(item)}</span>" for item in info]) + "</div>",
            unsafe_allow_html=True,
        )

    with st.expander(Tget(T, "detailed_summary", "Detailed narrative"), expanded=False):
        if summary_text:
            st.write(summary_text)
        else:
            st.caption(Tget(T, "empty_summary", "No summary available yet."))


def _render_segment_cards(T: Dict[str, Any], annotations: List[Dict[str, Any]], session_key: str) -> None:
    """Render segment cards without per-segment tag clutter."""

    st.markdown(f"<div class='section-title'>{E(Tget(T, 'segments_title', 'Segments and clips'))}</div>", unsafe_allow_html=True)
    if not annotations:
        st.caption(Tget(T, "no_annotations", "No annotations available."))
        return

    for idx, ann in enumerate(annotations[:24]):
        start_sec = float(ann.get("start_sec", 0.0) or 0.0)
        end_sec = float(ann.get("end_sec", 0.0) or 0.0)
        caption = str(ann.get("normalized_caption") or ann.get("description") or "").strip()
        segment_id = str(ann.get("segment_id") or "").strip()
        cols = st.columns([5, 1], gap="small")
        with cols[0]:
            st.markdown(
                f"""
                <div class="segment-card">
                    <div class="segment-time">{E(mmss(start_sec))} - {E(mmss(end_sec))}</div>
                    <div class="segment-title">{E(segment_id or f"SEGMENT {idx + 1}")}</div>
                    <div class="segment-text">{E(_clip_text(caption, 220))}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with cols[1]:
            if st.button(Tget(T, "open_clip", "Open"), key=f"{session_key}_{idx}", use_container_width=True):
                st.session_state["video_seek_sec"] = int(start_sec)


def _render_video_library(
    T: Dict[str, Any],
    client: BackendClient,
    videos: List[Dict[str, Any]],
    selected_video_id: str,
) -> None:
    """Render the left-side library cards."""

    st.markdown(f"<div class='section-title'>{E(Tget(T, 'library_title', 'Library'))}</div>", unsafe_allow_html=True)
    for item in videos:
        if not isinstance(item, dict):
            continue
        video_id = str(item.get("video_id") or "")
        active = " active" if video_id == selected_video_id else ""
        path = Path(str(item.get("path") or ""))
        meta = get_video_meta(str(path), _mtime(path)) if path.exists() else {}
        summary = ""
        try:
            outputs = client.get_video_outputs(video_id, "en")
            summary = _first_sentence(str(outputs.get("global_summary") or ""))
        except Exception:
            summary = ""

        st.markdown(f"<div class='library-card{active}'>", unsafe_allow_html=True)
        st.markdown(f"<div class='library-card-title'>{E(video_id)}</div>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='library-card-sub'>{E(fmt_bytes(meta.get('size_bytes')))} · {E(hms(meta.get('duration_sec', 0.0)) if meta.get('duration_sec') else '-')}</div>",
            unsafe_allow_html=True,
        )
        if summary:
            st.markdown(f"<div class='library-card-copy'>{E(_clip_text(summary, 100))}</div>", unsafe_allow_html=True)
        if st.button(Tget(T, "open_video", "Open"), key=f"pick_video_{video_id}", use_container_width=True):
            st.session_state["selected_video_id"] = video_id
            st.session_state["video_seek_sec"] = 0
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)


def _render_upload_panel(T: Dict[str, Any], client: BackendClient) -> None:
    """Render the upload widget and backend upload flow."""

    st.markdown(f"<div class='section-title'>{E(Tget(T, 'upload_title', 'Upload video'))}</div>", unsafe_allow_html=True)
    upload = st.file_uploader(
        Tget(T, "upload_hint", "Choose a video file"),
        type=["mp4", "mov", "mkv", "avi"],
        key="library_upload",
        label_visibility="collapsed",
    )
    if upload is not None:
        if st.button(Tget(T, "upload_action", "Save to library"), key="library_upload_submit", use_container_width=True):
            try:
                payload = client.upload_video(upload.name, upload.getvalue())
                st.session_state["selected_video_id"] = str(payload.get("video_id") or Path(upload.name).stem)
                soft_note(Tget(T, "upload_done", "Video uploaded successfully."), kind="ok")
                st.rerun()
            except Exception as exc:
                soft_note(f"{Tget(T, 'error_prefix', 'Error')}: {exc}", kind="warn")


def _render_run_panel(T: Dict[str, Any], client: BackendClient, selected_video_id: str) -> None:
    """Render run controls for main and experimental modes."""

    st.markdown(f"<div class='section-title'>{E(Tget(T, 'run_title', 'Run pipeline'))}</div>", unsafe_allow_html=True)
    profile = st.radio(
        Tget(T, "profile", "Profile"),
        options=["main", "experimental"],
        horizontal=True,
        key="run_profile",
        label_visibility="collapsed",
    )
    if st.button(Tget(T, "run_action", "Start processing"), key="run_video_btn", use_container_width=True):
        try:
            response = client.create_job(
                selected_video_id,
                extra={"force_overwrite": False},
                profile=profile,
                variant=None,
            )
            soft_note(f"{Tget(T, 'job_created', 'Job queued')}: {response.get('job_id')}", kind="ok")
        except Exception as exc:
            soft_note(f"{Tget(T, 'error_prefix', 'Error')}: {exc}", kind="warn")

    try:
        queue = client.queue_list()
    except Exception:
        queue = {}
    running = queue.get("running") if isinstance(queue, dict) else None
    if isinstance(running, dict) and running.get("job_id"):
        st.markdown(
            f"<div class='queue-inline'>{E(Tget(T, 'running_now', 'Running now'))}: {E(str(running.get('video_id') or ''))} · {E(str(running.get('stage') or ''))}</div>",
            unsafe_allow_html=True,
        )


def _render_player(path: Path, start_sec: int) -> None:
    """Render the video player for the given file."""

    try:
        st.video(str(path), start_time=int(max(0, start_sec)))
    except TypeError:
        st.video(str(path))


def gallery_tab(client: BackendClient, cfg: Any, ui_text: Dict[str, Dict[str, Any]]) -> None:
    """Render the main video library and playback page."""

    T = get_T(ui_text, str(st.session_state.get("ui_lang") or cfg.ui.default_lang or "en"))
    _render_language_switcher(T, cfg)

    videos = client.list_videos()
    if not videos:
        _render_upload_panel(T, client)
        soft_note(Tget(T, "empty_library", "The library is empty. Upload a video to begin."), kind="info")
        return

    video_ids = [str(item.get("video_id") or "") for item in videos if str(item.get("video_id") or "")]
    selected = str(st.session_state.get("selected_video_id") or (video_ids[0] if video_ids else ""))
    if selected not in video_ids and video_ids:
        selected = video_ids[0]
    st.session_state["selected_video_id"] = selected

    selected_video = next((item for item in videos if str(item.get("video_id") or "") == selected), None) or {}
    variant_tokens = _video_variant_tokens(selected_video)
    current_variant_token = str(st.session_state.get("library_variant") or variant_tokens[0])
    if current_variant_token not in variant_tokens:
        current_variant_token = variant_tokens[0]
    st.session_state["library_variant"] = current_variant_token
    current_variant = _variant_from_token(current_variant_token)

    available_languages = _collect_available_languages(selected_video, current_variant)
    current_lang = str(st.session_state.get("library_lang") or available_languages[0])
    if current_lang not in available_languages:
        current_lang = available_languages[0]
    st.session_state["library_lang"] = current_lang

    left, right = st.columns([1.05, 1.95], gap="large")
    with left:
        _render_upload_panel(T, client)
        _render_run_panel(T, client, selected)
        _render_video_library(T, client, videos, selected)

    with right:
        toolbar = st.columns([2, 1, 1], gap="small")
        with toolbar[0]:
            st.selectbox(
                Tget(T, "selected_video", "Selected video"),
                options=video_ids,
                key="selected_video_id",
            )
        with toolbar[1]:
            st.selectbox(
                Tget(T, "variant", "Variant"),
                options=variant_tokens,
                format_func=_variant_label,
                key="library_variant",
            )
        with toolbar[2]:
            st.selectbox(
                Tget(T, "output_language", "Output language"),
                options=available_languages,
                key="library_lang",
            )

        selected_video = next((item for item in videos if str(item.get("video_id") or "") == st.session_state["selected_video_id"]), None) or {}
        raw_path = Path(str(selected_video.get("path") or ""))
        play_path = _ensure_browser_video(raw_path) if raw_path.exists() else raw_path
        meta = get_video_meta(str(play_path), _mtime(play_path)) if play_path.exists() else {}

        outputs = {}
        try:
            outputs = client.get_video_outputs(
                str(st.session_state.get("selected_video_id") or ""),
                str(st.session_state.get("library_lang") or "en"),
                variant=_variant_from_token(st.session_state.get("library_variant")),
            )
        except Exception as exc:
            soft_note(f"{Tget(T, 'error_prefix', 'Error')}: {exc}", kind="warn")
            outputs = {}

        if play_path.exists():
            _render_player(play_path, int(st.session_state.get("video_seek_sec") or 0))
        else:
            soft_note(Tget(T, "raw_missing", "Source video not found."), kind="warn")

        _render_scene_panel(T, outputs, meta)
        annotations = outputs.get("annotations") if isinstance(outputs.get("annotations"), list) else []
        _render_segment_cards(T, annotations, "library_segment")


def search_tab(client: BackendClient, cfg: Any, ui_text: Dict[str, Dict[str, Any]]) -> None:
    """Render the analytics and retrieval view."""

    T = get_T(ui_text, str(st.session_state.get("ui_lang") or cfg.ui.default_lang or "en"))
    _render_language_switcher(T, cfg)

    videos = client.list_videos()
    if not videos:
        soft_note(Tget(T, "empty_library", "The library is empty. Upload a video to begin."), kind="info")
        return

    video_ids = [""] + [str(item.get("video_id") or "") for item in videos if str(item.get("video_id") or "")]
    selected_video_id = str(st.session_state.get("search_video_id") or "")
    if selected_video_id not in video_ids:
        selected_video_id = ""
    selected_item = next((item for item in videos if str(item.get("video_id") or "") == selected_video_id), None)
    variant_tokens = _video_variant_tokens(selected_item)
    current_variant_token = str(st.session_state.get("search_variant") or variant_tokens[0])
    if current_variant_token not in variant_tokens:
        current_variant_token = variant_tokens[0]
    st.session_state["search_variant"] = current_variant_token
    lang = str(st.session_state.get("ui_lang") or cfg.ui.default_lang or "en")

    top = st.columns([2.2, 1, 1, 1], gap="small")
    with top[0]:
        st.text_input(Tget(T, "query", "Query"), key="search_query_box", placeholder=Tget(T, "search_placeholder", "Find a person, vehicle, crowd, object..."))
    with top[1]:
        st.selectbox(Tget(T, "video_filter", "Video"), options=video_ids, key="search_video_id", format_func=lambda x: x or Tget(T, "all_videos", "All videos"))
    with top[2]:
        st.selectbox(Tget(T, "variant", "Variant"), options=variant_tokens, key="search_variant", format_func=_variant_label)
    with top[3]:
        st.number_input(Tget(T, "topk", "Top-K"), min_value=1, max_value=20, value=int(st.session_state.get("search_topk") or 8), key="search_topk")

    filters = st.columns(3, gap="small")
    with filters[0]:
        event_type = st.text_input(Tget(T, "event_filter", "Event type"), key="search_event_type")
    with filters[1]:
        risk_level = st.selectbox(
            Tget(T, "risk_filter", "Risk"),
            options=["", "normal", "attention", "warning", "critical"],
            key="search_risk_level",
            format_func=lambda x: x or Tget(T, "any_value", "Any"),
        )
    with filters[2]:
        anomaly_only = st.checkbox(Tget(T, "anomaly_only", "Anomalies only"), key="search_anomaly_only")

    if st.button(Tget(T, "search_action", "Run search"), key="search_run_btn", use_container_width=True):
        try:
            hits = client.search(
                query=str(st.session_state.get("search_query_box") or "").strip(),
                top_k=int(st.session_state.get("search_topk") or 8),
                video_id=str(st.session_state.get("search_video_id") or "") or None,
                language=lang,
                variant=_variant_from_token(st.session_state.get("search_variant")),
                event_type=str(event_type or "").strip() or None,
                risk_level=str(risk_level or "").strip() or None,
                anomaly_only=bool(anomaly_only),
            )
            st.session_state["search_hits"] = hits
        except Exception as exc:
            st.session_state["search_hits"] = []
            soft_note(f"{Tget(T, 'error_prefix', 'Error')}: {exc}", kind="warn")

    hits = st.session_state.get("search_hits") or []
    left, right = st.columns([1.2, 1.8], gap="large")
    with left:
        st.markdown(f"<div class='section-title'>{E(Tget(T, 'results_title', 'Results'))}</div>", unsafe_allow_html=True)
        if not hits:
            st.caption(Tget(T, "no_results", "No results available."))
        for idx, hit in enumerate(hits[:20]):
            if not isinstance(hit, dict):
                continue
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-head">
                        <span>{E(str(hit.get('video_id') or '-'))}</span>
                        <span>{E(mmss(hit.get('start_sec', 0.0)))} - {E(mmss(hit.get('end_sec', 0.0)))}</span>
                    </div>
                    <div class="result-text">{E(_clip_text(str(hit.get('description') or ''), 180))}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            _render_capsules(_hit_capsules(hit))
            if st.button(Tget(T, "open_clip", "Open"), key=f"search_open_{idx}", use_container_width=True):
                st.session_state["selected_video_id"] = str(hit.get("video_id") or "")
                st.session_state["video_seek_sec"] = int(float(hit.get("start_sec", 0.0) or 0.0))
                st.query_params["tab"] = "storage"
                st.rerun()

    with right:
        st.markdown(f"<div class='section-title'>{E(Tget(T, 'inspector_title', 'Inspector'))}</div>", unsafe_allow_html=True)
        if hits:
            first = hits[0] if isinstance(hits[0], dict) else {}
            st.markdown(f"<div class='scene-headline'>{E(_clip_text(str(first.get('description') or ''), 220))}</div>", unsafe_allow_html=True)
            chips: List[str] = []
            for key in ("event_type", "risk_level", "people_count_bucket", "motion_type"):
                value = str(first.get(key) or "").strip()
                if value:
                    chips.append(value)
            if bool(first.get("anomaly_flag", False)):
                chips.append("anomaly")
            _render_capsules(chips)
            st.caption(f"{Tget(T, 'score', 'Score')}: {float(first.get('score', 0.0) or 0.0):.3f}")
            st.caption(
                f"{Tget(T, 'time_filter', 'Time')}: {mmss(first.get('start_sec', 0.0))} - {mmss(first.get('end_sec', 0.0))}"
            )
        else:
            st.caption(Tget(T, "pick_result", "Run a search to inspect the top result."))


def assistant_tab(client: BackendClient, cfg: Any, ui_text: Dict[str, Dict[str, Any]]) -> None:
    """Render the grounded report / QA / RAG page."""

    T = get_T(ui_text, str(st.session_state.get("ui_lang") or cfg.ui.default_lang or "en"))
    _render_language_switcher(T, cfg)

    videos = client.list_videos()
    if not videos:
        soft_note(Tget(T, "empty_library", "The library is empty. Upload a video to begin."), kind="info")
        return

    video_ids = [str(item.get("video_id") or "") for item in videos if str(item.get("video_id") or "")]
    selected = str(st.session_state.get("assistant_video_id") or (video_ids[0] if video_ids else ""))
    if selected not in video_ids and video_ids:
        selected = video_ids[0]
    st.session_state["assistant_video_id"] = selected

    selected_item = next((item for item in videos if str(item.get("video_id") or "") == selected), None)
    variant_tokens = _video_variant_tokens(selected_item)
    current_variant_token = str(st.session_state.get("assistant_variant") or variant_tokens[0])
    if current_variant_token not in variant_tokens:
        current_variant_token = variant_tokens[0]
    st.session_state["assistant_variant"] = current_variant_token

    top = st.columns([2, 1, 1], gap="small")
    with top[0]:
        st.selectbox(Tget(T, "selected_video", "Selected video"), options=video_ids, key="assistant_video_id")
    with top[1]:
        st.selectbox(Tget(T, "variant", "Variant"), options=variant_tokens, key="assistant_variant", format_func=_variant_label)
    with top[2]:
        st.number_input(Tget(T, "topk", "Top-K"), min_value=1, max_value=20, value=int(st.session_state.get("assistant_top_k") or 6), key="assistant_top_k")

    left, right = st.columns([1.1, 1.4], gap="large")
    with left:
        st.markdown(f"<div class='section-title'>{E(Tget(T, 'reports_title', 'Grounded report'))}</div>", unsafe_allow_html=True)
        st.text_input(Tget(T, "report_prompt", "Report focus"), key="assistant_report_query", placeholder=Tget(T, "report_placeholder", "Summarize the visible activity near the entrance"))
        if st.button(Tget(T, "build_report", "Build report"), key="assistant_build_report", use_container_width=True):
            try:
                st.session_state["assistant_report_payload"] = client.build_report(
                    video_id=str(st.session_state.get("assistant_video_id") or "") or None,
                    language=str(st.session_state.get("ui_lang") or cfg.ui.default_lang or "en"),
                    variant=_variant_from_token(st.session_state.get("assistant_variant")),
                    query=str(st.session_state.get("assistant_report_query") or "").strip() or None,
                    top_k=int(st.session_state.get("assistant_top_k") or 6),
                )
            except Exception as exc:
                soft_note(f"{Tget(T, 'error_prefix', 'Error')}: {exc}", kind="warn")

        report_payload = st.session_state.get("assistant_report_payload") or {}
        report_text = str(report_payload.get("report") or "").strip()
        if report_text:
            st.markdown(f"<div class='scene-headline'>{E(_clip_text(report_text, 240))}</div>", unsafe_allow_html=True)
            st.caption(
                f"mode={report_payload.get('mode', 'deterministic')} · "
                f"latency={float(report_payload.get('latency_sec', 0.0) or 0.0):.2f}s · "
                f"hits={int(report_payload.get('hit_count', 0) or 0)}"
            )
            with st.expander(Tget(T, "full_report", "Full report"), expanded=False):
                st.write(report_text)

        st.markdown(f"<div class='section-title'>{E(Tget(T, 'metrics', 'Metrics'))}</div>", unsafe_allow_html=True)
        try:
            metrics = client.get_metrics_summary(
                str(st.session_state.get("assistant_video_id") or ""),
                str(st.session_state.get("ui_lang") or cfg.ui.default_lang or "en"),
                variant=_variant_from_token(st.session_state.get("assistant_variant")),
            )
        except Exception:
            metrics = {}
        if metrics:
            timings = metrics.get("timings_sec") or {}
            rows = [
                ("PREPROCESS", timings.get("preprocess_time_sec")),
                ("VLM", timings.get("model_time_sec")),
                ("POSTPROCESS", timings.get("postprocess_time_sec")),
                ("TOTAL", timings.get("total_time_sec")),
            ]
            stat_html = "".join(
                [f"<span class='stat-chip'>{E(name)}: {float(value or 0.0):.2f}s</span>" for name, value in rows if value is not None]
            )
            if stat_html:
                st.markdown(f"<div class='stat-row'>{stat_html}</div>", unsafe_allow_html=True)
        else:
            st.caption(Tget(T, "metrics_empty", "Metrics appear after the first completed run."))

    with right:
        st.markdown(f"<div class='section-title'>{E(Tget(T, 'qa_title', 'Grounded QA and RAG'))}</div>", unsafe_allow_html=True)
        st.text_area(
            Tget(T, "assistant_prompt", "Ask about the video"),
            key="assistant_question",
            height=120,
            placeholder=Tget(T, "assistant_placeholder", "What happens near the entrance?"),
        )
        qa_col, rag_col = st.columns(2, gap="small")
        with qa_col:
            if st.button(Tget(T, "ask", "Ask"), key="assistant_ask_btn", use_container_width=True):
                try:
                    st.session_state["assistant_qa_payload"] = client.ask_qa(
                        question=str(st.session_state.get("assistant_question") or ""),
                        language=str(st.session_state.get("ui_lang") or cfg.ui.default_lang or "en"),
                        variant=_variant_from_token(st.session_state.get("assistant_variant")),
                        video_id=str(st.session_state.get("assistant_video_id") or "") or None,
                        top_k=int(st.session_state.get("assistant_top_k") or 6),
                    )
                except Exception as exc:
                    soft_note(f"{Tget(T, 'error_prefix', 'Error')}: {exc}", kind="warn")
        with rag_col:
            if st.button(Tget(T, "rag", "RAG"), key="assistant_rag_btn", use_container_width=True):
                try:
                    st.session_state["assistant_rag_payload"] = client.ask_rag(
                        query=str(st.session_state.get("assistant_question") or ""),
                        language=str(st.session_state.get("ui_lang") or cfg.ui.default_lang or "en"),
                        variant=_variant_from_token(st.session_state.get("assistant_variant")),
                        video_id=str(st.session_state.get("assistant_video_id") or "") or None,
                        top_k=int(st.session_state.get("assistant_top_k") or 6),
                    )
                except Exception as exc:
                    soft_note(f"{Tget(T, 'error_prefix', 'Error')}: {exc}", kind="warn")

        qa_payload = st.session_state.get("assistant_qa_payload") or {}
        qa_text = str(qa_payload.get("answer") or "").strip()
        if qa_text:
            st.markdown(f"<div class='answer-block'>{E(_clip_text(qa_text, 260))}</div>", unsafe_allow_html=True)
            st.caption(
                f"QA · mode={qa_payload.get('mode', 'deterministic')} · "
                f"latency={float(qa_payload.get('latency_sec', 0.0) or 0.0):.2f}s"
            )
            context = str(qa_payload.get("context") or "").strip()
            if context:
                with st.expander(Tget(T, "qa_context", "QA context"), expanded=False):
                    st.code(context)

        rag_payload = st.session_state.get("assistant_rag_payload") or {}
        rag_text = str(rag_payload.get("answer") or "").strip()
        if rag_text:
            st.markdown(f"<div class='answer-block alt'>{E(_clip_text(rag_text, 260))}</div>", unsafe_allow_html=True)
            st.caption(
                f"RAG · mode={rag_payload.get('mode', 'deterministic')} · "
                f"latency={float(rag_payload.get('latency_sec', 0.0) or 0.0):.2f}s"
            )
            context = str(rag_payload.get("context") or "").strip()
            if context:
                with st.expander(Tget(T, "rag_context", "RAG context"), expanded=False):
                    st.code(context)
