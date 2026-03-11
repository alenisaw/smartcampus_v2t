# app/pages/storage.py
"""
Storage page for SmartCampus V2T Streamlit UI.

Purpose:
- Render video browsing, processing controls, and artifact inspection flows.
- Keep storage-oriented operator tasks separate from analytics and assistant pages.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import streamlit as st

from app.api_client import BackendClient
from app.components.chrome import soft_note
from app.components.common import render_metrics_summary, render_outputs_overview, render_scene_panel, render_segment_cards
from app.components.storage import render_player
from app.lib.formatters import E, collect_available_languages, variant_from_token, variant_label, video_variant_tokens
from app.lib.i18n import get_T, Tget
from app.lib.media import ensure_browser_video, get_video_meta, mtime


def render_page(client: BackendClient, cfg: Any, ui_text: Dict[str, Dict[str, Any]]) -> None:
    """Render the main video library and playback page."""

    T = get_T(ui_text, str(st.session_state.get("ui_lang") or cfg.ui.default_lang or "en"))
    videos = client.list_videos()
    if not videos:
        render_upload_panel(T, client)
        soft_note(Tget(T, "empty_library", "The library is empty. Upload a video to begin."), kind="info")
        return

    video_ids = [str(item.get("video_id") or "") for item in videos if str(item.get("video_id") or "")]
    selected = str(st.session_state.get("selected_video_id") or (video_ids[0] if video_ids else ""))
    if selected not in video_ids and video_ids:
        selected = video_ids[0]
    st.session_state["selected_video_id"] = selected

    selected_video = next((item for item in videos if str(item.get("video_id") or "") == selected), None) or {}
    variant_tokens = video_variant_tokens(selected_video)
    current_variant_token = str(st.session_state.get("library_variant") or variant_tokens[0])
    if current_variant_token not in variant_tokens:
        current_variant_token = variant_tokens[0]
    st.session_state["library_variant"] = current_variant_token
    current_variant = variant_from_token(current_variant_token)

    available_languages = collect_available_languages(selected_video, current_variant)
    current_lang = str(st.session_state.get("library_lang") or available_languages[0])
    if current_lang not in available_languages:
        current_lang = available_languages[0]
    st.session_state["library_lang"] = current_lang

    st.markdown(f"<div class='section-title'>{E(Tget(T, 'video_analytics_title', 'Video Analytics'))}</div>", unsafe_allow_html=True)
    toolbar = st.columns([2.2, 1.0, 1.0], gap="small")
    with toolbar[0]:
        st.selectbox(
            Tget(T, "selected_video", "Selected video"),
            options=video_ids,
            key="selected_video_id",
            index=video_ids.index(selected),
        )
    with toolbar[1]:
        st.selectbox(
            Tget(T, "variant", "Variant"),
            options=variant_tokens,
            format_func=variant_label,
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
    play_path = ensure_browser_video(raw_path) if raw_path.exists() else raw_path
    meta = get_video_meta(str(play_path), mtime(play_path)) if play_path.exists() else {}

    outputs: Dict[str, Any] = {}
    try:
        outputs = client.get_video_outputs(
            str(st.session_state.get("selected_video_id") or ""),
            str(st.session_state.get("library_lang") or "en"),
            variant=variant_from_token(st.session_state.get("library_variant")),
        )
    except Exception as exc:
        soft_note(f"{Tget(T, 'error_prefix', 'Error')}: {exc}", kind="warn")

    if play_path.exists():
        render_player(play_path, int(st.session_state.get("video_seek_sec") or 0))
    else:
        soft_note(Tget(T, "raw_missing", "Source video not found."), kind="warn")

    render_outputs_overview(T, outputs, str(st.session_state.get("library_lang") or "en"))
    render_scene_panel(T, outputs, meta)
    render_metrics_summary(T, outputs)

    batch_manifest = outputs.get("batch_manifest") if isinstance(outputs.get("batch_manifest"), dict) else {}
    if batch_manifest:
        with st.expander(Tget(T, "batch_manifest", "Batch manifest"), expanded=False):
            st.json(batch_manifest)

    annotations = outputs.get("annotations") if isinstance(outputs.get("annotations"), list) else []
    render_segment_cards(T, annotations, "library_segment")
