# app/pages/processing.py
"""
Processing page for SmartCampus V2T Streamlit UI.

Purpose:
- Render upload, queue, library, and run-control workflows in one operational surface.
- Keep processing and asset-management tasks separate from analytics inspection.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import streamlit as st

from app.api_client import BackendClient
from app.components.chrome import soft_note
from app.components.storage import render_queue_panel, render_run_panel, render_upload_panel, render_video_library
from app.lib.i18n import Tget, get_T


def _filter_videos(videos: List[Dict[str, Any]], query: str, readiness: str) -> List[Dict[str, Any]]:
    """Filter library items by search text and output readiness."""

    text = str(query or "").strip().lower()
    filtered: List[Dict[str, Any]] = []
    for item in videos:
        if not isinstance(item, dict):
            continue
        video_id = str(item.get("video_id") or "")
        langs = item.get("languages") if isinstance(item.get("languages"), list) else []
        has_outputs = bool(langs)
        if readiness == "ready" and not has_outputs:
            continue
        if readiness == "raw" and has_outputs:
            continue
        if text:
            hay = " ".join(
                [
                    video_id.lower(),
                    " ".join(str(lang).lower() for lang in langs),
                    str(item.get("path") or "").lower(),
                ]
            )
            if text not in hay:
                continue
        filtered.append(item)
    return filtered


def render_page(client: BackendClient, cfg: Any, ui_text: Dict[str, Dict[str, Any]]) -> None:
    """Render the processing and queue-management page."""

    T = get_T(ui_text, str(st.session_state.get("ui_lang") or cfg.ui.default_lang or "en"))
    videos = client.list_videos()
    try:
        queue = client.queue_list()
    except Exception:
        queue = {}

    video_ids = [str(item.get("video_id") or "") for item in videos if str(item.get("video_id") or "")]
    selected = str(st.session_state.get("selected_video_id") or (video_ids[0] if video_ids else ""))
    if selected not in video_ids and video_ids:
        selected = video_ids[0]
    st.session_state["selected_video_id"] = selected

    st.markdown(f"<div class='section-title'>{Tget(T, 'processing_title', 'Processing')}</div>", unsafe_allow_html=True)

    control_cols = st.columns([1.1, 1.2], gap="large")
    with control_cols[0]:
        render_upload_panel(T, client)
    with control_cols[1]:
        if video_ids:
            render_run_panel(T, client, cfg, str(st.session_state.get("selected_video_id") or selected))
        else:
            soft_note(Tget(T, "empty_library", "The library is empty. Upload a video to begin."), kind="info")

    filter_cols = st.columns([2.4, 1.0], gap="small")
    with filter_cols[0]:
        st.text_input(Tget(T, "library_search", "Search videos"), key="processing_search_query", placeholder=Tget(T, "library_search_placeholder", "Search by video id, path, or language"))
    with filter_cols[1]:
        readiness_options = ["all", "ready", "raw"]
        readiness_labels = {
            "all": Tget(T, "filter_all", "All"),
            "ready": Tget(T, "filter_ready", "Ready"),
            "raw": Tget(T, "filter_raw", "Raw only"),
        }
        st.selectbox(
            Tget(T, "library_filter", "Filter"),
            options=readiness_options,
            key="processing_readiness_filter",
            format_func=lambda key: readiness_labels.get(key, key),
        )

    filtered_videos = _filter_videos(
        videos,
        str(st.session_state.get("processing_search_query") or ""),
        str(st.session_state.get("processing_readiness_filter") or "all"),
    )

    left, right = st.columns([1.9, 0.95], gap="large")
    with left:
        if filtered_videos:
            render_video_library(
                T,
                client,
                filtered_videos,
                str(st.session_state.get("selected_video_id") or selected),
                Path(cfg.paths.thumbs_dir),
            )
        else:
            soft_note(Tget(T, "library_no_match", "No videos matched the current search/filter."), kind="info")

    with right:
        render_queue_panel(T, client, queue)
