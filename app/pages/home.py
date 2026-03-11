# app/pages/home.py
"""
Home page for SmartCampus V2T Streamlit UI.

Purpose:
- Introduce the project, pipeline, and operating model before users enter task-specific views.
- Surface the main system contours in one dashboard-style overview page.
"""

from __future__ import annotations

from typing import Any, Dict

import streamlit as st

from app.api_client import BackendClient
from app.components.chrome import soft_note
from app.lib.formatters import E
from app.lib.i18n import Tget, get_T


def _metric_card(label: str, value: str, meta: str) -> str:
    """Render one overview metric card."""

    return (
        "<div class='home-metric-card'>"
        f"<div class='home-metric-label'>{E(label)}</div>"
        f"<div class='home-metric-value'>{E(value)}</div>"
        f"<div class='home-metric-meta'>{E(meta)}</div>"
        "</div>"
    )


def render_page(client: BackendClient, cfg: Any, ui_text: Dict[str, Dict[str, Any]]) -> None:
    """Render the overview page with project and pipeline context."""

    T = get_T(ui_text, str(st.session_state.get("ui_lang") or cfg.ui.default_lang or "en"))
    try:
        videos = client.list_videos()
    except Exception as exc:
        soft_note(f"{Tget(T, 'error_prefix', 'Error')}: {exc}", kind="warn")
        videos = []
    try:
        index_status = client.index_status()
    except Exception:
        index_status = {}

    index_langs = index_status.get("languages") if isinstance(index_status, dict) else {}

    st.markdown(
        f"""
        <div class="project-story">
            <div class="project-story-title">{E(Tget(T, 'home_title', 'End-to-end local video understanding workspace'))}</div>
            <div class="project-story-copy">{E(Tget(T, 'home_copy', 'SmartCampus V2T processes video into structured events, multilingual outputs, hybrid search surfaces, grounded reports, QA, and RAG workflows through one local operator console.'))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div class='home-metric-grid'>"
        + _metric_card(Tget(T, "home_videos", "Video library"), str(len(videos)), Tget(T, "home_videos_meta", "registered source videos"))
        + _metric_card(Tget(T, "home_index", "Index views"), str(len(index_langs) if isinstance(index_langs, dict) else 0), Tget(T, "home_index_meta", "searchable language surfaces"))
        + _metric_card(Tget(T, "home_surface_video", "Video Analytics"), Tget(T, "home_video_metric", "Inspection"), Tget(T, "home_video_metric_meta", "player, outputs, metrics, timeline"))
        + _metric_card(Tget(T, "home_surface_search", "Search"), Tget(T, "home_search_metric", "Retrieval"), Tget(T, "home_search_metric_meta", "search, report, QA, RAG"))
        + "</div>",
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.15, 1.0], gap="large")
    with left:
        st.markdown(f"<div class='section-title'>{E(Tget(T, 'home_pipeline_title', 'Pipeline'))}</div>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="surface-list">
                <div class="surface-list-item"><strong>{E(Tget(T, 'stage_decode', 'Decode'))}</strong><span>{E(Tget(T, 'stage_decode_copy', 'FFmpeg normalization, frame filtering, and browser-safe video preparation.'))}</span></div>
                <div class="surface-list-item"><strong>{E(Tget(T, 'stage_caption', 'Caption'))}</strong><span>{E(Tget(T, 'stage_caption_copy', 'Clip-level VLM descriptions, structuring, anomaly hints, and summaries.'))}</span></div>
                <div class="surface-list-item"><strong>{E(Tget(T, 'stage_translate', 'Translate'))}</strong><span>{E(Tget(T, 'stage_translate_copy', 'Additional language views through MT and selective post-edit.'))}</span></div>
                <div class="surface-list-item"><strong>{E(Tget(T, 'stage_search', 'Search and ground'))}</strong><span>{E(Tget(T, 'stage_search_copy', 'Hybrid index build, retrieval, grounded report, QA, and RAG.'))}</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(f"<div class='section-title'>{E(Tget(T, 'home_surfaces_title', 'Workspace surfaces'))}</div>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="surface-list compact">
                <div class="surface-list-item"><strong>{E(Tget(T, 'home_surface_overview', 'Overview'))}</strong><span>{E(Tget(T, 'home_surface_overview_copy', 'Project context, live queue state, and searchable index readiness.'))}</span></div>
                <div class="surface-list-item"><strong>{E(Tget(T, 'processing_title', 'Processing'))}</strong><span>{E(Tget(T, 'home_surface_processing_copy', 'Upload videos, manage queue state, launch inference, and control the library.'))}</span></div>
                <div class="surface-list-item"><strong>{E(Tget(T, 'home_surface_video', 'Video Analytics'))}</strong><span>{E(Tget(T, 'home_surface_video_copy', 'Primary video player, outputs, run controls, queue rail, and artifact inspection.'))}</span></div>
                <div class="surface-list-item"><strong>{E(Tget(T, 'home_surface_search', 'Search'))}</strong><span>{E(Tget(T, 'home_surface_search_copy', 'Retrieval inspection plus grounded report, QA, RAG, and pipeline metrics.'))}</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
