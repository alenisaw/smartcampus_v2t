"""Shared mid-level UI components used across Streamlit pages."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import streamlit as st

from app.lib.formatters import E, clip_text, collect_scene_capsules, first_sentence, hms, hit_capsules, mmss
from app.lib.i18n import Tget


def render_capsules(items: Iterable[str], *, accent: bool = False) -> None:
    """Render capsule tags as a single responsive row."""

    filtered = [str(item).strip().upper() for item in items if str(item).strip()]
    if not filtered:
        return
    css = "capsule accent" if accent else "capsule"
    html_items = "".join([f"<span class='{css}'>{E(item)}</span>" for item in filtered])
    st.markdown(f"<div class='capsule-row'>{html_items}</div>", unsafe_allow_html=True)


def render_stat_row(items: Iterable[str]) -> None:
    """Render compact stat chips."""

    clean = [str(item).strip() for item in items if str(item).strip()]
    if not clean:
        return
    st.markdown(
        "<div class='stat-row'>" + "".join([f"<span class='stat-chip'>{E(item)}</span>" for item in clean]) + "</div>",
        unsafe_allow_html=True,
    )


def render_status_overview(T: Dict[str, Any], queue: Dict[str, Any], index_status: Dict[str, Any]) -> None:
    """Render queue and index state as compact operator cards."""

    running = queue.get("running") if isinstance(queue, dict) else None
    queued = queue.get("queued") if isinstance(queue, dict) else []
    status = queue.get("status") if isinstance(queue, dict) else {}
    index_langs = index_status.get("languages") if isinstance(index_status, dict) else {}

    cards = [
        (
            "Queue",
            "Paused" if bool((status or {}).get("paused")) else "Active",
            f"{len(queued) if isinstance(queued, list) else 0} queued",
        ),
        (
            "Running",
            str((running or {}).get("video_id") or "-"),
            str((running or {}).get("stage") or "idle"),
        ),
        (
            "Index",
            f"{len(index_langs) if isinstance(index_langs, dict) else 0} language views",
            "ready" if not index_status.get("last_error") else "error",
        ),
    ]

    html_cards = []
    for title, value, meta in cards:
        html_cards.append(
            f"""
            <div class="overview-card">
                <div class="overview-label">{E(title)}</div>
                <div class="overview-value">{E(value)}</div>
                <div class="overview-meta">{E(meta)}</div>
            </div>
            """
        )
    st.markdown(f"<div class='overview-grid'>{''.join(html_cards)}</div>", unsafe_allow_html=True)

    if isinstance(running, dict) and running.get("job_id"):
        st.markdown(
            f"<div class='queue-inline'>{E(Tget(T, 'running_now', 'Running now'))}: {E(str(running.get('video_id') or ''))} · {E(str(running.get('stage') or ''))}</div>",
            unsafe_allow_html=True,
        )


def render_outputs_overview(T: Dict[str, Any], outputs: Dict[str, Any], selected_lang: str) -> None:
    """Render manifest and runtime details for the selected output set."""

    manifest = outputs.get("manifest") if isinstance(outputs.get("manifest"), dict) else {}
    run_manifest = outputs.get("run_manifest") if isinstance(outputs.get("run_manifest"), dict) else {}
    metrics = outputs.get("metrics") if isinstance(outputs.get("metrics"), dict) else {}
    languages = manifest.get("languages") if isinstance(manifest.get("languages"), dict) else {}

    cards = [
        ("Profile", str(run_manifest.get("profile") or "main")),
        ("Variant", str(outputs.get("variant") or "base")),
        ("Language", str(selected_lang or "en")),
        ("Config", str(run_manifest.get("config_fingerprint") or "-")[:12] or "-"),
    ]
    html_cards = []
    for title, value in cards:
        html_cards.append(
            f"""
            <div class="surface-card compact">
                <div class="surface-label">{E(title)}</div>
                <div class="surface-value">{E(value)}</div>
            </div>
            """
        )
    st.markdown(f"<div class='surface-grid'>{''.join(html_cards)}</div>", unsafe_allow_html=True)

    if languages:
        status_items = []
        for lang, payload in sorted(languages.items()):
            if not isinstance(payload, dict):
                continue
            status_items.append(f"{lang}: {payload.get('status', 'unknown')}")
        render_capsules(status_items)

    stage_stats = metrics.get("stage_stats_sec") if isinstance(metrics.get("stage_stats_sec"), dict) else {}
    if stage_stats:
        rows: List[str] = []
        for stage_name, payload in stage_stats.items():
            if not isinstance(payload, dict):
                continue
            mean_sec = payload.get("mean_sec")
            if mean_sec is None:
                continue
            rows.append(f"{stage_name}: {float(mean_sec):.2f}s")
        render_stat_row(rows)


def render_metrics_summary(T: Dict[str, Any], outputs: Dict[str, Any]) -> None:
    """Render runtime metrics from stored process outputs."""

    metrics = outputs.get("metrics") if isinstance(outputs.get("metrics"), dict) else {}
    if not metrics:
        st.caption(Tget(T, "metrics_empty", "Metrics appear after the first completed run."))
        return

    translations = metrics.get("translations") if isinstance(metrics.get("translations"), dict) else {}
    indexing = metrics.get("indexing") if isinstance(metrics.get("indexing"), dict) else {}

    cards = [
        ("Frames", metrics.get("num_frames")),
        ("Clips", metrics.get("num_clips")),
        ("Process", f"{float(metrics.get('total_time_sec') or 0.0):.2f}s" if metrics.get("total_time_sec") is not None else "-"),
        ("Translations", len(translations)),
        ("Index views", len(indexing)),
    ]
    html_cards = []
    for title, value in cards:
        html_cards.append(
            f"""
            <div class="metric-card">
                <div class="metric-label">{E(title)}</div>
                <div class="metric-value">{E(value)}</div>
            </div>
            """
        )
    st.markdown(f"<div class='metric-grid'>{''.join(html_cards)}</div>", unsafe_allow_html=True)

    extra = metrics.get("extra") if isinstance(metrics.get("extra"), dict) else {}
    detail_items = []
    for label, value in (
        ("decode", extra.get("decode_backend")),
        ("source fps", extra.get("source_fps")),
        ("processed fps", extra.get("processed_fps")),
        ("drop dark", extra.get("dark_drop_ratio")),
        ("drop lazy", extra.get("lazy_drop_ratio")),
    ):
        if value not in (None, ""):
            detail_items.append(f"{label}: {value}")
    render_stat_row(detail_items)


def render_scene_panel(T: Dict[str, Any], outputs: Dict[str, Any], video_meta: Dict[str, Any]) -> None:
    """Render a scene overview with headline, summary, and top-level capsules."""

    summary_text = str(outputs.get("global_summary") or "").strip()
    headline = first_sentence(summary_text) or Tget(T, "empty_summary", "No summary available yet.")

    st.markdown(f"<div class='section-title'>{E(Tget(T, 'scene_overview', 'Scene overview'))}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='scene-headline'>{E(headline)}</div>", unsafe_allow_html=True)
    render_capsules(collect_scene_capsules(outputs), accent=True)

    metrics = outputs.get("metrics") if isinstance(outputs.get("metrics"), dict) else {}
    info: List[str] = []
    if video_meta.get("duration_sec"):
        info.append(f"{Tget(T, 'meta_duration', 'Duration')}: {hms(video_meta.get('duration_sec', 0.0))}")
    if video_meta.get("width") and video_meta.get("height"):
        info.append(f"{Tget(T, 'meta_resolution', 'Resolution')}: {video_meta['width']}x{video_meta['height']}")
    if metrics.get("num_clips") is not None:
        info.append(f"{Tget(T, 'metric_clips', 'Clips')}: {metrics.get('num_clips')}")
    if metrics.get("total_time_sec") is not None:
        info.append(f"{Tget(T, 'metrics_total', 'Total time')}: {float(metrics.get('total_time_sec') or 0.0):.2f}s")
    render_stat_row(info)

    with st.expander(Tget(T, "detailed_summary", "Detailed narrative"), expanded=False):
        if summary_text:
            st.write(summary_text)
        else:
            st.caption(Tget(T, "empty_summary", "No summary available yet."))


def render_segment_cards(T: Dict[str, Any], annotations: List[Dict[str, Any]], session_key: str) -> None:
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
                    <div class="segment-text">{E(clip_text(caption, 220))}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with cols[1]:
            if st.button(Tget(T, "open_clip", "Open"), key=f"{session_key}_{idx}", use_container_width=True):
                st.session_state["video_seek_sec"] = int(start_sec)


def render_search_results(T: Dict[str, Any], hits: List[Dict[str, Any]], *, open_prefix: str) -> Optional[Dict[str, Any]]:
    """Render retrieval results and return the currently inspected hit."""

    st.markdown(f"<div class='section-title'>{E(Tget(T, 'results_title', 'Results'))}</div>", unsafe_allow_html=True)
    if not hits:
        st.caption(Tget(T, "no_results", "No results available."))
        return None

    selected_hit = hits[0] if isinstance(hits[0], dict) else None
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
                <div class="result-text">{E(clip_text(str(hit.get('description') or ''), 180))}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        render_capsules(hit_capsules(hit))
        st.caption(
            f"hybrid={float(hit.get('score', 0.0) or 0.0):.3f} · "
            f"sparse={float(hit.get('sparse_score', 0.0) or 0.0):.3f} · "
            f"dense={float(hit.get('dense_score', 0.0) or 0.0):.3f}"
        )
        if st.button(Tget(T, "open_clip", "Open"), key=f"{open_prefix}_{idx}", use_container_width=True):
            st.session_state["selected_video_id"] = str(hit.get("video_id") or "")
            st.session_state["video_seek_sec"] = int(float(hit.get("start_sec", 0.0) or 0.0))
            st.query_params["tab"] = "storage"
            st.rerun()
    return selected_hit


def render_hit_inspector(T: Dict[str, Any], hit: Optional[Dict[str, Any]]) -> None:
    """Render detailed metadata for one retrieved hit."""

    st.markdown(f"<div class='section-title'>{E(Tget(T, 'inspector_title', 'Inspector'))}</div>", unsafe_allow_html=True)
    if not hit:
        st.caption(Tget(T, "pick_result", "Run a search to inspect the top result."))
        return

    st.markdown(f"<div class='scene-headline'>{E(clip_text(str(hit.get('description') or ''), 220))}</div>", unsafe_allow_html=True)
    chips: List[str] = []
    for key in ("event_type", "risk_level", "people_count_bucket", "motion_type"):
        value = str(hit.get(key) or "").strip()
        if value:
            chips.append(value)
    if bool(hit.get("anomaly_flag", False)):
        chips.append("anomaly")
    render_capsules(chips)
    render_stat_row(
        [
            f"{Tget(T, 'score', 'Score')}: {float(hit.get('score', 0.0) or 0.0):.3f}",
            f"sparse: {float(hit.get('sparse_score', 0.0) or 0.0):.3f}",
            f"dense: {float(hit.get('dense_score', 0.0) or 0.0):.3f}",
            f"{Tget(T, 'time_filter', 'Time')}: {mmss(hit.get('start_sec', 0.0))} - {mmss(hit.get('end_sec', 0.0))}",
        ]
    )


def render_supporting_hits(title: str, hits: List[Dict[str, Any]]) -> None:
    """Render compact evidence cards for grounded outputs."""

    if not hits:
        return
    st.markdown(f"<div class='section-title'>{E(title)}</div>", unsafe_allow_html=True)
    for hit in hits[:6]:
        if not isinstance(hit, dict):
            continue
        st.markdown(
            f"""
            <div class="evidence-card">
                <div class="result-head">
                    <span>{E(str(hit.get('video_id') or '-'))}</span>
                    <span>{E(mmss(hit.get('start_sec', 0.0)))} - {E(mmss(hit.get('end_sec', 0.0)))}</span>
                </div>
                <div class="result-text">{E(clip_text(str(hit.get('description') or ''), 180))}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        render_capsules(hit_capsules(hit))
