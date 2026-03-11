# app/components/common.py
"""
Shared UI components for SmartCampus V2T.

Purpose:
- Render reusable mid-level Streamlit blocks across multiple pages.
- Keep repeated UI patterns out of page modules.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import streamlit as st

from app.lib.formatters import E, clip_text, collect_scene_capsules, first_sentence, hms, hit_capsules, humanize_token, mmss
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
    index_error = str(index_status.get("last_error") or "").strip() if isinstance(index_status, dict) else ""

    cards = [
        (
            Tget(T, "queue_title", "Queue"),
            Tget(T, "paused", "Paused") if bool((status or {}).get("paused")) else Tget(T, "active", "Active"),
            f"{len(queued) if isinstance(queued, list) else 0} {Tget(T, 'queued_short', 'queued')}",
        ),
        (Tget(T, "home_running", "Running"), str((running or {}).get("job_type") or "-"), str((running or {}).get("stage") or Tget(T, "idle", "idle"))),
        (
            Tget(T, "home_index", "Index views"),
            str(len(index_langs) if isinstance(index_langs, dict) else 0),
            "ready" if not index_error else "error",
        ),
    ]

    cols = st.columns(len(cards), gap="small")
    for col, (title, value, meta) in zip(cols, cards):
        with col:
            st.metric(title, value, meta, delta_color="off")

    if isinstance(running, dict) and running.get("job_id"):
        st.markdown(
            f"<div class='queue-inline'>{E(Tget(T, 'running_now', 'Running now'))}: {E(str(running.get('video_id') or ''))} | {E(str(running.get('stage') or ''))}</div>",
            unsafe_allow_html=True,
        )


def render_outputs_overview(T: Dict[str, Any], outputs: Dict[str, Any], selected_lang: str) -> None:
    """Render manifest and runtime details for the selected output set."""

    manifest = outputs.get("manifest") if isinstance(outputs.get("manifest"), dict) else {}
    run_manifest = outputs.get("run_manifest") if isinstance(outputs.get("run_manifest"), dict) else {}
    metrics = outputs.get("metrics") if isinstance(outputs.get("metrics"), dict) else {}
    languages = manifest.get("languages") if isinstance(manifest.get("languages"), dict) else {}

    cards = [
        (Tget(T, "profile", "Profile"), str(run_manifest.get("profile") or "main")),
        (Tget(T, "variant", "Variant"), str(outputs.get("variant") or "base")),
        (Tget(T, "output_language", "Output language"), str(selected_lang or "en")),
        (Tget(T, "config_label", "Config"), str(run_manifest.get("config_fingerprint") or "-")[:12] or "-"),
    ]
    cols = st.columns(len(cards), gap="small")
    for col, (title, value) in zip(cols, cards):
        with col:
            st.caption(title)
            st.markdown(f"**{value}**")

    if languages:
        status_items = []
        for lang, payload in sorted(languages.items()):
            if not isinstance(payload, dict):
                continue
            status_items.append(f"{lang}: {payload.get('status', 'unknown')}")
        render_capsules(status_items)


def render_metrics_summary(T: Dict[str, Any], outputs: Dict[str, Any]) -> None:
    """Render runtime metrics from stored process outputs."""

    metrics = outputs.get("metrics") if isinstance(outputs.get("metrics"), dict) else {}
    if not metrics:
        st.caption(Tget(T, "metrics_empty", "Metrics appear after the first completed run."))
        return

    translations = metrics.get("translations") if isinstance(metrics.get("translations"), dict) else {}
    indexing = metrics.get("indexing") if isinstance(metrics.get("indexing"), dict) else {}

    extra = metrics.get("extra") if isinstance(metrics.get("extra"), dict) else {}
    stat_items = []
    for label, value in (
        (Tget(T, "metric_frames", "Frames"), metrics.get("num_frames")),
        (Tget(T, "metric_clips", "Clips"), metrics.get("num_clips")),
        (Tget(T, "metric_total_time", "Total"), f"{float(metrics.get('total_time_sec') or 0.0):.2f}s" if metrics.get("total_time_sec") is not None else None),
        (Tget(T, "metric_translations", "Translations"), len(translations) if translations else None),
        (Tget(T, "metric_index_views", "Index views"), len(indexing) if indexing else None),
        (Tget(T, "metric_decode", "Decode"), extra.get("decode_backend")),
        (Tget(T, "metric_source_fps", "Source FPS"), extra.get("source_fps")),
        (Tget(T, "metric_processed_fps", "Processed FPS"), extra.get("processed_fps")),
        (Tget(T, "metric_dark_drop", "Dark drop"), extra.get("dark_drop_ratio")),
        (Tget(T, "metric_lazy_drop", "Lazy drop"), extra.get("lazy_drop_ratio")),
    ):
        if value not in (None, "", 0):
            stat_items.append(f"{label}: {value}")

    stage_stats = metrics.get("stage_stats_sec") if isinstance(metrics.get("stage_stats_sec"), dict) else {}
    stage_label_map = {
        "preprocess_video": Tget(T, "stage_preprocess_video", "Video preprocess"),
        "build_clips": Tget(T, "stage_build_clips", "Build clips"),
        "caption_clips": Tget(T, "stage_caption_clips", "Caption clips"),
        "structure_segments": Tget(T, "stage_structure_segments", "Structure segments"),
        "generate_summary": Tget(T, "stage_generate_summary", "Generate summary"),
        "translate_outputs": Tget(T, "stage_translate_outputs", "Translate outputs"),
        "build_index": Tget(T, "stage_build_index", "Build index"),
    }
    for stage_name, payload in stage_stats.items():
        if not isinstance(payload, dict):
            continue
        mean_sec = payload.get("mean_sec")
        if mean_sec is None:
            continue
        label = stage_label_map.get(stage_name, humanize_token(stage_name))
        stat_items.append(f"{label}: {float(mean_sec):.2f}s")

    st.markdown(f"<div class='metric-summary-line'>{E(Tget(T, 'metrics', 'Metrics'))}:</div>", unsafe_allow_html=True)
    render_stat_row(stat_items)


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
        st.markdown("<div class='segment-card'>", unsafe_allow_html=True)
        if st.button(f"{mmss(start_sec)} - {mmss(end_sec)}", key=f"{session_key}_{idx}", use_container_width=True):
            st.session_state["video_seek_sec"] = int(start_sec)
            st.rerun()
        st.markdown(f"<div class='segment-text'>{E(clip_text(caption, 240))}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


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
            f"hybrid={float(hit.get('score', 0.0) or 0.0):.3f} | "
            f"sparse={float(hit.get('sparse_score', 0.0) or 0.0):.3f} | "
            f"dense={float(hit.get('dense_score', 0.0) or 0.0):.3f}"
        )
        if st.button(Tget(T, "open_clip", "Open"), key=f"{open_prefix}_{idx}", use_container_width=True):
            st.session_state["selected_video_id"] = str(hit.get("video_id") or "")
            st.session_state["video_seek_sec"] = int(float(hit.get("start_sec", 0.0) or 0.0))
            st.query_params["tab"] = "video"
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
