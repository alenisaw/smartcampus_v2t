# app/components/storage.py
"""
Storage-page UI components for SmartCampus V2T.

Purpose:
- Render video inventory, outputs, and artifact inspection widgets.
- Keep storage-specific presentation logic out of generic UI modules.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

from app.api_client import BackendClient
from app.components.chrome import soft_note
from app.lib.formatters import E, available_variant_ids, clip_text, first_sentence, fmt_bytes, hms, variant_from_token, variant_label
from app.lib.i18n import Tget
from app.lib.media import get_video_meta, mtime


def render_video_library(
    T: Dict[str, Any],
    client: BackendClient,
    videos: List[Dict[str, Any]],
    selected_video_id: str,
    thumbs_dir: Path,
) -> None:
    """Render the processing-library cards."""

    st.markdown(f"<div class='section-title'>{E(Tget(T, 'library_title', 'Library'))}</div>", unsafe_allow_html=True)
    for item in videos:
        if not isinstance(item, dict):
            continue
        video_id = str(item.get("video_id") or "")
        active = " active" if video_id == selected_video_id else ""
        path = Path(str(item.get("path") or ""))
        meta = get_video_meta(str(path), mtime(path)) if path.exists() else {}
        thumb_path = thumbs_dir / f"{video_id}.jpg"
        summary = first_sentence(str(item.get("summary") or ""))
        st.markdown(f"<div class='library-card{active}'>", unsafe_allow_html=True)
        card_cols = st.columns([1.1, 2.8, 0.8], gap="small")
        with card_cols[0]:
            if thumb_path.exists():
                st.image(str(thumb_path), use_container_width=True)
            else:
                st.markdown(f"<div class='thumb-fallback'>{E(video_id)}</div>", unsafe_allow_html=True)
        with card_cols[1]:
            st.markdown(f"<div class='library-card-title'>{E(video_id)}</div>", unsafe_allow_html=True)
            brief_items = [
                fmt_bytes(meta.get("size_bytes")),
                hms(meta.get("duration_sec", 0.0)) if meta.get("duration_sec") else "-",
            ]
            langs = item.get("languages") if isinstance(item.get("languages"), list) else []
            if langs:
                brief_items.append("/".join(str(lang).upper() for lang in langs[:3]))
            st.markdown(f"<div class='library-card-sub'>{E(' | '.join(brief_items))}</div>", unsafe_allow_html=True)
            if summary:
                st.markdown(f"<div class='library-card-copy'>{E(clip_text(summary, 120))}</div>", unsafe_allow_html=True)
        with card_cols[2]:
            if st.button("+", key=f"queue_video_{video_id}", use_container_width=True, help=Tget(T, "queue_job", "Add to queue")):
                try:
                    st.session_state["selected_video_id"] = video_id
                    payload = client.create_job(video_id)
                    soft_note(f"{Tget(T, 'job_created', 'Job queued')}: {payload.get('job_id')}", kind="ok")
                except Exception as exc:
                    soft_note(f"{Tget(T, 'error_prefix', 'Error')}: {exc}", kind="warn")
            if st.button(">", key=f"open_video_analytics_{video_id}", use_container_width=True, help=Tget(T, "go_to_analytics", "Open analytics")):
                st.session_state["selected_video_id"] = video_id
                st.session_state["video_seek_sec"] = 0
                st.query_params["tab"] = "video"
                st.rerun()
            if st.button("x", key=f"delete_video_trigger_{video_id}", use_container_width=True, help=Tget(T, "delete_video_action", "Delete video")):
                st.session_state["confirm_delete_video_id"] = video_id
            if st.session_state.get("confirm_delete_video_id") == video_id:
                confirm_cols = st.columns(2, gap="small")
                with confirm_cols[0]:
                    if st.button(Tget(T, "confirm_delete", "Confirm"), key=f"confirm_delete_{video_id}", use_container_width=True):
                        try:
                            client.delete_video(video_id)
                            if st.session_state.get("selected_video_id") == video_id:
                                st.session_state["selected_video_id"] = ""
                            st.session_state["confirm_delete_video_id"] = ""
                            soft_note(Tget(T, "video_deleted", "Video deleted"), kind="ok")
                            st.rerun()
                        except Exception as exc:
                            soft_note(f"{Tget(T, 'error_prefix', 'Error')}: {exc}", kind="warn")
                with confirm_cols[1]:
                    if st.button(Tget(T, "cancel", "Cancel"), key=f"cancel_delete_{video_id}", use_container_width=True):
                        st.session_state["confirm_delete_video_id"] = ""
                        st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)


def render_upload_panel(T: Dict[str, Any], client: BackendClient) -> None:
    """Render the upload widget and backend upload flow."""

    st.markdown(f"<div class='section-title'>{E(Tget(T, 'upload_title', 'Upload video'))}</div>", unsafe_allow_html=True)
    upload = st.file_uploader(
        Tget(T, "upload_hint", "Choose a video file"),
        type=["mp4", "mov", "mkv", "avi"],
        key="library_upload",
        label_visibility="collapsed",
    )
    if upload is not None and st.button(Tget(T, "upload_action", "Save to library"), key="library_upload_submit", use_container_width=True):
        try:
            payload = client.upload_video(upload.name, upload.getvalue())
            st.session_state["selected_video_id"] = str(payload.get("video_id") or Path(upload.name).stem)
            soft_note(Tget(T, "upload_done", "Video uploaded successfully."), kind="ok")
            st.rerun()
        except Exception as exc:
            soft_note(f"{Tget(T, 'error_prefix', 'Error')}: {exc}", kind="warn")


def render_run_panel(T: Dict[str, Any], client: BackendClient, cfg: Any, selected_video_id: str) -> None:
    """Render run controls for main and experimental modes."""

    st.markdown(f"<div class='section-title'>{E(Tget(T, 'run_title', 'Run pipeline'))}</div>", unsafe_allow_html=True)

    profile = st.radio(
        Tget(T, "profile", "Profile"),
        options=["main", "experimental"],
        horizontal=True,
        key="run_profile",
        label_visibility="collapsed",
    )
    variants = available_variant_ids(cfg)
    variant_options = ["__base__"] if profile == "main" else ["__fanout__"] + variants
    default_variant = "__base__" if profile == "main" else "__fanout__"
    current_variant = str(st.session_state.get("run_variant") or default_variant)
    if current_variant not in variant_options:
        current_variant = default_variant
    st.session_state["run_variant"] = current_variant

    variant_labels = {"__base__": Tget(T, "base_variant", "BASE"), "__fanout__": Tget(T, "all_variants", "ALL VARIANTS")}
    st.selectbox(
        Tget(T, "variant", "Variant"),
        options=variant_options,
        index=variant_options.index(current_variant),
        key="run_variant",
        format_func=lambda token: variant_labels.get(token, variant_label(token)),
    )
    st.checkbox(Tget(T, "force_overwrite", "Force overwrite outputs"), key="run_force_overwrite")

    col_run, col_index = st.columns(2, gap="small")
    with col_run:
        if st.button(Tget(T, "run_action", "Start processing"), key="run_video_btn", use_container_width=True):
            try:
                selected_variant: Optional[str]
                if profile == "main":
                    selected_variant = None
                elif st.session_state.get("run_variant") == "__fanout__":
                    selected_variant = None
                else:
                    selected_variant = variant_from_token(st.session_state.get("run_variant"))

                response = client.create_job(
                    selected_video_id,
                    extra={"force_overwrite": bool(st.session_state.get("run_force_overwrite", False))},
                    profile=profile,
                    variant=selected_variant,
                )
                soft_note(f"{Tget(T, 'job_created', 'Job queued')}: {response.get('job_id')}", kind="ok")
            except Exception as exc:
                soft_note(f"{Tget(T, 'error_prefix', 'Error')}: {exc}", kind="warn")
    with col_index:
        if st.button(Tget(T, "rebuild_index", "Rebuild index"), key="run_rebuild_index", use_container_width=True):
            try:
                payload = client.index_rebuild()
                soft_note(f"{Tget(T, 'rebuild_index', 'Rebuild index')}: {payload.get('ok')}", kind="ok")
            except Exception as exc:
                soft_note(f"{Tget(T, 'error_prefix', 'Error')}: {exc}", kind="warn")


def render_player(path: Path, start_sec: int) -> None:
    """Render the video player for the given file."""

    try:
        st.video(str(path), start_time=int(max(0, start_sec)))
    except TypeError:
        st.video(str(path))


def render_queue_panel(T: Dict[str, Any], client: BackendClient, queue: Dict[str, Any]) -> None:
    """Render a compact queue control rail for the storage page."""

    status = queue.get("status") if isinstance(queue, dict) else {}
    running = queue.get("running") if isinstance(queue, dict) else {}
    queued = queue.get("queued") if isinstance(queue, dict) else []

    st.markdown(f"<div class='section-title'>{E(Tget(T, 'queue_title', 'Queue'))}</div>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="queue-side-card">
            <div class="queue-side-top">
                <div>
                    <div class="queue-side-label">{E(Tget(T, 'worker_state', 'Worker state'))}</div>
                    <div class="queue-side-value">{E(Tget(T, 'paused', 'Paused') if bool((status or {}).get('paused')) else Tget(T, 'active', 'Active'))}</div>
                </div>
                <div class="queue-side-meta">{E(str(len(queued) if isinstance(queued, list) else 0))} {E(Tget(T, 'queued_short', 'queued'))}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    actions = st.columns(2, gap="small")
    with actions[0]:
        if bool((status or {}).get("paused")):
            if st.button(Tget(T, "resume", "Resume"), key="queue_resume_btn", use_container_width=True):
                try:
                    client.queue_resume()
                    st.rerun()
                except Exception as exc:
                    soft_note(f"{Tget(T, 'error_prefix', 'Error')}: {exc}", kind="warn")
        else:
            if st.button(Tget(T, "pause", "Pause"), key="queue_pause_btn", use_container_width=True):
                try:
                    client.queue_pause()
                    st.rerun()
                except Exception as exc:
                    soft_note(f"{Tget(T, 'error_prefix', 'Error')}: {exc}", kind="warn")
    with actions[1]:
        if st.button(Tget(T, "refresh", "Refresh"), key="queue_refresh_btn", use_container_width=True):
            st.rerun()

    if isinstance(running, dict) and running.get("job_id"):
        st.markdown(
            f"""
            <div class="queue-job-card running">
                <div class="queue-job-title">{E(str(running.get('job_type') or 'running'))}</div>
                <div class="queue-job-copy">{E(str(running.get('video_id') or '-'))}</div>
                <div class="queue-job-meta">{E(str(running.get('stage') or 'running'))} | {float(running.get('progress') or 0.0):.0%}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if not isinstance(queued, list) or not queued:
        st.caption(Tget(T, "queue_empty", "Queue is empty."))
        return

    for idx, item in enumerate(queued[:6]):
        if not isinstance(item, dict):
            continue
        job_id = str(item.get("job_id") or "")
        st.markdown(
            f"""
            <div class="queue-job-card">
                <div class="queue-job-title">{E(str(item.get('job_type') or 'job'))}</div>
                <div class="queue-job-copy">{E(str(item.get('video_id') or '-'))}</div>
                <div class="queue-job-meta">{E(str(item.get('profile') or 'main'))} | {E(str(item.get('variant') or 'base'))}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        row = st.columns(3, gap="small")
        with row[0]:
            if st.button(Tget(T, "move_up", "Up"), key=f"queue_up_{job_id}_{idx}", use_container_width=True):
                try:
                    client.queue_move(job_id, "up")
                    st.rerun()
                except Exception as exc:
                    soft_note(f"{Tget(T, 'error_prefix', 'Error')}: {exc}", kind="warn")
        with row[1]:
            if st.button(Tget(T, "move_down", "Down"), key=f"queue_down_{job_id}_{idx}", use_container_width=True):
                try:
                    client.queue_move(job_id, "down")
                    st.rerun()
                except Exception as exc:
                    soft_note(f"{Tget(T, 'error_prefix', 'Error')}: {exc}", kind="warn")
        with row[2]:
            if st.button(Tget(T, "drop", "Drop"), key=f"queue_drop_{job_id}_{idx}", use_container_width=True):
                try:
                    client.queue_remove(job_id)
                    st.rerun()
                except Exception as exc:
                    soft_note(f"{Tget(T, 'error_prefix', 'Error')}: {exc}", kind="warn")
