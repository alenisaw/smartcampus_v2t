"""Storage-page specific UI components."""

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
) -> None:
    """Render the left-side library cards."""

    st.markdown(f"<div class='section-title'>{E(Tget(T, 'library_title', 'Library'))}</div>", unsafe_allow_html=True)
    for item in videos:
        if not isinstance(item, dict):
            continue
        video_id = str(item.get("video_id") or "")
        active = " active" if video_id == selected_video_id else ""
        path = Path(str(item.get("path") or ""))
        meta = get_video_meta(str(path), mtime(path)) if path.exists() else {}
        summary = ""
        try:
            outputs = client.get_video_outputs(video_id, "en")
            summary = first_sentence(str(outputs.get("global_summary") or ""))
        except Exception:
            summary = ""

        st.markdown(f"<div class='library-card{active}'>", unsafe_allow_html=True)
        st.markdown(f"<div class='library-card-title'>{E(video_id)}</div>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='library-card-sub'>{E(fmt_bytes(meta.get('size_bytes')))} · {E(hms(meta.get('duration_sec', 0.0)) if meta.get('duration_sec') else '-')}</div>",
            unsafe_allow_html=True,
        )
        if summary:
            st.markdown(f"<div class='library-card-copy'>{E(clip_text(summary, 100))}</div>", unsafe_allow_html=True)
        if st.button(Tget(T, "open_video", "Open"), key=f"pick_video_{video_id}", use_container_width=True):
            st.session_state["selected_video_id"] = video_id
            st.session_state["video_seek_sec"] = 0
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
    if upload is not None:
        if st.button(Tget(T, "upload_action", "Save to library"), key="library_upload_submit", use_container_width=True):
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

    variant_labels = {
        "__base__": "BASE",
        "__fanout__": "ALL VARIANTS",
    }
    st.selectbox(
        Tget(T, "variant", "Variant"),
        options=variant_options,
        index=variant_options.index(current_variant),
        key="run_variant",
        format_func=lambda token: variant_labels.get(token, variant_label(token)),
    )
    st.checkbox("Force overwrite outputs", key="run_force_overwrite")

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
        if st.button("Rebuild index", key="run_rebuild_index", use_container_width=True):
            try:
                payload = client.index_rebuild()
                soft_note(f"Index rebuild: {payload.get('ok')}", kind="ok")
            except Exception as exc:
                soft_note(f"{Tget(T, 'error_prefix', 'Error')}: {exc}", kind="warn")


def render_player(path: Path, start_sec: int) -> None:
    """Render the video player for the given file."""

    try:
        st.video(str(path), start_time=int(max(0, start_sec)))
    except TypeError:
        st.video(str(path))
