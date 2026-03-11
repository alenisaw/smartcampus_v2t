# app/lib/media.py
"""
Media and asset helpers for SmartCampus V2T Streamlit UI.

Purpose:
- Load CSS, images, and media-related presentation assets.
- Keep asset and browser rendering helpers out of page modules.
"""

from __future__ import annotations

import base64
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import streamlit as st


def mtime(path: Path) -> float:
    """Return file mtime or zero on failure."""

    try:
        return float(path.stat().st_mtime)
    except Exception:
        return 0.0


def load_and_apply_css(styles_path: Path) -> None:
    """Load and inject UI CSS."""

    if not styles_path.exists():
        return
    css = styles_path.read_text(encoding="utf-8")
    if css.strip():
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def img_to_data_uri(path: Path) -> Optional[str]:
    """Convert an image file into a data URI."""

    if not path.exists():
        return None
    try:
        return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("utf-8")
    except Exception:
        return None


def get_video_meta(video_path_str: str, mtime_ns: float) -> Dict[str, Any]:
    """Read basic video metadata through OpenCV."""

    _ = mtime_ns
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


def ensure_browser_video(video_path: Path) -> Path:
    """Ensure the selected video has a browser-friendly container for playback."""

    ext = video_path.suffix.lower()
    if ext == ".mp4":
        return video_path

    mp4_path = video_path.with_suffix(".mp4")
    if mp4_path.exists() and mtime(mp4_path) >= mtime(video_path):
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
