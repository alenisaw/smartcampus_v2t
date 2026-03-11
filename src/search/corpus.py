# src/search/corpus.py
"""
Search corpus assembly helpers for SmartCampus V2T.

Purpose:
- Build normalized search documents and dense-input text from stored segment payloads.
- Keep metadata extraction and keyframe-derived text augmentation out of the index builder.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import cv2
except Exception:
    cv2 = None


def coerce_str_list(value: Any) -> List[str]:
    out: List[str] = []
    if isinstance(value, list):
        for item in value:
            text = str(item or "").strip()
            if text:
                out.append(text)
    elif value is not None:
        text = str(value).strip()
        if text:
            out.append(text)
    return out


def extract_doc_metadata(item: Dict[str, Any], variant: Optional[str]) -> Dict[str, Any]:
    base_extra = item.get("extra")
    meta: Dict[str, Any] = dict(base_extra) if isinstance(base_extra, dict) else {}

    for key in ("segment_id", "event_type", "risk_level", "people_count_bucket", "motion_type"):
        value = item.get(key)
        if value is not None:
            meta[key] = str(value)

    for key in ("tags", "objects", "anomaly_notes"):
        values = coerce_str_list(item.get(key))
        if values:
            meta[key] = values
        elif key not in meta:
            meta[key] = []

    anomaly_flag = item.get("anomaly_flag")
    if anomaly_flag is not None:
        meta["anomaly_flag"] = bool(anomaly_flag)

    if variant is not None:
        meta["variant"] = str(variant)

    return meta


def build_searchable_text(display_text: str, meta: Dict[str, Any]) -> str:
    parts: List[str] = []
    base = str(display_text or "").strip()
    if base:
        parts.append(base)

    for key in ("event_type", "risk_level", "people_count_bucket", "motion_type"):
        value = str(meta.get(key) or "").strip()
        if value:
            parts.append(value)

    for key in ("tags", "objects"):
        for value in coerce_str_list(meta.get(key)):
            parts.append(value)

    return " \n ".join(parts)


def resolve_keyframe_path(
    *,
    raw_path: Optional[str],
    videos_root: Path,
    video_id: str,
    segment_file: Path,
) -> Optional[str]:
    """Resolve a keyframe reference into an existing absolute path when possible."""

    text = str(raw_path or "").strip()
    if not text:
        return None
    candidate = Path(text)
    if candidate.is_absolute() and candidate.exists():
        return str(candidate)

    checks = [
        Path(videos_root) / video_id / text,
        segment_file.parent / text,
        segment_file.parent.parent / text,
        segment_file.parent.parent.parent / text,
    ]
    for item in checks:
        try:
            if item.exists():
                return str(item.resolve())
        except Exception:
            continue
    return None


def keyframe_visual_tokens(path: Optional[str]) -> List[str]:
    """Create compact visual tokens from a keyframe for multimodal dense input."""

    if not path or cv2 is None:
        return []
    try:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            return []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean_luma = float(np.mean(gray))
        if mean_luma < 55:
            brightness = "vf_brightness_dark"
        elif mean_luma < 120:
            brightness = "vf_brightness_dim"
        elif mean_luma < 190:
            brightness = "vf_brightness_normal"
        else:
            brightness = "vf_brightness_bright"

        edges = cv2.Canny(gray, 80, 160)
        edge_ratio = float(np.mean(edges > 0))
        if edge_ratio < 0.03:
            edge_tag = "vf_edges_low"
        elif edge_ratio < 0.08:
            edge_tag = "vf_edges_mid"
        else:
            edge_tag = "vf_edges_high"

        means = image.reshape(-1, 3).mean(axis=0)
        dominant_idx = int(np.argmax(means))
        dominant = ("vf_color_blue", "vf_color_green", "vf_color_red")[dominant_idx]
        return [brightness, edge_tag, dominant]
    except Exception:
        return []


def build_dense_text(display_text: str, *, dense_input_mode: str, keyframe_path: Optional[str]) -> str:
    """Build dense embedding text in text-only or text+keyframe mode."""

    base = str(display_text or "").strip()
    mode = str(dense_input_mode or "text").strip().lower() or "text"
    if mode != "text_keyframe":
        return base
    tokens = keyframe_visual_tokens(keyframe_path)
    if not tokens:
        return base
    return f"{base} {' '.join(tokens)}".strip()
