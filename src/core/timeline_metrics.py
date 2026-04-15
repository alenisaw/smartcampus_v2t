"""
Intrinsic semantic timeline metrics for SmartCampus V2T.

Purpose:
- Reuse one deterministic implementation for post-generation timeline analysis.
- Support threshold ablations without touching expensive video inference stages.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np


def _float_or_none(value: Any) -> Optional[float]:
    """Return a float value when possible."""

    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        text = str(value or "").strip()
        if not text:
            return None
        return float(text)
    except Exception:
        return None


def _normalized(vec: Any) -> Optional[np.ndarray]:
    """Return an L2-normalized vector or None when invalid."""

    if vec is None:
        return None
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return None
    norm = float(np.linalg.norm(arr))
    if norm <= 0.0:
        return None
    return arr / norm


def cosine_similarity(left: Any, right: Any) -> Optional[float]:
    """Return cosine similarity for two vectors or None when undefined."""

    a = _normalized(left)
    b = _normalized(right)
    if a is None or b is None:
        return None
    return float(np.dot(a, b))


def mean_ignore_none(values: List[Optional[float]]) -> Optional[float]:
    """Average numeric values while skipping missing items."""

    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None
    return float(sum(clean) / len(clean))


def segment_embeddings_from_members(
    raw_embeddings: List[Optional[np.ndarray]],
    member_groups: List[List[int]],
) -> List[Optional[np.ndarray]]:
    """Aggregate raw clip embeddings into one embedding per merged segment."""

    out: List[Optional[np.ndarray]] = []
    for members in member_groups:
        rows: List[np.ndarray] = []
        for index in members:
            if index < 0 or index >= len(raw_embeddings):
                continue
            vec = _normalized(raw_embeddings[index])
            if vec is not None:
                rows.append(vec)
        if not rows:
            out.append(None)
            continue
        merged = np.mean(np.stack(rows, axis=0), axis=0)
        out.append(_normalized(merged))
    return out


def intrinsic_timeline_metrics(
    *,
    duration_seconds: Any,
    raw_clips: int,
    merged_segments: List[Dict[str, Any]],
    segment_embeddings: List[Optional[np.ndarray]],
) -> Dict[str, Optional[float]]:
    """Compute intrinsic timeline metrics for one merged semantic timeline."""

    duration = _float_or_none(duration_seconds) or 0.0
    final_segments = int(len(merged_segments))
    segment_durations = [
        max(
            0.0,
            float(_float_or_none(item.get("end_sec")) or 0.0) - float(_float_or_none(item.get("start_sec")) or 0.0),
        )
        for item in merged_segments
    ]

    adjacent_scores = [
        cosine_similarity(segment_embeddings[index], segment_embeddings[index + 1])
        for index in range(max(0, len(segment_embeddings) - 1))
    ]
    srr_scores = [
        cosine_similarity(segment_embeddings[index], segment_embeddings[index + 2])
        for index in range(max(0, len(segment_embeddings) - 2))
    ]

    sns_scores: List[Optional[float]] = []
    running_rows: List[np.ndarray] = []
    for vec in segment_embeddings:
        current = _normalized(vec)
        if current is None:
            continue
        if running_rows:
            previous_mean = _normalized(np.mean(np.stack(running_rows, axis=0), axis=0))
            sim = cosine_similarity(current, previous_mean)
            sns_scores.append(None if sim is None else float(1.0 - sim))
        running_rows.append(current)

    tcs = mean_ignore_none(adjacent_scores)
    srr = mean_ignore_none(srr_scores) if len(segment_embeddings) >= 3 else None
    sns = mean_ignore_none(sns_scores)

    return {
        "duration_seconds": duration if duration > 0.0 else None,
        "raw_clips": int(raw_clips),
        "final_segments": int(final_segments),
        "compression_ratio": (float(raw_clips) / float(final_segments)) if final_segments > 0 else None,
        "dd": (60.0 * float(final_segments) / float(duration)) if duration > 0.0 else None,
        "mean_segment_duration": (float(sum(segment_durations)) / float(len(segment_durations))) if segment_durations else None,
        "tcs": tcs,
        "srr": srr,
        "sns": sns,
        "sdi": (float(1.0 - tcs) if tcs is not None else None),
    }
