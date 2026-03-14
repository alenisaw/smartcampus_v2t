# backend/jobs/stage_metrics.py
"""
Stage metrics helpers for SmartCampus V2T backend.

Purpose:
- Aggregate per-stage timing samples into consistent metrics payloads.
- Keep timing-statistics logic separate from executor job flows.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List


def stage_stats(samples: List[float]) -> Dict[str, float]:
    """Compute count/mean/std for one stage timing sample list."""

    vals = [float(x) for x in samples if isinstance(x, (int, float))]
    if not vals:
        return {"count": 0, "mean_sec": 0.0, "std_sec": 0.0}
    mean = float(sum(vals) / len(vals))
    if len(vals) == 1:
        return {"count": 1, "mean_sec": mean, "std_sec": 0.0}
    var = float(sum((v - mean) ** 2 for v in vals) / len(vals))
    return {"count": int(len(vals)), "mean_sec": mean, "std_sec": float(math.sqrt(max(0.0, var)))}


def record_stage(metric_payload: Dict[str, Any], stage_name: str, elapsed_sec: float) -> None:
    """Append one stage sample and refresh aggregated stage statistics."""

    samples_map = metric_payload.setdefault("stage_samples_sec", {})
    if not isinstance(samples_map, dict):
        samples_map = {}
        metric_payload["stage_samples_sec"] = samples_map
    stats_map = metric_payload.setdefault("stage_stats_sec", {})
    if not isinstance(stats_map, dict):
        stats_map = {}
        metric_payload["stage_stats_sec"] = stats_map

    arr = samples_map.get(stage_name)
    if not isinstance(arr, list):
        arr = []
    arr.append(float(elapsed_sec))
    samples_map[stage_name] = arr
    stats_map[stage_name] = stage_stats(arr)


def finalize_stage_stats(metric_payload: Dict[str, Any], *, keep_samples: bool) -> None:
    """Ensure stage statistics are consistent and optionally drop raw samples."""

    samples_map = metric_payload.get("stage_samples_sec")
    if not isinstance(samples_map, dict):
        metric_payload["stage_samples_sec"] = {}
        samples_map = metric_payload["stage_samples_sec"]
    stats_map = metric_payload.get("stage_stats_sec")
    if not isinstance(stats_map, dict):
        stats_map = {}
    for stage_name, values in samples_map.items():
        if not isinstance(values, list):
            continue
        stats_map[str(stage_name)] = stage_stats(values)
    metric_payload["stage_stats_sec"] = stats_map
    if not keep_samples:
        metric_payload.pop("stage_samples_sec", None)


def merge_stage_payload(dst: Dict[str, Any], src: Dict[str, Any]) -> None:
    """Merge stage samples from one metrics payload into another."""

    samples_map = src.get("stage_samples_sec")
    if not isinstance(samples_map, dict):
        return
    for stage_name, values in samples_map.items():
        if not isinstance(values, list):
            continue
        for value in values:
            if isinstance(value, (int, float)):
                record_stage(dst, str(stage_name), float(value))
