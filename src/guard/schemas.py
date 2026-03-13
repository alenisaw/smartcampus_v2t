# src/guard/schemas.py
"""
Schema guards for SmartCampus V2T model outputs.

Purpose:
- Normalize and validate model-generated payloads before they move between stages.
- Keep schema repair logic separate from policy-oriented text guards.
"""

from __future__ import annotations

from typing import Any, Dict, List


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value or "").strip().lower()
    return text in {"1", "true", "yes", "y", "on"}


def _to_float_01(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        out = 0.0
    if out < 0.0:
        return 0.0
    if out > 1.0:
        return 1.0
    return out


def _to_string_list(value: Any, *, max_items: int = 6) -> List[str]:
    if value is None:
        items: List[str] = []
    elif isinstance(value, str):
        items = [value]
    elif isinstance(value, (list, tuple, set)):
        items = [str(item) for item in value]
    else:
        items = [str(value)]

    out: List[str] = []
    for item in items:
        cleaned = _clean_text(item)
        if cleaned and cleaned not in out:
            out.append(cleaned)
        if len(out) >= max(1, int(max_items)):
            break
    return out


def validate_clip_observation(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a raw clip-observation payload into the strict VLM contract."""

    normalized = dict(payload or {})
    normalized["description"] = _clean_text(normalized.get("description"))
    normalized["anomaly_flag"] = _to_bool(normalized.get("anomaly_flag", False))
    normalized["anomaly_confidence"] = _to_float_01(normalized.get("anomaly_confidence", 0.0))
    normalized["anomaly_notes"] = _to_string_list(normalized.get("anomaly_notes"), max_items=4)
    if not normalized["description"]:
        normalized["description"] = "No clear visible activity is described."
    if not normalized["anomaly_flag"]:
        normalized["anomaly_confidence"] = min(float(normalized["anomaly_confidence"]), 0.5)
        normalized["anomaly_notes"] = []
    return normalized


def validate_segment_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize anomaly-related fields inside a structured segment payload."""

    normalized = dict(payload or {})
    normalized["description"] = _clean_text(
        normalized.get("description")
        or normalized.get("normalized_caption")
        or normalized.get("normalized_description")
        or normalized.get("raw_caption")
    )
    normalized["normalized_caption"] = _clean_text(
        normalized.get("normalized_caption")
        or normalized.get("normalized_description")
        or normalized.get("description")
        or normalized.get("raw_caption")
    )
    normalized["normalized_description"] = _clean_text(
        normalized.get("normalized_description")
        or normalized.get("normalized_caption")
        or normalized.get("description")
    )
    normalized["anomaly_flag"] = _to_bool(normalized.get("anomaly_flag", False))
    normalized["anomaly_confidence"] = _to_float_01(normalized.get("anomaly_confidence", 0.0))
    normalized["anomaly_notes"] = _to_string_list(normalized.get("anomaly_notes"), max_items=6)
    if not normalized["anomaly_flag"]:
        normalized["anomaly_confidence"] = min(float(normalized["anomaly_confidence"]), 0.5)
    normalized["tags"] = _to_string_list(normalized.get("tags"), max_items=8)
    normalized["objects"] = _to_string_list(normalized.get("objects"), max_items=8)
    return normalized


def validate_summary_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize top-level textual summary fields."""

    normalized = dict(payload or {})
    global_summary = _clean_text(normalized.get("global_summary") or normalized.get("summary"))
    normalized["global_summary"] = global_summary
    normalized["summary"] = global_summary
    if not isinstance(normalized.get("key_events"), list):
        normalized["key_events"] = []
    if not isinstance(normalized.get("citations"), list):
        normalized["citations"] = []
    if not isinstance(normalized.get("risk_overview"), dict):
        normalized["risk_overview"] = {}
    if not isinstance(normalized.get("statistics"), dict):
        normalized["statistics"] = {}
    if not isinstance(normalized.get("translation_views"), dict):
        normalized["translation_views"] = {}
    return normalized
