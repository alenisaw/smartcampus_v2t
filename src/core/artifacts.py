# src/core/artifacts.py
"""
Artifact schema builders for SmartCampus V2T.

Purpose:
- Convert pipeline outputs into canonical segment and video summary artifacts.
- Keep artifact-shape assembly under the new compact `src.core` layout.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def _safe_ratio(numerator: float, denominator: float) -> float:
    """Return a safe ratio for metrics and distribution fields."""

    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _models_block(cfg: Any) -> Dict[str, Any]:
    """Build a consistent models metadata block for persisted artifacts."""

    return {
        "vision_model": str(getattr(cfg.model, "model_name_or_path", "")),
        "structuring_model": str(getattr(cfg.llm, "model_id", "")),
        "embedding_model": str(getattr(cfg.search, "embedding_model_id", "")),
        "reranker_model": str(getattr(cfg.search, "reranker_model_id", "")),
        "mt_ru_en": str(getattr(cfg.translation, "ru_en_model_id", "")),
        "mt_en_ru": str(getattr(cfg.translation, "en_ru_model_id", "")),
        "mt_kk_ru": str(getattr(cfg.translation, "kk_ru_model_id", "")),
        "mt_ru_kk": str(getattr(cfg.translation, "ru_kk_model_id", "")),
        "guard_model": str(getattr(cfg.guard, "model_id", "")),
    }


def build_segment_schema_v2(
    *,
    cfg: Any,
    video_id: str,
    segment_index: int,
    start_sec: float,
    end_sec: float,
    raw_caption: str,
    normalized_caption: str,
    keyframe_path: Optional[str],
    preprocessing_meta: Optional[Dict[str, Any]] = None,
    metrics_meta: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
    language: str = "en",
    anomaly_flag: bool = False,
    anomaly_confidence: float = 0.0,
    anomaly_notes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Build one Segment Schema v2 object with safe defaults."""

    segment_id = f"seg_{int(segment_index):06d}"
    normalized_caption = str(normalized_caption or raw_caption or "").strip()
    raw_caption = str(raw_caption or normalized_caption).strip()
    preprocessing_meta = dict(preprocessing_meta or {})
    metrics_meta = dict(metrics_meta or {})
    extra = dict(extra or {})
    anomaly_notes = [str(item).strip() for item in (anomaly_notes or []) if str(item).strip()]

    merged_from = extra.get("merged_from")
    if not isinstance(merged_from, list):
        merged_from = []

    payload = {
        "schema_version": "2.0",
        "video_id": str(video_id),
        "segment_id": segment_id,
        "start_sec": float(start_sec),
        "end_sec": float(end_sec),
        "language": str(language),
        "raw_caption": raw_caption,
        "normalized_caption": normalized_caption,
        "event_type": "unclassified",
        "risk_level": "normal",
        "tags": [],
        "objects": [],
        "people_count_bucket": "none",
        "motion_type": "stable",
        "anomaly_flag": bool(anomaly_flag),
        "anomaly_confidence": float(anomaly_confidence),
        "anomaly_notes": anomaly_notes,
        "severity_reason": "",
        "needs_attention": bool(anomaly_flag),
        "evidence": {
            "keyframe_path": keyframe_path,
        },
        "preprocessing_meta": {
            "fps": float(preprocessing_meta.get("fps", 0.0)),
            "resize": str(preprocessing_meta.get("resize", "")),
            "dark_drop_ratio": float(preprocessing_meta.get("dark_drop_ratio", 0.0)),
            "lazy_drop_ratio": float(preprocessing_meta.get("lazy_drop_ratio", 0.0)),
            "blur_flag_ratio": float(preprocessing_meta.get("blur_flag_ratio", 0.0)),
            "anonymized": bool(preprocessing_meta.get("anonymized", False)),
        },
        "models": _models_block(cfg),
        "translation_meta": {
            "available_languages": [str(language)],
            "translated_from": None,
            "mt_engine": None,
            "post_edited_by_llm": False,
        },
        "metrics_meta": {
            "preprocess_time_sec": float(metrics_meta.get("preprocess_time_sec", 0.0)),
            "vlm_time_sec": float(metrics_meta.get("vlm_time_sec", 0.0)),
            "postprocess_time_sec": float(metrics_meta.get("postprocess_time_sec", 0.0)),
            "structuring_time_sec": float(metrics_meta.get("structuring_time_sec", 0.0)),
            "embedding_time_sec": float(metrics_meta.get("embedding_time_sec", 0.0)),
        },
        "extra": {
            "merge_group_id": extra.get("merge_group_id"),
            "merged_from": merged_from,
        },
        "description": normalized_caption,
    }
    return payload


def build_video_summary_v2(
    *,
    cfg: Any,
    video_id: str,
    language: str,
    duration_sec: float,
    summary_text: str,
    segments: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build a minimal but valid Video Summary Schema v2 from current pipeline outputs."""

    summary_text = str(summary_text or "").strip()
    citations = []
    key_events = []
    risk_distribution = {"normal": 0, "attention": 0, "warning": 0, "critical": 0}
    people_positive = 0

    for item in segments[:10]:
        segment_id = str(item.get("segment_id") or "")
        start_sec = float(item.get("start_sec", 0.0))
        end_sec = float(item.get("end_sec", 0.0))
        citations.append(
            {
                "segment_id": segment_id,
                "start_sec": start_sec,
                "end_sec": end_sec,
            }
        )

    for item in segments[:5]:
        risk = str(item.get("risk_level", "normal"))
        if risk not in risk_distribution:
            risk = "normal"
        risk_distribution[risk] += 1
        if str(item.get("people_count_bucket", "none")) != "none":
            people_positive += 1

    for item in segments[:3]:
        key_events.append(
            {
                "event_type": str(item.get("event_type", "unclassified")),
                "description": str(item.get("normalized_caption") or item.get("description") or ""),
                "risk_level": str(item.get("risk_level", "normal")),
                "time_spans": [
                    {
                        "start_sec": float(item.get("start_sec", 0.0)),
                        "end_sec": float(item.get("end_sec", 0.0)),
                    }
                ],
                "supporting_segments": [str(item.get("segment_id") or "")],
            }
        )

    return {
        "schema_version": "2.0",
        "video_id": str(video_id),
        "language": str(language),
        "time_range_sec": {
            "start": 0.0,
            "end": float(duration_sec),
        },
        "global_summary": summary_text,
        "key_events": key_events,
        "risk_overview": {
            "distribution": risk_distribution,
            "notable_segments": [item["segment_id"] for item in segments[:1] if item.get("segment_id")],
        },
        "statistics": {
            "segments_total": int(len(segments)),
            "people_presence_ratio": _safe_ratio(people_positive, len(segments)),
            "activity_peaks": [],
        },
        "citations": citations[: max(5, len(citations))],
        "translation_views": {},
        "models": {
            "summary_llm": str(getattr(cfg.llm, "model_id", "")),
            "mt_ru_en": str(getattr(cfg.translation, "ru_en_model_id", "")),
            "mt_en_ru": str(getattr(cfg.translation, "en_ru_model_id", "")),
            "mt_kk_ru": str(getattr(cfg.translation, "kk_ru_model_id", "")),
            "mt_ru_kk": str(getattr(cfg.translation, "ru_kk_model_id", "")),
        },
        "summary": summary_text,
    }
