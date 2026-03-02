# src/pipeline/structuring.py
"""
Structuring layer for enriching Segment Schema v2 with event/risk fields.

Purpose:
- Provide a rule-based baseline now.
- Attempt strict-JSON LLM extraction when available.
- Keep a robust rule-based fallback and repair path.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.llm.client import LLMClient


def _contains_any(text: str, terms: List[str]) -> bool:
    """Check whether any keyword appears in normalized lowercase text."""

    low = str(text or "").lower()
    return any(term in low for term in terms)


def _infer_objects(text: str) -> List[str]:
    """Infer a coarse object list from caption keywords."""

    mapping = {
        "person": ["person", "people", "man", "woman", "pedestrian"],
        "vehicle": ["car", "vehicle", "truck", "bus", "motorcycle"],
        "bicycle": ["bicycle", "bike", "cyclist"],
        "bag": ["bag", "backpack", "package"],
        "door": ["door", "gate", "entrance"],
    }
    found = [label for label, terms in mapping.items() if _contains_any(text, terms)]
    return found


def _infer_tags(text: str) -> List[str]:
    """Infer a lightweight tag set from action words."""

    tags: List[str] = []
    if _contains_any(text, ["walk", "walking", "move", "moving", "approach"]):
        tags.append("movement")
    if _contains_any(text, ["stand", "standing", "wait", "waiting"]):
        tags.append("idle_presence")
    if _contains_any(text, ["enter", "exit", "leave"]):
        tags.append("entry_exit")
    if _contains_any(text, ["car", "vehicle", "truck", "bus"]):
        tags.append("vehicle")
    return tags


def _infer_people_bucket(text: str) -> str:
    """Infer a coarse people count bucket from caption phrasing."""

    if not _contains_any(text, ["person", "people", "man", "woman", "pedestrian"]):
        return "none"
    if _contains_any(text, ["crowd", "many people", "group of people"]):
        return "high"
    if _contains_any(text, ["two people", "three people", "several people", "group"]):
        return "mid"
    return "low"


def _infer_motion_type(text: str) -> str:
    """Infer a coarse motion type from caption wording."""

    if _contains_any(text, ["run", "running", "rush", "fast", "chaotic"]):
        return "dynamic"
    if _contains_any(text, ["walk", "walking", "move", "moving", "drive"]):
        return "stable"
    return "static"


def _infer_event_type(text: str, objects: List[str]) -> str:
    """Infer a coarse event type for filters and summaries."""

    if "vehicle" in objects:
        return "vehicle_activity"
    if "person" in objects:
        if _contains_any(text, ["enter", "exit", "leave"]):
            return "entry_exit"
        return "person_activity"
    return "scene_observation"


def _infer_risk(text: str) -> tuple[str, bool, List[str]]:
    """Infer risk level and anomaly markers from explicit risk keywords."""

    low = str(text or "").lower()
    anomaly_terms = []
    if any(term in low for term in ["fight", "fall", "fire", "smoke", "alarm", "collision"]):
        anomaly_terms.append("explicit_anomaly_keyword")
        return "warning", True, anomaly_terms
    if any(term in low for term in ["running", "rush", "loiter", "unauthorized", "trespass"]):
        anomaly_terms.append("attention_keyword")
        return "attention", True, anomaly_terms
    return "normal", False, anomaly_terms


def validate_segment_schema(segment: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize required fields into valid Segment Schema v2 values."""

    allowed_risk = {"normal", "attention", "warning", "critical"}
    allowed_people = {"none", "low", "mid", "high"}
    allowed_motion = {"static", "stable", "dynamic"}

    segment["risk_level"] = str(segment.get("risk_level", "normal"))
    if segment["risk_level"] not in allowed_risk:
        segment["risk_level"] = "normal"

    segment["people_count_bucket"] = str(segment.get("people_count_bucket", "none"))
    if segment["people_count_bucket"] not in allowed_people:
        segment["people_count_bucket"] = "none"

    segment["motion_type"] = str(segment.get("motion_type", "static"))
    if segment["motion_type"] not in allowed_motion:
        segment["motion_type"] = "static"

    for key in ("tags", "objects", "anomaly_notes"):
        if not isinstance(segment.get(key), list):
            segment[key] = []
    segment["anomaly_flag"] = bool(segment.get("anomaly_flag", False))
    segment["event_type"] = str(segment.get("event_type", "unclassified") or "unclassified")
    return segment


def _structuring_prompt(segment: Dict[str, Any]) -> str:
    """Build the strict-JSON extraction prompt for one segment."""

    payload = {
        "video_id": segment.get("video_id"),
        "segment_id": segment.get("segment_id"),
        "start_sec": segment.get("start_sec"),
        "end_sec": segment.get("end_sec"),
        "raw_caption": segment.get("raw_caption"),
        "normalized_caption": segment.get("normalized_caption"),
    }
    return (
        "Return only JSON with keys: "
        "event_type, risk_level, tags, objects, people_count_bucket, motion_type, anomaly_flag, anomaly_notes.\n"
        "Use only the provided caption. Do not invent facts.\n"
        f"INPUT={json.dumps(payload, ensure_ascii=False)}"
    )


def _repair_llm_payload(data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Repair a partially valid LLM JSON payload into the expected schema subset."""

    src = dict(data or {})
    repaired: Dict[str, Any] = {
        "event_type": str(src.get("event_type", "unclassified") or "unclassified"),
        "risk_level": str(src.get("risk_level", "normal") or "normal"),
        "tags": src.get("tags") if isinstance(src.get("tags"), list) else [],
        "objects": src.get("objects") if isinstance(src.get("objects"), list) else [],
        "people_count_bucket": str(src.get("people_count_bucket", "none") or "none"),
        "motion_type": str(src.get("motion_type", "static") or "static"),
        "anomaly_flag": bool(src.get("anomaly_flag", False)),
        "anomaly_notes": src.get("anomaly_notes") if isinstance(src.get("anomaly_notes"), list) else [],
    }
    return validate_segment_schema(repaired)


@dataclass
class StructuringService:
    """Segment enrichment service with optional LLM hook and robust fallback."""

    cfg: Any
    llm_client: Optional[LLMClient] = None

    @classmethod
    def from_config(cls, cfg: Any) -> "StructuringService":
        """Build the service and attach an LLM client when configured."""

        client: Optional[LLMClient] = None
        try:
            client = LLMClient.from_config(cfg)
        except Exception:
            client = None
        return cls(cfg=cfg, llm_client=client)

    def enrich_segments(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enrich all segments using fallback rules and future LLM hooks."""

        out: List[Dict[str, Any]] = []
        for segment in segments:
            out.append(self.enrich_segment(segment))
        return out

    def enrich_segment(self, segment: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich one segment in-place and return the normalized result."""

        payload = dict(segment or {})
        text = str(payload.get("normalized_caption") or payload.get("description") or "")

        llm_payload = self._try_llm_structuring(payload)
        if llm_payload:
            payload.update(llm_payload)
            metrics_meta = payload.get("metrics_meta")
            if isinstance(metrics_meta, dict):
                metrics_meta.setdefault("structuring_time_sec", 0.0)
                payload["metrics_meta"] = metrics_meta
            return validate_segment_schema(payload)

        objects = _infer_objects(text)
        tags = _infer_tags(text)
        people_bucket = _infer_people_bucket(text)
        motion_type = _infer_motion_type(text)
        event_type = _infer_event_type(text, objects)
        risk_level, anomaly_flag, anomaly_notes = _infer_risk(text)

        payload["objects"] = objects
        payload["tags"] = tags
        payload["people_count_bucket"] = people_bucket
        payload["motion_type"] = motion_type
        payload["event_type"] = event_type
        payload["risk_level"] = risk_level
        payload["anomaly_flag"] = anomaly_flag
        payload["anomaly_notes"] = anomaly_notes

        metrics_meta = payload.get("metrics_meta")
        if isinstance(metrics_meta, dict):
            metrics_meta.setdefault("structuring_time_sec", 0.0)
            payload["metrics_meta"] = metrics_meta

        return validate_segment_schema(payload)

    def _try_llm_structuring(self, segment: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Attempt strict-JSON extraction via LLM, then repair the payload."""

        if self.llm_client is None:
            return None

        try:
            response = self.llm_client.generate_json(_structuring_prompt(segment))
        except Exception:
            return None
        if not isinstance(response, dict):
            return None
        return _repair_llm_payload(response)
