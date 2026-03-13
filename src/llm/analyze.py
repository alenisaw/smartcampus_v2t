# src/llm/analyze.py
"""
Semantic segment analysis for SmartCampus V2T.

Purpose:
- Normalize segment descriptions after the VLM observation stage.
- Classify event type, risk, and internal search tags from segment evidence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.guard.schemas import validate_segment_payload
from src.llm.client import LLMClient


def _contains_any(text: str, terms: List[str]) -> bool:
    """Check whether any keyword appears in normalized lowercase text."""

    low = str(text or "").lower()
    return any(term in low for term in terms)


def _infer_objects(text: str) -> List[str]:
    """Infer a coarse object list from description keywords."""

    mapping = {
        "person": ["person", "people", "man", "woman", "pedestrian"],
        "vehicle": ["car", "vehicle", "truck", "bus", "motorcycle"],
        "bicycle": ["bicycle", "bike", "cyclist"],
        "bag": ["bag", "backpack", "package"],
        "door": ["door", "gate", "entrance"],
    }
    return [label for label, terms in mapping.items() if _contains_any(text, terms)]


def _infer_tags(text: str, *, anomaly_flag: bool) -> List[str]:
    """Infer a lightweight internal tag set from text and anomaly cues."""

    tags: List[str] = []
    if _contains_any(text, ["walk", "walking", "move", "moving", "approach"]):
        tags.append("movement")
    if _contains_any(text, ["stand", "standing", "wait", "waiting"]):
        tags.append("idle_presence")
    if _contains_any(text, ["enter", "exit", "leave"]):
        tags.append("entry_exit")
    if _contains_any(text, ["car", "vehicle", "truck", "bus"]):
        tags.append("vehicle")
    if anomaly_flag:
        tags.append("attention")
    return tags


def _infer_people_bucket(text: str) -> str:
    """Infer a coarse people count bucket from description phrasing."""

    if not _contains_any(text, ["person", "people", "man", "woman", "pedestrian"]):
        return "none"
    if _contains_any(text, ["crowd", "many people", "group of people"]):
        return "high"
    if _contains_any(text, ["two people", "three people", "several people", "group"]):
        return "mid"
    return "low"


def _infer_motion_type(text: str) -> str:
    """Infer a coarse motion type from description wording."""

    if _contains_any(text, ["run", "running", "rush", "fast", "chaotic"]):
        return "dynamic"
    if _contains_any(text, ["walk", "walking", "move", "moving", "drive"]):
        return "stable"
    return "static"


def _infer_event_type(text: str, objects: List[str], *, anomaly_flag: bool) -> str:
    """Infer a coarse event type for filters and summaries."""

    if anomaly_flag and _contains_any(text, ["fall", "fight", "smoke", "fire", "intrusion", "trespass"]):
        return "suspicious_activity"
    if "vehicle" in objects:
        return "vehicle_activity"
    if "person" in objects:
        if _contains_any(text, ["enter", "exit", "leave"]):
            return "entry_exit"
        return "person_activity"
    return "scene_observation"


def _infer_risk(
    text: str,
    *,
    anomaly_flag: bool,
    anomaly_confidence: float,
    anomaly_notes: List[str],
) -> tuple[str, str]:
    """Infer risk level and a short reason from observation evidence."""

    low = str(text or "").lower()
    if any(term in low for term in ["fight", "fire", "smoke", "alarm", "collision"]):
        return "warning", "Visible signs point to an explicitly unsafe event."
    if anomaly_flag and anomaly_confidence >= 0.75:
        return "warning", "The observation stage marked the segment as highly suspicious."
    if anomaly_flag or anomaly_confidence >= 0.35:
        if anomaly_notes:
            return "attention", str(anomaly_notes[0])
        return "attention", "The observation stage detected suspicious visual signals."
    if any(term in low for term in ["running", "rush", "loiter", "unauthorized", "trespass"]):
        return "attention", "Movement or behavior suggests a segment worth operator attention."
    return "normal", "No strong warning signals are visible in the segment."


def validate_segment_schema(segment: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize required fields into valid structured segment values."""

    allowed_risk = {"normal", "attention", "warning", "critical"}
    allowed_people = {"none", "low", "mid", "high"}
    allowed_motion = {"static", "stable", "dynamic"}

    normalized = validate_segment_payload(segment)
    normalized["risk_level"] = str(normalized.get("risk_level", "normal") or "normal")
    if normalized["risk_level"] not in allowed_risk:
        normalized["risk_level"] = "normal"

    normalized["people_count_bucket"] = str(normalized.get("people_count_bucket", "none") or "none")
    if normalized["people_count_bucket"] not in allowed_people:
        normalized["people_count_bucket"] = "none"

    normalized["motion_type"] = str(normalized.get("motion_type", "static") or "static")
    if normalized["motion_type"] not in allowed_motion:
        normalized["motion_type"] = "static"

    normalized["event_type"] = str(normalized.get("event_type", "unclassified") or "unclassified")
    normalized["severity_reason"] = str(normalized.get("severity_reason", "") or "").strip()
    normalized["needs_attention"] = bool(
        normalized.get("needs_attention", normalized["risk_level"] != "normal" or normalized["anomaly_flag"])
    )
    return normalized


def _structuring_prompt(segment: Dict[str, Any]) -> str:
    """Build the strict-JSON analysis prompt for one segment."""

    payload = {
        "video_id": segment.get("video_id"),
        "segment_id": segment.get("segment_id"),
        "start_sec": segment.get("start_sec"),
        "end_sec": segment.get("end_sec"),
        "raw_caption": segment.get("raw_caption"),
        "normalized_caption": segment.get("normalized_caption"),
        "anomaly_flag": segment.get("anomaly_flag"),
        "anomaly_confidence": segment.get("anomaly_confidence"),
        "anomaly_notes": segment.get("anomaly_notes"),
    }
    return (
        "Return only JSON with keys: "
        "normalized_description, event_type, risk_level, severity_reason, tags, objects, "
        "people_count_bucket, motion_type, needs_attention.\n"
        "Use only the provided caption and anomaly cues. Do not invent hidden facts or motives.\n"
        f"INPUT={json.dumps(payload, ensure_ascii=False)}"
    )


def _repair_llm_payload(data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Repair a partially valid LLM JSON payload into the expected schema subset."""

    src = dict(data or {})
    repaired: Dict[str, Any] = {
        "normalized_description": str(
            src.get("normalized_description")
            or src.get("normalized_caption")
            or src.get("description")
            or ""
        ).strip(),
        "event_type": str(src.get("event_type", "unclassified") or "unclassified"),
        "risk_level": str(src.get("risk_level", "normal") or "normal"),
        "severity_reason": str(src.get("severity_reason", "") or "").strip(),
        "tags": src.get("tags") if isinstance(src.get("tags"), list) else [],
        "objects": src.get("objects") if isinstance(src.get("objects"), list) else [],
        "people_count_bucket": str(src.get("people_count_bucket", "none") or "none"),
        "motion_type": str(src.get("motion_type", "static") or "static"),
        "needs_attention": bool(src.get("needs_attention", False)),
    }
    return validate_segment_schema(repaired)


@dataclass
class SegmentAnalysisService:
    """Segment enrichment service with optional LLM hook and robust fallback."""

    cfg: Any
    llm_client: Optional[LLMClient] = None

    @classmethod
    def from_config(cls, cfg: Any) -> "SegmentAnalysisService":
        """Build the service and attach an LLM client when configured."""

        client: Optional[LLMClient] = None
        try:
            client = LLMClient.from_config(cfg)
        except Exception:
            client = None
        return cls(cfg=cfg, llm_client=client)

    def enrich_segments(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enrich all segments using fallback rules and optional LLM extraction."""

        return [self.enrich_segment(segment) for segment in segments]

    def enrich_segment(self, segment: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich one segment and return the normalized result."""

        payload = validate_segment_schema(dict(segment or {}))
        text = str(
            payload.get("normalized_caption")
            or payload.get("normalized_description")
            or payload.get("description")
            or ""
        )

        llm_payload = self._try_llm_structuring(payload)
        if llm_payload:
            payload.update(llm_payload)
            payload["normalized_caption"] = str(
                payload.get("normalized_caption")
                or payload.get("normalized_description")
                or text
            )
            payload["normalized_description"] = str(
                payload.get("normalized_description")
                or payload.get("normalized_caption")
                or text
            )
            payload["description"] = str(payload.get("normalized_description") or text)
            return validate_segment_schema(payload)

        objects = _infer_objects(text)
        payload["objects"] = objects
        payload["tags"] = _infer_tags(text, anomaly_flag=bool(payload.get("anomaly_flag", False)))
        payload["people_count_bucket"] = _infer_people_bucket(text)
        payload["motion_type"] = _infer_motion_type(text)
        payload["event_type"] = _infer_event_type(
            text,
            objects,
            anomaly_flag=bool(payload.get("anomaly_flag", False)),
        )
        risk_level, severity_reason = _infer_risk(
            text,
            anomaly_flag=bool(payload.get("anomaly_flag", False)),
            anomaly_confidence=float(payload.get("anomaly_confidence", 0.0) or 0.0),
            anomaly_notes=list(payload.get("anomaly_notes") or []),
        )
        payload["risk_level"] = risk_level
        payload["severity_reason"] = severity_reason
        payload["needs_attention"] = bool(risk_level != "normal" or payload.get("anomaly_flag", False))
        payload["normalized_description"] = text
        payload["description"] = text
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


StructuringService = SegmentAnalysisService
