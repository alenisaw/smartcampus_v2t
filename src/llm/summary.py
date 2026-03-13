# src/llm/summary.py
"""
Video-level summary generation for SmartCampus V2T.

Purpose:
- Build canonical video summaries from structured segment facts.
- Keep global summary generation separate from the VLM observation stage.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.guard.schemas import validate_summary_payload
from src.llm.client import LLMClient
from src.core.artifacts import build_video_summary_v2


def _summary_prompt(video_id: str, duration_sec: float, segments: List[Dict[str, Any]]) -> str:
    """Build the strict-JSON summary prompt from compact segment facts."""

    compact_segments = []
    for item in segments[:20]:
        compact_segments.append(
            {
                "segment_id": item.get("segment_id"),
                "start_sec": item.get("start_sec"),
                "end_sec": item.get("end_sec"),
                "normalized_caption": item.get("normalized_caption"),
                "event_type": item.get("event_type"),
                "risk_level": item.get("risk_level"),
                "anomaly_flag": item.get("anomaly_flag"),
                "anomaly_confidence": item.get("anomaly_confidence"),
                "anomaly_notes": item.get("anomaly_notes"),
                "tags": item.get("tags"),
            }
        )

    return (
        "Return only JSON for Video Summary Schema v2. Use only the provided segment facts.\n"
        f"VIDEO={json.dumps({'video_id': video_id, 'duration_sec': duration_sec}, ensure_ascii=False)}\n"
        f"SEGMENTS={json.dumps(compact_segments, ensure_ascii=False)}"
    )


def _fallback_summary_text(segments: List[Dict[str, Any]]) -> str:
    """Build a short deterministic summary from structured segments."""

    if not segments:
        return "No significant activity was detected in the processed video."

    seen: List[str] = []
    for item in segments:
        text = str(item.get("normalized_caption") or item.get("description") or "").strip()
        if text and text not in seen:
            seen.append(text)
        if len(seen) >= 3:
            break

    summary = " ".join(seen).strip()
    attention_segments = [
        item
        for item in segments
        if bool(item.get("anomaly_flag", False)) or str(item.get("risk_level", "normal")) != "normal"
    ]
    if attention_segments:
        summary = (
            f"{summary} "
            f"{len(attention_segments)} segment(s) require attention based on visible anomaly signals."
        ).strip()
    return summary or "No significant activity was detected in the processed video."


def _repair_summary_payload(data: Optional[Dict[str, Any]], fallback: Dict[str, Any]) -> Dict[str, Any]:
    """Merge a partial LLM summary into a fully valid fallback summary."""

    if not isinstance(data, dict):
        return validate_summary_payload(fallback)

    merged = dict(fallback)
    for key in (
        "global_summary",
        "key_events",
        "risk_overview",
        "statistics",
        "citations",
        "translation_views",
        "models",
        "summary",
    ):
        if key in data:
            merged[key] = data[key]

    merged["schema_version"] = "2.0"
    merged["video_id"] = fallback.get("video_id")
    merged["language"] = fallback.get("language")
    merged.setdefault("summary", merged.get("global_summary", fallback.get("summary", "")))
    return validate_summary_payload(merged)


@dataclass
class SummaryService:
    """Generate a video summary with optional LLM and deterministic fallback."""

    cfg: Any
    llm_client: Optional[LLMClient] = None

    @classmethod
    def from_config(cls, cfg: Any) -> "SummaryService":
        """Build the service and attach an LLM client when possible."""

        client: Optional[LLMClient] = None
        try:
            client = LLMClient.from_config(cfg)
        except Exception:
            client = None
        return cls(cfg=cfg, llm_client=client)

    def build_summary(
        self,
        *,
        video_id: str,
        language: str,
        duration_sec: float,
        segments: List[Dict[str, Any]],
        summary_text: str = "",
    ) -> Dict[str, Any]:
        """Build a summary from structured segments, preferring LLM JSON output."""

        fallback_text = str(summary_text or "").strip() or _fallback_summary_text(segments)
        fallback = build_video_summary_v2(
            cfg=self.cfg,
            video_id=video_id,
            language=language,
            duration_sec=duration_sec,
            summary_text=fallback_text,
            segments=segments,
        )
        llm_payload = self._try_llm_summary(video_id=video_id, duration_sec=duration_sec, segments=segments)
        return _repair_summary_payload(llm_payload, fallback)

    def _try_llm_summary(
        self,
        *,
        video_id: str,
        duration_sec: float,
        segments: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Attempt strict-JSON summary generation via LLM."""

        if self.llm_client is None:
            return None
        try:
            return self.llm_client.generate_json(_summary_prompt(video_id, duration_sec, segments))
        except Exception:
            return None
