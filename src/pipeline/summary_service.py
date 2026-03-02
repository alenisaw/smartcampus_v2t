# src/pipeline/summary_service.py
"""
Summary layer for producing Video Summary Schema v2 artifacts.

Purpose:
- Attempt strict-JSON summary generation through LLMClient when available.
- Fall back to deterministic schema v2 assembly from segments.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.llm.client import LLMClient
from src.pipeline.schema_v2 import build_video_summary_v2


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
                "tags": item.get("tags"),
            }
        )

    return (
        "Return only JSON for Video Summary Schema v2. Use only provided segment facts.\n"
        f"VIDEO={json.dumps({'video_id': video_id, 'duration_sec': duration_sec}, ensure_ascii=False)}\n"
        f"SEGMENTS={json.dumps(compact_segments, ensure_ascii=False)}"
    )


def _repair_summary_payload(data: Optional[Dict[str, Any]], fallback: Dict[str, Any]) -> Dict[str, Any]:
    """Merge a partial LLM summary into a fully valid fallback summary."""

    if not isinstance(data, dict):
        return fallback

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

    if not isinstance(merged.get("key_events"), list):
        merged["key_events"] = fallback.get("key_events", [])
    if not isinstance(merged.get("citations"), list):
        merged["citations"] = fallback.get("citations", [])
    if not isinstance(merged.get("translation_views"), dict):
        merged["translation_views"] = fallback.get("translation_views", {})
    if not isinstance(merged.get("risk_overview"), dict):
        merged["risk_overview"] = fallback.get("risk_overview", {})
    if not isinstance(merged.get("statistics"), dict):
        merged["statistics"] = fallback.get("statistics", {})

    merged["schema_version"] = "2.0"
    merged["video_id"] = fallback.get("video_id")
    merged["language"] = fallback.get("language")
    merged.setdefault("summary", merged.get("global_summary", fallback.get("summary", "")))
    return merged


@dataclass
class SummaryService:
    """Generate a schema v2 video summary with optional LLM and deterministic fallback."""

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
        summary_text: str,
        segments: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Build a summary, preferring strict-JSON LLM output and falling back deterministically."""

        fallback = build_video_summary_v2(
            cfg=self.cfg,
            video_id=video_id,
            language=language,
            duration_sec=duration_sec,
            summary_text=summary_text,
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
