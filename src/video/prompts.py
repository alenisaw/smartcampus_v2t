# src/video/prompts.py
"""
Prompting and output normalization for the VLM observation layer.

Purpose:
- Keep the VLM prompt strict and observation-only.
- Normalize raw model text into the compact clip-observation contract.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from src.guard.schemas import validate_clip_observation

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_WS_RE = re.compile(r"\s+")
_LIST_PREFIX_RE = re.compile(r"^\s*[-*]+\s*")
_ENUM_PREFIX_RE = re.compile(r"^\s*\(?\d+\)?[.)]\s*")
_GENERIC_OPENERS_RE = re.compile(
    r"^(in the frame|in the image|in the video|we can see|there is|there are|visible is|shown is)\s*[:,\-]?\s*",
    flags=re.IGNORECASE,
)


def build_clip_observation_prompt(lang: str) -> str:
    """Return the strict JSON prompt for clip observation generation."""

    _ = lang
    return (
        "Return only JSON with keys: description, anomaly_flag, anomaly_confidence, anomaly_notes.\n"
        "Rules:\n"
        "- description: 1-2 short natural English sentences about visible activity only.\n"
        "- anomaly_flag: true only if something looks unusual, suspicious, unsafe, or out of the ordinary.\n"
        "- anomaly_confidence: number from 0.0 to 1.0.\n"
        "- anomaly_notes: short list of visible suspicious signals; use [] when nothing stands out.\n"
        "- Use only visible facts about people, vehicles, objects, motion, and scene changes.\n"
        "- Do not guess intent, identity, roles, emotions, cause, or hidden context.\n"
        "- Do not generate event classes, risk classes, summaries, or tags."
    )


def strip_prefix(text: str) -> str:
    """Drop timeline-like prefixes from model text."""

    return re.sub(r"^\[[0-9:\s\-]+\]\s*", "", text or "")


def _collapse_spaces(text: str) -> str:
    """Collapse whitespace runs into one space."""

    normalized = (text or "").replace("\n", " ").replace("\r", " ").replace("\t", " ")
    return _WS_RE.sub(" ", normalized).strip()


def _remove_list_prefixes(text: str) -> str:
    """Drop bullet and numbered-list prefixes from one line."""

    out = _LIST_PREFIX_RE.sub("", text or "")
    out = _ENUM_PREFIX_RE.sub("", out)
    return out.strip()


def _strip_generic_openers(text: str) -> str:
    """Remove generic lead-ins from VLM output."""

    return _GENERIC_OPENERS_RE.sub("", text or "").strip()


def _force_max_sentences(text: str, max_sentences: int = 2) -> str:
    """Clamp free text to at most `max_sentences` sentences."""

    raw = (text or "").strip()
    if not raw:
        return raw
    parts = [part.strip() for part in _SENT_SPLIT_RE.split(raw) if part.strip()]
    kept = parts[: max(1, int(max_sentences))] if parts else [raw]
    out = _collapse_spaces(" ".join(kept))
    if out and out[-1] not in ".!?":
        out += "."
    return out


def _dedupe_repeated_words(text: str, max_repeat: int = 2) -> str:
    """Trim pathological repeated-word fragments."""

    words = (text or "").split()
    out: List[str] = []
    run_word: Optional[str] = None
    run_len = 0
    for word in words:
        key = word.lower()
        if key == run_word:
            run_len += 1
            if run_len >= max_repeat:
                continue
        else:
            run_word = key
            run_len = 0
        out.append(word)
    merged = " ".join(out).strip()
    if merged and merged[-1] not in ".!?":
        merged += "."
    return merged


def sanitize_clip_description(text: str) -> str:
    """Normalize one raw clip description into a short English observation."""

    normalized = strip_prefix(text)
    normalized = _collapse_spaces(normalized)
    normalized = _strip_generic_openers(normalized)
    normalized = _remove_list_prefixes(normalized)
    normalized = _force_max_sentences(normalized, max_sentences=2)
    normalized = _dedupe_repeated_words(normalized, max_repeat=2)
    return normalized


def _parse_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Parse a JSON object from raw text or fenced code blocks."""

    raw = str(text or "").strip()
    if not raw:
        return None

    candidates = [raw]
    if "```" in raw:
        stripped = raw.replace("```json", "```").replace("```JSON", "```")
        parts = [part.strip() for part in stripped.split("```") if part.strip()]
        candidates.extend(parts)

    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(raw[start : end + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def parse_clip_observation(raw_text: str) -> Dict[str, Any]:
    """Parse and validate one raw VLM response into the clip-observation schema."""

    parsed = _parse_json_object(raw_text)
    if isinstance(parsed, dict):
        if "description" not in parsed:
            parsed["description"] = sanitize_clip_description(str(raw_text or ""))
        return validate_clip_observation(parsed)
    return validate_clip_observation(
        {
            "description": sanitize_clip_description(str(raw_text or "")),
            "anomaly_flag": False,
            "anomaly_confidence": 0.0,
            "anomaly_notes": [],
        }
    )
