# app/lib/formatters.py
"""
Formatting helpers for SmartCampus V2T Streamlit UI.

Purpose:
- Normalize text, metrics, and artifact payloads into UI-friendly shapes.
- Keep display formatting logic out of page and component modules.
"""

from __future__ import annotations

import html
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional


def E(value: Any) -> str:
    """Escape text for safe HTML rendering."""

    return html.escape("" if value is None else str(value), quote=True)


def mmss(sec: float) -> str:
    """Format seconds as M:SS."""

    total = max(0, int(round(float(sec or 0.0))))
    return f"{total // 60}:{total % 60:02d}"


def hms(sec: float) -> str:
    """Format seconds as H:MM:SS or M:SS."""

    total = max(0, int(round(float(sec or 0.0))))
    hours = total // 3600
    mins = (total % 3600) // 60
    secs = total % 60
    if hours > 0:
        return f"{hours}:{mins:02d}:{secs:02d}"
    return f"{mins}:{secs:02d}"


def fmt_bytes(num: Optional[float]) -> str:
    """Format byte counts for UI display."""

    if num is None:
        return "-"
    value = float(num)
    units = ["B", "KB", "MB", "GB", "TB"]
    idx = 0
    while value >= 1024.0 and idx < len(units) - 1:
        value /= 1024.0
        idx += 1
    if idx == 0:
        return f"{int(value)} {units[idx]}"
    return f"{value:.2f} {units[idx]}"


def first_sentence(text: str) -> str:
    """Extract a short headline from a longer summary."""

    clean = " ".join(str(text or "").split()).strip()
    if not clean:
        return ""
    for sep in (". ", "! ", "? ", " - "):
        idx = clean.find(sep)
        if idx > 0:
            return clean[:idx].strip(" .")
    return clean[:160].strip()


def clip_text(text: str, limit: int = 260) -> str:
    """Clip long text for compact card display."""

    clean = " ".join(str(text or "").split()).strip()
    if len(clean) <= limit:
        return clean
    return clean[: limit - 3].rstrip() + "..."


def humanize_token(value: Any) -> str:
    """Convert snake_case tokens into compact display labels."""

    text = str(value or "").strip()
    if not text:
        return ""
    return " ".join(part for part in text.replace("-", "_").split("_") if part).replace("  ", " ").strip().title()


def variant_token(value: Optional[str]) -> str:
    """Map optional variant ids to a stable UI token."""

    text = str(value or "").strip().lower()
    return text if text else "__base__"


def variant_from_token(token: Optional[str]) -> Optional[str]:
    """Map a UI token back to an optional variant id."""

    text = str(token or "").strip().lower()
    return None if (not text or text == "__base__") else text


def variant_label(token: Optional[str]) -> str:
    """Human-readable variant label."""

    variant = variant_from_token(token)
    if variant is None:
        return "BASE"
    aliases = {
        "fast": "Fast",
        "throughput": "Throughput",
        "max_quality": "Max Quality",
    }
    return aliases.get(variant, humanize_token(variant))


def video_variant_tokens(video_item: Optional[Dict[str, Any]]) -> List[str]:
    """Build available variant tokens for a video."""

    tokens = ["__base__"]
    if not isinstance(video_item, dict):
        return tokens
    variants = video_item.get("variants") or {}
    if not isinstance(variants, dict):
        return tokens
    for key in sorted(str(item).strip().lower() for item in variants.keys() if str(item).strip()):
        token = variant_token(key)
        if token not in tokens:
            tokens.append(token)
    return tokens


def collect_available_languages(video_item: Dict[str, Any], selected_variant: Optional[str]) -> List[str]:
    """Return available output languages for the current selection."""

    if selected_variant:
        variants = video_item.get("variants") or {}
        bucket = variants.get(selected_variant) if isinstance(variants, dict) else {}
        langs = bucket.get("languages") if isinstance(bucket, dict) else []
    else:
        langs = video_item.get("languages") or []
    clean = [str(item).strip().lower() for item in langs if str(item).strip()]
    return clean or ["en"]


def hit_capsules(hit: Dict[str, Any]) -> List[str]:
    """Build compact search-result capsules from structured hit fields."""

    tags: List[str] = []
    for key in ("event_type", "risk_level", "people_count_bucket", "motion_type"):
        value = str(hit.get(key) or "").strip()
        if value:
            tags.append(value)
    if bool(hit.get("anomaly_flag", False)):
        tags.append("anomaly")
    variant = str(hit.get("variant") or "").strip()
    if variant:
        tags.append(variant)
    return tags[:6]


def collect_scene_capsules(outputs: Dict[str, Any]) -> List[str]:
    """Derive scene-level capsules from structured outputs."""

    annotations = outputs.get("annotations") if isinstance(outputs.get("annotations"), list) else []
    event_counter: Counter[str] = Counter()
    risk_counter: Counter[str] = Counter()
    people_counter: Counter[str] = Counter()
    motion_counter: Counter[str] = Counter()
    anomaly_notes: List[str] = []
    anomaly_count = 0

    for ann in annotations:
        if not isinstance(ann, dict):
            continue
        for key, counter in (
            ("event_type", event_counter),
            ("risk_level", risk_counter),
            ("people_count_bucket", people_counter),
            ("motion_type", motion_counter),
        ):
            value = str(ann.get(key) or "").strip()
            if value:
                counter[value] += 1
        if bool(ann.get("anomaly_flag", False)):
            anomaly_count += 1
        notes = ann.get("anomaly_notes") or []
        if isinstance(notes, list):
            for note in notes:
                text = str(note).strip()
                if text:
                    anomaly_notes.append(text)

    tags: List[str] = []
    if event_counter:
        tags.extend([item for item, _ in event_counter.most_common(2)])
    if risk_counter:
        tags.append(next(iter(risk_counter.most_common(1)))[0])
    if people_counter:
        tags.append(next(iter(people_counter.most_common(1)))[0])
    if motion_counter:
        tags.append(next(iter(motion_counter.most_common(1)))[0])
    if anomaly_count > 0:
        if anomaly_notes:
            note_counts = Counter(anomaly_notes)
            tags.extend([item for item, _ in note_counts.most_common(2)])
        else:
            tags.append("anomaly detected")

    run_manifest = outputs.get("run_manifest") if isinstance(outputs.get("run_manifest"), dict) else {}
    profile = str(run_manifest.get("profile") or "").strip()
    variant = str(outputs.get("variant") or "").strip()
    if profile:
        tags.append(profile)
    if variant:
        tags.append(variant)

    deduped: List[str] = []
    seen = set()
    for item in tags:
        key = str(item).strip().lower()
        if key and key not in seen:
            seen.add(key)
            deduped.append(str(item))
    return deduped[:8]


def available_variant_ids(cfg: Any) -> List[str]:
    """List configured experimental variant ids with filesystem fallback."""

    variant_ids = list(getattr(getattr(cfg, "experiment", None), "variant_ids", []) or [])
    clean = [str(item).strip().lower() for item in variant_ids if str(item).strip()]
    if clean:
        return clean
    variants_dir = Path(getattr(getattr(cfg, "paths", None), "variants_dir", ""))
    if variants_dir.exists():
        return sorted([path.stem.lower() for path in variants_dir.glob("*.yaml")])
    return []


def available_profile_ids(cfg: Any) -> List[str]:
    """List configured profile ids with a stable, user-facing order."""

    profiles_dir = Path(getattr(getattr(cfg, "paths", None), "profiles_dir", ""))
    discovered = sorted([path.stem.lower() for path in profiles_dir.glob("*.yaml")]) if profiles_dir.exists() else []
    preferred = ["main", "experimental"]
    ordered = [profile for profile in preferred if profile in discovered]
    for profile in discovered:
        if profile not in ordered:
            ordered.append(profile)
    return ordered or preferred
