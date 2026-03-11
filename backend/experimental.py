# backend/experimental.py
"""
Experimental run helpers for SmartCampus V2T backend.

Purpose:
- Build child jobs and manifests for multi-variant experiment execution.
- Keep experimental orchestration logic out of the main worker loop.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.deps import now_ts, read_json
from src.utils.video_store import batch_manifest_path, write_batch_manifest


def should_expand_experimental_job(job_type: str, cfg: Any, variant_id: Optional[str]) -> bool:
    """Return whether a process job should fan out into multiple variants."""

    return (
        str(job_type) == "process"
        and str(getattr(cfg, "active_profile", "")) == "experimental"
        and variant_id is None
        and bool(getattr(getattr(cfg, "experiment", None), "compare_on_single_video", False))
    )


def build_variant_child_specs(
    *,
    cfg: Any,
    job: Dict[str, Any],
    extra: Dict[str, Any],
    language: str,
    source_language: Optional[str],
) -> List[Dict[str, Any]]:
    """Build child job specs for each configured experimental variant."""

    specs: List[Dict[str, Any]] = []
    for variant_name in getattr(cfg.experiment, "variant_ids", []) or []:
        child_extra = dict(extra or {})
        child_extra["profile"] = cfg.active_profile
        child_extra["variant"] = str(variant_name)
        specs.append(
            {
                "video_id": str(job.get("video_id") or ""),
                "job_type": "process",
                "profile": str(cfg.active_profile),
                "variant": str(variant_name),
                "language": str(language),
                "source_language": source_language,
                "extra": child_extra,
            }
        )
    return specs


def build_batch_manifest_payload(cfg: Any, parent_job_id: str, children: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build the initial experimental batch manifest payload."""

    variants: Dict[str, Any] = {}
    for child in children:
        variants[str(child["variant"])] = {
            "job_id": str(child["job_id"]),
            "status": "queued",
            "updated_at": now_ts(),
        }
    return {
        "profile": str(cfg.active_profile),
        "config_fingerprint": str(cfg.config_fingerprint),
        "parent_job_id": str(parent_job_id),
        "status": "queued_children",
        "variants": variants,
    }


def update_batch_variant_status(
    videos_dir: Path,
    video_id: str,
    variant: Optional[str],
    status: str,
    **extra: Any,
) -> None:
    """Update the top-level experimental batch manifest for one variant."""

    if not variant:
        return

    current = read_json(batch_manifest_path(videos_dir, video_id), default={}) or {}
    if not isinstance(current, dict):
        current = {}

    variants = current.get("variants")
    if not isinstance(variants, dict):
        variants = {}

    entry = variants.get(str(variant))
    if not isinstance(entry, dict):
        entry = {}

    entry["status"] = str(status)
    entry["updated_at"] = now_ts()
    for key, value in extra.items():
        if value is not None:
            entry[str(key)] = value
    variants[str(variant)] = entry

    overall_status = current.get("status")
    states = [str(item.get("status") or "") for item in variants.values() if isinstance(item, dict)]
    if states:
        if all(state == "done" for state in states):
            overall_status = "done"
        elif any(state == "failed" for state in states):
            overall_status = "partial_failure"
        elif any(state in {"running", "indexing", "translating"} for state in states):
            overall_status = "running"
        else:
            overall_status = current.get("status") or "queued_children"

    write_batch_manifest(
        videos_dir,
        video_id,
        {
            "status": overall_status,
            "variants": variants,
        },
    )
