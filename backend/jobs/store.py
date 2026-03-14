# backend/jobs/store.py
"""
Filesystem job and queue persistence helpers for SmartCampus V2T backend.

Purpose:
- Keep persisted job-record and queue-file primitives in one shared module.
- Remove duplicate filesystem helpers from worker-control and API queue runtime layers.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import HTTPException

from backend.deps import atomic_write_json, now_ts, read_json


def job_path(paths: Any, job_id: str) -> Path:
    """Return the filesystem path for one persisted job record."""

    return paths.jobs_dir / f"{job_id}.json"


def read_job(paths: Any, job_id: str) -> Dict[str, Any]:
    """Load a job payload or raise if the file is invalid."""

    path = job_path(paths, job_id)
    job = read_json(path, default=None)
    if not isinstance(job, dict):
        raise RuntimeError(f"Job file missing or corrupted: {path}")
    return job


def read_job_or_404(paths: Any, job_id: str) -> Dict[str, Any]:
    """Load a job record or raise an HTTP 404 error."""

    job = read_json(job_path(paths, job_id), default=None)
    if not isinstance(job, dict):
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    return job


def write_job(paths: Any, job: Dict[str, Any]) -> None:
    """Persist one job record atomically."""

    atomic_write_json(job_path(paths, str(job["job_id"])), job)


def build_job_record(
    *,
    job_id: str,
    video_id: str,
    job_type: str,
    profile: str,
    variant: Optional[str],
    language: str,
    source_language: Optional[str],
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a normalized queued job record for API submission and worker dispatch."""

    timestamp = now_ts()
    return {
        "job_id": str(job_id),
        "video_id": str(video_id),
        "job_type": str(job_type),
        "profile": str(profile),
        "variant": variant,
        "language": str(language),
        "source_language": source_language,
        "state": "queued",
        "stage": "queued",
        "progress": 0.0,
        "message": "Queued",
        "created_at": timestamp,
        "updated_at": timestamp,
        "started_at": None,
        "finished_at": None,
        "error": None,
        "cancel_requested": False,
        "lease": None,
        "extra": dict(extra or {}),
    }


def time_tag() -> str:
    """Build a stable queue filename timestamp token."""

    return time.strftime("%Y%m%dT%H%M%S", time.localtime())


def enqueue_job(paths: Any, job_id: str, *, priority: str = "010") -> None:
    """Append one job to the filesystem queue."""

    paths.queue_dir.mkdir(parents=True, exist_ok=True)
    qname = f"p{str(priority)}__{time_tag()}__{job_id}.q"
    (paths.queue_dir / qname).write_text(job_id, encoding="utf-8")


def list_queue_items(queue_dir: Path) -> List[Path]:
    """Return queue items in stable filesystem order."""

    return sorted(queue_dir.glob("p*__*__*.q"))


def parse_job_id_from_queue_file(qf: Path) -> Optional[str]:
    """Extract the job id from one queue filename."""

    try:
        parts = qf.name.split("__")
        if len(parts) >= 3:
            return parts[2].replace(".q", "")
    except Exception:
        return None
    return None


def queue_files_with_ids(paths: Any) -> List[Tuple[Path, str]]:
    """Return queue files paired with parsed job ids."""

    out: List[Tuple[Path, str]] = []
    for qf in list_queue_items(paths.queue_dir):
        job_id = parse_job_id_from_queue_file(qf)
        if job_id:
            out.append((qf, job_id))
    return out


def find_queue_files_for_job(paths: Any, job_id: str) -> List[Path]:
    """List queue files that reference a specific job id."""

    return sorted(paths.queue_dir.glob(f"p*__*__{job_id}.q"))


def queue_status(paths: Any) -> Dict[str, Any]:
    """Load queue pause state with defaults."""

    state = read_json(paths.queue_state_path, default={"paused": False, "updated_at": None}) or {}
    if "paused" not in state:
        state["paused"] = False
    if "updated_at" not in state:
        state["updated_at"] = now_ts()
    return state


def set_queue_paused(paths: Any, paused: bool) -> Dict[str, Any]:
    """Persist queue pause state."""

    state = queue_status(paths)
    state["paused"] = bool(paused)
    state["updated_at"] = now_ts()
    atomic_write_json(paths.queue_state_path, state)
    return state
