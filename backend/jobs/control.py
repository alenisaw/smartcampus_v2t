# backend/jobs/control.py
"""
Job control helpers for SmartCampus V2T backend.

Purpose:
- Centralize job state, queue lock, lease, and webhook side effects.
- Provide one shared control surface for worker and executor modules.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from backend.deps import new_job_id, now_ts
from backend.jobs.store import (
    build_job_record as _build_job_record,
    enqueue_job,
    job_path,
    list_queue_items,
    parse_job_id_from_queue_file as parse_job_id_from_queue_item,
    queue_status,
    read_job,
    write_job,
)


def job_state(job: Dict[str, Any]) -> str:
    """Return the normalized state string for a job."""

    return str(job.get("state", "queued"))


def is_terminal(job: Dict[str, Any]) -> bool:
    """Return whether the job is in a terminal state."""

    return job_state(job) in {"done", "failed", "canceled"}


def lock_path(paths: Any, job_id: str) -> Path:
    """Return the worker lock path for a job."""

    return paths.locks_dir / f"{job_id}.lock"


def try_create_lock(paths: Any, job_id: str, owner: str) -> bool:
    """Try to create an exclusive worker lock file."""

    path = lock_path(paths, job_id)
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    try:
        fd = os.open(str(path), flags)
        try:
            os.write(fd, f"{owner}\n{now_ts()}\n".encode("utf-8", errors="ignore"))
        finally:
            os.close(fd)
        return True
    except FileExistsError:
        return False
    except Exception:
        return False


def remove_lock_if_exists(paths: Any, job_id: str) -> None:
    """Best-effort lock cleanup."""

    try:
        lock_path(paths, job_id).unlink(missing_ok=True)
    except Exception:
        pass


def lease_expired(job: Dict[str, Any]) -> bool:
    """Return whether the current job lease has expired."""

    lease = job.get("lease") or {}
    expires_at = lease.get("expires_at")
    try:
        return (expires_at is None) or (float(expires_at) <= now_ts())
    except Exception:
        return True


def try_lease(paths: Any, job_id: str, owner: str, lease_sec: float) -> bool:
    """Acquire or refresh a job lease if it is available."""

    job = read_job(paths, job_id)
    if is_terminal(job):
        return False

    lease = job.get("lease") or {}
    current_owner = lease.get("owner")
    if current_owner and current_owner != owner and not lease_expired(job):
        return False

    job["lease"] = {"owner": owner, "expires_at": now_ts() + float(lease_sec)}
    job["state"] = "leased"
    job["stage"] = "leased"
    job["updated_at"] = now_ts()
    write_job(paths, job)
    return True


def check_cancel(paths: Any, job_id: str) -> bool:
    """Return whether a cancel was requested for the job."""

    job = read_job(paths, job_id)
    return bool(job.get("cancel_requested", False))


def set_state(
    paths: Any,
    job_id: str,
    state: str,
    *,
    stage: Optional[str] = None,
    progress: Optional[float] = None,
    message: Optional[str] = None,
    error: Optional[str] = None,
) -> None:
    """Update job state, timestamps, and optional error payload."""

    job = read_job(paths, job_id)
    job["state"] = state
    if stage is not None:
        job["stage"] = stage
    if progress is not None:
        job["progress"] = float(progress)
    if message is not None:
        job["message"] = message
    job["updated_at"] = now_ts()
    if state == "running" and job.get("started_at") is None:
        job["started_at"] = now_ts()
    if state in {"done", "failed", "canceled"}:
        job["finished_at"] = now_ts()
    if error is not None:
        job["error"] = {"type": "WorkerError", "message": error}
    write_job(paths, job)


def create_job(
    paths: Any,
    *,
    video_id: str,
    job_type: str,
    profile: str,
    variant: Optional[str],
    language: str,
    source_language: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create and enqueue a new job record."""

    job_id = new_job_id("job")
    job = _build_job_record(
        job_id=job_id,
        video_id=video_id,
        job_type=job_type,
        profile=profile,
        variant=variant,
        language=language,
        source_language=source_language,
        extra=extra,
    )
    write_job(paths, job)
    enqueue_job(paths, job_id)
    return job


def notify_webhook(webhook_cfg: Dict[str, Any], event: str, job: Dict[str, Any]) -> None:
    """Send a best-effort webhook event for job lifecycle changes."""

    url = str(webhook_cfg.get("url") or "").strip()
    if not url or not bool(webhook_cfg.get("enabled", False)):
        return

    timeout = float(webhook_cfg.get("timeout_sec", 5.0) or 5.0)
    payload = {
        "event": str(event),
        "job_id": job.get("job_id"),
        "video_id": job.get("video_id"),
        "state": job.get("state"),
        "stage": job.get("stage"),
        "progress": job.get("progress"),
        "message": job.get("message"),
        "job_type": job.get("job_type"),
        "language": job.get("language"),
        "source_language": job.get("source_language"),
        "created_at": job.get("created_at"),
        "started_at": job.get("started_at"),
        "updated_at": job.get("updated_at"),
        "finished_at": job.get("finished_at"),
    }
    try:
        requests.post(url, json=payload, timeout=timeout)
    except Exception:
        return


def queue_paused(paths: Any) -> bool:
    """Return whether queue processing is paused."""

    return bool(queue_status(paths).get("paused", False))
