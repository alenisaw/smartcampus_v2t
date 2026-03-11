# backend/job_queue_runtime.py
"""
Filesystem job and queue helpers for SmartCampus V2T backend.

Purpose:
- Manage persisted job records and filesystem queue state for API routes.
- Keep queue ordering, pause state, and running-job discovery out of `backend/api.py`.
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
    extra: Dict[str, Any],
) -> Dict[str, Any]:
    """Build a normalized queued job record for API submission."""

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


def enqueue_job(paths: Any, job_id: str) -> None:
    """Append one processing job to the filesystem queue."""

    paths.queue_dir.mkdir(parents=True, exist_ok=True)
    qname = f"p010__{time_tag()}__{job_id}.q"
    (paths.queue_dir / qname).write_text(job_id, encoding="utf-8")


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
    for qf in sorted(paths.queue_dir.glob("p*__*__*.q")):
        job_id = parse_job_id_from_queue_file(qf)
        if job_id:
            out.append((qf, job_id))
    return out


def queue_snapshot(paths: Any) -> Dict[str, Any]:
    """Build a queue status snapshot with queued and running job payloads."""

    queued: List[Dict[str, Any]] = []
    for _queue_path, job_id in queue_files_with_ids(paths):
        try:
            job = read_json(job_path(paths, job_id), default={}) or {}
        except Exception:
            job = {}
        queued.append(
            {
                "job_id": str(job_id),
                "video_id": job.get("video_id"),
                "job_type": job.get("job_type"),
                "profile": job.get("profile"),
                "variant": job.get("variant"),
                "language": job.get("language"),
                "state": job.get("state"),
                "created_at": job.get("created_at"),
            }
        )

    running = find_running_job(paths)
    running_payload: Optional[Dict[str, Any]] = None
    if isinstance(running, dict) and running.get("job_id"):
        running_payload = {
            "job_id": str(running.get("job_id")),
            "video_id": running.get("video_id"),
            "job_type": running.get("job_type"),
            "profile": running.get("profile"),
            "variant": running.get("variant"),
            "language": running.get("language"),
            "state": str(running.get("state") or "running"),
            "stage": running.get("stage"),
            "progress": float(running.get("progress") or 0.0),
            "message": running.get("message"),
            "created_at": running.get("created_at"),
            "started_at": running.get("started_at"),
            "updated_at": running.get("updated_at"),
        }

    return {
        "status": queue_status(paths),
        "queued": queued,
        "running": running_payload,
    }


def move_job_in_queue(paths: Any, job_id: str, direction: str, steps: int) -> Dict[str, Any]:
    """Reorder one queued job and return move metadata."""

    normalized_direction = str(direction or "").strip().lower()
    if normalized_direction not in {"up", "down", "top", "bottom"}:
        raise HTTPException(status_code=400, detail="direction must be one of: up, down, top, bottom")

    items = queue_files_with_ids(paths)
    ordered = [queued_job_id for _, queued_job_id in items]
    if not ordered:
        raise HTTPException(status_code=404, detail="Queue is empty")

    normalized_job_id = str(job_id)
    if normalized_job_id not in ordered:
        raise HTTPException(status_code=404, detail=f"Job not found in queue: {normalized_job_id}")

    old_index = ordered.index(normalized_job_id)
    step_count = max(1, int(steps or 1))

    if normalized_direction == "top":
        new_index = 0
    elif normalized_direction == "bottom":
        new_index = len(ordered) - 1
    elif normalized_direction == "up":
        new_index = max(0, old_index - step_count)
    else:
        new_index = min(len(ordered) - 1, old_index + step_count)

    if new_index != old_index:
        ordered.pop(old_index)
        ordered.insert(new_index, normalized_job_id)
        rewrite_queue_order(paths, ordered)

    return {
        "job_id": normalized_job_id,
        "old_index": int(old_index),
        "new_index": int(new_index),
        "queued_count": len(ordered),
    }


def cancel_queued_job(paths: Any, job_id: str) -> Dict[str, Any]:
    """Remove one non-running queued job and persist the canceled state."""

    job = read_job_or_404(paths, job_id)
    state = str(job.get("state") or "unknown")
    if state in {"running", "leased", "indexing"}:
        raise HTTPException(status_code=409, detail=f"Job is already running: {job_id} ({state})")

    for queue_file in find_queue_files_for_job(paths, job_id):
        try:
            queue_file.unlink(missing_ok=True)
        except Exception:
            pass

    timestamp = now_ts()
    job["state"] = "canceled"
    job["stage"] = "canceled"
    job["progress"] = 0.0
    job["message"] = "Canceled (removed from queue)"
    job["cancel_requested"] = True
    job["updated_at"] = timestamp
    job["finished_at"] = timestamp
    write_job(paths, job)
    return {"ok": True, "job_id": str(job_id), "state": "canceled"}


def rewrite_queue_order(paths: Any, ordered_job_ids: List[str]) -> None:
    """Rewrite queue file order using the provided job ordering."""

    paths.queue_dir.mkdir(parents=True, exist_ok=True)

    existing = queue_files_with_ids(paths)
    id_to_paths: Dict[str, List[Path]] = {}
    for path, job_id in existing:
        id_to_paths.setdefault(job_id, []).append(path)

    for _job_id, path_list in id_to_paths.items():
        for extra in path_list[1:]:
            try:
                extra.unlink(missing_ok=True)
            except Exception:
                pass

    tmp_map: Dict[str, Path] = {}
    for index, job_id in enumerate(ordered_job_ids):
        srcs = id_to_paths.get(job_id) or []
        if not srcs:
            continue
        src = srcs[0]
        tmp = paths.queue_dir / f"_tmp__{job_id}__{index}.q"
        try:
            src.replace(tmp)
            tmp_map[job_id] = tmp
        except Exception:
            tmp_map[job_id] = src

    tag = time_tag()
    for index, job_id in enumerate(ordered_job_ids):
        src = tmp_map.get(job_id)
        if not src:
            continue
        dst = paths.queue_dir / f"p{index:03d}__{tag}__{job_id}.q"
        try:
            src.replace(dst)
        except Exception:
            try:
                dst.write_text(job_id, encoding="utf-8")
                if src.exists():
                    src.unlink(missing_ok=True)
            except Exception:
                pass


def find_running_job(paths: Any) -> Optional[Dict[str, Any]]:
    """Return the most recently updated running-like job."""

    states = {"running", "leased", "indexing"}
    best: Optional[Dict[str, Any]] = None
    best_key = -1.0

    for path in sorted(paths.jobs_dir.glob("job_*.json")):
        job = read_json(path, default=None)
        if not isinstance(job, dict):
            continue
        state = str(job.get("state") or "")
        if state not in states:
            continue
        key = float(job.get("updated_at") or job.get("started_at") or job.get("created_at") or 0.0)
        if key > best_key:
            best_key = key
            best = job
    return best
