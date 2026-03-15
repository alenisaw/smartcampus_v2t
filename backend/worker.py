# backend/worker.py
"""
Background worker entrypoint for SmartCampus V2T.

Purpose:
- Poll the filesystem queue and lease jobs safely.
- Dispatch process, translate, and index work to dedicated executors.
"""

from __future__ import annotations

import os
import threading
import time
import traceback
from typing import Any, Optional

from backend.deps import get_backend_paths, host_id, load_cfg_and_raw
from backend.jobs.experimental import should_expand_experimental_job
from backend.jobs.control import (
    check_cancel,
    list_queue_items,
    parse_job_id_from_queue_item,
    queue_paused,
    read_job,
    remove_lock_if_exists,
    renew_lease,
    set_state,
    try_create_lock,
    try_lease,
    lease_expired,
)
from backend.jobs.process_runtime import run_process_job
from backend.jobs.runtime_common import run_index_job
from backend.jobs.summary_polish_runtime import run_summary_polish_job
from backend.jobs.store import enqueue_job, find_queue_files_for_job
from backend.jobs.translate_runtime import run_translate_job
from backend.jobs.worker_runtime import (
    _cleanup_services_after_job,
    _prepare_services_for_job,
    build_worker_context,
    expand_experimental_job,
    handle_job_failure,
    mark_job_canceled,
    resolve_worker_context,
)


def _worker_role() -> str:
    """Return the current worker role selector."""

    role = str(os.environ.get("SMARTCAMPUS_WORKER_ROLE", "all") or "all").strip().lower()
    return role or "all"


def _job_allowed_for_role(job_type: str, role: str) -> bool:
    """Check whether a worker role should execute a job type."""

    jt = str(job_type or "").strip().lower()
    rr = str(role or "all").strip().lower() or "all"
    if rr in {"", "all", "any"}:
        return True
    if rr in {"gpu", "worker_gpu"}:
        return jt in {"process", "summary_polish"}
    if rr in {"mt", "worker_mt"}:
        return jt == "translate"
    if rr in {"cpu", "worker_cpu"}:
        return jt in {"index", "summary_polish"}
    return True


def _priority_for_job_type(job_type: str) -> str:
    """Return the default queue priority for a job type."""

    jt = str(job_type or "").strip().lower()
    if jt == "translate":
        return "020"
    if jt == "summary_polish":
        return "025"
    if jt == "index":
        return "030"
    return "010"


def _requeue_for_other_role(paths: Any, job_id: str, job_type: str) -> None:
    """Put a job back to queue when the current worker role does not match."""

    priority = _priority_for_job_type(job_type)
    tag = str(int(time.time() * 1000))
    paths.queue_dir.mkdir(parents=True, exist_ok=True)
    (paths.queue_dir / f"p{priority}__{tag}__{job_id}.q").write_text(job_id, encoding="utf-8")


def _recover_stale_jobs(paths: Any) -> None:
    """Requeue jobs whose lease expired after a worker crash or forced stop."""

    for path in sorted(paths.jobs_dir.glob("job_*.json")):
        try:
            job = read_job(paths, path.stem)
        except Exception:
            continue
        state = str(job.get("state") or "").strip().lower()
        if state not in {"running", "leased", "indexing"}:
            continue
        if not lease_expired(job):
            continue

        job_id = str(job.get("job_id") or path.stem)
        if not find_queue_files_for_job(paths, job_id):
            enqueue_job(paths, job_id, priority=_priority_for_job_type(str(job.get("job_type") or "")))
        set_state(
            paths,
            job_id,
            "queued",
            stage="queued",
            progress=float(job.get("progress") or 0.0),
            message="Recovered after stale lease",
        )
        remove_lock_if_exists(paths, job_id)


def _start_lease_heartbeat(
    paths: Any,
    *,
    job_id: str,
    owner: str,
    lease_sec: float,
    heartbeat_sec: float,
) -> tuple[threading.Event, threading.Thread]:
    """Start a background lease refresher for one active job."""

    stop_event = threading.Event()

    def _run() -> None:
        while not stop_event.wait(max(0.5, float(heartbeat_sec))):
            try:
                if not renew_lease(paths, job_id, owner, lease_sec):
                    return
            except Exception:
                return

    thread = threading.Thread(target=_run, name=f"lease-heartbeat-{job_id}", daemon=True)
    thread.start()
    return stop_event, thread


def _lease_next_job(paths: Any, owner: str, lease_sec: float) -> Optional[str]:
    """Take the next queue item, lock it, and return the leased job id."""

    items = list_queue_items(paths.queue_dir)
    if not items:
        return None

    queue_item = items[0]
    job_id = parse_job_id_from_queue_item(queue_item)
    if not job_id:
        try:
            queue_item.unlink(missing_ok=True)
        except Exception:
            pass
        return None

    if not try_create_lock(paths, job_id, owner):
        try:
            if lease_expired(read_job(paths, job_id)):
                remove_lock_if_exists(paths, job_id)
        except Exception:
            pass
        time.sleep(0.1)
        return None

    try:
        queue_item.unlink(missing_ok=True)
    except Exception:
        pass

    if not try_lease(paths, job_id, owner, lease_sec):
        remove_lock_if_exists(paths, job_id)
        return None

    return job_id


def _dispatch_job(
    *,
    context: Any,
    default_profile: str,
    default_variant: Optional[str],
    paths: Any,
    job_id: str,
    role: str,
) -> Any:
    """Resolve the effective context and execute one leased job."""

    set_state(paths, job_id, "running", stage="starting", progress=0.01, message="Started")

    job = read_job(paths, job_id)
    context = resolve_worker_context(
        context=context,
        default_profile=default_profile,
        default_variant=default_variant,
        job=job,
    )
    cfg = context.cfg

    extra = job.get("extra") or {}
    job_type = str(job.get("job_type") or extra.get("job_type") or "process").strip().lower() or "process"
    language = str(job.get("language") or extra.get("language") or cfg.model.language).strip().lower()
    source_language = str(job.get("source_language") or extra.get("source_language") or "").strip().lower() or None
    device = str(extra.get("device") or cfg.model.device)
    force_overwrite = bool(extra.get("force_overwrite", cfg.runtime.overwrite_existing))
    index_only = bool(extra.get("index_only", False)) or job_type == "index"

    if not _job_allowed_for_role(job_type, role):
        _requeue_for_other_role(paths, job_id, job_type)
        set_state(
            paths,
            job_id,
            "queued",
            stage="queued",
            progress=0.0,
            message=f"Waiting for worker role for job_type={job_type}",
        )
        remove_lock_if_exists(paths, job_id)
        time.sleep(0.1)
        return context

    if should_expand_experimental_job(job_type, cfg, cfg.active_variant):
        expand_experimental_job(
            cfg=cfg,
            paths=paths,
            job_id=job_id,
            job=job,
            language=language,
            source_language=source_language,
        )
        return context

    if check_cancel(paths, job_id):
        mark_job_canceled(paths, job_id, context.webhook_cfg, "Canceled before start")
        return context

    _prepare_services_for_job(context.services, job_type)
    try:
        if index_only:
            run_index_job(
                cfg=cfg,
                paths=paths,
                cfg_fp=context.cfg_fp,
                job_id=job_id,
                webhook_cfg=context.webhook_cfg,
            )
            return context

        video_id = str(job.get("video_id") or "")
        if not video_id:
            raise RuntimeError("Missing video_id in job")

        if job_type == "translate":
            run_translate_job(
                cfg=cfg,
                paths=paths,
                cfg_fp=context.cfg_fp,
                job_id=job_id,
                job=job,
                language=language,
                source_language=source_language,
                force_overwrite=force_overwrite,
                auto_index=context.auto_index,
                webhook_cfg=context.webhook_cfg,
                translation_service=context.services.translation_service,
            )
            return context

        if job_type == "summary_polish":
            run_summary_polish_job(
                cfg=cfg,
                paths=paths,
                job_id=job_id,
                job=job,
                webhook_cfg=context.webhook_cfg,
                summary_service=context.services.summary_service,
                guard_service=context.services.guard_service,
            )
            return context

        run_process_job(
            cfg=cfg,
            paths=paths,
            cfg_fp=context.cfg_fp,
            job_id=job_id,
            job=job,
            device=device,
            force_overwrite=force_overwrite,
            auto_index=context.auto_index,
            webhook_cfg=context.webhook_cfg,
            services=context.services,
        )
    finally:
        _cleanup_services_after_job(context.services)
    return context


def worker_main() -> None:
    """Run the main worker loop."""

    cfg, raw = load_cfg_and_raw()
    default_profile = cfg.active_profile
    default_variant = cfg.active_variant
    paths = get_backend_paths(cfg, raw)

    paths.jobs_dir.mkdir(parents=True, exist_ok=True)
    paths.queue_dir.mkdir(parents=True, exist_ok=True)
    paths.locks_dir.mkdir(parents=True, exist_ok=True)

    worker_cfg = raw.get("worker") or {}
    poll_interval = float(worker_cfg.get("poll_interval_sec", 1))
    lease_sec = float(worker_cfg.get("lease_sec", 30))
    heartbeat_sec = float(worker_cfg.get("heartbeat_sec", 5))
    role = _worker_role()
    owner = host_id()
    context = build_worker_context(cfg, raw)

    print(
        f"[worker] owner={owner} role={role} jobs={paths.jobs_dir} "
        f"queue={paths.queue_dir} locks={paths.locks_dir}"
    )
    active_job_id: Optional[str] = None
    heartbeat_stop: Optional[threading.Event] = None
    heartbeat_thread: Optional[threading.Thread] = None

    while True:
        try:
            if queue_paused(paths):
                time.sleep(max(0.4, poll_interval))
                continue

            _recover_stale_jobs(paths)
            job_id = _lease_next_job(paths, owner, lease_sec)
            if not job_id:
                time.sleep(poll_interval)
                continue

            active_job_id = job_id
            heartbeat_stop, heartbeat_thread = _start_lease_heartbeat(
                paths,
                job_id=job_id,
                owner=owner,
                lease_sec=lease_sec,
                heartbeat_sec=heartbeat_sec,
            )
            context = _dispatch_job(
                context=context,
                default_profile=default_profile,
                default_variant=default_variant,
                paths=paths,
                job_id=job_id,
                role=role,
            )
            if heartbeat_stop is not None:
                heartbeat_stop.set()
            if heartbeat_thread is not None:
                heartbeat_thread.join(timeout=0.2)
            heartbeat_stop = None
            heartbeat_thread = None
            active_job_id = None

        except Exception as exc:
            error_text = f"{exc}\n{traceback.format_exc()}"
            if heartbeat_stop is not None:
                heartbeat_stop.set()
            if heartbeat_thread is not None:
                heartbeat_thread.join(timeout=0.2)
            heartbeat_stop = None
            heartbeat_thread = None
            if active_job_id:
                try:
                    handle_job_failure(
                        cfg=cfg,
                        paths=paths,
                        job_id=active_job_id,
                        error=exc,
                        webhook_cfg=context.webhook_cfg,
                    )
                except Exception:
                    pass
                active_job_id = None
            print(f"[worker] ERROR:\n{error_text}")
            time.sleep(0.5)


if __name__ == "__main__":
    worker_main()
