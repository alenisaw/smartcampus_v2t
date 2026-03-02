# backend/worker.py
"""
Background worker entrypoint for SmartCampus V2T.

Purpose:
- Poll the filesystem queue and lease jobs safely.
- Expand experimental batch jobs into variant runs.
- Dispatch index, translate, and process jobs to dedicated executors.
"""

from __future__ import annotations

import time
import traceback
from typing import Optional

from backend.deps import get_backend_paths, host_id, load_cfg_and_raw
from backend.experimental import should_expand_experimental_job
from backend.job_control import (
    check_cancel,
    list_queue_items,
    parse_job_id_from_queue_item,
    queue_paused,
    read_job,
    remove_lock_if_exists,
    set_state,
    try_create_lock,
    try_lease,
    lease_expired,
)
from backend.job_executors import run_index_job, run_process_job, run_translate_job
from backend.worker_runtime import (
    build_worker_context,
    expand_experimental_job,
    handle_job_failure,
    mark_job_canceled,
    resolve_worker_context,
)


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
    owner = host_id()
    context = build_worker_context(cfg, raw)

    print(f"[worker] owner={owner} jobs={paths.jobs_dir} queue={paths.queue_dir} locks={paths.locks_dir}")
    active_job_id: Optional[str] = None

    while True:
        try:
            if queue_paused(paths):
                time.sleep(max(0.4, poll_interval))
                continue

            items = list_queue_items(paths.queue_dir)
            if not items:
                time.sleep(poll_interval)
                continue

            queue_item = items[0]
            job_id = parse_job_id_from_queue_item(queue_item)
            if not job_id:
                try:
                    queue_item.unlink(missing_ok=True)
                except Exception:
                    pass
                continue

            if not try_create_lock(paths, job_id, owner):
                try:
                    if lease_expired(read_job(paths, job_id)):
                        remove_lock_if_exists(paths, job_id)
                except Exception:
                    pass
                time.sleep(0.1)
                continue

            try:
                queue_item.unlink(missing_ok=True)
            except Exception:
                pass

            if not try_lease(paths, job_id, owner, lease_sec):
                remove_lock_if_exists(paths, job_id)
                continue

            active_job_id = job_id
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

            if should_expand_experimental_job(job_type, cfg, cfg.active_variant):
                expand_experimental_job(
                    cfg=cfg,
                    paths=paths,
                    job_id=job_id,
                    job=job,
                    language=language,
                    source_language=source_language,
                )
                active_job_id = None
                continue

            if check_cancel(paths, job_id):
                mark_job_canceled(paths, job_id, context.webhook_cfg, "Canceled before start")
                active_job_id = None
                continue

            if index_only:
                run_index_job(
                    cfg=cfg,
                    paths=paths,
                    cfg_fp=context.cfg_fp,
                    job_id=job_id,
                    webhook_cfg=context.webhook_cfg,
                )
                active_job_id = None
                continue

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
                active_job_id = None
                continue

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
            active_job_id = None

        except Exception as exc:
            error_text = f"{exc}\n{traceback.format_exc()}"
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
