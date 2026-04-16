# backend/jobs/runtime_common.py
"""
Shared runtime helpers for SmartCampus V2T worker job execution.

Purpose:
- Hold shared worker service wiring and job-runtime helpers used by multiple job families.
- Keep common execution utilities out of process- and translate-specific runtime modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from backend.deps import atomic_write_json
from backend.jobs.index_runtime import rebuild_index_status as _rebuild_index_status
from backend.jobs.control import notify_webhook, read_job, remove_lock_if_exists, set_state
from scripts.collect_metrics import export_metrics_bundle


@dataclass
class WorkerServices:
    """Service bundle reused across worker job executors."""

    pipeline: Any = None
    structuring_service: Any = None
    summary_service: Any = None
    translation_service: Any = None
    guard_service: Any = None


def finalize_job_state(
    paths: Any,
    job_id: str,
    *,
    state: str,
    stage: str,
    progress: float,
    message: str,
    webhook_cfg: Optional[Dict[str, Any]] = None,
    webhook_event: Optional[str] = None,
    error: Optional[str] = None,
) -> None:
    """Persist final job state, optionally notify listeners, and always release the job lock."""

    state_payload = {
        "stage": stage,
        "progress": float(progress),
        "message": message,
    }
    if error is not None:
        state_payload["error"] = error
    set_state(paths, job_id, state, **state_payload)

    if webhook_cfg and webhook_event:
        try:
            notify_webhook(webhook_cfg, webhook_event, read_job(paths, job_id))
        except Exception:
            pass

    remove_lock_if_exists(paths, job_id)


def refresh_research_metrics(cfg: Any) -> None:
    """Refresh the aggregate research metrics snapshot after completed runs."""

    try:
        export_metrics_bundle(
            videos_dir=Path(cfg.paths.videos_dir),
            out_dir=Path(cfg.paths.data_dir) / "research",
        )
    except Exception:
        return


def build_runtime_metric_payload(
    cfg: Any,
    *,
    language: str,
    variant: Optional[str],
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the common metrics payload header used by worker job runtimes."""

    payload: Dict[str, Any] = {
        "language": str(language),
        "profile": cfg.active_profile,
        "variant": variant,
        "config_fingerprint": cfg.config_fingerprint,
        "runtime": {
            "metrics_repeats": int(getattr(cfg.runtime, "metrics_repeats", 1)),
            "metrics_store_samples": bool(getattr(cfg.runtime, "metrics_store_samples", True)),
        },
    }
    if device is not None:
        payload["device"] = str(device)
    return payload


def runtime_keep_samples(cfg: Any) -> bool:
    """Return whether stage timing samples should remain in persisted metrics."""

    return bool(getattr(cfg.runtime, "metrics_store_samples", True))


def run_index_job(
    *,
    cfg: Any,
    paths: Any,
    cfg_fp: str,
    job_id: str,
    language: Optional[str],
    webhook_cfg: Dict[str, Any],
) -> None:
    """Execute a standalone index rebuild job."""

    set_state(paths, job_id, "indexing", stage="indexing", progress=0.2, message="Index rebuild")
    target_language = str(language or "").strip().lower() or None
    if target_language:
        from backend.jobs.index_runtime import build_index_for_language, write_index_state

        payload = build_index_for_language(cfg=cfg, cfg_fp=cfg_fp, language=target_language)
        status = write_index_state(
            paths,
            built=True,
            last_error=None,
            language=target_language,
            payload=payload,
        )
    else:
        status = _rebuild_index_status(cfg=cfg, cfg_fp=cfg_fp)
        atomic_write_json(paths.index_state_path, status)
    finalize_job_state(
        paths,
        job_id,
        state="done",
        stage="indexing",
        progress=1.0,
        message="Index rebuilt",
        webhook_cfg=webhook_cfg,
        webhook_event="job_done",
    )
