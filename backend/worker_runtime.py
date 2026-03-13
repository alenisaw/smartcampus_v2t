# backend/worker_runtime.py
"""
Worker runtime helpers for SmartCampus V2T backend.

Purpose:
- Build the effective runtime context for each job.
- Keep expansion, cancellation, and failure side effects out of the worker loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from backend.deps import load_cfg_and_raw
from backend.experimental import (
    build_batch_manifest_payload,
    build_variant_child_specs,
    update_batch_variant_status,
)
from backend.index_runtime import write_index_state as _write_index_state
from backend.job_control import create_job, notify_webhook, read_job, remove_lock_if_exists, set_state, write_job
from backend.job_executors import WorkerServices
from src.guard.service import GuardService
from src.llm.analyze import StructuringService
from src.llm.summary import SummaryService
from src.video.describe import VideoToTextPipeline
from src.search.builder import search_config_fingerprint
from src.translation.service import TranslationService
from src.utils.video_store import update_outputs_manifest, write_batch_manifest


@dataclass
class WorkerContext:
    """Resolved runtime dependencies for one effective config."""

    cfg: Any
    raw: Dict[str, Any]
    cfg_fp: str
    auto_index: bool
    webhook_cfg: Dict[str, Any]
    services: WorkerServices


def _build_services(cfg: Any) -> WorkerServices:
    """Build the service bundle for the current effective config."""

    return WorkerServices(
        pipeline=VideoToTextPipeline(cfg),
        structuring_service=StructuringService.from_config(cfg),
        summary_service=SummaryService.from_config(cfg),
        translation_service=TranslationService(cfg),
        guard_service=GuardService.from_config(cfg),
    )


def build_worker_context(cfg: Any, raw: Dict[str, Any]) -> WorkerContext:
    """Build a runtime context from an already loaded config bundle."""

    return WorkerContext(
        cfg=cfg,
        raw=raw,
        cfg_fp=search_config_fingerprint(cfg),
        auto_index=bool((raw.get("index") or {}).get("auto_update_on_done", True)),
        webhook_cfg=raw.get("webhook") or {},
        services=_build_services(cfg),
    )


def resolve_worker_context(
    *,
    context: WorkerContext,
    default_profile: str,
    default_variant: Optional[str],
    job: Dict[str, Any],
) -> WorkerContext:
    """Reload the context when a job targets another profile or variant."""

    extra = job.get("extra") or {}
    job_profile = str(job.get("profile") or extra.get("profile") or default_profile).strip().lower() or default_profile

    job_variant = job.get("variant")
    if job_variant is None and "variant" in extra:
        job_variant = extra.get("variant")
    if job_variant is None:
        job_variant = default_variant
    if job_variant is not None:
        job_variant = str(job_variant).strip().lower() or None

    if job_profile == context.cfg.active_profile and job_variant == context.cfg.active_variant:
        return context

    cfg, raw = load_cfg_and_raw(profile=job_profile, variant=job_variant)
    return build_worker_context(cfg, raw)


def mark_job_canceled(paths: Any, job_id: str, webhook_cfg: Dict[str, Any], message: str) -> None:
    """Mark a job as canceled and release its lock."""

    set_state(paths, job_id, "canceled", stage="canceled", progress=0.0, message=message)
    try:
        notify_webhook(webhook_cfg, "job_canceled", read_job(paths, job_id))
    except Exception:
        pass
    remove_lock_if_exists(paths, job_id)


def expand_experimental_job(
    *,
    cfg: Any,
    paths: Any,
    job_id: str,
    job: Dict[str, Any],
    language: str,
    source_language: Optional[str],
) -> None:
    """Expand one experimental parent job into per-variant child jobs."""

    extra = job.get("extra") or {}
    child_job_ids = []
    children = []
    for spec in build_variant_child_specs(
        cfg=cfg,
        job=job,
        extra=extra,
        language=language,
        source_language=source_language,
    ):
        child = create_job(paths, **spec)
        child_job_ids.append(str(child["job_id"]))
        children.append({"job_id": child["job_id"], **spec})

    write_batch_manifest(
        Path(cfg.paths.videos_dir),
        str(job.get("video_id") or ""),
        build_batch_manifest_payload(cfg, job_id, children),
    )

    batch_job = read_job(paths, job_id)
    batch_job["extra"] = dict(batch_job.get("extra") or {})
    batch_job["extra"]["child_job_ids"] = child_job_ids
    write_job(paths, batch_job)
    set_state(
        paths,
        job_id,
        "done",
        stage="expanded",
        progress=1.0,
        message=f"Expanded into variants: {', '.join(cfg.experiment.variant_ids)}",
    )
    remove_lock_if_exists(paths, job_id)


def handle_job_failure(
    *,
    cfg: Any,
    paths: Any,
    job_id: str,
    error: Exception,
    webhook_cfg: Dict[str, Any],
) -> None:
    """Persist failure state, manifests, and index error status for a crashed job."""

    job = read_job(paths, job_id)
    job_type = str(job.get("job_type") or "").lower()
    video_id = str(job.get("video_id") or "")
    language = str(job.get("language") or "").lower()
    source_language = str(job.get("source_language") or "").lower()
    if video_id and job_type in {"process", "translate"} and language:
        update_outputs_manifest(
            Path(cfg.paths.videos_dir),
            video_id,
            language,
            variant=(job.get("variant") or None),
            source_lang=(source_language or None),
            model_name=str(cfg.translation.model_name_or_path if job_type == "translate" else cfg.model.model_name_or_path),
            status="failed",
            error=str(error),
            job_id=job_id,
            note=job_type,
        )
    if video_id and job_type == "process":
        update_batch_variant_status(
            Path(cfg.paths.videos_dir),
            video_id,
            (job.get("variant") or None),
            "failed",
            job_id=job_id,
            config_fingerprint=cfg.config_fingerprint,
            error=str(error),
        )

    set_state(
        paths,
        job_id,
        "failed",
        stage="failed",
        progress=0.0,
        message="Failed",
        error=str(error),
    )
    try:
        notify_webhook(webhook_cfg, "job_failed", read_job(paths, job_id))
    except Exception:
        pass
    remove_lock_if_exists(paths, job_id)

    _write_index_state(paths, built=False, last_error=str(error))
