# backend/jobs/worker_runtime.py
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
from backend.jobs.experimental import (
    build_batch_manifest_payload,
    build_variant_child_specs,
    update_batch_variant_status,
)
from backend.jobs.index_runtime import write_index_state as _write_index_state
from backend.jobs.control import create_job, read_job, remove_lock_if_exists, set_state, write_job
from backend.jobs.runtime_common import WorkerServices, finalize_job_state
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


def _cuda_memory_snapshot() -> Optional[Dict[str, float]]:
    """Return a compact CUDA memory snapshot when CUDA is available."""

    try:
        import torch
    except Exception:
        return None

    try:
        if not torch.cuda.is_available():
            return None
        free_bytes, total_bytes = torch.cuda.mem_get_info()
    except Exception:
        return None

    used_bytes = max(0, int(total_bytes) - int(free_bytes))
    gib = float(1024**3)
    return {
        "free_bytes": float(free_bytes),
        "total_bytes": float(total_bytes),
        "used_bytes": float(used_bytes),
        "free_gib": float(free_bytes) / gib,
        "used_gib": float(used_bytes) / gib,
        "free_ratio": (float(free_bytes) / float(total_bytes)) if float(total_bytes) > 0 else 0.0,
    }


def _cuda_headroom_ok(*, min_free_gib: float = 2.5, min_free_ratio: float = 0.18) -> bool:
    """Return whether CUDA has enough free headroom to keep idle models resident."""

    snapshot = _cuda_memory_snapshot()
    if snapshot is None:
        return True
    return bool(
        float(snapshot.get("free_gib", 0.0)) >= float(min_free_gib)
        and float(snapshot.get("free_ratio", 0.0)) >= float(min_free_ratio)
    )


def _release_service(service: Any) -> None:
    """Release one service cache if it exposes a release hook."""

    if service is None or not hasattr(service, "release"):
        return
    try:
        service.release()
    except Exception:
        return


def _release_inactive_services(services: WorkerServices, *, keep: set[str]) -> None:
    """Release cached models for services that are not needed for the active job."""

    service_map = {
        "pipeline": getattr(services, "pipeline", None),
        "summary_service": getattr(services, "summary_service", None),
        "guard_service": getattr(services, "guard_service", None),
        "translation_service": getattr(services, "translation_service", None),
    }
    for name, service in service_map.items():
        if name in keep:
            continue
        _release_service(service)


def _prepare_services_for_job(services: WorkerServices, job_type: str) -> None:
    """Trim resident model caches before a job when VRAM headroom is tight."""

    if _cuda_headroom_ok():
        return

    jt = str(job_type or "").strip().lower()
    keep: set[str]
    if jt == "process":
        keep = {"pipeline"}
    elif jt == "summary_polish":
        keep = {"summary_service", "guard_service"}
    elif jt == "translate":
        keep = {"translation_service"}
    else:
        keep = set()
    _release_inactive_services(services, keep=keep)


def _cleanup_services_after_job(services: WorkerServices) -> None:
    """Release idle model caches after a job when VRAM headroom is limited."""

    if _cuda_headroom_ok():
        return
    _release_inactive_services(services, keep=set())


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

    finalize_job_state(
        paths,
        job_id,
        state="canceled",
        stage="canceled",
        progress=0.0,
        message=message,
        webhook_cfg=webhook_cfg,
        webhook_event="job_canceled",
    )


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

    finalize_job_state(
        paths,
        job_id,
        state="failed",
        stage="failed",
        progress=0.0,
        message="Failed",
        webhook_cfg=webhook_cfg,
        webhook_event="job_failed",
        error=str(error),
    )

    _write_index_state(paths, built=False, last_error=str(error))
