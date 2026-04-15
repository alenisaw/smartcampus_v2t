# backend/jobs/summary_polish_runtime.py
"""
Purpose:
- Refine an already persisted deterministic summary after the main process job is complete.
- Keep optional LLM summary polishing out of the critical process runtime path.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional

from backend.jobs.control import check_cancel, set_state
from backend.jobs.runtime_common import finalize_job_state
from src.utils.video_store import (
    metrics_path,
    read_metrics,
    read_segments,
    read_summary,
    segments_path,
    summary_path,
    update_metrics,
    update_outputs_manifest,
    write_summary,
)


def _guard_summary_payload(cfg: Any, payload: Optional[Dict[str, Any]], guard_service: Any) -> Optional[Dict[str, Any]]:
    """Apply the lightweight worker-side output guard to polished summary text."""

    if not isinstance(payload, dict):
        return payload
    if guard_service is None or not bool(getattr(cfg.guard, "enabled", False)) or not bool(getattr(cfg.guard, "output_gate", False)):
        return payload

    backend_pref = str(getattr(getattr(cfg, "runtime", None), "worker_output_guard_backend", "rules") or "rules")
    guarded = dict(payload)
    for key in ("global_summary", "summary"):
        text = str(guarded.get(key) or "")
        if text:
            guarded[key] = guard_service.sanitize_output(text, backend_pref=backend_pref)
    return guarded


def _estimate_duration_sec(segments: list[Dict[str, Any]], metrics: Dict[str, Any]) -> float:
    """Estimate video duration for the summary prompt without reopening the raw video."""

    metrics_duration = metrics.get("video_duration_sec") if isinstance(metrics, dict) else None
    try:
        if metrics_duration is not None and float(metrics_duration) > 0:
            return float(metrics_duration)
    except Exception:
        pass

    end_points = []
    for segment in segments:
        try:
            end_points.append(float(segment.get("end_sec") or 0.0))
        except Exception:
            continue
    return max(end_points) if end_points else 0.0


def run_summary_polish_job(
    *,
    cfg: Any,
    paths: Any,
    job_id: str,
    job: Dict[str, Any],
    webhook_cfg: Dict[str, Any],
    summary_service: Any,
    guard_service: Any,
) -> None:
    """Refine the canonical summary artifact after the main process outputs are already usable."""

    video_id = str(job.get("video_id") or "")
    language = str(job.get("language") or cfg.translation.source_lang or cfg.model.language or "en").strip().lower() or "en"
    variant = job.get("variant") or cfg.active_variant
    videos_dir = Path(cfg.paths.videos_dir)

    set_state(paths, job_id, "running", stage="summary_polish", progress=0.25, message="Polishing summary")

    segments = read_segments(segments_path(videos_dir, video_id, language, variant=variant))
    if not segments:
        finalize_job_state(
            paths,
            job_id,
            state="done",
            stage="summary_polish",
            progress=1.0,
            message="Summary polish skipped (segments unavailable)",
            webhook_cfg=webhook_cfg,
            webhook_event="job_done",
        )
        return

    current_summary_payload = read_summary(summary_path(videos_dir, video_id, language, variant=variant)) or {}
    current_summary = str(
        current_summary_payload.get("global_summary", current_summary_payload.get("summary", ""))
        if isinstance(current_summary_payload, dict)
        else ""
    ).strip()
    metric_payload = read_metrics(metrics_path(videos_dir, video_id, variant=variant)) or {}

    if check_cancel(paths, job_id):
        finalize_job_state(
            paths,
            job_id,
            state="canceled",
            stage="summary_polish",
            progress=0.0,
            message="Canceled before summary polish",
            webhook_cfg=webhook_cfg,
            webhook_event="job_canceled",
        )
        return

    polish_started = time.perf_counter()
    polished_payload = summary_service.polish_summary(
        video_id=video_id,
        language=language,
        duration_sec=_estimate_duration_sec(segments, metric_payload),
        segments=segments,
        summary_text=current_summary,
    )
    polish_elapsed = time.perf_counter() - polish_started

    if not isinstance(polished_payload, dict):
        finalize_job_state(
            paths,
            job_id,
            state="done",
            stage="summary_polish",
            progress=1.0,
            message="Summary polish skipped",
            webhook_cfg=webhook_cfg,
            webhook_event="job_done",
        )
        return

    if check_cancel(paths, job_id):
        finalize_job_state(
            paths,
            job_id,
            state="canceled",
            stage="summary_polish",
            progress=0.0,
            message="Canceled during summary polish",
            webhook_cfg=webhook_cfg,
            webhook_event="job_canceled",
        )
        return

    polished_payload = _guard_summary_payload(cfg, polished_payload, guard_service)
    write_summary(summary_path(videos_dir, video_id, language, variant=variant), polished_payload, language)
    update_outputs_manifest(
        videos_dir,
        video_id,
        language,
        variant=variant,
        source_lang=None,
        model_name=str(cfg.llm.model_id),
        status="ready",
        job_id=job_id,
        note="summary_polished",
    )
    update_metrics(
        metrics_path(videos_dir, video_id, variant=variant),
        {"summary_polish": {"time_sec": float(polish_elapsed), "updated_at": float(time.time()), "applied": True}},
    )
    finalize_job_state(
        paths,
        job_id,
        state="done",
        stage="summary_polish",
        progress=1.0,
        message="Summary polished",
        webhook_cfg=webhook_cfg,
        webhook_event="job_done",
    )
