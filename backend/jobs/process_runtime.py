# backend/jobs/process_runtime.py
"""
Process job runtime for SmartCampus V2T worker execution.

Purpose:
- Execute the main preprocess, VLM, structuring, persistence, and indexing workflow.
- Keep process-specific pipeline logic out of the worker loop and backend route modules.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.jobs.experimental import update_batch_variant_status
from backend.jobs.index_runtime import (
    build_index_for_language as _run_index_build,
    write_index_metrics as _write_index_metrics,
    write_index_state as _write_index_state,
)
from backend.jobs.control import check_cancel, create_job, remove_lock_if_exists, set_state
from backend.jobs.runtime_common import (
    WorkerServices,
    build_runtime_metric_payload,
    finalize_job_state,
    refresh_research_metrics,
    runtime_keep_samples,
)
from backend.jobs.stage_metrics import (
    finalize_stage_stats as _finalize_stage_stats,
    merge_stage_payload as _merge_stage_payload,
    record_stage as _record_stage,
)
from src.core.artifacts import build_segment_schema_v2
from src.video.clips import build_clips_from_video_meta
from src.video.io import preprocess_video
from src.utils.video_store import (
    clip_observations_path,
    find_video_file,
    metrics_path,
    outputs_manifest_path,
    segments_path,
    summary_path,
    update_metrics,
    update_outputs_manifest,
    write_clip_observations,
    write_metrics,
    write_run_manifest,
    write_segments,
    write_summary,
)


@dataclass
class _ProcessJobContext:
    cfg: Any
    paths: Any
    cfg_fp: str
    job_id: str
    video_id: str
    video_path: Path
    videos_dir: Path
    base_lang: str
    device: str
    variant_id: Optional[str]
    force_overwrite: bool
    auto_index: bool
    webhook_cfg: Dict[str, Any]
    services: WorkerServices
    job_extra: Dict[str, Any]


def _guard_summary_payload(cfg: Any, payload: Optional[Dict[str, Any]], guard_service: Any) -> Optional[Dict[str, Any]]:
    """Apply a compact output guard to stored summary text when enabled."""

    if not isinstance(payload, dict):
        return payload
    if guard_service is None or not bool(getattr(cfg.guard, "enabled", False)) or not bool(getattr(cfg.guard, "output_gate", False)):
        return payload

    guarded = dict(payload)
    for key in ("global_summary", "summary"):
        text = str(guarded.get(key) or "")
        if text:
            guarded[key] = guard_service.sanitize_output(text)
    return guarded


def _prepare_process_metadata(
    cfg: Any,
    *,
    video_meta: Any,
    metric_payload: Dict[str, Any],
    base_lang: str,
    device: str,
    variant_id: Optional[str],
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Prepare normalized preprocessing and metrics metadata for stored outputs."""

    metric_payload.setdefault("language", base_lang)
    metric_payload.setdefault("device", str(device))
    metric_payload.setdefault("profile", cfg.active_profile)
    metric_payload.setdefault("variant", variant_id)
    metric_payload.setdefault("config_fingerprint", cfg.config_fingerprint)
    metric_payload.setdefault(
        "runtime",
        {
            "metrics_repeats": int(getattr(cfg.runtime, "metrics_repeats", 1)),
            "metrics_store_samples": bool(getattr(cfg.runtime, "metrics_store_samples", True)),
        },
    )

    frame_stats = dict((video_meta.extra or {}).get("frame_stats") or {})
    decode_meta = dict((video_meta.extra or {}).get("decode") or {})
    raw_frames = float(frame_stats.get("raw_read_frames") or frame_stats.get("num_raw_frames") or 0.0)
    preprocessing_meta = {
        "fps": float(getattr(video_meta, "processed_fps", 0.0) or 0.0),
        "resize": str(
            decode_meta.get("resolution", f"{int(cfg.video.decode_resolution[0])}x{int(cfg.video.decode_resolution[1])}")
        ),
        "dark_drop_ratio": float(frame_stats.get("num_dark_skipped", 0.0)) / raw_frames if raw_frames > 0 else 0.0,
        "lazy_drop_ratio": float(frame_stats.get("num_lazy_skipped", 0.0)) / raw_frames if raw_frames > 0 else 0.0,
        "blur_flag_ratio": float(frame_stats.get("num_blur_flagged", 0.0)) / raw_frames if raw_frames > 0 else 0.0,
        "anonymized": bool(getattr(cfg.video, "anonymization_enabled", False)),
    }
    metrics_meta = {
        "preprocess_time_sec": float(metric_payload.get("preprocess_time_sec", 0.0)),
        "vlm_time_sec": float(metric_payload.get("model_time_sec", 0.0)),
        "postprocess_time_sec": float(metric_payload.get("postprocess_time_sec", 0.0)),
        "structuring_time_sec": 0.0,
        "embedding_time_sec": 0.0,
    }

    metrics_extra = metric_payload.get("extra")
    if not isinstance(metrics_extra, dict):
        metrics_extra = {}
    metrics_extra.update(
        {
            "processed_fps": float(getattr(video_meta, "processed_fps", 0.0) or 0.0),
            "source_fps": float(frame_stats.get("source_fps", 0.0) or decode_meta.get("source_fps", 0.0) or 0.0),
            "decode_backend": str(decode_meta.get("backend", "opencv") or "opencv"),
            "decode_time_sec": float(decode_meta.get("time_sec", 0.0) or 0.0),
            "decode_resolution": str(decode_meta.get("resolution", preprocessing_meta["resize"])),
            "pixel_format": str(decode_meta.get("pixel_format", getattr(cfg.video, "pixel_format", "")) or ""),
            "dark_drop_ratio": float(preprocessing_meta["dark_drop_ratio"]),
            "lazy_drop_ratio": float(preprocessing_meta["lazy_drop_ratio"]),
            "blur_flag_ratio": float(preprocessing_meta["blur_flag_ratio"]),
            "anonymized": bool(preprocessing_meta["anonymized"]),
            "raw_frames_read": int(frame_stats.get("raw_read_frames", frame_stats.get("num_raw_frames", 0)) or 0),
            "frames_saved": int(getattr(video_meta, "num_frames", 0) or 0),
        }
    )
    metric_payload["extra"] = metrics_extra
    return preprocessing_meta, metrics_meta


def _build_process_segment_payloads(
    cfg: Any,
    *,
    video_meta: Any,
    annotations: List[Any],
    clip_keyframes: List[str],
    base_lang: str,
    preprocessing_meta: Dict[str, Any],
    metrics_meta: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Build stored segment payloads from pipeline annotations."""

    segment_payloads: List[Dict[str, Any]] = []
    for index, annotation in enumerate(annotations or [], start=1):
        if isinstance(annotation, dict):
            start_sec = float(annotation.get("start_sec", 0.0))
            end_sec = float(annotation.get("end_sec", 0.0))
            text = str(annotation.get("normalized_caption") or annotation.get("description", ""))
            extra_payload = annotation.get("extra")
            anomaly_flag = bool(annotation.get("anomaly_flag", False))
            anomaly_confidence = float(annotation.get("anomaly_confidence", 0.0) or 0.0)
            anomaly_notes = list(annotation.get("anomaly_notes") or [])
        else:
            start_sec = float(getattr(annotation, "start_sec", 0.0))
            end_sec = float(getattr(annotation, "end_sec", 0.0))
            text = str(getattr(annotation, "description", ""))
            extra_payload = getattr(annotation, "extra", None)
            anomaly_flag = bool(getattr(annotation, "anomaly_flag", False))
            anomaly_confidence = float(getattr(annotation, "anomaly_confidence", 0.0) or 0.0)
            anomaly_notes = list(getattr(annotation, "anomaly_notes", []) or [])

        keyframe_path = clip_keyframes[index - 1] if index - 1 < len(clip_keyframes) else None
        if not keyframe_path:
            frames = getattr(video_meta, "frames", None) or []
            if index - 1 < len(frames):
                try:
                    keyframe_path = str(getattr(frames[index - 1], "path", None) or "")
                except Exception:
                    keyframe_path = None

        segment_payloads.append(
            build_segment_schema_v2(
                cfg=cfg,
                video_id=video_meta.video_id,
                segment_index=index,
                start_sec=start_sec,
                end_sec=end_sec,
                raw_caption=text,
                normalized_caption=text,
                keyframe_path=keyframe_path,
                preprocessing_meta=preprocessing_meta,
                metrics_meta=metrics_meta,
                extra=extra_payload,
                language=base_lang,
                anomaly_flag=anomaly_flag,
                anomaly_confidence=anomaly_confidence,
                anomaly_notes=anomaly_notes,
            )
        )
    return segment_payloads


def _build_clip_observation_payloads(observations: List[Any]) -> List[Dict[str, Any]]:
    """Build persisted raw clip-observation rows from the VLM observation stage."""

    rows: List[Dict[str, Any]] = []
    for index, item in enumerate(observations or [], start=1):
        if isinstance(item, dict):
            rows.append(
                {
                    "clip_id": str(item.get("clip_id") or f"clip_{index:06d}"),
                    "video_id": str(item.get("video_id") or ""),
                    "start_sec": float(item.get("start_sec", 0.0) or 0.0),
                    "end_sec": float(item.get("end_sec", 0.0) or 0.0),
                    "description": str(item.get("description") or item.get("normalized_caption") or ""),
                    "anomaly_flag": bool(item.get("anomaly_flag", False)),
                    "anomaly_confidence": float(item.get("anomaly_confidence", 0.0) or 0.0),
                    "anomaly_notes": list(item.get("anomaly_notes") or []),
                }
            )
            continue

        rows.append(
            {
                "clip_id": f"clip_{index:06d}",
                "video_id": str(getattr(item, "video_id", "") or ""),
                "start_sec": float(getattr(item, "start_sec", 0.0) or 0.0),
                "end_sec": float(getattr(item, "end_sec", 0.0) or 0.0),
                "description": str(getattr(item, "description", "") or ""),
                "anomaly_flag": bool(getattr(item, "anomaly_flag", False)),
                "anomaly_confidence": float(getattr(item, "anomaly_confidence", 0.0) or 0.0),
                "anomaly_notes": list(getattr(item, "anomaly_notes", []) or []),
            }
        )
    return rows


def _enqueue_translation_jobs(
    *,
    paths: Any,
    cfg: Any,
    video_id: str,
    base_lang: str,
    variant_id: Optional[str],
    force_overwrite: bool,
    job_extra: Dict[str, Any],
) -> None:
    """Enqueue missing translation jobs for target languages."""

    for lang in cfg.translation.target_langs:
        tgt_lang = str(lang).strip().lower()
        if not tgt_lang or tgt_lang == base_lang:
            continue
        target_path = segments_path(Path(cfg.paths.videos_dir), video_id, tgt_lang, variant=variant_id)
        if target_path.exists() and not force_overwrite:
            continue
        create_job(
            paths,
            video_id=video_id,
            job_type="translate",
            profile=cfg.active_profile,
            variant=cfg.active_variant,
            language=tgt_lang,
            source_language=base_lang,
            extra={**job_extra, "force_overwrite": force_overwrite},
        )


def _cancel_job_and_release(paths: Any, job_id: str, webhook_cfg: Dict[str, Any], message: str) -> None:
    """Persist a cancel state, notify listeners, and release the worker lock."""

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


def _reuse_existing_process_outputs(
    *,
    cfg: Any,
    paths: Any,
    job_id: str,
    video_id: str,
    variant_id: Optional[str],
    base_lang: str,
    force_overwrite: bool,
    job_extra: Dict[str, Any],
) -> bool:
    """Short-circuit processing when reusable base outputs already exist."""

    base_segments_path = segments_path(Path(cfg.paths.videos_dir), video_id, base_lang, variant=variant_id)
    if not (base_segments_path.exists() and not force_overwrite):
        return False

    set_state(paths, job_id, "done", stage="skip", progress=1.0, message="Outputs already exist")
    update_batch_variant_status(
        Path(cfg.paths.videos_dir),
        video_id,
        variant_id,
        "done",
        job_id=job_id,
        config_fingerprint=cfg.config_fingerprint,
        reused_existing=True,
    )
    _enqueue_translation_jobs(
        paths=paths,
        cfg=cfg,
        video_id=video_id,
        base_lang=base_lang,
        variant_id=variant_id,
        force_overwrite=force_overwrite,
        job_extra=job_extra,
    )
    remove_lock_if_exists(paths, job_id)
    return True


def _build_process_context(
    *,
    cfg: Any,
    paths: Any,
    cfg_fp: str,
    job_id: str,
    job: Dict[str, Any],
    device: str,
    force_overwrite: bool,
    auto_index: bool,
    webhook_cfg: Dict[str, Any],
    services: WorkerServices,
) -> _ProcessJobContext:
    video_id = str(job.get("video_id") or "")
    videos_dir = Path(cfg.paths.videos_dir)
    video_path = find_video_file(videos_dir, video_id)
    if video_path is None:
        raise RuntimeError(f"Video not found for job video_id={video_id}")

    base_lang = str(cfg.translation.source_lang or cfg.model.language or "en").strip().lower()
    cfg.model.device = str(device)
    cfg.model.language = base_lang

    return _ProcessJobContext(
        cfg=cfg,
        paths=paths,
        cfg_fp=cfg_fp,
        job_id=job_id,
        video_id=video_id,
        video_path=video_path,
        videos_dir=videos_dir,
        base_lang=base_lang,
        device=str(device),
        variant_id=cfg.active_variant,
        force_overwrite=force_overwrite,
        auto_index=auto_index,
        webhook_cfg=webhook_cfg,
        services=services,
        job_extra=dict(job.get("extra") or {}),
    )


def _mark_process_running(context: _ProcessJobContext) -> None:
    update_outputs_manifest(
        context.videos_dir,
        context.video_id,
        context.base_lang,
        variant=context.variant_id,
        source_lang=None,
        model_name=str(context.cfg.model.model_name_or_path),
        status="processing",
        job_id=context.job_id,
        note="process",
    )
    update_batch_variant_status(
        context.videos_dir,
        context.video_id,
        context.variant_id,
        "running",
        job_id=context.job_id,
        profile=context.cfg.active_profile,
        config_fingerprint=context.cfg.config_fingerprint,
    )


def _build_process_metric_payload(context: _ProcessJobContext) -> Dict[str, Any]:
    return build_runtime_metric_payload(
        context.cfg,
        language=context.base_lang,
        variant=context.variant_id,
        device=context.device,
    )


def _cancel_if_requested(context: _ProcessJobContext, message: str) -> bool:
    if not check_cancel(context.paths, context.job_id):
        return False
    _cancel_job_and_release(context.paths, context.job_id, context.webhook_cfg, message)
    return True


def _run_preprocess_stage(context: _ProcessJobContext, metric_payload: Dict[str, Any]) -> Any:
    set_state(context.paths, context.job_id, "running", stage="preprocess", progress=0.08, message="Preprocessing")
    preprocess_started = time.perf_counter()
    video_meta = (
        context.services.pipeline.preprocess_video(context.video_path)
        if hasattr(context.services.pipeline, "preprocess_video")
        else None
    )
    if video_meta is None:
        video_meta = preprocess_video(context.video_path, context.cfg)
    _record_stage(metric_payload, "preprocess_video", time.perf_counter() - preprocess_started)
    return video_meta


def _run_clip_build_stage(
    context: _ProcessJobContext,
    video_meta: Any,
    metric_payload: Dict[str, Any],
) -> tuple[List[Any], List[Any], List[Any]]:
    clip_build_started = time.perf_counter()
    clips_data = build_clips_from_video_meta(
        video_meta=video_meta,
        window_sec=context.cfg.clips.window_sec,
        stride_sec=context.cfg.clips.stride_sec,
        min_clip_frames=context.cfg.clips.min_clip_frames,
        max_clip_frames=context.cfg.clips.max_clip_frames,
        keyframe_policy=str(getattr(context.cfg.clips, "keyframe_policy", "middle")),
        return_keyframes=True,
    )
    if isinstance(clips_data, tuple) and len(clips_data) == 3:
        clips, clip_timestamps, clip_keyframes = clips_data
    else:
        clips, clip_timestamps = clips_data  # type: ignore[misc]
        clip_keyframes = [None] * len(clips)
    _record_stage(metric_payload, "build_clips", time.perf_counter() - clip_build_started)
    return list(clips), list(clip_timestamps), list(clip_keyframes)


def _run_inference_stage(
    context: _ProcessJobContext,
    video_meta: Any,
    clips: List[Any],
    clip_timestamps: List[Any],
    metric_payload: Dict[str, Any],
) -> tuple[List[Any], List[Any], Any]:
    set_state(context.paths, context.job_id, "running", stage="inference", progress=0.20, message="Inference")
    inference_started = time.perf_counter()
    annotations, raw_observations, metrics = context.services.pipeline.run(
        video_id=video_meta.video_id,
        video_duration_sec=float(video_meta.duration_sec),
        clips=clips,
        clip_timestamps=clip_timestamps,
        preprocess_time_sec=float((video_meta.extra or {}).get("preprocess_time_sec", 0.0)),
    )
    _record_stage(metric_payload, "run_vlm_pipeline", time.perf_counter() - inference_started)
    return list(annotations or []), list(raw_observations or []), metrics


def _normalize_process_metric_payload(metric_payload: Dict[str, Any], metrics: Any) -> Dict[str, Any]:
    staged_timings_snapshot = {
        "stage_samples_sec": dict(metric_payload.get("stage_samples_sec") or {}),
        "stage_stats_sec": dict(metric_payload.get("stage_stats_sec") or {}),
    }
    normalized = (
        metrics
        if isinstance(metrics, dict)
        else getattr(metrics, "model_dump", lambda: None)() or getattr(metrics, "__dict__", {})
    )
    if not isinstance(normalized, dict):
        normalized = {}
    _merge_stage_payload(normalized, staged_timings_snapshot)
    return normalized


def _persist_process_outputs(
    *,
    context: _ProcessJobContext,
    video_meta: Any,
    annotations: List[Any],
    raw_observations: List[Any],
    clip_keyframes: List[Any],
    metric_payload: Dict[str, Any],
) -> Path:
    set_state(context.paths, context.job_id, "running", stage="saving", progress=0.85, message="Saving outputs")
    preprocessing_meta, metrics_meta = _prepare_process_metadata(
        context.cfg,
        video_meta=video_meta,
        metric_payload=metric_payload,
        base_lang=context.base_lang,
        device=context.device,
        variant_id=context.variant_id,
    )

    build_segments_started = time.perf_counter()
    observation_rows = _build_clip_observation_payloads(raw_observations)
    if observation_rows:
        observations_write_started = time.perf_counter()
        write_clip_observations(
            clip_observations_path(context.videos_dir, context.video_id, variant=context.variant_id),
            observation_rows,
        )
        _record_stage(metric_payload, "write_clip_observations", time.perf_counter() - observations_write_started)

    segment_payloads = _build_process_segment_payloads(
        context.cfg,
        video_meta=video_meta,
        annotations=annotations,
        clip_keyframes=clip_keyframes,
        base_lang=context.base_lang,
        preprocessing_meta=preprocessing_meta,
        metrics_meta=metrics_meta,
    )
    _record_stage(metric_payload, "build_segment_schema", time.perf_counter() - build_segments_started)

    structuring_started = time.perf_counter()
    segment_payloads = context.services.structuring_service.enrich_segments(segment_payloads)
    structuring_elapsed = time.perf_counter() - structuring_started
    _record_stage(metric_payload, "structuring_segments", structuring_elapsed)
    if segment_payloads:
        avg_structuring_time = float(structuring_elapsed) / float(max(1, len(segment_payloads)))
        for segment in segment_payloads:
            mm = segment.get("metrics_meta")
            if not isinstance(mm, dict):
                mm = {}
            mm["structuring_time_sec"] = avg_structuring_time
            segment["metrics_meta"] = mm

    write_segments_started = time.perf_counter()
    base_segments_path = segments_path(context.videos_dir, context.video_id, context.base_lang, variant=context.variant_id)
    write_segments(base_segments_path, segment_payloads)
    _record_stage(metric_payload, "write_segments", time.perf_counter() - write_segments_started)

    summary_started = time.perf_counter()
    summary_payload = context.services.summary_service.build_summary(
        video_id=context.video_id,
        language=context.base_lang,
        duration_sec=float(getattr(video_meta, "duration_sec", 0.0) or 0.0),
        segments=segment_payloads,
    )
    _record_stage(metric_payload, "build_summary", time.perf_counter() - summary_started)

    guard_started = time.perf_counter()
    summary_payload = _guard_summary_payload(context.cfg, summary_payload, context.services.guard_service)
    _record_stage(metric_payload, "guard_summary", time.perf_counter() - guard_started)

    summary_write_started = time.perf_counter()
    write_summary(
        summary_path(context.videos_dir, context.video_id, context.base_lang, variant=context.variant_id),
        summary_payload,
        context.base_lang,
    )
    _record_stage(metric_payload, "write_summary", time.perf_counter() - summary_write_started)

    run_manifest_started = time.perf_counter()
    write_run_manifest(
        context.videos_dir,
        context.video_id,
        {
            "profile": context.cfg.active_profile,
            "config_fingerprint": context.cfg.config_fingerprint,
            "language": context.base_lang,
            "job_id": context.job_id,
            "status": "ready",
            "model_ids": {
                "vision": str(context.cfg.model.model_name_or_path),
                "llm": str(context.cfg.llm.model_id),
                "embedding": str(context.cfg.search.embedding_model_id),
                "reranker": str(context.cfg.search.reranker_model_id),
                "translation_backend": str(context.cfg.translation.backend),
            },
            "output_paths": {
                "clip_observations": str(clip_observations_path(context.videos_dir, context.video_id, variant=context.variant_id)),
                "segments": str(base_segments_path),
                "summary": str(summary_path(context.videos_dir, context.video_id, context.base_lang, variant=context.variant_id)),
                "metrics": str(metrics_path(context.videos_dir, context.video_id, variant=context.variant_id)),
                "manifest": str(outputs_manifest_path(context.videos_dir, context.video_id, variant=context.variant_id)),
            },
        },
        variant=context.variant_id,
    )
    _record_stage(metric_payload, "write_run_manifest", time.perf_counter() - run_manifest_started)

    outputs_manifest_started = time.perf_counter()
    update_outputs_manifest(
        context.videos_dir,
        context.video_id,
        context.base_lang,
        variant=context.variant_id,
        source_lang=None,
        model_name=str(context.cfg.model.model_name_or_path),
        status="ready",
        job_id=context.job_id,
        note="process",
    )
    _record_stage(metric_payload, "update_outputs_manifest", time.perf_counter() - outputs_manifest_started)
    return base_segments_path


def _run_optional_index(context: _ProcessJobContext, metric_payload: Dict[str, Any]) -> Optional[float]:
    if not context.auto_index:
        return None

    set_state(context.paths, context.job_id, "indexing", stage="indexing", progress=0.92, message="Index update")
    index_started = time.perf_counter()
    index_time = _run_index_build(cfg=context.cfg, cfg_fp=context.cfg_fp, language=context.base_lang)
    _record_stage(metric_payload, "index_build", time.perf_counter() - index_started)
    _write_index_state(context.paths, built=True, last_error=None)
    return index_time


def _enqueue_process_translations(context: _ProcessJobContext, metric_payload: Dict[str, Any]) -> None:
    enqueue_started = time.perf_counter()
    _enqueue_translation_jobs(
        paths=context.paths,
        cfg=context.cfg,
        video_id=context.video_id,
        base_lang=context.base_lang,
        variant_id=context.variant_id,
        force_overwrite=context.force_overwrite,
        job_extra=context.job_extra,
    )
    _record_stage(metric_payload, "enqueue_translation_jobs", time.perf_counter() - enqueue_started)


def _persist_process_metrics(
    context: _ProcessJobContext,
    metric_payload: Dict[str, Any],
    *,
    process_started_at: float,
    index_time: Optional[float],
) -> Path:
    _record_stage(metric_payload, "process_total", time.perf_counter() - process_started_at)
    _finalize_stage_stats(metric_payload, keep_samples=runtime_keep_samples(context.cfg))

    metrics_target = metrics_path(context.videos_dir, context.video_id, variant=context.variant_id)
    write_metrics_started = time.perf_counter()
    write_metrics(metrics_target, metric_payload or {})
    _record_stage(metric_payload, "write_metrics", time.perf_counter() - write_metrics_started)
    _finalize_stage_stats(metric_payload, keep_samples=runtime_keep_samples(context.cfg))
    write_metrics(metrics_target, metric_payload or {})

    if context.auto_index and index_time is not None:
        _write_index_metrics(
            cfg=context.cfg,
            video_id=context.video_id,
            language=context.base_lang,
            variant=context.variant_id,
            index_time=index_time,
        )
    return metrics_target


def _finalize_process_success(context: _ProcessJobContext, metrics_target: Path) -> None:
    update_batch_variant_status(
        context.videos_dir,
        context.video_id,
        context.variant_id,
        "done",
        job_id=context.job_id,
        config_fingerprint=context.cfg.config_fingerprint,
        metrics_path=str(metrics_target),
        manifest_path=str(outputs_manifest_path(context.videos_dir, context.video_id, variant=context.variant_id)),
    )
    refresh_research_metrics(context.cfg)
    finalize_job_state(
        context.paths,
        context.job_id,
        state="done",
        stage="completed",
        progress=1.0,
        message="Finished",
        webhook_cfg=context.webhook_cfg,
        webhook_event="job_done",
    )


def run_process_job(
    *,
    cfg: Any,
    paths: Any,
    cfg_fp: str,
    job_id: str,
    job: Dict[str, Any],
    device: str,
    force_overwrite: bool,
    auto_index: bool,
    webhook_cfg: Dict[str, Any],
    services: WorkerServices,
) -> None:
    """Execute the main preprocess -> VLM -> structuring -> save pipeline."""

    process_started_at = time.perf_counter()
    context = _build_process_context(
        cfg=cfg,
        paths=paths,
        cfg_fp=cfg_fp,
        job_id=job_id,
        job=job,
        device=device,
        force_overwrite=force_overwrite,
        auto_index=auto_index,
        webhook_cfg=webhook_cfg,
        services=services,
    )

    if _reuse_existing_process_outputs(
        cfg=context.cfg,
        paths=context.paths,
        job_id=context.job_id,
        video_id=context.video_id,
        variant_id=context.variant_id,
        base_lang=context.base_lang,
        force_overwrite=context.force_overwrite,
        job_extra=context.job_extra,
    ):
        return

    _mark_process_running(context)
    metric_payload = _build_process_metric_payload(context)

    video_meta = _run_preprocess_stage(context, metric_payload)
    if _cancel_if_requested(context, "Canceled after preprocess"):
        return

    clips, clip_timestamps, clip_keyframes = _run_clip_build_stage(context, video_meta, metric_payload)
    annotations, raw_observations, metrics = _run_inference_stage(
        context,
        video_meta,
        clips,
        clip_timestamps,
        metric_payload,
    )
    if _cancel_if_requested(context, "Canceled after inference"):
        return

    metric_payload = _normalize_process_metric_payload(metric_payload, metrics)
    _persist_process_outputs(
        context=context,
        video_meta=video_meta,
        annotations=annotations,
        raw_observations=raw_observations,
        clip_keyframes=clip_keyframes,
        metric_payload=metric_payload,
    )

    index_time = _run_optional_index(context, metric_payload)
    _enqueue_process_translations(context, metric_payload)
    metrics_target = _persist_process_metrics(
        context,
        metric_payload,
        process_started_at=process_started_at,
        index_time=index_time,
    )
    _finalize_process_success(context, metrics_target)
