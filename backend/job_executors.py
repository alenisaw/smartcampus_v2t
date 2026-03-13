# backend/job_executors.py
"""
Job executors for SmartCampus V2T backend worker.

Purpose:
- Run process, translate, and index job families outside the worker loop.
- Keep per-job execution logic modular while preserving runtime behavior.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.deps import atomic_write_json, now_ts, read_json
from backend.experimental import update_batch_variant_status
from backend.index_runtime import (
    build_index_for_language as _run_index_build,
    rebuild_index_status as _rebuild_index_status,
    write_index_metrics as _write_index_metrics,
    write_index_state as _write_index_state,
)
from backend.job_control import check_cancel, create_job, notify_webhook, read_job, remove_lock_if_exists, set_state
from backend.stage_metrics import (
    finalize_stage_stats as _finalize_stage_stats,
    merge_stage_payload as _merge_stage_payload,
    record_stage as _record_stage,
)
from scripts.collect_metrics import export_metrics_bundle
from src.core.artifacts import build_segment_schema_v2
from src.video.clips import build_clips_from_video_meta
from src.video.io import preprocess_video
from src.utils.video_store import (
    find_video_file,
    metrics_path,
    outputs_manifest_path,
    read_segments,
    read_summary,
    clip_observations_path,
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


@dataclass
class WorkerServices:
    """Service bundle reused across job executors."""

    pipeline: Any
    structuring_service: Any
    summary_service: Any
    translation_service: Any
    guard_service: Any


def _refresh_research_metrics(cfg: Any) -> None:
    """Refresh the aggregate research metrics snapshot after completed runs."""

    try:
        export_metrics_bundle(
            videos_dir=Path(cfg.paths.videos_dir),
            out_dir=Path(cfg.paths.data_dir) / "research",
        )
    except Exception:
        return


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


def run_index_job(
    *,
    cfg: Any,
    paths: Any,
    cfg_fp: str,
    job_id: str,
    webhook_cfg: Dict[str, Any],
) -> None:
    """Execute a standalone index rebuild job."""

    set_state(paths, job_id, "indexing", stage="indexing", progress=0.2, message="Index rebuild")
    status = _rebuild_index_status(cfg=cfg, cfg_fp=cfg_fp)
    atomic_write_json(paths.index_state_path, status)
    set_state(paths, job_id, "done", stage="indexing", progress=1.0, message="Index rebuilt")
    try:
        notify_webhook(webhook_cfg, "job_done", read_job(paths, job_id))
    except Exception:
        pass
    remove_lock_if_exists(paths, job_id)


def run_translate_job(
    *,
    cfg: Any,
    paths: Any,
    cfg_fp: str,
    job_id: str,
    job: Dict[str, Any],
    language: str,
    source_language: Optional[str],
    force_overwrite: bool,
    auto_index: bool,
    webhook_cfg: Dict[str, Any],
    translation_service: Any,
) -> None:
    """Execute a translation job for segments and summary outputs."""

    video_id = str(job.get("video_id") or "")
    variant_id = cfg.active_variant
    tgt_lang = str(language)
    src_lang = source_language or str(cfg.translation.source_lang or cfg.model.language or "en").lower()
    if not tgt_lang or not src_lang:
        raise RuntimeError("Missing translation languages")
    if tgt_lang == src_lang:
        set_state(paths, job_id, "done", stage="translate", progress=1.0, message="Source equals target")
        remove_lock_if_exists(paths, job_id)
        return

    target_path = segments_path(Path(cfg.paths.videos_dir), video_id, tgt_lang, variant=variant_id)
    if target_path.exists() and not force_overwrite:
        set_state(paths, job_id, "done", stage="translate", progress=1.0, message="Already translated")
        remove_lock_if_exists(paths, job_id)
        return

    source_path = segments_path(Path(cfg.paths.videos_dir), video_id, src_lang, variant=variant_id)
    source_load_started = time.perf_counter()
    source_segments = read_segments(source_path)
    source_load_elapsed = time.perf_counter() - source_load_started
    if not source_segments and not source_path.exists():
        raise RuntimeError(f"Source segments not found: {video_id} ({src_lang})")

    set_state(paths, job_id, "running", stage="translate", progress=0.2, message="Translating")
    update_outputs_manifest(
        Path(cfg.paths.videos_dir),
        video_id,
        tgt_lang,
        variant=variant_id,
        source_lang=src_lang,
        model_name=str(cfg.translation.model_name_or_path),
        status="translating",
        job_id=job_id,
        note="translate",
    )
    translate_started_at = time.perf_counter()
    metric_payload: Dict[str, Any] = {
        "language": tgt_lang,
        "profile": cfg.active_profile,
        "variant": variant_id,
        "config_fingerprint": cfg.config_fingerprint,
        "runtime": {
            "metrics_repeats": int(getattr(cfg.runtime, "metrics_repeats", 1)),
            "metrics_store_samples": bool(getattr(cfg.runtime, "metrics_store_samples", True)),
        },
    }
    _record_stage(metric_payload, "load_source_segments", source_load_elapsed)

    texts = [str(item.get("normalized_caption") or item.get("description", "")) for item in source_segments]
    mt_started = time.perf_counter()
    translated = translation_service.translate(
        texts,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        batch_size=int(cfg.translation.batch_size),
        max_new_tokens=int(cfg.translation.max_new_tokens),
        use_cache=bool(cfg.translation.cache_enabled),
    )
    _record_stage(metric_payload, "translate_segments_mt", time.perf_counter() - mt_started)

    selected_candidate_indices: List[int] = []
    for idx, segment in enumerate(source_segments):
        if not isinstance(segment, dict):
            continue
        risk = str(segment.get("risk_level", "normal")).strip().lower()
        anomaly_flag = bool(segment.get("anomaly_flag", False))
        if anomaly_flag or risk in {"attention", "warning", "critical"}:
            selected_candidate_indices.append(idx)

    selected_post_edit_edited = 0
    selected_post_edit_edited_indices: set[int] = set()
    if selected_candidate_indices and hasattr(translation_service, "post_edit_many"):
        selected_src = [texts[i] for i in selected_candidate_indices]
        selected_mt_before = [translated[i] for i in selected_candidate_indices]
        post_edit_started = time.perf_counter()
        selected_mt_after, selected_post_edit_edited = translation_service.post_edit_many(
            selected_src,
            selected_mt_before,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            target_name="selected_segments",
        )
        _record_stage(metric_payload, "post_edit_selected_segments", time.perf_counter() - post_edit_started)
        for local_idx, global_idx in enumerate(selected_candidate_indices):
            if local_idx >= len(selected_mt_after):
                continue
            translated[global_idx] = selected_mt_after[local_idx]
            if str(selected_mt_after[local_idx]) != str(selected_mt_before[local_idx]):
                selected_post_edit_edited_indices.add(int(global_idx))

    out_segments: List[Dict[str, Any]] = []
    write_segments_started = time.perf_counter()
    for index, segment in enumerate(source_segments):
        if not isinstance(segment, dict):
            continue
        translated_text = translated[index] if index < len(translated) else str(segment.get("normalized_caption") or segment.get("description", ""))
        out_segment = dict(segment)
        out_segment["language"] = tgt_lang
        out_segment["normalized_caption"] = translated_text
        out_segment["description"] = translated_text
        translation_meta = out_segment.get("translation_meta")
        if not isinstance(translation_meta, dict):
            translation_meta = {}
        available_languages = translation_meta.get("available_languages")
        if not isinstance(available_languages, list):
            available_languages = []
        if tgt_lang not in available_languages:
            available_languages.append(tgt_lang)
        translation_meta["available_languages"] = available_languages
        translation_meta["translated_from"] = src_lang
        translation_meta["mt_engine"] = str(cfg.translation.backend)
        translation_meta["post_edited_by_llm"] = bool(index in selected_post_edit_edited_indices)
        out_segment["translation_meta"] = translation_meta
        out_segments.append(out_segment)

    write_segments(target_path, out_segments)
    _record_stage(metric_payload, "write_segments", time.perf_counter() - write_segments_started)

    source_summary_load_started = time.perf_counter()
    source_summary = read_summary(summary_path(Path(cfg.paths.videos_dir), video_id, src_lang, variant=variant_id))
    _record_stage(metric_payload, "load_source_summary", time.perf_counter() - source_summary_load_started)
    summary_text = None
    if isinstance(source_summary, dict):
        summary_text = source_summary.get("global_summary", source_summary.get("summary"))
    translated_summary_payload: Optional[Dict[str, Any]] = None
    summary_post_edited = False
    if summary_text:
        summary_mt_started = time.perf_counter()
        translated_summary = translation_service.translate(
            [str(summary_text)],
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            batch_size=1,
            max_new_tokens=int(cfg.translation.max_new_tokens),
            use_cache=bool(cfg.translation.cache_enabled),
        )
        _record_stage(metric_payload, "translate_summary_mt", time.perf_counter() - summary_mt_started)
        if translated_summary:
            summary_text_translated = str(translated_summary[0])
            if hasattr(translation_service, "post_edit_many"):
                summary_post_edit_started = time.perf_counter()
                summary_post = translation_service.post_edit_many(
                    [str(summary_text)],
                    [summary_text_translated],
                    src_lang=src_lang,
                    tgt_lang=tgt_lang,
                    target_name="summary",
                )
                _record_stage(metric_payload, "post_edit_summary", time.perf_counter() - summary_post_edit_started)
                if isinstance(summary_post, tuple):
                    edited_texts, edited_count = summary_post
                    if edited_texts:
                        summary_text_translated = str(edited_texts[0])
                    summary_post_edited = bool(edited_count and summary_text_translated)

            if isinstance(source_summary, dict):
                translated_summary_payload = dict(source_summary)
                translated_summary_payload["language"] = tgt_lang
                translated_summary_payload["global_summary"] = summary_text_translated
                translated_summary_payload["summary"] = summary_text_translated
                translation_views = translated_summary_payload.get("translation_views")
                if not isinstance(translation_views, dict):
                    translation_views = {}
                translation_views[tgt_lang] = {
                    "mt_text": summary_text_translated,
                    "post_edited": bool(summary_post_edited),
                }
                translated_summary_payload["translation_views"] = translation_views
            else:
                translated_summary_payload = {
                    "language": tgt_lang,
                    "global_summary": summary_text_translated,
                    "summary": summary_text_translated,
                }
            write_summary_started = time.perf_counter()
            write_summary(
                summary_path(Path(cfg.paths.videos_dir), video_id, tgt_lang, variant=variant_id),
                translated_summary_payload,
                tgt_lang,
                extra={"source_lang": src_lang},
            )
            _record_stage(metric_payload, "write_summary", time.perf_counter() - write_summary_started)

    update_outputs_manifest(
        Path(cfg.paths.videos_dir),
        video_id,
        tgt_lang,
        variant=variant_id,
        source_lang=src_lang,
        model_name=str(cfg.translation.model_name_or_path),
        status="ready",
        job_id=job_id,
        note="translate",
    )

    if auto_index:
        set_state(paths, job_id, "indexing", stage="indexing", progress=0.88, message="Index update")
        index_started = time.perf_counter()
        index_time = _run_index_build(cfg=cfg, cfg_fp=cfg_fp, language=tgt_lang)
        _record_stage(metric_payload, "index_build", time.perf_counter() - index_started)
        _write_index_state(paths, built=False, last_error=None)
        _write_index_metrics(cfg=cfg, video_id=video_id, language=tgt_lang, variant=variant_id, index_time=index_time)

    _record_stage(metric_payload, "translate_total", time.perf_counter() - translate_started_at)
    _finalize_stage_stats(
        metric_payload,
        keep_samples=bool(getattr(cfg.runtime, "metrics_store_samples", True)),
    )

    update_metrics(
        metrics_path(Path(cfg.paths.videos_dir), video_id, variant=variant_id),
        {
            "translations": {
                tgt_lang: {
                    "source_lang": src_lang,
                    "model_name": str(cfg.translation.backend),
                    "time_sec": float(time.perf_counter() - translate_started_at),
                    "num_segments": int(len(texts)),
                    "summary_available": bool(summary_text),
                    "selected_segments_candidates": int(len(selected_candidate_indices)),
                    "selected_segments_post_edited": int(selected_post_edit_edited),
                    "summary_post_edited": bool(summary_post_edited),
                    "stage_stats_sec": metric_payload.get("stage_stats_sec", {}),
                    "stage_samples_sec": metric_payload.get("stage_samples_sec", {}),
                }
            }
        },
    )

    set_state(paths, job_id, "done", stage="translate", progress=1.0, message="Translated")
    _refresh_research_metrics(cfg)
    try:
        notify_webhook(webhook_cfg, "job_done", read_job(paths, job_id))
    except Exception:
        pass
    remove_lock_if_exists(paths, job_id)


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
    video_id = str(job.get("video_id") or "")
    variant_id = cfg.active_variant
    job_extra = dict(job.get("extra") or {})
    video_path = find_video_file(Path(cfg.paths.videos_dir), video_id)
    if video_path is None:
        raise RuntimeError(f"Video not found for job video_id={video_id}")

    base_lang = str(cfg.translation.source_lang or cfg.model.language or "en").strip().lower()
    cfg.model.device = str(device)
    cfg.model.language = base_lang

    base_segments_path = segments_path(Path(cfg.paths.videos_dir), video_id, base_lang, variant=variant_id)
    if base_segments_path.exists() and not force_overwrite:
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
        for lang in cfg.translation.target_langs:
            tgt_lang = str(lang).strip().lower()
            if tgt_lang and tgt_lang != base_lang:
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
        remove_lock_if_exists(paths, job_id)
        return

    update_outputs_manifest(
        Path(cfg.paths.videos_dir),
        video_id,
        base_lang,
        variant=variant_id,
        source_lang=None,
        model_name=str(cfg.model.model_name_or_path),
        status="processing",
        job_id=job_id,
        note="process",
    )
    update_batch_variant_status(
        Path(cfg.paths.videos_dir),
        video_id,
        variant_id,
        "running",
        job_id=job_id,
        profile=cfg.active_profile,
        config_fingerprint=cfg.config_fingerprint,
    )

    metric_payload: Dict[str, Any] = {
        "language": base_lang,
        "device": str(device),
        "profile": cfg.active_profile,
        "variant": variant_id,
        "config_fingerprint": cfg.config_fingerprint,
        "runtime": {
            "metrics_repeats": int(getattr(cfg.runtime, "metrics_repeats", 1)),
            "metrics_store_samples": bool(getattr(cfg.runtime, "metrics_store_samples", True)),
        },
    }

    set_state(paths, job_id, "running", stage="preprocess", progress=0.08, message="Preprocessing")
    preprocess_started = time.perf_counter()
    video_meta = services.pipeline.preprocess_video(video_path) if hasattr(services.pipeline, "preprocess_video") else None
    if video_meta is None:
        video_meta = preprocess_video(video_path, cfg)
    _record_stage(metric_payload, "preprocess_video", time.perf_counter() - preprocess_started)

    if check_cancel(paths, job_id):
        set_state(paths, job_id, "canceled", stage="canceled", progress=0.0, message="Canceled after preprocess")
        try:
            notify_webhook(webhook_cfg, "job_canceled", read_job(paths, job_id))
        except Exception:
            pass
        remove_lock_if_exists(paths, job_id)
        return

    clip_build_started = time.perf_counter()
    clips_data = build_clips_from_video_meta(
        video_meta=video_meta,
        window_sec=cfg.clips.window_sec,
        stride_sec=cfg.clips.stride_sec,
        min_clip_frames=cfg.clips.min_clip_frames,
        max_clip_frames=cfg.clips.max_clip_frames,
        keyframe_policy=str(getattr(cfg.clips, "keyframe_policy", "middle")),
        return_keyframes=True,
    )
    if isinstance(clips_data, tuple) and len(clips_data) == 3:
        clips, clip_timestamps, clip_keyframes = clips_data
    else:
        clips, clip_timestamps = clips_data  # type: ignore[misc]
        clip_keyframes = [None] * len(clips)
    _record_stage(metric_payload, "build_clips", time.perf_counter() - clip_build_started)

    set_state(paths, job_id, "running", stage="inference", progress=0.20, message="Inference")
    inference_started = time.perf_counter()
    annotations, raw_observations, metrics = services.pipeline.run(
        video_id=video_meta.video_id,
        video_duration_sec=float(video_meta.duration_sec),
        clips=clips,
        clip_timestamps=clip_timestamps,
        preprocess_time_sec=float((video_meta.extra or {}).get("preprocess_time_sec", 0.0)),
    )
    _record_stage(metric_payload, "run_vlm_pipeline", time.perf_counter() - inference_started)

    if check_cancel(paths, job_id):
        set_state(paths, job_id, "canceled", stage="canceled", progress=0.0, message="Canceled after inference")
        try:
            notify_webhook(webhook_cfg, "job_canceled", read_job(paths, job_id))
        except Exception:
            pass
        remove_lock_if_exists(paths, job_id)
        return

    set_state(paths, job_id, "running", stage="saving", progress=0.85, message="Saving outputs")
    staged_timings_snapshot = {
        "stage_samples_sec": dict(metric_payload.get("stage_samples_sec") or {}),
        "stage_stats_sec": dict(metric_payload.get("stage_stats_sec") or {}),
    }
    metric_payload = (
        metrics
        if isinstance(metrics, dict)
        else getattr(metrics, "model_dump", lambda: None)() or getattr(metrics, "__dict__", {})
    )
    if not isinstance(metric_payload, dict):
        metric_payload = {}
    _merge_stage_payload(metric_payload, staged_timings_snapshot)
    preprocessing_meta, metrics_meta = _prepare_process_metadata(
        cfg,
        video_meta=video_meta,
        metric_payload=metric_payload,
        base_lang=base_lang,
        device=str(device),
        variant_id=variant_id,
    )

    build_segments_started = time.perf_counter()
    observation_rows = _build_clip_observation_payloads(list(raw_observations or []))
    if observation_rows:
        observations_write_started = time.perf_counter()
        write_clip_observations(
            clip_observations_path(Path(cfg.paths.videos_dir), video_id, variant=variant_id),
            observation_rows,
        )
        _record_stage(metric_payload, "write_clip_observations", time.perf_counter() - observations_write_started)

    segment_payloads = _build_process_segment_payloads(
        cfg,
        video_meta=video_meta,
        annotations=list(annotations or []),
        clip_keyframes=list(clip_keyframes or []),
        base_lang=base_lang,
        preprocessing_meta=preprocessing_meta,
        metrics_meta=metrics_meta,
    )
    _record_stage(metric_payload, "build_segment_schema", time.perf_counter() - build_segments_started)

    structuring_started = time.perf_counter()
    segment_payloads = services.structuring_service.enrich_segments(segment_payloads)
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
    write_segments(base_segments_path, segment_payloads)
    _record_stage(metric_payload, "write_segments", time.perf_counter() - write_segments_started)

    summary_started = time.perf_counter()
    summary_payload = services.summary_service.build_summary(
        video_id=video_id,
        language=base_lang,
        duration_sec=float(getattr(video_meta, "duration_sec", 0.0) or 0.0),
        segments=segment_payloads,
    )
    _record_stage(metric_payload, "build_summary", time.perf_counter() - summary_started)

    guard_started = time.perf_counter()
    summary_payload = _guard_summary_payload(cfg, summary_payload, services.guard_service)
    _record_stage(metric_payload, "guard_summary", time.perf_counter() - guard_started)

    summary_write_started = time.perf_counter()
    write_summary(
        summary_path(Path(cfg.paths.videos_dir), video_id, base_lang, variant=variant_id),
        summary_payload,
        base_lang,
    )
    _record_stage(metric_payload, "write_summary", time.perf_counter() - summary_write_started)

    run_manifest_started = time.perf_counter()
    write_run_manifest(
        Path(cfg.paths.videos_dir),
        video_id,
        {
            "profile": cfg.active_profile,
            "config_fingerprint": cfg.config_fingerprint,
            "language": base_lang,
            "job_id": job_id,
            "status": "ready",
            "model_ids": {
                "vision": str(cfg.model.model_name_or_path),
                "llm": str(cfg.llm.model_id),
                "embedding": str(cfg.search.embedding_model_id),
                "reranker": str(cfg.search.reranker_model_id),
                "translation_backend": str(cfg.translation.backend),
            },
            "output_paths": {
                "clip_observations": str(clip_observations_path(Path(cfg.paths.videos_dir), video_id, variant=variant_id)),
                "segments": str(base_segments_path),
                "summary": str(summary_path(Path(cfg.paths.videos_dir), video_id, base_lang, variant=variant_id)),
                "metrics": str(metrics_path(Path(cfg.paths.videos_dir), video_id, variant=variant_id)),
                "manifest": str(outputs_manifest_path(Path(cfg.paths.videos_dir), video_id, variant=variant_id)),
            },
        },
        variant=variant_id,
    )
    _record_stage(metric_payload, "write_run_manifest", time.perf_counter() - run_manifest_started)

    outputs_manifest_started = time.perf_counter()
    update_outputs_manifest(
        Path(cfg.paths.videos_dir),
        video_id,
        base_lang,
        variant=variant_id,
        source_lang=None,
        model_name=str(cfg.model.model_name_or_path),
        status="ready",
        job_id=job_id,
        note="process",
    )
    _record_stage(metric_payload, "update_outputs_manifest", time.perf_counter() - outputs_manifest_started)

    index_time: Optional[float] = None
    if auto_index:
        set_state(paths, job_id, "indexing", stage="indexing", progress=0.92, message="Index update")
        index_started = time.perf_counter()
        index_time = _run_index_build(cfg=cfg, cfg_fp=cfg_fp, language=base_lang)
        _record_stage(metric_payload, "index_build", time.perf_counter() - index_started)
        _write_index_state(paths, built=True, last_error=None)

    enqueue_started = time.perf_counter()
    _enqueue_translation_jobs(
        paths=paths,
        cfg=cfg,
        video_id=video_id,
        base_lang=base_lang,
        variant_id=variant_id,
        force_overwrite=force_overwrite,
        job_extra=job_extra,
    )
    _record_stage(metric_payload, "enqueue_translation_jobs", time.perf_counter() - enqueue_started)

    _record_stage(metric_payload, "process_total", time.perf_counter() - process_started_at)
    _finalize_stage_stats(
        metric_payload,
        keep_samples=bool(getattr(cfg.runtime, "metrics_store_samples", True)),
    )

    metrics_target = metrics_path(Path(cfg.paths.videos_dir), video_id, variant=variant_id)
    write_metrics_started = time.perf_counter()
    write_metrics(metrics_target, metric_payload or {})
    _record_stage(metric_payload, "write_metrics", time.perf_counter() - write_metrics_started)
    _finalize_stage_stats(
        metric_payload,
        keep_samples=bool(getattr(cfg.runtime, "metrics_store_samples", True)),
    )
    write_metrics(metrics_target, metric_payload or {})
    if auto_index and index_time is not None:
        _write_index_metrics(cfg=cfg, video_id=video_id, language=base_lang, variant=variant_id, index_time=index_time)

    set_state(paths, job_id, "done", stage="completed", progress=1.0, message="Finished")
    update_batch_variant_status(
        Path(cfg.paths.videos_dir),
        video_id,
        variant_id,
        "done",
        job_id=job_id,
        config_fingerprint=cfg.config_fingerprint,
        metrics_path=str(metrics_path(Path(cfg.paths.videos_dir), video_id, variant=variant_id)),
        manifest_path=str(outputs_manifest_path(Path(cfg.paths.videos_dir), video_id, variant=variant_id)),
    )
    _refresh_research_metrics(cfg)
    try:
        notify_webhook(webhook_cfg, "job_done", read_job(paths, job_id))
    except Exception:
        pass
    remove_lock_if_exists(paths, job_id)
