# backend/job_executors.py
"""
Per-job execution routines for the backend worker.

Purpose:
- Move process/translate/index logic out of the main worker loop.
- Keep one function per job family so worker dispatch stays thin.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.deps import atomic_write_json, now_ts, read_json
from backend.experimental import update_batch_variant_status
from backend.job_control import check_cancel, create_job, notify_webhook, read_job, remove_lock_if_exists, set_state
from scripts.collect_metrics import export_metrics_bundle
from src.pipeline.schema_v2 import build_segment_schema_v2
from src.pipeline.video_to_text import build_clips_from_video_meta
from src.preprocessing.video_io import preprocess_video
from src.search import build_or_update_index
from src.search.index_builder import select_embedding_model_ref
from src.utils.video_store import (
    find_video_file,
    metrics_path,
    outputs_manifest_path,
    read_segments,
    read_summary,
    segments_path,
    summary_path,
    update_metrics,
    update_outputs_manifest,
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


def _run_index_build(
    *,
    cfg: Any,
    cfg_fp: str,
    language: str,
) -> float:
    """Build or refresh the search index for one language and return elapsed time."""

    started_at = time.perf_counter()
    build_or_update_index(
        videos_root=Path(cfg.paths.videos_dir),
        index_dir=Path(cfg.paths.indexes_dir),
        model_name=select_embedding_model_ref(cfg.search, models_dir=Path(cfg.paths.models_dir)),
        embedding_backend=str(getattr(cfg.search, "embedding_backend", "auto")),
        fallback_model_name=str(cfg.search.embed_model_name),
        config_fingerprint=cfg_fp,
        variant=cfg.active_variant,
        language=str(language),
        query_prefix=str(getattr(cfg.search, "query_prefix", "query: ")),
        passage_prefix=str(getattr(cfg.search, "passage_prefix", "passage: ")),
        normalize_text=bool(getattr(cfg.search, "normalize_text", True)),
        lemmatize=bool(getattr(cfg.search, "lemmatize", False)),
        cache_dir=Path(cfg.paths.cache_dir),
        use_embed_cache=bool(getattr(cfg.search, "embed_cache", True)),
    )
    return time.perf_counter() - started_at


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
    status = {"languages": {}, "updated_at": now_ts(), "last_error": None}
    built_times: List[float] = []
    for lang in cfg.ui.langs:
        try:
            _run_index_build(cfg=cfg, cfg_fp=cfg_fp, language=str(lang))
            built_at = now_ts()
            built_times.append(built_at)
            status["languages"][str(lang)] = {"built_at": built_at, "last_error": None}
        except Exception as exc:
            status["languages"][str(lang)] = {"built_at": None, "last_error": str(exc)}
            status["last_error"] = str(exc)
    status["built_at"] = max(built_times) if built_times else None
    status["version"] = now_ts()
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
    source_segments = read_segments(source_path)
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
    texts = [str(item.get("normalized_caption") or item.get("description", "")) for item in source_segments]
    translated = translation_service.translate(
        texts,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        batch_size=int(cfg.translation.batch_size),
        max_new_tokens=int(cfg.translation.max_new_tokens),
        use_cache=bool(cfg.translation.cache_enabled),
    )

    out_segments: List[Dict[str, Any]] = []
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
        translation_meta["post_edited_by_llm"] = False
        out_segment["translation_meta"] = translation_meta
        out_segments.append(out_segment)

    write_segments(target_path, out_segments)

    source_summary = read_summary(summary_path(Path(cfg.paths.videos_dir), video_id, src_lang, variant=variant_id))
    summary_text = None
    if isinstance(source_summary, dict):
        summary_text = source_summary.get("global_summary", source_summary.get("summary"))
    translated_summary_payload: Optional[Dict[str, Any]] = None
    if summary_text:
        translated_summary = translation_service.translate(
            [str(summary_text)],
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            batch_size=1,
            max_new_tokens=int(cfg.translation.max_new_tokens),
            use_cache=bool(cfg.translation.cache_enabled),
        )
        if translated_summary:
            if isinstance(source_summary, dict):
                translated_summary_payload = dict(source_summary)
                translated_summary_payload["language"] = tgt_lang
                translated_summary_payload["global_summary"] = translated_summary[0]
                translated_summary_payload["summary"] = translated_summary[0]
                translation_views = translated_summary_payload.get("translation_views")
                if not isinstance(translation_views, dict):
                    translation_views = {}
                translation_views[tgt_lang] = {
                    "mt_text": translated_summary[0],
                    "post_edited": False,
                }
                translated_summary_payload["translation_views"] = translation_views
            else:
                translated_summary_payload = {
                    "language": tgt_lang,
                    "global_summary": translated_summary[0],
                    "summary": translated_summary[0],
                }
            write_summary(
                summary_path(Path(cfg.paths.videos_dir), video_id, tgt_lang, variant=variant_id),
                translated_summary_payload,
                tgt_lang,
                extra={"source_lang": src_lang},
            )

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
        index_time = _run_index_build(cfg=cfg, cfg_fp=cfg_fp, language=tgt_lang)
        index_state = read_json(paths.index_state_path, default={}) or {}
        index_state["updated_at"] = now_ts()
        index_state["last_error"] = None
        atomic_write_json(paths.index_state_path, index_state)
        update_metrics(
            metrics_path(Path(cfg.paths.videos_dir), video_id, variant=variant_id),
            {"indexing": {tgt_lang: {"time_sec": float(index_time), "built_at": float(now_ts())}}},
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

    video_id = str(job.get("video_id") or "")
    variant_id = cfg.active_variant
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
                    extra={"force_overwrite": force_overwrite},
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

    set_state(paths, job_id, "running", stage="preprocess", progress=0.08, message="Preprocessing")
    video_meta = services.pipeline.preprocess_video(video_path) if hasattr(services.pipeline, "preprocess_video") else None
    if video_meta is None:
        video_meta = preprocess_video(video_path, cfg)

    if check_cancel(paths, job_id):
        set_state(paths, job_id, "canceled", stage="canceled", progress=0.0, message="Canceled after preprocess")
        try:
            notify_webhook(webhook_cfg, "job_canceled", read_job(paths, job_id))
        except Exception:
            pass
        remove_lock_if_exists(paths, job_id)
        return

    clips, clip_timestamps = build_clips_from_video_meta(
        video_meta=video_meta,
        window_sec=cfg.clips.window_sec,
        stride_sec=cfg.clips.stride_sec,
        min_clip_frames=cfg.clips.min_clip_frames,
        max_clip_frames=cfg.clips.max_clip_frames,
    )

    set_state(paths, job_id, "running", stage="inference", progress=0.20, message="Inference")
    annotations, metrics = services.pipeline.run(
        video_id=video_meta.video_id,
        video_duration_sec=float(video_meta.duration_sec),
        clips=clips,
        clip_timestamps=clip_timestamps,
        preprocess_time_sec=float((video_meta.extra or {}).get("preprocess_time_sec", 0.0)),
    )

    if check_cancel(paths, job_id):
        set_state(paths, job_id, "canceled", stage="canceled", progress=0.0, message="Canceled after inference")
        try:
            notify_webhook(webhook_cfg, "job_canceled", read_job(paths, job_id))
        except Exception:
            pass
        remove_lock_if_exists(paths, job_id)
        return

    set_state(paths, job_id, "running", stage="saving", progress=0.85, message="Saving outputs")
    metric_payload = (
        metrics
        if isinstance(metrics, dict)
        else getattr(metrics, "model_dump", lambda: None)() or getattr(metrics, "__dict__", {})
    )
    if not isinstance(metric_payload, dict):
        metric_payload = {}
    metric_payload.setdefault("language", base_lang)
    metric_payload.setdefault("device", str(device))
    metric_payload.setdefault("profile", cfg.active_profile)
    metric_payload.setdefault("variant", variant_id)
    metric_payload.setdefault("config_fingerprint", cfg.config_fingerprint)

    frame_stats = dict((video_meta.extra or {}).get("frame_stats") or {})
    raw_frames = float(frame_stats.get("raw_read_frames") or frame_stats.get("num_raw_frames") or 0.0)
    preprocessing_meta = {
        "fps": float(getattr(video_meta, "processed_fps", 0.0) or 0.0),
        "resize": f"{int(cfg.video.decode_resolution[0])}x{int(cfg.video.decode_resolution[1])}",
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

    segment_payloads: List[Dict[str, Any]] = []
    for index, annotation in enumerate(annotations or [], start=1):
        if isinstance(annotation, dict):
            start_sec = float(annotation.get("start_sec", 0.0))
            end_sec = float(annotation.get("end_sec", 0.0))
            text = str(annotation.get("normalized_caption") or annotation.get("description", ""))
            extra_payload = annotation.get("extra")
        else:
            start_sec = float(getattr(annotation, "start_sec", 0.0))
            end_sec = float(getattr(annotation, "end_sec", 0.0))
            text = str(getattr(annotation, "description", ""))
            extra_payload = getattr(annotation, "extra", None)

        keyframe_path = None
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
            )
        )

    segment_payloads = services.structuring_service.enrich_segments(segment_payloads)
    write_segments(base_segments_path, segment_payloads)
    write_metrics(metrics_path(Path(cfg.paths.videos_dir), video_id, variant=variant_id), metric_payload or {})

    summary_text = (metric_payload.get("extra") or {}).get("global_summary")
    if summary_text:
        summary_payload = services.summary_service.build_summary(
            video_id=video_id,
            language=base_lang,
            duration_sec=float(getattr(video_meta, "duration_sec", 0.0) or 0.0),
            summary_text=str(summary_text),
            segments=segment_payloads,
        )
        summary_payload = _guard_summary_payload(cfg, summary_payload, services.guard_service)
        write_summary(
            summary_path(Path(cfg.paths.videos_dir), video_id, base_lang, variant=variant_id),
            summary_payload,
            base_lang,
        )

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
                "segments": str(base_segments_path),
                "summary": str(summary_path(Path(cfg.paths.videos_dir), video_id, base_lang, variant=variant_id)),
                "metrics": str(metrics_path(Path(cfg.paths.videos_dir), video_id, variant=variant_id)),
                "manifest": str(outputs_manifest_path(Path(cfg.paths.videos_dir), video_id, variant=variant_id)),
            },
        },
        variant=variant_id,
    )
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

    if auto_index:
        set_state(paths, job_id, "indexing", stage="indexing", progress=0.92, message="Index update")
        index_time = _run_index_build(cfg=cfg, cfg_fp=cfg_fp, language=base_lang)
        index_state = {"built_at": now_ts(), "updated_at": now_ts(), "version": now_ts(), "last_error": None}
        atomic_write_json(paths.index_state_path, index_state)
        update_metrics(
            metrics_path(Path(cfg.paths.videos_dir), video_id, variant=variant_id),
            {"indexing": {base_lang: {"time_sec": float(index_time), "built_at": float(index_state["built_at"])}}},
        )

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
            extra={"force_overwrite": force_overwrite},
        )

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
