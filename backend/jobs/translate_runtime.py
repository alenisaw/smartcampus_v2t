# backend/jobs/translate_runtime.py
"""
Translation job runtime for SmartCampus V2T worker execution.

Purpose:
- Execute persisted translation jobs for segments and summaries.
- Keep translation-specific workflow and persistence logic out of the worker loop.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.jobs.index_runtime import (
    build_index_for_language as _run_index_build,
    write_index_metrics as _write_index_metrics,
    write_index_state as _write_index_state,
)
from backend.jobs.control import set_state
from backend.jobs.runtime_common import (
    build_runtime_metric_payload,
    finalize_job_state,
    refresh_research_metrics,
    runtime_keep_samples,
)
from backend.jobs.stage_metrics import finalize_stage_stats as _finalize_stage_stats, record_stage as _record_stage
from src.utils.video_store import (
    metrics_path,
    read_segments,
    read_summary,
    segments_path,
    summary_path,
    update_metrics,
    update_outputs_manifest,
    write_segments,
    write_summary,
)


@dataclass
class _TranslateJobContext:
    cfg: Any
    paths: Any
    cfg_fp: str
    job_id: str
    video_id: str
    videos_dir: Path
    tgt_lang: str
    src_lang: str
    variant_id: Optional[str]
    force_overwrite: bool
    auto_index: bool
    webhook_cfg: Dict[str, Any]
    translation_service: Any


def _selected_translation_candidate_indices(source_segments: List[Dict[str, Any]]) -> List[int]:
    """Return source-segment indices that should receive selective post-editing."""

    selected_candidate_indices: List[int] = []
    for idx, segment in enumerate(source_segments):
        if not isinstance(segment, dict):
            continue
        risk = str(segment.get("risk_level", "normal")).strip().lower()
        anomaly_flag = bool(segment.get("anomaly_flag", False))
        if anomaly_flag or risk in {"attention", "warning", "critical"}:
            selected_candidate_indices.append(idx)
    return selected_candidate_indices


def _post_edit_selected_segments(
    *,
    translation_service: Any,
    texts: List[str],
    translated: List[str],
    src_lang: str,
    tgt_lang: str,
    source_segments: List[Dict[str, Any]],
    metric_payload: Dict[str, Any],
) -> tuple[List[str], int, set[int]]:
    """Selective post-edit for high-signal translated segments."""

    selected_candidate_indices = _selected_translation_candidate_indices(source_segments)
    selected_post_edit_edited = 0
    selected_post_edit_edited_indices: set[int] = set()
    if not selected_candidate_indices or not hasattr(translation_service, "post_edit_many"):
        return translated, selected_post_edit_edited, selected_post_edit_edited_indices

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
    return translated, int(selected_post_edit_edited), selected_post_edit_edited_indices


def _build_translated_segments(
    *,
    source_segments: List[Dict[str, Any]],
    translated: List[str],
    src_lang: str,
    tgt_lang: str,
    translation_backend: str,
    selected_post_edit_edited_indices: set[int],
) -> List[Dict[str, Any]]:
    """Build translated segment payloads from source segments and translated text."""

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
        translation_meta["mt_engine"] = translation_backend
        translation_meta["post_edited_by_llm"] = bool(index in selected_post_edit_edited_indices)
        out_segment["translation_meta"] = translation_meta
        out_segments.append(out_segment)
    return out_segments


def _translate_summary_payload(
    *,
    cfg: Any,
    translation_service: Any,
    video_id: str,
    src_lang: str,
    tgt_lang: str,
    variant_id: Optional[str],
    metric_payload: Dict[str, Any],
) -> tuple[Optional[str], bool]:
    """Translate and persist the summary payload when the source summary exists."""

    source_summary_load_started = time.perf_counter()
    source_summary = read_summary(summary_path(Path(cfg.paths.videos_dir), video_id, src_lang, variant=variant_id))
    _record_stage(metric_payload, "load_source_summary", time.perf_counter() - source_summary_load_started)
    summary_text = None
    if isinstance(source_summary, dict):
        summary_text = source_summary.get("global_summary", source_summary.get("summary"))
    translated_summary_payload: Optional[Dict[str, Any]] = None
    summary_post_edited = False
    if not summary_text:
        return None, summary_post_edited

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
    if not translated_summary:
        return str(summary_text), summary_post_edited

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
    return str(summary_text), summary_post_edited


def _build_translate_context(
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
) -> _TranslateJobContext:
    tgt_lang = str(language)
    src_lang = source_language or str(cfg.translation.source_lang or cfg.model.language or "en").lower()
    if not tgt_lang or not src_lang:
        raise RuntimeError("Missing translation languages")

    return _TranslateJobContext(
        cfg=cfg,
        paths=paths,
        cfg_fp=cfg_fp,
        job_id=job_id,
        video_id=str(job.get("video_id") or ""),
        videos_dir=Path(cfg.paths.videos_dir),
        tgt_lang=tgt_lang,
        src_lang=src_lang,
        variant_id=cfg.active_variant,
        force_overwrite=force_overwrite,
        auto_index=auto_index,
        webhook_cfg=webhook_cfg,
        translation_service=translation_service,
    )


def _complete_translation_job(paths: Any, job_id: str, message: str) -> None:
    finalize_job_state(
        paths,
        job_id,
        state="done",
        stage="translate",
        progress=1.0,
        message=message,
    )


def _load_source_segments(
    context: _TranslateJobContext,
    metric_payload: Dict[str, Any],
) -> List[Dict[str, Any]]:
    source_path = segments_path(context.videos_dir, context.video_id, context.src_lang, variant=context.variant_id)
    source_load_started = time.perf_counter()
    source_segments = read_segments(source_path)
    _record_stage(metric_payload, "load_source_segments", time.perf_counter() - source_load_started)
    if not source_segments and not source_path.exists():
        raise RuntimeError(f"Source segments not found: {context.video_id} ({context.src_lang})")
    return source_segments


def _mark_translation_running(context: _TranslateJobContext) -> None:
    set_state(context.paths, context.job_id, "running", stage="translate", progress=0.2, message="Translating")
    update_outputs_manifest(
        context.videos_dir,
        context.video_id,
        context.tgt_lang,
        variant=context.variant_id,
        source_lang=context.src_lang,
        model_name=str(context.cfg.translation.model_name_or_path),
        status="translating",
        job_id=context.job_id,
        note="translate",
    )


def _build_translation_metric_payload(context: _TranslateJobContext) -> Dict[str, Any]:
    return build_runtime_metric_payload(
        context.cfg,
        language=context.tgt_lang,
        variant=context.variant_id,
    )


def _translate_segment_texts(
    context: _TranslateJobContext,
    source_segments: List[Dict[str, Any]],
    metric_payload: Dict[str, Any],
) -> tuple[List[str], List[str], int, List[int], set[int]]:
    texts = [str(item.get("normalized_caption") or item.get("description", "")) for item in source_segments]
    mt_started = time.perf_counter()
    translated = context.translation_service.translate(
        texts,
        src_lang=context.src_lang,
        tgt_lang=context.tgt_lang,
        batch_size=int(context.cfg.translation.batch_size),
        max_new_tokens=int(context.cfg.translation.max_new_tokens),
        use_cache=bool(context.cfg.translation.cache_enabled),
    )
    _record_stage(metric_payload, "translate_segments_mt", time.perf_counter() - mt_started)

    translated, selected_post_edit_edited, selected_post_edit_edited_indices = _post_edit_selected_segments(
        translation_service=context.translation_service,
        texts=texts,
        translated=list(translated),
        src_lang=context.src_lang,
        tgt_lang=context.tgt_lang,
        source_segments=source_segments,
        metric_payload=metric_payload,
    )
    selected_candidate_indices = _selected_translation_candidate_indices(source_segments)
    return texts, translated, selected_post_edit_edited, selected_candidate_indices, selected_post_edit_edited_indices


def _persist_translated_outputs(
    context: _TranslateJobContext,
    *,
    source_segments: List[Dict[str, Any]],
    translated: List[str],
    selected_post_edit_edited_indices: set[int],
    metric_payload: Dict[str, Any],
) -> tuple[Optional[str], bool]:
    write_segments_started = time.perf_counter()
    out_segments = _build_translated_segments(
        source_segments=source_segments,
        translated=translated,
        src_lang=context.src_lang,
        tgt_lang=context.tgt_lang,
        translation_backend=str(context.cfg.translation.backend),
        selected_post_edit_edited_indices=selected_post_edit_edited_indices,
    )
    write_segments(
        segments_path(context.videos_dir, context.video_id, context.tgt_lang, variant=context.variant_id),
        out_segments,
    )
    _record_stage(metric_payload, "write_segments", time.perf_counter() - write_segments_started)

    summary_text, summary_post_edited = _translate_summary_payload(
        cfg=context.cfg,
        translation_service=context.translation_service,
        video_id=context.video_id,
        src_lang=context.src_lang,
        tgt_lang=context.tgt_lang,
        variant_id=context.variant_id,
        metric_payload=metric_payload,
    )

    update_outputs_manifest(
        context.videos_dir,
        context.video_id,
        context.tgt_lang,
        variant=context.variant_id,
        source_lang=context.src_lang,
        model_name=str(context.cfg.translation.model_name_or_path),
        status="ready",
        job_id=context.job_id,
        note="translate",
    )
    return summary_text, summary_post_edited


def _run_optional_translation_index(
    context: _TranslateJobContext,
    metric_payload: Dict[str, Any],
) -> None:
    if not context.auto_index:
        return

    set_state(context.paths, context.job_id, "indexing", stage="indexing", progress=0.88, message="Index update")
    index_started = time.perf_counter()
    index_time = _run_index_build(cfg=context.cfg, cfg_fp=context.cfg_fp, language=context.tgt_lang)
    _record_stage(metric_payload, "index_build", time.perf_counter() - index_started)
    _write_index_state(context.paths, built=False, last_error=None)
    _write_index_metrics(
        cfg=context.cfg,
        video_id=context.video_id,
        language=context.tgt_lang,
        variant=context.variant_id,
        index_time=index_time,
    )


def _persist_translation_metrics(
    context: _TranslateJobContext,
    *,
    metric_payload: Dict[str, Any],
    texts: List[str],
    selected_candidate_indices: List[int],
    selected_post_edit_edited: int,
    summary_text: Optional[str],
    summary_post_edited: bool,
    translate_started_at: float,
) -> None:
    _record_stage(metric_payload, "translate_total", time.perf_counter() - translate_started_at)
    _finalize_stage_stats(metric_payload, keep_samples=runtime_keep_samples(context.cfg))
    update_metrics(
        metrics_path(context.videos_dir, context.video_id, variant=context.variant_id),
        {
            "translations": {
                context.tgt_lang: {
                    "source_lang": context.src_lang,
                    "model_name": str(context.cfg.translation.backend),
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


def _finalize_translation_success(context: _TranslateJobContext) -> None:
    refresh_research_metrics(context.cfg)
    finalize_job_state(
        context.paths,
        context.job_id,
        state="done",
        stage="translate",
        progress=1.0,
        message="Translated",
        webhook_cfg=context.webhook_cfg,
        webhook_event="job_done",
    )


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

    context = _build_translate_context(
        cfg=cfg,
        paths=paths,
        cfg_fp=cfg_fp,
        job_id=job_id,
        job=job,
        language=language,
        source_language=source_language,
        force_overwrite=force_overwrite,
        auto_index=auto_index,
        webhook_cfg=webhook_cfg,
        translation_service=translation_service,
    )

    if context.tgt_lang == context.src_lang:
        _complete_translation_job(context.paths, context.job_id, "Source equals target")
        return

    target_path = segments_path(context.videos_dir, context.video_id, context.tgt_lang, variant=context.variant_id)
    if target_path.exists() and not context.force_overwrite:
        _complete_translation_job(context.paths, context.job_id, "Already translated")
        return

    _mark_translation_running(context)
    translate_started_at = time.perf_counter()
    metric_payload = _build_translation_metric_payload(context)
    source_segments = _load_source_segments(context, metric_payload)

    texts, translated, selected_post_edit_edited, selected_candidate_indices, selected_post_edit_edited_indices = _translate_segment_texts(
        context,
        source_segments,
        metric_payload,
    )
    summary_text, summary_post_edited = _persist_translated_outputs(
        context,
        source_segments=source_segments,
        translated=translated,
        selected_post_edit_edited_indices=selected_post_edit_edited_indices,
        metric_payload=metric_payload,
    )
    _run_optional_translation_index(context, metric_payload)
    _persist_translation_metrics(
        context,
        metric_payload=metric_payload,
        texts=texts,
        selected_candidate_indices=selected_candidate_indices,
        selected_post_edit_edited=selected_post_edit_edited,
        summary_text=summary_text,
        summary_post_edited=summary_post_edited,
        translate_started_at=translate_started_at,
    )
    _finalize_translation_success(context)
