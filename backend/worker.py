# backend/worker.py
"""
Background worker for SmartCampus V2T.

Purpose:
- Consume filesystem queue (data/queue/*.q).
- Maintain per-job lock + lease (Windows-safe).
- Run preprocess -> clips -> inference -> write run outputs.
- Optionally update search index after successful run.
- Support queue pause/resume via data/queue_state.json.
"""
from __future__ import annotations

import os
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Optional, List

import requests

from backend.deps import (
    atomic_write_json,
    get_backend_paths,
    host_id,
    load_cfg_and_raw,
    new_job_id,
    now_ts,
    read_json,
)

from src.preprocessing.video_io import preprocess_video
from src.pipeline.video_to_text import VideoToTextPipeline, build_clips_from_video_meta
from src.search import build_or_update_index
from src.search.index_builder import search_config_fingerprint
from src.translation.nllb_translator import NLLBTranslator
from src.translation.translation_cache import TranslationCache
from src.utils.video_store import (
    find_video_file,
    metrics_path,
    read_segments,
    read_summary,
    segments_path,
    summary_path,
    update_outputs_manifest,
    update_metrics,
    write_metrics,
    write_segments,
    write_summary,
)


def _list_queue_items(queue_dir: Path) -> List[Path]:
    return sorted(queue_dir.glob("p*__*__*.q"))


def _parse_job_id_from_queue_item(p: Path) -> Optional[str]:
    try:
        parts = p.name.split("__")
        if len(parts) < 3:
            return None
        job_part = parts[2]
        if job_part.endswith(".q"):
            job_part = job_part[:-2]
        return job_part
    except Exception:
        return None


def _job_path(paths, job_id: str) -> Path:
    return paths.jobs_dir / f"{job_id}.json"


def _read_job(paths, job_id: str) -> Dict[str, Any]:
    p = _job_path(paths, job_id)
    job = read_json(p, default=None)
    if not isinstance(job, dict):
        raise RuntimeError(f"Job file missing or corrupted: {p}")
    return job


def _write_job(paths, job: Dict[str, Any]) -> None:
    atomic_write_json(_job_path(paths, job["job_id"]), job)


def _job_state(job: Dict[str, Any]) -> str:
    return str(job.get("state", "queued"))


def _is_terminal(job: Dict[str, Any]) -> bool:
    return _job_state(job) in {"done", "failed", "canceled"}


def _lock_path(paths, job_id: str) -> Path:
    return paths.locks_dir / f"{job_id}.lock"


def _try_create_lock(paths, job_id: str, owner: str) -> bool:
    lp = _lock_path(paths, job_id)
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    try:
        fd = os.open(str(lp), flags)
        try:
            os.write(fd, f"{owner}\n{now_ts()}\n".encode("utf-8", errors="ignore"))
        finally:
            os.close(fd)
        return True
    except FileExistsError:
        return False
    except Exception:
        return False


def _remove_lock_if_exists(paths, job_id: str) -> None:
    try:
        _lock_path(paths, job_id).unlink(missing_ok=True)
    except Exception:
        pass


def _lease_expired(job: Dict[str, Any]) -> bool:
    lease = job.get("lease") or {}
    exp = lease.get("expires_at")
    try:
        return (exp is None) or (float(exp) <= now_ts())
    except Exception:
        return True


def _try_lease(paths, job_id: str, owner: str, lease_sec: float) -> bool:
    job = _read_job(paths, job_id)
    if _is_terminal(job):
        return False

    lease = job.get("lease") or {}
    cur_owner = lease.get("owner")
    if cur_owner and cur_owner != owner and not _lease_expired(job):
        return False

    job["lease"] = {"owner": owner, "expires_at": now_ts() + float(lease_sec)}
    job["state"] = "leased"
    job["stage"] = "leased"
    job["updated_at"] = now_ts()
    _write_job(paths, job)
    return True


def _check_cancel(paths, job_id: str) -> bool:
    job = _read_job(paths, job_id)
    return bool(job.get("cancel_requested", False))


def _set_state(
    paths,
    job_id: str,
    state: str,
    *,
    stage: Optional[str] = None,
    progress: Optional[float] = None,
    message: Optional[str] = None,
    error: Optional[str] = None,
) -> None:
    job = _read_job(paths, job_id)
    job["state"] = state
    if stage is not None:
        job["stage"] = stage
    if progress is not None:
        job["progress"] = float(progress)
    if message is not None:
        job["message"] = message
    job["updated_at"] = now_ts()
    if state == "running" and job.get("started_at") is None:
        job["started_at"] = now_ts()
    if state in {"done", "failed", "canceled"}:
        job["finished_at"] = now_ts()
    if error is not None:
        job["error"] = {"type": "WorkerError", "message": error}
    _write_job(paths, job)


def _time_tag() -> str:
    return time.strftime("%Y%m%dT%H%M%S", time.localtime())


def _enqueue_job(paths, job_id: str) -> None:
    paths.queue_dir.mkdir(parents=True, exist_ok=True)
    qname = f"p010__{_time_tag()}__{job_id}.q"
    (paths.queue_dir / qname).write_text(job_id, encoding="utf-8")


def _create_job(
    paths,
    *,
    video_id: str,
    job_type: str,
    language: str,
    source_language: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    job_id = new_job_id("job")
    job = {
        "job_id": job_id,
        "video_id": video_id,
        "job_type": job_type,
        "language": language,
        "source_language": source_language,
        "state": "queued",
        "stage": "queued",
        "progress": 0.0,
        "message": "Queued",
        "created_at": now_ts(),
        "updated_at": now_ts(),
        "started_at": None,
        "finished_at": None,
        "error": None,
        "cancel_requested": False,
        "lease": None,
        "extra": extra or {},
    }
    _write_job(paths, job)
    _enqueue_job(paths, job_id)
    return job


def _notify_webhook(webhook_cfg: Dict[str, Any], event: str, job: Dict[str, Any]) -> None:
    url = str(webhook_cfg.get("url") or "").strip()
    if not url:
        return
    enabled = bool(webhook_cfg.get("enabled", False))
    if not enabled:
        return
    timeout = float(webhook_cfg.get("timeout_sec", 5.0) or 5.0)
    payload = {
        "event": str(event),
        "job_id": job.get("job_id"),
        "video_id": job.get("video_id"),
        "state": job.get("state"),
        "stage": job.get("stage"),
        "progress": job.get("progress"),
        "message": job.get("message"),
        "job_type": job.get("job_type"),
        "language": job.get("language"),
        "source_language": job.get("source_language"),
        "created_at": job.get("created_at"),
        "started_at": job.get("started_at"),
        "updated_at": job.get("updated_at"),
        "finished_at": job.get("finished_at"),
    }
    try:
        requests.post(url, json=payload, timeout=timeout)
    except Exception:
        return
    try:
        cur = read_json(mp, default=None)
        if not isinstance(cur, dict):
            return
        cur.update(payload)
        atomic_write_json(mp, cur)
    except Exception:
        return


def _queue_paused(paths) -> bool:
    stt = read_json(paths.queue_state_path, default={"paused": False}) or {}
    return bool(stt.get("paused", False))


def worker_main() -> None:
    cfg, raw = load_cfg_and_raw()
    paths = get_backend_paths(cfg, raw)

    paths.jobs_dir.mkdir(parents=True, exist_ok=True)
    paths.queue_dir.mkdir(parents=True, exist_ok=True)
    paths.locks_dir.mkdir(parents=True, exist_ok=True)

    wcfg = raw.get("worker") or {}
    poll_interval = float(wcfg.get("poll_interval_sec", 1))
    lease_sec = float(wcfg.get("lease_sec", 30))
    auto_index = bool((raw.get("index") or {}).get("auto_update_on_done", True))
    webhook_cfg = raw.get("webhook") or {}

    owner = host_id()
    cfg_fp = search_config_fingerprint(cfg)
    pipeline = VideoToTextPipeline(cfg)

    print(f"[worker] owner={owner} jobs={paths.jobs_dir} queue={paths.queue_dir} locks={paths.locks_dir}")
    active_job_id: Optional[str] = None

    while True:
        try:
            if _queue_paused(paths):
                time.sleep(max(0.4, poll_interval))
                continue

            items = _list_queue_items(paths.queue_dir)
            if not items:
                time.sleep(poll_interval)
                continue

            qitem = items[0]
            job_id = _parse_job_id_from_queue_item(qitem)
            if not job_id:
                try:
                    qitem.unlink(missing_ok=True)
                except Exception:
                    pass
                continue

            if not _try_create_lock(paths, job_id, owner):
                try:
                    job = _read_job(paths, job_id)
                    if _lease_expired(job):
                        _remove_lock_if_exists(paths, job_id)
                except Exception:
                    pass
                time.sleep(0.1)
                continue

            try:
                qitem.unlink(missing_ok=True)
            except Exception:
                pass

            if not _try_lease(paths, job_id, owner, lease_sec):
                _remove_lock_if_exists(paths, job_id)
                continue

            active_job_id = job_id
            _set_state(paths, job_id, "running", stage="starting", progress=0.01, message="Started")

            job = _read_job(paths, job_id)
            extra = job.get("extra") or {}
            job_type = str(job.get("job_type") or extra.get("job_type") or "process").strip().lower() or "process"
            language = str(job.get("language") or extra.get("language") or cfg.model.language).strip().lower()
            source_language = str(job.get("source_language") or extra.get("source_language") or "").strip().lower() or None
            device = extra.get("device") or cfg.model.device
            force_overwrite = bool(extra.get("force_overwrite", cfg.runtime.overwrite_existing))
            index_only = bool(extra.get("index_only", False)) or job_type == "index"

            if _check_cancel(paths, job_id):
                _set_state(paths, job_id, "canceled", stage="canceled", progress=0.0, message="Canceled before start")
                try:
                    _notify_webhook(webhook_cfg, "job_canceled", _read_job(paths, job_id))
                except Exception:
                    pass
                _remove_lock_if_exists(paths, job_id)
                active_job_id = None
                continue

            if index_only:
                _set_state(paths, job_id, "indexing", stage="indexing", progress=0.2, message="Index rebuild")
                status = {"languages": {}, "updated_at": now_ts(), "last_error": None}
                built_times: List[float] = []
                for lang in cfg.ui.langs:
                    try:
                        build_or_update_index(
                            videos_root=Path(cfg.paths.videos_dir),
                            index_dir=Path(cfg.paths.indexes_dir),
                            model_name=cfg.search.embed_model_name,
                            config_fingerprint=cfg_fp,
                            language=str(lang),
                            query_prefix=str(getattr(cfg.search, "query_prefix", "query: ")),
                            passage_prefix=str(getattr(cfg.search, "passage_prefix", "passage: ")),
                            normalize_text=bool(getattr(cfg.search, "normalize_text", True)),
                            lemmatize=bool(getattr(cfg.search, "lemmatize", False)),
                            cache_dir=Path(cfg.paths.cache_dir),
                            use_embed_cache=bool(getattr(cfg.search, "embed_cache", True)),
                        )
                        built_at = now_ts()
                        built_times.append(built_at)
                        status["languages"][str(lang)] = {"built_at": built_at, "last_error": None}
                    except Exception as e:
                        status["languages"][str(lang)] = {"built_at": None, "last_error": str(e)}
                        status["last_error"] = str(e)
                status["built_at"] = max(built_times) if built_times else None
                status["version"] = now_ts()
                atomic_write_json(paths.index_state_path, status)
                _set_state(paths, job_id, "done", stage="indexing", progress=1.0, message="Index rebuilt")
                try:
                    _notify_webhook(webhook_cfg, "job_done", _read_job(paths, job_id))
                except Exception:
                    pass
                _remove_lock_if_exists(paths, job_id)
                active_job_id = None
                continue

            video_id = str(job.get("video_id") or "")
            if not video_id:
                raise RuntimeError("Missing video_id in job")

            if job_type == "translate":
                tgt_lang = language
                src_lang = source_language or str(cfg.translation.source_lang or cfg.model.language or "en").lower()
                if not tgt_lang or not src_lang:
                    raise RuntimeError("Missing translation languages")
                if tgt_lang == src_lang:
                    _set_state(paths, job_id, "done", stage="translate", progress=1.0, message="Source equals target")
                    _remove_lock_if_exists(paths, job_id)
                    active_job_id = None
                    continue

                tgt_path = segments_path(Path(cfg.paths.videos_dir), video_id, tgt_lang)
                if tgt_path.exists() and not force_overwrite:
                    _set_state(paths, job_id, "done", stage="translate", progress=1.0, message="Already translated")
                    _remove_lock_if_exists(paths, job_id)
                    active_job_id = None
                    continue

                src_path = segments_path(Path(cfg.paths.videos_dir), video_id, src_lang)
                src_segments = read_segments(src_path)
                if not src_segments and not src_path.exists():
                    raise RuntimeError(f"Source segments not found: {video_id} ({src_lang})")

                _set_state(paths, job_id, "running", stage="translate", progress=0.2, message="Translating")
                update_outputs_manifest(
                    Path(cfg.paths.videos_dir),
                    video_id,
                    tgt_lang,
                    source_lang=src_lang,
                    model_name=str(cfg.translation.model_name_or_path),
                    status="translating",
                    job_id=job_id,
                    note="translate",
                )
                t_translate0 = time.perf_counter()
                translator = NLLBTranslator(
                    model_name_or_path=str(cfg.translation.model_name_or_path),
                    device=str(cfg.translation.device),
                    dtype=str(cfg.translation.dtype),
                )

                texts = [str(x.get("description", "")) for x in src_segments]
                translated: List[str] = [""] * len(texts)
                if cfg.translation.cache_enabled and cfg.paths.cache_dir:
                    cache = TranslationCache(
                        Path(cfg.paths.cache_dir),
                        str(cfg.translation.model_name_or_path),
                        src_lang,
                        tgt_lang,
                    )
                    cached_map = cache.get_many(texts)

                    def _hash_text(t: str) -> str:
                        import hashlib

                        return hashlib.sha1(t.encode("utf-8")).hexdigest()

                    miss_idx: List[int] = []
                    miss_texts: List[str] = []
                    hashes = [_hash_text(t) for t in texts]
                    for i, h in enumerate(hashes):
                        cached = cached_map.get(h)
                        if cached:
                            translated[i] = cached
                        else:
                            miss_idx.append(i)
                            miss_texts.append(texts[i])
                    if miss_texts:
                        miss_translated = translator.translate(
                            miss_texts,
                            src_lang=src_lang,
                            tgt_lang=tgt_lang,
                            batch_size=int(cfg.translation.batch_size),
                            max_new_tokens=int(cfg.translation.max_new_tokens),
                        )
                        for i, t in zip(miss_idx, miss_translated):
                            translated[i] = t
                        cache.put_many(miss_texts, miss_translated)
                else:
                    translated = translator.translate(
                        texts,
                        src_lang=src_lang,
                        tgt_lang=tgt_lang,
                        batch_size=int(cfg.translation.batch_size),
                        max_new_tokens=int(cfg.translation.max_new_tokens),
                    )

                out_segments: List[Dict[str, Any]] = []
                for i, seg in enumerate(src_segments):
                    if not isinstance(seg, dict):
                        continue
                    out = dict(seg)
                    out["description"] = translated[i] if i < len(translated) else str(seg.get("description", ""))
                    out_segments.append(out)

                write_segments(tgt_path, out_segments)

                src_summary = read_summary(summary_path(Path(cfg.paths.videos_dir), video_id, src_lang))
                summary_text = None
                if isinstance(src_summary, dict):
                    summary_text = src_summary.get("summary")
                if summary_text:
                    if cfg.translation.cache_enabled and cfg.paths.cache_dir:
                        cache = TranslationCache(
                            Path(cfg.paths.cache_dir),
                            str(cfg.translation.model_name_or_path),
                            src_lang,
                            tgt_lang,
                        )
                        cached = cache.get_many([str(summary_text)])
                        def _hash_text(t: str) -> str:
                            import hashlib
                            return hashlib.sha1(t.encode("utf-8")).hexdigest()
                        h = _hash_text(str(summary_text))
                        cached_text = cached.get(h)
                        translated_summary = [cached_text] if cached_text else translator.translate(
                            [str(summary_text)],
                            src_lang=src_lang,
                            tgt_lang=tgt_lang,
                            batch_size=1,
                            max_new_tokens=int(cfg.translation.max_new_tokens),
                        )
                        if translated_summary and translated_summary[0] and not cached_text:
                            cache.put_many([str(summary_text)], [translated_summary[0]])
                    else:
                        translated_summary = translator.translate(
                            [str(summary_text)],
                            src_lang=src_lang,
                            tgt_lang=tgt_lang,
                            batch_size=1,
                            max_new_tokens=int(cfg.translation.max_new_tokens),
                        )
                    if translated_summary:
                        write_summary(
                            summary_path(Path(cfg.paths.videos_dir), video_id, tgt_lang),
                            translated_summary[0],
                            tgt_lang,
                            extra={"source_lang": src_lang},
                        )

                update_outputs_manifest(
                    Path(cfg.paths.videos_dir),
                    video_id,
                    tgt_lang,
                    source_lang=src_lang,
                    model_name=str(cfg.translation.model_name_or_path),
                    status="ready",
                    job_id=job_id,
                    note="translate",
                )

                if auto_index:
                    _set_state(paths, job_id, "indexing", stage="indexing", progress=0.88, message="Index update")
                    t_index0 = time.perf_counter()
                    build_or_update_index(
                        videos_root=Path(cfg.paths.videos_dir),
                        index_dir=Path(cfg.paths.indexes_dir),
                        model_name=cfg.search.embed_model_name,
                        config_fingerprint=cfg_fp,
                        language=tgt_lang,
                        query_prefix=str(getattr(cfg.search, "query_prefix", "query: ")),
                        passage_prefix=str(getattr(cfg.search, "passage_prefix", "passage: ")),
                        normalize_text=bool(getattr(cfg.search, "normalize_text", True)),
                        lemmatize=bool(getattr(cfg.search, "lemmatize", False)),
                        cache_dir=Path(cfg.paths.cache_dir),
                        use_embed_cache=bool(getattr(cfg.search, "embed_cache", True)),
                    )
                    index_time = time.perf_counter() - t_index0
                    stt = read_json(paths.index_state_path, default={}) or {}
                    stt["updated_at"] = now_ts()
                    stt["last_error"] = None
                    atomic_write_json(paths.index_state_path, stt)
                    update_metrics(
                        metrics_path(Path(cfg.paths.videos_dir), video_id),
                        {"indexing": {tgt_lang: {"time_sec": float(index_time), "built_at": float(now_ts())}}},
                    )

                update_metrics(
                    metrics_path(Path(cfg.paths.videos_dir), video_id),
                    {
                        "translations": {
                            tgt_lang: {
                                "source_lang": src_lang,
                                "model_name": str(cfg.translation.model_name_or_path),
                                "time_sec": float(time.perf_counter() - t_translate0),
                                "num_segments": int(len(texts)),
                            }
                        }
                    },
                )

                _set_state(paths, job_id, "done", stage="translate", progress=1.0, message="Translated")
                try:
                    _notify_webhook(webhook_cfg, "job_done", _read_job(paths, job_id))
                except Exception:
                    pass
                _remove_lock_if_exists(paths, job_id)
                active_job_id = None
                continue
            video_path = find_video_file(Path(cfg.paths.videos_dir), video_id)
            if video_path is None:
                raise RuntimeError(f"Video not found for job video_id={video_id}")

            base_lang = str(cfg.translation.source_lang or cfg.model.language or "en").strip().lower()
            cfg.model.device = str(device)
            cfg.model.language = base_lang

            base_segments_path = segments_path(Path(cfg.paths.videos_dir), video_id, base_lang)
            if base_segments_path.exists() and not force_overwrite:
                _set_state(paths, job_id, "done", stage="skip", progress=1.0, message="Outputs already exist")
                for lang in cfg.translation.target_langs:
                    tgt_lang = str(lang).strip().lower()
                    if tgt_lang and tgt_lang != base_lang:
                        tgt_path = segments_path(Path(cfg.paths.videos_dir), video_id, tgt_lang)
                        if tgt_path.exists() and not force_overwrite:
                            continue
                        _create_job(
                            paths,
                            video_id=video_id,
                            job_type="translate",
                            language=tgt_lang,
                            source_language=base_lang,
                            extra={"force_overwrite": force_overwrite},
                        )
                _remove_lock_if_exists(paths, job_id)
                active_job_id = None
                continue

            update_outputs_manifest(
                Path(cfg.paths.videos_dir),
                video_id,
                base_lang,
                source_lang=None,
                model_name=str(cfg.model.model_name_or_path),
                status="processing",
                job_id=job_id,
                note="process",
            )

            _set_state(paths, job_id, "running", stage="preprocess", progress=0.08, message="Preprocessing")
            video_meta = preprocess_video(video_path, cfg)

            if _check_cancel(paths, job_id):
                _set_state(paths, job_id, "canceled", stage="canceled", progress=0.0, message="Canceled after preprocess")
                try:
                    _notify_webhook(webhook_cfg, "job_canceled", _read_job(paths, job_id))
                except Exception:
                    pass
                _remove_lock_if_exists(paths, job_id)
                active_job_id = None
                continue

            clips, clip_ts = build_clips_from_video_meta(
                video_meta=video_meta,
                window_sec=cfg.clips.window_sec,
                stride_sec=cfg.clips.stride_sec,
                min_clip_frames=cfg.clips.min_clip_frames,
                max_clip_frames=cfg.clips.max_clip_frames,
            )

            _set_state(paths, job_id, "running", stage="inference", progress=0.20, message="Inference")

            annotations, metrics = pipeline.run(
                video_id=video_meta.video_id,
                video_duration_sec=float(video_meta.duration_sec),
                clips=clips,
                clip_timestamps=clip_ts,
                preprocess_time_sec=float((video_meta.extra or {}).get("preprocess_time_sec", 0.0)),
            )

            if _check_cancel(paths, job_id):
                _set_state(paths, job_id, "canceled", stage="canceled", progress=0.0, message="Canceled after inference")
                try:
                    _notify_webhook(webhook_cfg, "job_canceled", _read_job(paths, job_id))
                except Exception:
                    pass
                _remove_lock_if_exists(paths, job_id)
                active_job_id = None
                continue

            _set_state(paths, job_id, "running", stage="saving", progress=0.85, message="Saving outputs")

            ann_payload: List[Dict[str, Any]] = []
            for a in annotations or []:
                if isinstance(a, dict):
                    ann_payload.append(a)
                else:
                    ann_payload.append(
                        {
                            "video_id": getattr(a, "video_id", video_meta.video_id),
                            "start_sec": float(getattr(a, "start_sec", 0.0)),
                            "end_sec": float(getattr(a, "end_sec", 0.0)),
                            "description": str(getattr(a, "description", "")),
                            "extra": getattr(a, "extra", None),
                        }
                    )

            met_payload = (
                metrics
                if isinstance(metrics, dict)
                else getattr(metrics, "model_dump", lambda: None)() or getattr(metrics, "__dict__", {})
            )
            if isinstance(met_payload, dict):
                met_payload.setdefault("language", base_lang)
                met_payload.setdefault("device", str(device))
            write_segments(base_segments_path, ann_payload)
            write_metrics(metrics_path(Path(cfg.paths.videos_dir), video_id), met_payload or {})

            summary_text = None
            if isinstance(met_payload, dict):
                summary_text = (met_payload.get("extra") or {}).get("global_summary")
            if summary_text:
                write_summary(
                    summary_path(Path(cfg.paths.videos_dir), video_id, base_lang),
                    str(summary_text),
                    base_lang,
                )
            update_outputs_manifest(
                Path(cfg.paths.videos_dir),
                video_id,
                base_lang,
                source_lang=None,
                model_name=str(cfg.model.model_name_or_path),
                status="ready",
                job_id=job_id,
                note="process",
            )

            if auto_index:
                _set_state(paths, job_id, "indexing", stage="indexing", progress=0.92, message="Index update")
                t_index0 = time.perf_counter()
                build_or_update_index(
                    videos_root=Path(cfg.paths.videos_dir),
                    index_dir=Path(cfg.paths.indexes_dir),
                    model_name=cfg.search.embed_model_name,
                    config_fingerprint=cfg_fp,
                    language=base_lang,
                    query_prefix=str(getattr(cfg.search, "query_prefix", "query: ")),
                    passage_prefix=str(getattr(cfg.search, "passage_prefix", "passage: ")),
                    normalize_text=bool(getattr(cfg.search, "normalize_text", True)),
                    lemmatize=bool(getattr(cfg.search, "lemmatize", False)),
                    cache_dir=Path(cfg.paths.cache_dir),
                    use_embed_cache=bool(getattr(cfg.search, "embed_cache", True)),
                )
                index_time = time.perf_counter() - t_index0
                stt = {"built_at": now_ts(), "updated_at": now_ts(), "version": now_ts(), "last_error": None}
                atomic_write_json(paths.index_state_path, stt)
                update_metrics(
                    metrics_path(Path(cfg.paths.videos_dir), video_id),
                    {"indexing": {base_lang: {"time_sec": float(index_time), "built_at": float(stt["built_at"])}}},
                )

            for lang in cfg.translation.target_langs:
                tgt_lang = str(lang).strip().lower()
                if not tgt_lang or tgt_lang == base_lang:
                    continue
                tgt_path = segments_path(Path(cfg.paths.videos_dir), video_id, tgt_lang)
                if tgt_path.exists() and not force_overwrite:
                    continue
                _create_job(
                    paths,
                    video_id=video_id,
                    job_type="translate",
                    language=tgt_lang,
                    source_language=base_lang,
                    extra={"force_overwrite": force_overwrite},
                )

            _set_state(paths, job_id, "done", stage="completed", progress=1.0, message="Finished")
            try:
                _notify_webhook(webhook_cfg, "job_done", _read_job(paths, job_id))
            except Exception:
                pass

            _remove_lock_if_exists(paths, job_id)
            active_job_id = None

        except Exception as e:
            err = f"{e}\n{traceback.format_exc()}"
            if active_job_id:
                try:
                    job = _read_job(paths, active_job_id)
                    job_type = str(job.get("job_type") or "").lower()
                    v_id = str(job.get("video_id") or "")
                    lang = str(job.get("language") or "").lower()
                    src_lang = str(job.get("source_language") or "").lower()
                    if v_id and job_type in {"process", "translate"} and lang:
                        update_outputs_manifest(
                            Path(cfg.paths.videos_dir),
                            v_id,
                            lang,
                            source_lang=(src_lang or None),
                            model_name=str(cfg.translation.model_name_or_path if job_type == "translate" else cfg.model.model_name_or_path),
                            status="failed",
                            error=str(e),
                            job_id=active_job_id,
                            note=job_type,
                        )
                    _set_state(
                        paths,
                        active_job_id,
                        "failed",
                        stage="failed",
                        progress=0.0,
                        message="Failed",
                        error=str(e),
                    )
                    try:
                        _notify_webhook(webhook_cfg, "job_failed", _read_job(paths, active_job_id))
                    except Exception:
                        pass
                except Exception:
                    pass
                try:
                    _remove_lock_if_exists(paths, active_job_id)
                except Exception:
                    pass
                active_job_id = None

            try:
                stt = read_json(paths.index_state_path, default={}) or {}
                stt["updated_at"] = now_ts()
                stt["last_error"] = str(e)
                atomic_write_json(paths.index_state_path, stt)
            except Exception:
                pass

            print(f"[worker] ERROR:\n{err}")
            time.sleep(0.5)


if __name__ == "__main__":
    worker_main()
