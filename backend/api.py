# backend/api.py
"""
FastAPI backend for SmartCampus V2T.

Purpose:
- Provide HTTP API for Streamlit UI:
  - videos: list/upload/delete
  - runs: list/get outputs/delete
  - jobs: enqueue processing (worker consumes filesystem queue), status, cancel
  - queue: pause/resume/status/list/remove queued job + reorder
  - search: execute hybrid search over built index
  - index: rebuild/status
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
import time
from typing import Any, Dict, Optional, List

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from backend.deps import (
    atomic_write_json,
    get_backend_paths,
    list_videos,
    load_cfg_and_raw,
    new_job_id,
    now_ts,
    read_json,
    read_video_outputs,
)
from backend.index_runtime import rebuild_index_status as _rebuild_index_status, write_index_state as _write_index_state
from backend.job_queue_runtime import (
    build_job_record as _build_job_record,
    cancel_queued_job as _cancel_queued_job,
    enqueue_job as _enqueue_job,
    move_job_in_queue as _move_job_in_queue,
    queue_snapshot as _queue_snapshot,
    queue_status as _queue_status,
    read_job_or_404 as _read_job,
    set_queue_paused as _set_queue_paused,
    write_job as _write_job,
)
from backend.retrieval_runtime import (
    annotation_hits as _annotation_hits,
    generate_grounded_text as _generate_grounded_text,
    resolve_request_language as _resolve_request_language,
    search_hits as _search_hits,
    search_hits_with_fallback as _search_hits_with_fallback,
    guard_output_text as _guard_output_text,
    guard_query_text as _guard_query_text,
    hit_to_citation as _hit_to_citation,
    hit_to_schema as _hit_to_schema,
    metrics_summary_from_outputs as _metrics_summary_from_outputs,
    qa_text_from_hits as _qa_text_from_hits,
    report_text_from_hits as _report_text_from_hits,
    translate_output_text as _translate_output_text,
)
from backend.schemas import (
    Citation,
    MetricsSummaryResponse,
    VideoItem,
    VideoOutputs,
    JobCreateRequest,
    JobCreateResponse,
    JobStatus,
    JobCancelResponse,
    SearchRequest,
    SearchResponse,
    SearchHit,
    IndexStatus,
    IndexRebuildResponse,
    ReportRequest,
    ReportResponse,
    QaRequest,
    QaResponse,
    RagRequest,
    RagResponse,
    QueueStatus,
    QueueItem,
    QueueRunningItem,
    QueueListResponse,
    QueueMoveRequest,
    QueueMoveResponse,
)


app = FastAPI(title="SmartCampus V2T Backend", version="0.1.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.get("/v1/health")
def v1_health():
    return {"ok": True}


@app.get("/v1/healthz")
def v1_healthz():
    return {"ok": True}


def _get_cfg_paths():
    cfg, raw = load_cfg_and_raw()
    paths = get_backend_paths(cfg, raw)
    return cfg, raw, paths

def _normalize_uploaded_video(raw_path: Path) -> Path:
    """Convert uploaded videos to mp4 when ffmpeg is available."""

    if raw_path.suffix.lower() == ".mp4":
        return raw_path
    if not shutil.which("ffmpeg"):
        return raw_path

    mp4_path = raw_path.with_suffix(".mp4")
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(raw_path),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        str(mp4_path),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode == 0 and mp4_path.exists() and mp4_path.stat().st_size > 0:
            try:
                raw_path.unlink(missing_ok=True)
            except Exception:
                pass
            return mp4_path
    except Exception:
        pass
    return raw_path


@app.get("/v1/videos", response_model=List[VideoItem])
def videos_list():
    cfg, raw, paths = _get_cfg_paths()
    items = list_videos(paths.videos_dir)
    return [VideoItem(**x) for x in items]


@app.post("/v1/videos/upload", response_model=VideoItem)
def videos_upload(file: UploadFile = File(...)):
    cfg, raw, paths = _get_cfg_paths()
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")
    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".mpeg", ".mpg"}:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")

    from src.utils.video_store import ensure_video_dirs, list_output_languages, video_manifest_path

    video_id = Path(file.filename).stem
    dirs = ensure_video_dirs(Path(paths.videos_dir), video_id)
    target = dirs["raw"] / Path(file.filename).name
    try:
        with target.open("wb") as out:
            shutil.copyfileobj(file.file, out, length=1024 * 1024)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            file.file.close()
        except Exception:
            pass

    target = _normalize_uploaded_video(target)
    stt = target.stat()
    atomic_write_json(
        video_manifest_path(Path(paths.videos_dir), video_id),
        {
            "video_id": video_id,
            "filename": target.name,
            "path": str(target),
            "size_bytes": int(stt.st_size),
            "mtime": float(stt.st_mtime),
            "uploaded_at": now_ts(),
        },
    )

    return VideoItem(
        video_id=video_id,
        filename=target.name,
        path=str(target),
        size_bytes=int(stt.st_size),
        mtime=float(stt.st_mtime),
        languages=list_output_languages(Path(paths.videos_dir), video_id),
    )


@app.delete("/v1/videos/{video_id}")
def videos_delete(video_id: str):
    cfg, raw, paths = _get_cfg_paths()
    vdir = Path(paths.videos_dir) / video_id
    if not vdir.exists():
        raise HTTPException(status_code=404, detail=f"Video not found: {video_id}")
    try:
        shutil.rmtree(vdir, ignore_errors=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"ok": True}


@app.get("/v1/videos/{video_id}/outputs", response_model=VideoOutputs)
def video_outputs(video_id: str, lang: str, variant: Optional[str] = None):
    cfg, raw, paths = _get_cfg_paths()
    lang = (lang or "").strip().lower()
    variant = (variant or "").strip().lower() or None
    if not lang:
        raise HTTPException(status_code=400, detail="Missing language")

    out = read_video_outputs(Path(paths.videos_dir), video_id, lang, variant=variant)
    if (
        not out.get("manifest")
        and not out.get("batch_manifest")
        and not out.get("annotations")
        and out.get("global_summary") is None
    ):
        suffix = f", variant={variant}" if variant else ""
        raise HTTPException(status_code=404, detail=f"Outputs not found: {video_id} ({lang}{suffix})")
    return VideoOutputs(**out)


@app.get("/v1/videos/{video_id}/batch-manifest")
def video_batch_manifest(video_id: str):
    cfg, raw, paths = _get_cfg_paths()
    from src.utils.video_store import batch_manifest_path

    payload = read_json(batch_manifest_path(Path(paths.videos_dir), video_id), default=None)
    if not isinstance(payload, dict):
        raise HTTPException(status_code=404, detail=f"Batch manifest not found: {video_id}")
    return payload


@app.get("/v1/videos/{video_id}/metrics-summary", response_model=MetricsSummaryResponse)
def video_metrics_summary(video_id: str, lang: str = "en", variant: Optional[str] = None):
    cfg, raw, paths = _get_cfg_paths()
    language = (lang or "").strip().lower() or "en"
    resolved_variant = (variant or "").strip().lower() or None
    outputs = read_video_outputs(Path(paths.videos_dir), video_id, language, variant=resolved_variant)
    if not isinstance(outputs.get("metrics"), dict):
        suffix = f", variant={resolved_variant}" if resolved_variant else ""
        raise HTTPException(status_code=404, detail=f"Metrics not found: {video_id} ({language}{suffix})")
    return _metrics_summary_from_outputs(video_id, language, outputs)


@app.post("/v1/jobs", response_model=JobCreateResponse)
def jobs_create(req: JobCreateRequest):
    cfg, raw, paths = _get_cfg_paths()

    from src.utils.video_store import find_video_file

    if find_video_file(Path(paths.videos_dir), req.video_id) is None:
        raise HTTPException(status_code=404, detail=f"Video not found: {req.video_id}")

    extra = req.extra or {}
    job_type = str(extra.get("job_type", "process")).strip().lower() or "process"
    language = str(extra.get("language", cfg.model.language)).strip().lower() or str(cfg.model.language)
    source_language = extra.get("source_language")
    profile = str(req.profile or extra.get("profile") or cfg.active_profile).strip().lower() or cfg.active_profile
    variant = req.variant or extra.get("variant")
    if variant is not None:
        variant = str(variant).strip().lower() or None
    extra.setdefault("profile", profile)
    if variant:
        extra["variant"] = variant

    job_id = new_job_id("job")
    job = _build_job_record(
        job_id=job_id,
        video_id=req.video_id,
        job_type=job_type,
        profile=profile,
        variant=variant,
        language=language,
        source_language=source_language,
        extra=extra,
    )

    _write_job(paths, job)
    _enqueue_job(paths, job_id)
    return JobCreateResponse(job_id=job_id, state=job["state"], stage=job["stage"])


@app.get("/v1/jobs/{job_id}", response_model=JobStatus)
def jobs_status(job_id: str):
    cfg, raw, paths = _get_cfg_paths()
    job = _read_job(paths, job_id)
    return JobStatus(**job)


@app.post("/v1/jobs/{job_id}/cancel", response_model=JobCancelResponse)
def jobs_cancel(job_id: str):
    cfg, raw, paths = _get_cfg_paths()
    job = _read_job(paths, job_id)
    if job.get("state") in {"done", "failed", "canceled"}:
        return JobCancelResponse(job_id=job_id, state=str(job.get("state")))

    job["cancel_requested"] = True
    job["updated_at"] = now_ts()
    _write_job(paths, job)
    return JobCancelResponse(job_id=job_id, state="cancel_requested")


@app.get("/v1/queue", response_model=QueueListResponse)
def queue_list():
    cfg, raw, paths = _get_cfg_paths()
    snapshot = _queue_snapshot(paths)
    running_payload = snapshot.get("running")
    return QueueListResponse(
        status=QueueStatus(**dict(snapshot.get("status") or {})),
        running=QueueRunningItem(**running_payload) if isinstance(running_payload, dict) else None,
        queued=[QueueItem(**item) for item in list(snapshot.get("queued") or [])],
    )


@app.post("/v1/queue/pause", response_model=QueueStatus)
def queue_pause():
    cfg, raw, paths = _get_cfg_paths()
    stt = _set_queue_paused(paths, True)
    return QueueStatus(**stt)


@app.post("/v1/queue/resume", response_model=QueueStatus)
def queue_resume():
    cfg, raw, paths = _get_cfg_paths()
    stt = _set_queue_paused(paths, False)
    return QueueStatus(**stt)


@app.post("/v1/queue/move", response_model=QueueMoveResponse)
def queue_move(req: QueueMoveRequest):
    cfg, raw, paths = _get_cfg_paths()
    result = _move_job_in_queue(paths, req.job_id, req.direction, int(req.steps or 1))
    return QueueMoveResponse(ok=True, **result)


@app.delete("/v1/queue/{job_id}")
def queue_remove(job_id: str):
    cfg, raw, paths = _get_cfg_paths()
    return _cancel_queued_job(paths, job_id)


@app.get("/v1/index/status", response_model=IndexStatus)
def index_status():
    cfg, raw, paths = _get_cfg_paths()
    stt = read_json(paths.index_state_path, default={}) or {}
    return IndexStatus(**stt)


@app.post("/v1/index/rebuild", response_model=IndexRebuildResponse)
def index_rebuild():
    cfg, raw, paths = _get_cfg_paths()
    try:
        from src.search.builder import search_config_fingerprint

        cfg_fp = search_config_fingerprint(cfg)
        stt = _rebuild_index_status(cfg=cfg, cfg_fp=cfg_fp)
        atomic_write_json(paths.index_state_path, stt)
        return IndexRebuildResponse(ok=True, status=IndexStatus(**stt))
    except Exception as e:
        stt = _write_index_state(paths, built=False, last_error=str(e))
        return IndexRebuildResponse(ok=False, status=IndexStatus(**stt))


@app.post("/v1/search", response_model=SearchResponse)
def search(req: SearchRequest):
    cfg, raw, paths = _get_cfg_paths()
    lang = _resolve_request_language(cfg, req.language)
    _guard_query_text(cfg, req.query, endpoint="Search")

    query = req.query.strip()
    search_filters = {
        "start_sec": req.start_sec,
        "end_sec": req.end_sec,
        "min_duration_sec": req.min_duration_sec,
        "max_duration_sec": req.max_duration_sec,
        "event_type": req.event_type,
        "risk_level": req.risk_level,
        "tags": req.tags,
        "objects": req.objects,
        "people_count_bucket": req.people_count_bucket,
        "motion_type": req.motion_type,
        "anomaly_only": req.anomaly_only,
        "variant": req.variant,
    }
    hits = _search_hits_with_fallback(
        cfg,
        query=query,
        language=lang,
        variant=req.variant,
        top_k=max(1, int(req.top_k)),
        video_id=req.video_id,
        filters=search_filters,
        dedupe=bool(req.dedupe),
    )
    return SearchResponse(hits=hits)


@app.post("/v1/reports", response_model=ReportResponse)
def reports(req: ReportRequest):
    started_at = time.perf_counter()
    cfg, raw, paths = _get_cfg_paths()
    lang = _resolve_request_language(cfg, req.language)

    if not (str(req.query or "").strip() or str(req.video_id or "").strip()):
        raise HTTPException(status_code=400, detail="Report request requires a query or video_id.")
    if str(req.query or "").strip():
        _guard_query_text(cfg, str(req.query or ""), endpoint="Reports")

    supporting_hits: List[Any] = []
    if str(req.query or "").strip():
        supporting_hits = _search_hits(
            cfg,
            query=str(req.query or "").strip(),
            language=lang,
            variant=req.variant,
            top_k=max(1, int(req.top_k)),
            video_id=req.video_id,
            filters={"variant": req.variant} if req.variant else None,
            dedupe=True,
        )
    elif req.video_id:
        outputs = read_video_outputs(Path(paths.videos_dir), str(req.video_id), lang, variant=req.variant)
        supporting_hits = _annotation_hits(
            video_id=str(req.video_id),
            language=lang,
            annotations=list(outputs.get("annotations") or []),
            variant=req.variant,
            top_k=max(1, int(req.top_k)),
        )

    fallback_report = _report_text_from_hits(supporting_hits, req.video_id)
    report_text, mode, _context = _generate_grounded_text(
        cfg,
        task="grounded report",
        user_input=str(req.query or req.video_id or "report"),
        hits=supporting_hits,
        fallback_text=fallback_report,
    )
    report_text = _translate_output_text(cfg, report_text, target_lang=lang, target_name="reports")

    return ReportResponse(
        language=lang,
        variant=req.variant,
        report=_guard_output_text(cfg, report_text),
        mode=mode,
        latency_sec=float(time.perf_counter() - started_at),
        hit_count=int(len(supporting_hits)),
        citations=[_hit_to_citation(hit) for hit in supporting_hits],
        supporting_hits=[_hit_to_schema(hit) for hit in supporting_hits],
    )


@app.post("/v1/qa", response_model=QaResponse)
def qa(req: QaRequest):
    started_at = time.perf_counter()
    cfg, raw, paths = _get_cfg_paths()
    lang = _resolve_request_language(cfg, req.language)
    question = str(req.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is required.")
    _guard_query_text(cfg, question, endpoint="QA")
    if bool(getattr(cfg.guard, "enabled", False)) and bool(getattr(cfg.guard, "query_gate", False)) and len(question) < 3:
        raise HTTPException(status_code=400, detail="Question is too short for QA.")

    hits = _search_hits(
        cfg,
        query=question,
        language=lang,
        variant=req.variant,
        top_k=max(1, int(req.top_k)),
        video_id=req.video_id,
        filters={"variant": req.variant} if req.variant else None,
        dedupe=True,
    )

    fallback_answer = _qa_text_from_hits(question, hits)
    answer_text, mode, context = _generate_grounded_text(
        cfg,
        task="grounded question answering",
        user_input=question,
        hits=hits,
        fallback_text=fallback_answer,
    )
    answer_text = _translate_output_text(cfg, answer_text, target_lang=lang, target_name="qa")

    return QaResponse(
        language=lang,
        variant=req.variant,
        answer=_guard_output_text(cfg, answer_text),
        mode=mode,
        context=context,
        latency_sec=float(time.perf_counter() - started_at),
        hit_count=int(len(hits)),
        citations=[_hit_to_citation(hit) for hit in hits],
        supporting_hits=[_hit_to_schema(hit) for hit in hits],
    )


@app.post("/v1/rag", response_model=RagResponse)
def rag(req: RagRequest):
    started_at = time.perf_counter()
    cfg, raw, paths = _get_cfg_paths()
    lang = _resolve_request_language(cfg, req.language)
    query = str(req.query or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="RAG query is required.")
    _guard_query_text(cfg, query, endpoint="RAG")

    hits = _search_hits(
        cfg,
        query=query,
        language=lang,
        variant=req.variant,
        top_k=max(1, int(req.top_k)),
        video_id=req.video_id,
        filters={"variant": req.variant} if req.variant else None,
        dedupe=True,
    )

    fallback_answer = _qa_text_from_hits(query, hits)
    answer_text, mode, context = _generate_grounded_text(
        cfg,
        task="grounded retrieval augmented answer",
        user_input=query,
        hits=hits,
        fallback_text=fallback_answer,
    )
    answer_text = _translate_output_text(cfg, answer_text, target_lang=lang, target_name="qa")

    return RagResponse(
        language=lang,
        variant=req.variant,
        answer=_guard_output_text(cfg, answer_text),
        mode=mode,
        context=context,
        latency_sec=float(time.perf_counter() - started_at),
        hit_count=int(len(hits)),
        citations=[_hit_to_citation(hit) for hit in hits],
        supporting_hits=[_hit_to_schema(hit) for hit in hits],
    )
