# backend/api.py
"""
FastAPI backend for SmartCampus V2T.

Purpose:
- Provide the stable HTTP entrypoint for Streamlit UI and worker-facing backend operations.
- Keep route registration centralized while delegating shared logic to smaller backend modules.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict, Optional, List

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from backend.http.common import get_api_context, normalize_uploaded_video as _normalize_uploaded_video
from backend.http.grounded import build_qa_response, build_rag_response, build_report_response
from backend.deps import (
    atomic_write_json,
    list_videos,
    now_ts,
    read_json,
    read_video_outputs,
)
from backend.jobs.index_runtime import rebuild_index_status as _rebuild_index_status, write_index_state as _write_index_state
from backend.jobs.control import create_job as _create_job
from backend.jobs.queue_runtime import (
    cancel_queued_job as _cancel_queued_job,
    move_job_in_queue as _move_job_in_queue,
    queue_snapshot as _queue_snapshot,
    read_job_or_404 as _read_job,
    set_queue_paused as _set_queue_paused,
)
from backend.retrieval_runtime import (
    annotation_hits as _annotation_hits,
    resolve_request_language as _resolve_request_language,
    search_hits as _search_hits,
    search_hits_with_fallback as _search_hits_with_fallback,
    guard_query_text as _guard_query_text,
    metrics_summary_from_outputs as _metrics_summary_from_outputs,
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


@app.get("/v1/videos", response_model=List[VideoItem])
def videos_list():
    context = get_api_context()
    items = list_videos(context.paths.videos_dir)
    return [VideoItem(**x) for x in items]


@app.post("/v1/videos/upload", response_model=VideoItem)
def videos_upload(file: UploadFile = File(...)):
    context = get_api_context()
    paths = context.paths
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
    context = get_api_context()
    vdir = Path(context.paths.videos_dir) / video_id
    if not vdir.exists():
        raise HTTPException(status_code=404, detail=f"Video not found: {video_id}")
    try:
        shutil.rmtree(vdir, ignore_errors=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"ok": True}


@app.get("/v1/videos/{video_id}/outputs", response_model=VideoOutputs)
def video_outputs(video_id: str, lang: str, variant: Optional[str] = None):
    context = get_api_context()
    lang = (lang or "").strip().lower()
    variant = (variant or "").strip().lower() or None
    if not lang:
        raise HTTPException(status_code=400, detail="Missing language")

    out = read_video_outputs(Path(context.paths.videos_dir), video_id, lang, variant=variant)
    if (
        not out.get("manifest")
        and not out.get("batch_manifest")
        and not out.get("annotations")
        and not out.get("clip_observations")
        and out.get("global_summary") is None
    ):
        suffix = f", variant={variant}" if variant else ""
        raise HTTPException(status_code=404, detail=f"Outputs not found: {video_id} ({lang}{suffix})")
    return VideoOutputs(**out)


@app.get("/v1/videos/{video_id}/batch-manifest")
def video_batch_manifest(video_id: str):
    context = get_api_context()
    from src.utils.video_store import batch_manifest_path

    payload = read_json(batch_manifest_path(Path(context.paths.videos_dir), video_id), default=None)
    if not isinstance(payload, dict):
        raise HTTPException(status_code=404, detail=f"Batch manifest not found: {video_id}")
    return payload


@app.get("/v1/videos/{video_id}/metrics-summary", response_model=MetricsSummaryResponse)
def video_metrics_summary(video_id: str, lang: str = "en", variant: Optional[str] = None):
    context = get_api_context()
    language = (lang or "").strip().lower() or "en"
    resolved_variant = (variant or "").strip().lower() or None
    outputs = read_video_outputs(Path(context.paths.videos_dir), video_id, language, variant=resolved_variant)
    if not isinstance(outputs.get("metrics"), dict):
        suffix = f", variant={resolved_variant}" if resolved_variant else ""
        raise HTTPException(status_code=404, detail=f"Metrics not found: {video_id} ({language}{suffix})")
    return _metrics_summary_from_outputs(video_id, language, outputs)


@app.post("/v1/jobs", response_model=JobCreateResponse)
def jobs_create(req: JobCreateRequest):
    context = get_api_context()
    cfg = context.cfg
    paths = context.paths

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

    job = _create_job(
        paths,
        video_id=req.video_id,
        job_type=job_type,
        profile=profile,
        variant=variant,
        language=language,
        source_language=source_language,
        extra=extra,
    )
    return JobCreateResponse(job_id=job["job_id"], state=job["state"], stage=job["stage"])


@app.get("/v1/jobs/{job_id}", response_model=JobStatus)
def jobs_status(job_id: str):
    context = get_api_context()
    job = _read_job(context.paths, job_id)
    return JobStatus(**job)


@app.post("/v1/jobs/{job_id}/cancel", response_model=JobCancelResponse)
def jobs_cancel(job_id: str):
    context = get_api_context()
    paths = context.paths
    job = _read_job(paths, job_id)
    if job.get("state") in {"done", "failed", "canceled"}:
        return JobCancelResponse(job_id=job_id, state=str(job.get("state")))

    job["cancel_requested"] = True
    job["updated_at"] = now_ts()
    _write_job(paths, job)
    return JobCancelResponse(job_id=job_id, state="cancel_requested")


@app.get("/v1/queue", response_model=QueueListResponse)
def queue_list():
    context = get_api_context()
    snapshot = _queue_snapshot(context.paths)
    running_payload = snapshot.get("running")
    return QueueListResponse(
        status=QueueStatus(**dict(snapshot.get("status") or {})),
        running=QueueRunningItem(**running_payload) if isinstance(running_payload, dict) else None,
        queued=[QueueItem(**item) for item in list(snapshot.get("queued") or [])],
    )


@app.post("/v1/queue/pause", response_model=QueueStatus)
def queue_pause():
    context = get_api_context()
    stt = _set_queue_paused(context.paths, True)
    return QueueStatus(**stt)


@app.post("/v1/queue/resume", response_model=QueueStatus)
def queue_resume():
    context = get_api_context()
    stt = _set_queue_paused(context.paths, False)
    return QueueStatus(**stt)


@app.post("/v1/queue/move", response_model=QueueMoveResponse)
def queue_move(req: QueueMoveRequest):
    context = get_api_context()
    result = _move_job_in_queue(context.paths, req.job_id, req.direction, int(req.steps or 1))
    return QueueMoveResponse(ok=True, **result)


@app.delete("/v1/queue/{job_id}")
def queue_remove(job_id: str):
    context = get_api_context()
    return _cancel_queued_job(context.paths, job_id)


@app.get("/v1/index/status", response_model=IndexStatus)
def index_status():
    context = get_api_context()
    stt = read_json(context.paths.index_state_path, default={}) or {}
    return IndexStatus(**stt)


@app.post("/v1/index/rebuild", response_model=IndexRebuildResponse)
def index_rebuild():
    context = get_api_context()
    cfg = context.cfg
    paths = context.paths
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
    context = get_api_context()
    cfg = context.cfg
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
    return build_report_response(req, get_api_context())


@app.post("/v1/qa", response_model=QaResponse)
def qa(req: QaRequest):
    return build_qa_response(req, get_api_context())


@app.post("/v1/rag", response_model=RagResponse)
def rag(req: RagRequest):
    return build_rag_response(req, get_api_context())
