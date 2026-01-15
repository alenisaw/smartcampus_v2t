# backend/api.py
"""
FastAPI backend for SmartCampus V2T.

Purpose:
- Provide HTTP API for Streamlit UI:
  - videos: list/upload/delete
  - runs: list/get outputs/delete
  - jobs: enqueue processing (worker consumes filesystem queue), status, cancel
  - queue: pause/resume/status/list/remove queued job
  - search: execute hybrid search over built index
  - index: rebuild/status
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict, Optional, List

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from backend.deps import (
    atomic_write_json,
    get_backend_paths,
    list_all_runs,
    list_raw_videos,
    load_cfg_and_raw,
    new_job_id,
    now_ts,
    read_json,
    read_run_outputs,
)
from backend.schemas import (
    VideoItem,
    RunsMap,
    RunOutputs,
    JobCreateRequest,
    JobCreateResponse,
    JobStatus,
    JobCancelResponse,
    SearchRequest,
    SearchResponse,
    SearchHit,
    IndexStatus,
    IndexRebuildResponse,
    QueueStatus,
    QueueItem,
    QueueListResponse,
)

from src.search import QueryEngine, build_or_update_index


app = FastAPI(title="SmartCampus V2T Backend", version="0.1.0")

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


def _job_path(paths, job_id: str) -> Path:
    return paths.jobs_dir / f"{job_id}.json"


def _read_job(paths, job_id: str) -> Dict[str, Any]:
    job = read_json(_job_path(paths, job_id), default=None)
    if not isinstance(job, dict):
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    return job


def _write_job(paths, job: Dict[str, Any]) -> None:
    atomic_write_json(_job_path(paths, job["job_id"]), job)


def time_tag() -> str:
    import time
    return time.strftime("%Y%m%dT%H%M%S", time.localtime())


def _enqueue_job(paths, job_id: str) -> None:
    paths.queue_dir.mkdir(parents=True, exist_ok=True)
    qname = f"p010__{time_tag()}__{job_id}.q"
    (paths.queue_dir / qname).write_text(job_id, encoding="utf-8")


def _find_queue_files_for_job(paths, job_id: str) -> List[Path]:
    return sorted(paths.queue_dir.glob(f"p*__*__{job_id}.q"))


def _queue_status(paths) -> Dict[str, Any]:
    stt = read_json(paths.queue_state_path, default={"paused": False, "updated_at": None}) or {}
    if "paused" not in stt:
        stt["paused"] = False
    if "updated_at" not in stt:
        stt["updated_at"] = now_ts()
    return stt


def _set_queue_paused(paths, paused: bool) -> Dict[str, Any]:
    stt = _queue_status(paths)
    stt["paused"] = bool(paused)
    stt["updated_at"] = now_ts()
    atomic_write_json(paths.queue_state_path, stt)
    return stt


_ENGINE_CACHE: Dict[str, Any] = {"ver": None, "qe": None}


def _index_version(index_dir: Path) -> float:
    p1 = index_dir / "manifest.json"
    p2 = index_dir / "meta.json"
    v = 0.0
    for p in [p1, p2]:
        try:
            v = max(v, float(p.stat().st_mtime))
        except Exception:
            pass
    return v


def _get_engine(cfg) -> Optional[QueryEngine]:
    ver = _index_version(Path(cfg.paths.indexes_dir))
    if ver <= 0:
        return None
    if _ENGINE_CACHE["qe"] is None or _ENGINE_CACHE["ver"] != ver:
        _ENGINE_CACHE["qe"] = QueryEngine(
            index_dir=Path(cfg.paths.indexes_dir),
            w_bm25=float(cfg.search.w_bm25),
            w_dense=float(cfg.search.w_dense),
            candidate_k_sparse=int(getattr(cfg.search, "candidate_k_sparse", 200)),
            candidate_k_dense=int(getattr(cfg.search, "candidate_k_dense", 200)),
            fusion=str(getattr(cfg.search, "fusion", "rrf")),
            rrf_k=int(getattr(cfg.search, "rrf_k", 60)),
            dedupe_mode=str(getattr(cfg.search, "dedupe_mode", "span")),
            dedupe_tol_sec=float(getattr(cfg.search, "dedupe_tol_sec", 0.5)),
            dedupe_overlap_thr=float(getattr(cfg.search, "dedupe_overlap_thr", 0.3)),
            embed_model_name=str(cfg.search.embed_model_name),
        )
        _ENGINE_CACHE["ver"] = ver
    return _ENGINE_CACHE["qe"]


@app.get("/v1/videos", response_model=List[VideoItem])
def videos_list():
    cfg, raw, paths = _get_cfg_paths()
    items = list_raw_videos(paths.raw_dir)
    return [VideoItem(**x) for x in items]


@app.post("/v1/videos/upload", response_model=VideoItem)
def videos_upload(file: UploadFile = File(...)):
    cfg, raw, paths = _get_cfg_paths()
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")
    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".mp4", ".mov", ".mkv", ".avi"}:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")

    target = paths.raw_dir / Path(file.filename).name
    try:
        data = file.file.read()
        target.write_bytes(data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    stt = target.stat()
    return VideoItem(
        video_id=target.stem,
        filename=target.name,
        path=str(target),
        size_bytes=int(stt.st_size),
        mtime=float(stt.st_mtime),
    )


@app.delete("/v1/videos/{video_id}")
def videos_delete(video_id: str):
    cfg, raw, paths = _get_cfg_paths()
    p = None
    for ext in [".mp4", ".mov", ".mkv", ".avi"]:
        cand = paths.raw_dir / f"{video_id}{ext}"
        if cand.exists():
            p = cand
            break
    if p is None:
        raise HTTPException(status_code=404, detail=f"Video not found: {video_id}")

    try:
        p.unlink()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    try:
        shutil.rmtree(Path(paths.runs_dir) / video_id, ignore_errors=True)
    except Exception:
        pass

    return {"ok": True}


@app.get("/v1/runs", response_model=RunsMap)
def runs_list():
    cfg, raw, paths = _get_cfg_paths()
    return RunsMap(runs_map=list_all_runs(Path(paths.runs_dir)))


@app.get("/v1/runs/{video_id}/{run_id}", response_model=RunOutputs)
def runs_get(video_id: str, run_id: str):
    cfg, raw, paths = _get_cfg_paths()
    out = read_run_outputs(Path(paths.runs_dir), video_id, run_id)
    return RunOutputs(**out)


@app.delete("/v1/runs/{video_id}/{run_id}")
def runs_delete(video_id: str, run_id: str):
    cfg, raw, paths = _get_cfg_paths()
    run_dir = Path(paths.runs_dir) / video_id / run_id
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail=f"Run not found: {video_id}/{run_id}")
    try:
        shutil.rmtree(run_dir, ignore_errors=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"ok": True}


@app.post("/v1/jobs", response_model=JobCreateResponse)
def jobs_create(req: JobCreateRequest):
    cfg, raw, paths = _get_cfg_paths()

    video_exists = any((paths.raw_dir / f"{req.video_id}{ext}").exists() for ext in [".mp4", ".mov", ".mkv", ".avi"])
    if not video_exists:
        raise HTTPException(status_code=404, detail=f"Video not found: {req.video_id}")

    job_id = new_job_id("job")
    job = {
        "job_id": job_id,
        "video_id": req.video_id,
        "state": "queued",
        "stage": "queued",
        "progress": 0.0,
        "message": "Queued",
        "created_at": now_ts(),
        "updated_at": now_ts(),
        "started_at": None,
        "finished_at": None,
        "run_id": None,
        "error": None,
        "cancel_requested": False,
        "lease": None,
        "extra": req.extra or {},
    }

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
    stt = _queue_status(paths)

    queued_files = sorted(paths.queue_dir.glob("p*__*__*.q"))
    queued_items: List[QueueItem] = []
    for qf in queued_files:
        job_id = None
        try:
            parts = qf.name.split("__")
            if len(parts) >= 3:
                job_id = parts[2].replace(".q", "")
        except Exception:
            job_id = None
        if not job_id:
            continue
        try:
            job = read_json(_job_path(paths, job_id), default={}) or {}
        except Exception:
            job = {}
        queued_items.append(
            QueueItem(
                job_id=str(job_id),
                video_id=job.get("video_id"),
                state=job.get("state"),
                created_at=job.get("created_at"),
            )
        )

    return QueueListResponse(status=QueueStatus(**stt), queued=queued_items)


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


@app.delete("/v1/queue/{job_id}")
def queue_remove(job_id: str):
    cfg, raw, paths = _get_cfg_paths()
    job = _read_job(paths, job_id)
    state = str(job.get("state") or "unknown")

    if state in {"running", "leased", "indexing"}:
        raise HTTPException(status_code=409, detail=f"Job is already running: {job_id} ({state})")

    qfiles = _find_queue_files_for_job(paths, job_id)
    for qf in qfiles:
        try:
            qf.unlink(missing_ok=True)
        except Exception:
            pass

    job["state"] = "canceled"
    job["stage"] = "canceled"
    job["progress"] = 0.0
    job["message"] = "Canceled (removed from queue)"
    job["cancel_requested"] = True
    job["updated_at"] = now_ts()
    job["finished_at"] = now_ts()
    _write_job(paths, job)

    return {"ok": True, "job_id": job_id, "state": "canceled"}


@app.get("/v1/index/status", response_model=IndexStatus)
def index_status():
    cfg, raw, paths = _get_cfg_paths()
    stt = read_json(paths.index_state_path, default={}) or {}
    return IndexStatus(**stt)


@app.post("/v1/index/rebuild", response_model=IndexRebuildResponse)
def index_rebuild():
    cfg, raw, paths = _get_cfg_paths()
    try:
        build_or_update_index(
            ann_root=Path(cfg.paths.runs_dir),
            index_dir=Path(cfg.paths.indexes_dir),
            model_name=str(cfg.search.embed_model_name),
        )
        stt = {"built_at": now_ts(), "updated_at": now_ts(), "version": now_ts(), "last_error": None}
        atomic_write_json(paths.index_state_path, stt)
        return IndexRebuildResponse(ok=True, status=IndexStatus(**stt))
    except Exception as e:
        stt = read_json(paths.index_state_path, default={}) or {}
        stt["updated_at"] = now_ts()
        stt["last_error"] = str(e)
        atomic_write_json(paths.index_state_path, stt)
        return IndexRebuildResponse(ok=False, status=IndexStatus(**stt))


@app.post("/v1/search", response_model=SearchResponse)
def search(req: SearchRequest):
    cfg, raw, paths = _get_cfg_paths()
    qe = _get_engine(cfg)
    if qe is None:
        raise HTTPException(status_code=400, detail="Search index not found. Run /v1/index/rebuild first.")

    hits = qe.search(
        query=req.query.strip(),
        top_k=int(req.top_k),
        video_id=req.video_id,
        run_id=req.run_id,
        dedupe=bool(req.dedupe),
    )

    out: List[SearchHit] = []
    for h in hits:
        out.append(
            SearchHit(
                video_id=h.video_id,
                run_id=h.run_id,
                start_sec=float(h.start_sec),
                end_sec=float(h.end_sec),
                description=str(h.description),
                score=float(h.score),
                sparse_score=float(getattr(h, "sparse_score", 0.0)),
                dense_score=float(getattr(h, "dense_score", 0.0)),
            )
        )
    return SearchResponse(hits=out)
