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
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple, TYPE_CHECKING

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
from backend.schemas import (
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
    QueueStatus,
    QueueItem,
    QueueRunningItem,
    QueueListResponse,
    QueueMoveRequest,
    QueueMoveResponse,
)

if TYPE_CHECKING:
    from src.search import QueryEngine
    from src.translation.nllb_translator import NLLBTranslator
    from src.translation.translation_cache import TranslationCache


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


def _parse_job_id_from_queue_file(qf: Path) -> Optional[str]:
    try:
        parts = qf.name.split("__")
        if len(parts) >= 3:
            return parts[2].replace(".q", "")
    except Exception:
        return None
    return None


def _queue_files_with_ids(paths) -> List[Tuple[Path, str]]:
    out: List[Tuple[Path, str]] = []
    for qf in sorted(paths.queue_dir.glob("p*__*__*.q")):
        jid = _parse_job_id_from_queue_file(qf)
        if jid:
            out.append((qf, jid))
    return out


def _rewrite_queue_order(paths, ordered_job_ids: List[str]) -> None:
    paths.queue_dir.mkdir(parents=True, exist_ok=True)

    existing = _queue_files_with_ids(paths)
    id_to_paths: Dict[str, List[Path]] = {}
    for p, jid in existing:
        id_to_paths.setdefault(jid, []).append(p)

    for jid, ps in id_to_paths.items():
        for extra in ps[1:]:
            try:
                extra.unlink(missing_ok=True)
            except Exception:
                pass

    tmp_map: Dict[str, Path] = {}
    for i, jid in enumerate(ordered_job_ids):
        srcs = id_to_paths.get(jid) or []
        if not srcs:
            continue
        src = srcs[0]
        tmp = paths.queue_dir / f"_tmp__{jid}__{i}.q"
        try:
            src.replace(tmp)
            tmp_map[jid] = tmp
        except Exception:
            tmp_map[jid] = src


    tag = time_tag()
    for i, jid in enumerate(ordered_job_ids):
        src = tmp_map.get(jid)
        if not src:
            continue
        dst = paths.queue_dir / f"p{i:03d}__{tag}__{jid}.q"
        try:
            src.replace(dst)
        except Exception:
            try:
                dst.write_text(jid, encoding="utf-8")
                if src.exists():
                    src.unlink(missing_ok=True)
            except Exception:
                pass


def _find_running_job(paths) -> Optional[Dict[str, Any]]:
    states = {"running", "leased", "indexing"}
    best: Optional[Dict[str, Any]] = None
    best_key = -1.0

    for p in sorted(paths.jobs_dir.glob("job_*.json")):
        job = read_json(p, default=None)
        if not isinstance(job, dict):
            continue
        stt = str(job.get("state") or "")
        if stt not in states:
            continue
        k = float(job.get("updated_at") or job.get("started_at") or job.get("created_at") or 0.0)
        if k > best_key:
            best_key = k
            best = job
    return best


_ENGINE_CACHE: Dict[str, Dict[str, Any]] = {}
_TRANS_CACHE: Dict[str, Any] = {}


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


def _resolved_index_dir(cfg, language: str) -> Tuple[Path, str]:
    from src.search.index_builder import resolve_index_dir, search_config_fingerprint

    cfg_fp = search_config_fingerprint(cfg)
    return resolve_index_dir(Path(cfg.paths.indexes_dir), cfg_fp, language=language), cfg_fp


def _get_engine(cfg, language: str) -> Optional[QueryEngine]:
    from src.search import QueryEngine

    lang = (language or getattr(cfg.ui, "default_lang", None) or "en").strip().lower()
    idx_dir, cfg_fp = _resolved_index_dir(cfg, lang)
    ver = _index_version(idx_dir)
    if ver <= 0:
        return None
    cache = _ENGINE_CACHE.setdefault(lang, {"ver": None, "qe": None, "dir": None})
    if cache["qe"] is None or cache["ver"] != ver or cache["dir"] != str(idx_dir):
        cache["qe"] = QueryEngine(
            index_dir=Path(cfg.paths.indexes_dir),
            config_fingerprint=cfg_fp,
            language=lang,
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
        cache["ver"] = ver
        cache["dir"] = str(idx_dir)
    return cache["qe"]


def _get_translator(cfg) -> Optional[NLLBTranslator]:
    model_name = str(getattr(cfg.translation, "model_name_or_path", "") or "").strip()
    if not model_name:
        return None
    device = str(getattr(cfg.translation, "device", "cuda"))
    dtype = str(getattr(cfg.translation, "dtype", "fp16"))
    key = f"{model_name}::{device}::{dtype}"
    tr = _TRANS_CACHE.get(key)
    if tr is None:
        try:
            from src.translation.nllb_translator import NLLBTranslator
        except Exception:
            return None
        tr = NLLBTranslator(model_name_or_path=model_name, device=device, dtype=dtype)
        _TRANS_CACHE[key] = tr
    return tr


def _translate_query(cfg, query: str, src_lang: str, tgt_lang: str) -> str:
    if not query:
        return query
    src_lang = (src_lang or "").strip().lower()
    tgt_lang = (tgt_lang or "").strip().lower()
    if not src_lang or not tgt_lang or src_lang == tgt_lang:
        return query
    if not bool(getattr(cfg.search, "translate_queries", False)):
        return query
    try:
        translator = _get_translator(cfg)
    except Exception:
        return query
    if translator is None:
        return query

    max_new_tokens = int(getattr(cfg.translation, "max_new_tokens", 64))
    cache_enabled = bool(getattr(cfg.translation, "cache_enabled", False))
    cache_dir = getattr(cfg.paths, "cache_dir", None)
    if cache_enabled and cache_dir:
        try:
            from src.translation.translation_cache import TranslationCache

            cache = TranslationCache(Path(cache_dir), str(cfg.translation.model_name_or_path), src_lang, tgt_lang)

            def _hash_text(text: str) -> str:
                import hashlib

                return hashlib.sha1(text.encode("utf-8")).hexdigest()

            cached_map = cache.get_many([query])
            cached = cached_map.get(_hash_text(query))
            if cached:
                return cached
            tr = translator.translate([query], src_lang=src_lang, tgt_lang=tgt_lang, batch_size=1, max_new_tokens=max_new_tokens)
            if tr:
                cache.put_many([query], [tr[0]])
                return tr[0]
        except Exception:
            pass
    try:
        tr = translator.translate([query], src_lang=src_lang, tgt_lang=tgt_lang, batch_size=1, max_new_tokens=max_new_tokens)
        if tr:
            return tr[0]
    except Exception:
        return query
    return query


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
    if suffix not in {".mp4", ".mov", ".mkv", ".avi"}:
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
def video_outputs(video_id: str, lang: str):
    cfg, raw, paths = _get_cfg_paths()
    lang = (lang or "").strip().lower()
    if not lang:
        raise HTTPException(status_code=400, detail="Missing language")

    out = read_video_outputs(Path(paths.videos_dir), video_id, lang)
    if not out.get("manifest") and not out.get("annotations") and out.get("global_summary") is None:
        raise HTTPException(status_code=404, detail=f"Outputs not found: {video_id} ({lang})")
    return VideoOutputs(**out)


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

    job_id = new_job_id("job")
    job = {
        "job_id": job_id,
        "video_id": req.video_id,
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
        "extra": extra,
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

    queued_files = _queue_files_with_ids(paths)
    queued_items: List[QueueItem] = []
    for qf, job_id in queued_files:
        try:
            job = read_json(_job_path(paths, job_id), default={}) or {}
        except Exception:
            job = {}
        queued_items.append(
            QueueItem(
                job_id=str(job_id),
                video_id=job.get("video_id"),
                job_type=job.get("job_type"),
                language=job.get("language"),
                state=job.get("state"),
                created_at=job.get("created_at"),
            )
        )

    running_job = _find_running_job(paths)
    running_out: Optional[QueueRunningItem] = None
    if isinstance(running_job, dict) and running_job.get("job_id"):
        running_out = QueueRunningItem(
            job_id=str(running_job.get("job_id")),
            video_id=running_job.get("video_id"),
            job_type=running_job.get("job_type"),
            language=running_job.get("language"),
            state=str(running_job.get("state") or "running"),
            stage=running_job.get("stage"),
            progress=float(running_job.get("progress") or 0.0),
            message=running_job.get("message"),
            created_at=running_job.get("created_at"),
            started_at=running_job.get("started_at"),
            updated_at=running_job.get("updated_at"),
        )

    return QueueListResponse(status=QueueStatus(**stt), running=running_out, queued=queued_items)


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

    direction = (req.direction or "").strip().lower()
    if direction not in {"up", "down", "top", "bottom"}:
        raise HTTPException(status_code=400, detail="direction must be one of: up, down, top, bottom")

    items = _queue_files_with_ids(paths)
    ordered = [jid for _, jid in items]
    if not ordered:
        raise HTTPException(status_code=404, detail="Queue is empty")

    job_id = str(req.job_id)
    if job_id not in ordered:
        raise HTTPException(status_code=404, detail=f"Job not found in queue: {job_id}")

    old = ordered.index(job_id)
    steps = max(1, int(req.steps or 1))

    if direction == "top":
        new = 0
    elif direction == "bottom":
        new = len(ordered) - 1
    elif direction == "up":
        new = max(0, old - steps)
    else:
        new = min(len(ordered) - 1, old + steps)

    if new != old:
        ordered.pop(old)
        ordered.insert(new, job_id)
        _rewrite_queue_order(paths, ordered)

    return QueueMoveResponse(ok=True, job_id=job_id, old_index=int(old), new_index=int(new), queued_count=len(ordered))


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
        from src.search import build_or_update_index
        from src.search.index_builder import search_config_fingerprint

        cfg_fp = search_config_fingerprint(cfg)
        status = {"languages": {}, "updated_at": now_ts(), "last_error": None}
        built_times: List[float] = []
        for lang in cfg.ui.langs:
            try:
                build_or_update_index(
                    videos_root=Path(cfg.paths.videos_dir),
                    index_dir=Path(cfg.paths.indexes_dir),
                    model_name=str(cfg.search.embed_model_name),
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
        status["updated_at"] = now_ts()
        stt = status
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
    lang = (req.language or getattr(cfg.ui, "default_lang", None) or "en").strip().lower()
    qe = _get_engine(cfg, lang)
    if qe is None:
        raise HTTPException(status_code=400, detail="Search index not found. Run /v1/index/rebuild first.")

    query = req.query.strip()
    hits = qe.search(
        query=query,
        top_k=int(req.top_k),
        video_id=req.video_id,
        dedupe=bool(req.dedupe),
    )

    def _pass_filters(h) -> bool:
        start = float(getattr(h, "start_sec", 0.0) or 0.0)
        end = float(getattr(h, "end_sec", 0.0) or 0.0)
        if req.start_sec is not None and start < float(req.start_sec):
            return False
        if req.end_sec is not None and end > float(req.end_sec):
            return False
        dur = end - start
        if req.min_duration_sec is not None and dur < float(req.min_duration_sec):
            return False
        if req.max_duration_sec is not None and dur > float(req.max_duration_sec):
            return False
        return True

    out: List[SearchHit] = []
    seen = set()
    for h in hits:
        if not _pass_filters(h):
            continue
        key = (h.video_id, float(h.start_sec), float(h.end_sec), h.language)
        if key in seen:
            continue
        seen.add(key)
        out.append(
            SearchHit(
                video_id=h.video_id,
                language=h.language,
                start_sec=float(h.start_sec),
                end_sec=float(h.end_sec),
                description=str(h.description),
                score=float(h.score),
                sparse_score=float(getattr(h, "sparse_score", 0.0)),
                dense_score=float(getattr(h, "dense_score", 0.0)),
            )
        )

    if len(out) < int(req.top_k or 0):
        fallback_langs = [str(x).strip().lower() for x in (getattr(cfg.search, "fallback_langs", []) or [])]
        for fl in fallback_langs:
            if not fl or fl == lang:
                continue
            qe_fallback = _get_engine(cfg, fl)
            if qe_fallback is None:
                continue
            q_fallback = _translate_query(cfg, query, src_lang=lang, tgt_lang=fl)
            fhits = qe_fallback.search(
                query=q_fallback,
                top_k=int(req.top_k),
                video_id=req.video_id,
                dedupe=bool(req.dedupe),
            )
            for h in fhits:
                if not _pass_filters(h):
                    continue
                key = (h.video_id, float(h.start_sec), float(h.end_sec), h.language)
                if key in seen:
                    continue
                seen.add(key)
                out.append(
                    SearchHit(
                        video_id=h.video_id,
                        language=h.language,
                        start_sec=float(h.start_sec),
                        end_sec=float(h.end_sec),
                        description=str(h.description),
                        score=float(h.score),
                        sparse_score=float(getattr(h, "sparse_score", 0.0)),
                        dense_score=float(getattr(h, "dense_score", 0.0)),
                    )
                )
                if len(out) >= int(req.top_k):
                    break
            if len(out) >= int(req.top_k):
                break
    return SearchResponse(hits=out)
