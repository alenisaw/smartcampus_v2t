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
import re
import time
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

if TYPE_CHECKING:
    from src.guard.service import GuardService
    from src.llm.client import LLMClient
    from src.search import QueryEngine
    from src.translation.service import TranslationService
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
_LLM_CACHE: Dict[str, Any] = {}
_GUARD_CACHE: Dict[str, Any] = {}


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


def _resolved_index_dir(cfg, language: str, variant: Optional[str] = None) -> Tuple[Path, str]:
    from src.search.index_builder import resolve_index_dir, search_config_fingerprint

    cfg_fp = search_config_fingerprint(cfg)
    resolved_variant = str(variant).strip().lower() if variant else getattr(cfg, "active_variant", None)
    return resolve_index_dir(Path(cfg.paths.indexes_dir), cfg_fp, language=language, variant=resolved_variant), cfg_fp


def _get_engine(cfg, language: str, variant: Optional[str] = None) -> Optional[QueryEngine]:
    from src.search import QueryEngine

    lang = (language or getattr(cfg.ui, "default_lang", None) or "en").strip().lower()
    resolved_variant = str(variant).strip().lower() if variant else getattr(cfg, "active_variant", None)
    idx_dir, cfg_fp = _resolved_index_dir(cfg, lang, resolved_variant)
    ver = _index_version(idx_dir)
    if ver <= 0:
        return None
    cache_key = f"{lang}::{resolved_variant or 'base'}"
    cache = _ENGINE_CACHE.setdefault(cache_key, {"ver": None, "qe": None, "dir": None})
    if cache["qe"] is None or cache["ver"] != ver or cache["dir"] != str(idx_dir):
        cache["qe"] = QueryEngine(
            index_dir=Path(cfg.paths.indexes_dir),
            config_fingerprint=cfg_fp,
            language=lang,
            variant=resolved_variant,
            w_bm25=float(cfg.search.w_bm25),
            w_dense=float(cfg.search.w_dense),
            candidate_k_sparse=int(getattr(cfg.search, "candidate_k_sparse", 200)),
            candidate_k_dense=int(getattr(cfg.search, "candidate_k_dense", 200)),
            embedding_backend=str(getattr(cfg.search, "embedding_backend", "auto")),
            fallback_embed_model_name=str(cfg.search.embed_model_name),
            rerank_enabled=bool(getattr(cfg.search, "rerank_enabled", True)),
            rerank_top_k=int(getattr(cfg.search, "rerank_top_k", 30)),
            reranker_model_name=str(getattr(cfg.search, "reranker_model_id", "")),
            reranker_backend=str(getattr(cfg.search, "reranker_backend", "auto")),
            fusion=str(getattr(cfg.search, "fusion", "rrf")),
            rrf_k=int(getattr(cfg.search, "rrf_k", 60)),
            dedupe_mode=str(getattr(cfg.search, "dedupe_mode", "overlap")),
            dedupe_tol_sec=float(getattr(cfg.search, "dedupe_tol_sec", 0.5)),
            dedupe_overlap_thr=float(getattr(cfg.search, "dedupe_overlap_thr", 0.3)),
            embed_model_name=None,
        )
        cache["ver"] = ver
        cache["dir"] = str(idx_dir)
    return cache["qe"]


def _get_translation_service(cfg) -> Optional[TranslationService]:
    backend_name = str(getattr(cfg.translation, "backend", "") or "").strip().lower()
    if backend_name != "ctranslate2":
        return None
    key = f"{cfg.translation.backend}::{cfg.config_fingerprint}"
    service = _TRANS_CACHE.get(key)
    if service is None:
        try:
            from src.translation.service import TranslationService
        except Exception:
            return None
        service = TranslationService(cfg)
        _TRANS_CACHE[key] = service
    return service


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
        translator = _get_translation_service(cfg)
    except Exception:
        return query
    if translator is None:
        return query

    max_new_tokens = int(getattr(cfg.translation, "max_new_tokens", 64))
    cache_enabled = bool(getattr(cfg.translation, "cache_enabled", False))
    cache_dir = getattr(cfg.paths, "cache_dir", None)
    if cache_enabled and cache_dir:
        try:
            cache = translator.cache(src_lang=src_lang, tgt_lang=tgt_lang)
            cached_map = cache.get_many([query])
            import hashlib

            cached = cached_map.get(hashlib.sha1(query.encode("utf-8")).hexdigest())
            if cached:
                return cached
            tr = translator.translate(
                [query],
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                batch_size=1,
                max_new_tokens=max_new_tokens,
                use_cache=False,
            )
            if tr:
                cache.put_many([query], [tr[0]])
                return tr[0]
        except Exception:
            pass
    try:
        tr = translator.translate(
            [query],
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            batch_size=1,
            max_new_tokens=max_new_tokens,
            use_cache=cache_enabled,
        )
        if tr:
            return tr[0]
    except Exception:
        return query
    return query


def _translate_output_text(cfg, text: str, *, target_lang: str, target_name: str) -> str:
    """Translate generated report/qa text to the requested language with optional post-edit."""

    content = str(text or "").strip()
    if not content:
        return content
    tgt = str(target_lang or "").strip().lower()
    src = str(getattr(cfg.translation, "source_lang", None) or getattr(cfg.model, "language", "en")).strip().lower() or "en"
    if not tgt or tgt == src:
        return content

    translator = _get_translation_service(cfg)
    if translator is None:
        return content

    try:
        translated = translator.translate(
            [content],
            src_lang=src,
            tgt_lang=tgt,
            batch_size=1,
            max_new_tokens=int(getattr(cfg.translation, "max_new_tokens", 64)),
            use_cache=bool(getattr(cfg.translation, "cache_enabled", True)),
        )
        if translated:
            content = str(translated[0] or "").strip() or content
    except Exception:
        return content

    if hasattr(translator, "post_edit_many"):
        try:
            edited, _edited_count = translator.post_edit_many(
                [str(text or "")],
                [content],
                src_lang=src,
                tgt_lang=tgt,
                target_name=str(target_name or ""),
            )
            if edited:
                content = str(edited[0] or "").strip() or content
        except Exception:
            pass
    return content


def _get_llm_client(cfg) -> Optional[LLMClient]:
    """Build and cache the main text LLM client for grounded generation."""

    key = f"{cfg.llm.backend}::{cfg.llm.model_id}::{cfg.config_fingerprint}"
    client = _LLM_CACHE.get(key)
    if client is None:
        try:
            from src.llm.client import LLMClient
        except Exception:
            return None
        try:
            client = LLMClient.from_config(cfg)
        except Exception:
            return None
        _LLM_CACHE[key] = client
    return client


def _get_guard_service(cfg) -> Optional[GuardService]:
    """Build and cache the guard service for the active config."""

    if not bool(getattr(cfg.guard, "enabled", False)):
        return None
    key = f"{cfg.guard.model_id}::{cfg.config_fingerprint}"
    service = _GUARD_CACHE.get(key)
    if service is None:
        try:
            from src.guard.service import GuardService
        except Exception:
            return None
        try:
            service = GuardService.from_config(cfg)
        except Exception:
            return None
        _GUARD_CACHE[key] = service
    return service


def _guard_query_text(cfg: Any, text: str, *, endpoint: str) -> None:
    """Apply the active guard policy to user-provided text."""

    normalized = re.sub(r"\s+", " ", str(text or "").strip())
    if not normalized:
        raise HTTPException(status_code=400, detail=f"{endpoint} query is empty.")
    service = _get_guard_service(cfg)
    if service is None or not bool(getattr(cfg.guard, "query_gate", False)):
        return
    decision = service.inspect(normalized, mode="query")
    if not bool(decision.get("allowed", True)):
        raise HTTPException(status_code=400, detail=f"{endpoint} query blocked by guard policy.")


def _guard_output_text(cfg: Any, text: str) -> str:
    """Apply the active output guard to generated text."""

    service = _get_guard_service(cfg)
    if service is None:
        return str(text or "")
    return service.sanitize_output(str(text or ""))


def _build_context_block(hits: List[Any], *, max_hits: int = 6) -> str:
    """Serialize top evidence into a compact RAG context block."""

    lines: List[str] = []
    for idx, hit in enumerate(hits[: max(1, int(max_hits))], start=1):
        lines.append(
            f"{idx}. video={getattr(hit, 'video_id', '')} "
            f"time={float(getattr(hit, 'start_sec', 0.0)):.1f}-{float(getattr(hit, 'end_sec', 0.0)):.1f} "
            f"segment={getattr(hit, 'segment_id', '') or '-'} "
            f"text={str(getattr(hit, 'description', '') or '').strip()}"
        )
    return "\n".join(lines)


def _generate_grounded_text(
    cfg: Any,
    *,
    task: str,
    user_input: str,
    hits: List[Any],
    fallback_text: str,
) -> Tuple[str, str, str]:
    """Generate grounded text with the main LLM and fall back deterministically."""

    context = _build_context_block(hits)
    if not context.strip():
        return fallback_text, "deterministic", context

    client = _get_llm_client(cfg)
    if client is None:
        return fallback_text, "deterministic", context

    prompt = (
        f"You are a grounded surveillance analytics assistant.\n"
        f"Task: {task}.\n"
        "Use only the context below. Do not invent facts. Preserve time ranges and segment ids when relevant.\n"
        f"User input: {user_input}\n"
        f"Context:\n{context}\n"
        "Return only the final answer text."
    )

    try:
        generated = str(client.generate_text(prompt) or "").strip()
    except Exception:
        return fallback_text, "deterministic", context
    if not generated:
        return fallback_text, "deterministic", context
    return generated, "llm", context


def _hit_to_schema(hit: Any) -> SearchHit:
    """Convert an internal search hit into the API schema."""

    return SearchHit(
        video_id=str(getattr(hit, "video_id", "") or ""),
        language=str(getattr(hit, "language", "") or ""),
        start_sec=float(getattr(hit, "start_sec", 0.0) or 0.0),
        end_sec=float(getattr(hit, "end_sec", 0.0) or 0.0),
        description=str(getattr(hit, "description", "") or ""),
        score=float(getattr(hit, "score", 0.0) or 0.0),
        sparse_score=float(getattr(hit, "sparse_score", 0.0) or 0.0),
        dense_score=float(getattr(hit, "dense_score", 0.0) or 0.0),
        segment_id=getattr(hit, "segment_id", None),
        event_type=getattr(hit, "event_type", None),
        risk_level=getattr(hit, "risk_level", None),
        tags=list(getattr(hit, "tags", None) or []),
        objects=list(getattr(hit, "objects", None) or []),
        people_count_bucket=getattr(hit, "people_count_bucket", None),
        motion_type=getattr(hit, "motion_type", None),
        anomaly_flag=bool(getattr(hit, "anomaly_flag", False)),
        variant=getattr(hit, "variant", None),
    )


def _hit_to_citation(hit: Any) -> Citation:
    """Convert a hit into a compact citation payload."""

    return Citation(
        video_id=str(getattr(hit, "video_id", "") or ""),
        start_sec=float(getattr(hit, "start_sec", 0.0) or 0.0),
        end_sec=float(getattr(hit, "end_sec", 0.0) or 0.0),
        segment_id=getattr(hit, "segment_id", None),
        variant=getattr(hit, "variant", None),
    )


def _report_text_from_hits(hits: List[Any], video_id: Optional[str]) -> str:
    """Build a deterministic grounded report from search hits."""

    if not hits:
        return "No grounded evidence found for the requested scope."

    header = f"Grounded report for {video_id}." if video_id else "Grounded report."
    lines = [header]
    for idx, hit in enumerate(hits, start=1):
        fields: List[str] = []
        if getattr(hit, "event_type", None):
            fields.append(f"type={getattr(hit, 'event_type')}")
        if getattr(hit, "risk_level", None):
            fields.append(f"risk={getattr(hit, 'risk_level')}")
        if getattr(hit, "people_count_bucket", None):
            fields.append(f"people={getattr(hit, 'people_count_bucket')}")
        if getattr(hit, "motion_type", None):
            fields.append(f"motion={getattr(hit, 'motion_type')}")
        meta = f" [{' | '.join(fields)}]" if fields else ""
        lines.append(
            f"{idx}. {getattr(hit, 'video_id', '')} "
            f"[{float(getattr(hit, 'start_sec', 0.0)):.1f}-{float(getattr(hit, 'end_sec', 0.0)):.1f}] "
            f"{str(getattr(hit, 'description', '') or '').strip()}{meta}"
        )
    return "\n".join(lines)


def _qa_text_from_hits(question: str, hits: List[Any]) -> str:
    """Build a grounded answer from the top retrieved evidence only."""

    if not hits:
        return "I do not have grounded evidence to answer that question."

    lead = hits[0]
    lead_text = str(getattr(lead, "description", "") or "").strip()
    if len(hits) == 1:
        return f"Based on the top retrieved segment: {lead_text}"

    extras = []
    for hit in hits[1:3]:
        extras.append(str(getattr(hit, "description", "") or "").strip())
    joined = " ".join(x for x in extras if x)
    if joined:
        return f"Based on the retrieved evidence: {lead_text} Additional context: {joined}"
    return f"Based on the retrieved evidence: {lead_text}"


def _metrics_summary_from_outputs(video_id: str, language: str, outputs: Dict[str, Any]) -> MetricsSummaryResponse:
    """Convert stored metrics + manifests into one compact summary payload."""

    metrics = outputs.get("metrics") if isinstance(outputs.get("metrics"), dict) else {}
    run_manifest = outputs.get("run_manifest") if isinstance(outputs.get("run_manifest"), dict) else {}

    timings: Dict[str, float] = {}
    for key in (
        "preprocess_time_sec",
        "model_time_sec",
        "postprocess_time_sec",
        "total_time_sec",
    ):
        value = metrics.get(key)
        if isinstance(value, (int, float)):
            timings[key] = float(value)

    counters: Dict[str, Any] = {
        "num_frames": metrics.get("num_frames"),
        "num_clips": metrics.get("num_clips"),
        "avg_clip_duration_sec": metrics.get("avg_clip_duration_sec"),
        "language": metrics.get("language", language),
        "device": metrics.get("device"),
    }
    extra = metrics.get("extra")
    if isinstance(extra, dict):
        counters["extra"] = extra

    return MetricsSummaryResponse(
        video_id=video_id,
        language=language,
        variant=outputs.get("variant"),
        profile=run_manifest.get("profile"),
        config_fingerprint=run_manifest.get("config_fingerprint"),
        timings_sec=timings,
        counters=counters,
    )


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
    job = {
        "job_id": job_id,
        "video_id": req.video_id,
        "job_type": job_type,
        "profile": profile,
        "variant": variant,
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
                profile=job.get("profile"),
                variant=job.get("variant"),
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
            profile=running_job.get("profile"),
            variant=running_job.get("variant"),
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
        from src.search.index_builder import search_config_fingerprint, select_embedding_model_ref

        cfg_fp = search_config_fingerprint(cfg)
        status = {"languages": {}, "updated_at": now_ts(), "last_error": None}
        built_times: List[float] = []
        for lang in cfg.ui.langs:
            try:
                build_or_update_index(
                    videos_root=Path(cfg.paths.videos_dir),
                    index_dir=Path(cfg.paths.indexes_dir),
                    model_name=select_embedding_model_ref(cfg.search, models_dir=Path(cfg.paths.models_dir)),
                    embedding_backend=str(getattr(cfg.search, "embedding_backend", "auto")),
                    fallback_model_name=str(cfg.search.embed_model_name),
                    config_fingerprint=cfg_fp,
                    variant=cfg.active_variant,
                    language=str(lang),
                    query_prefix=str(getattr(cfg.search, "query_prefix", "query: ")),
                    passage_prefix=str(getattr(cfg.search, "passage_prefix", "passage: ")),
                    normalize_text=bool(getattr(cfg.search, "normalize_text", True)),
                    lemmatize=bool(getattr(cfg.search, "lemmatize", False)),
                    dense_input_mode=str(getattr(cfg.search, "dense_input_mode", "text")),
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
    _guard_query_text(cfg, req.query, endpoint="Search")
    qe = _get_engine(cfg, lang, req.variant)
    if qe is None:
        raise HTTPException(status_code=400, detail="Search index not found. Run /v1/index/rebuild first.")

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
    hits = qe.search(
        query=query,
        top_k=int(req.top_k),
        video_id=req.video_id,
        filters=search_filters,
        dedupe=bool(req.dedupe),
    )

    out: List[SearchHit] = []
    seen = set()
    for h in hits:
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
                segment_id=getattr(h, "segment_id", None),
                event_type=getattr(h, "event_type", None),
                risk_level=getattr(h, "risk_level", None),
                tags=list(getattr(h, "tags", None) or []),
                objects=list(getattr(h, "objects", None) or []),
                people_count_bucket=getattr(h, "people_count_bucket", None),
                motion_type=getattr(h, "motion_type", None),
                anomaly_flag=bool(getattr(h, "anomaly_flag", False)),
                variant=getattr(h, "variant", None),
            )
        )

    if len(out) < int(req.top_k or 0):
        fallback_langs = [str(x).strip().lower() for x in (getattr(cfg.search, "fallback_langs", []) or [])]
        for fl in fallback_langs:
            if not fl or fl == lang:
                continue
            qe_fallback = _get_engine(cfg, fl, req.variant)
            if qe_fallback is None:
                continue
            q_fallback = _translate_query(cfg, query, src_lang=lang, tgt_lang=fl)
            fhits = qe_fallback.search(
                query=q_fallback,
                top_k=int(req.top_k),
                video_id=req.video_id,
                filters=search_filters,
                dedupe=bool(req.dedupe),
            )
            for h in fhits:
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
                        segment_id=getattr(h, "segment_id", None),
                        event_type=getattr(h, "event_type", None),
                        risk_level=getattr(h, "risk_level", None),
                        tags=list(getattr(h, "tags", None) or []),
                        objects=list(getattr(h, "objects", None) or []),
                        people_count_bucket=getattr(h, "people_count_bucket", None),
                        motion_type=getattr(h, "motion_type", None),
                        anomaly_flag=bool(getattr(h, "anomaly_flag", False)),
                        variant=getattr(h, "variant", None),
                    )
                )
                if len(out) >= int(req.top_k):
                    break
            if len(out) >= int(req.top_k):
                break
    return SearchResponse(hits=out)


@app.post("/v1/reports", response_model=ReportResponse)
def reports(req: ReportRequest):
    started_at = time.perf_counter()
    cfg, raw, paths = _get_cfg_paths()
    lang = (req.language or getattr(cfg.ui, "default_lang", None) or "en").strip().lower()

    if not (str(req.query or "").strip() or str(req.video_id or "").strip()):
        raise HTTPException(status_code=400, detail="Report request requires a query or video_id.")
    if str(req.query or "").strip():
        _guard_query_text(cfg, str(req.query or ""), endpoint="Reports")

    supporting_hits: List[Any] = []
    if str(req.query or "").strip():
        qe = _get_engine(cfg, lang, req.variant)
        if qe is None:
            raise HTTPException(status_code=400, detail="Search index not found. Run /v1/index/rebuild first.")
        supporting_hits = qe.search(
            query=str(req.query or "").strip(),
            top_k=max(1, int(req.top_k)),
            video_id=req.video_id,
            filters={"variant": req.variant} if req.variant else None,
            dedupe=True,
        )
    elif req.video_id:
        outputs = read_video_outputs(Path(paths.videos_dir), str(req.video_id), lang, variant=req.variant)
        for ann in (outputs.get("annotations") or [])[: max(1, int(req.top_k))]:
            supporting_hits.append(
                type(
                    "ReportHit",
                    (),
                    {
                        "video_id": str(req.video_id),
                        "language": lang,
                        "start_sec": float(ann.get("start_sec", 0.0) or 0.0),
                        "end_sec": float(ann.get("end_sec", 0.0) or 0.0),
                        "description": str(ann.get("normalized_caption") or ann.get("description") or ""),
                        "score": 1.0,
                        "sparse_score": 1.0,
                        "dense_score": 1.0,
                        "segment_id": ann.get("segment_id"),
                        "event_type": ann.get("event_type"),
                        "risk_level": ann.get("risk_level"),
                        "tags": ann.get("tags") or [],
                        "objects": ann.get("objects") or [],
                        "people_count_bucket": ann.get("people_count_bucket"),
                        "motion_type": ann.get("motion_type"),
                        "anomaly_flag": bool(ann.get("anomaly_flag", False)),
                        "variant": req.variant,
                    },
                )()
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
    lang = (req.language or getattr(cfg.ui, "default_lang", None) or "en").strip().lower()
    question = str(req.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is required.")
    _guard_query_text(cfg, question, endpoint="QA")
    if bool(getattr(cfg.guard, "enabled", False)) and bool(getattr(cfg.guard, "query_gate", False)) and len(question) < 3:
        raise HTTPException(status_code=400, detail="Question is too short for QA.")

    qe = _get_engine(cfg, lang, req.variant)
    if qe is None:
        raise HTTPException(status_code=400, detail="Search index not found. Run /v1/index/rebuild first.")

    hits = qe.search(
        query=question,
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
    lang = (req.language or getattr(cfg.ui, "default_lang", None) or "en").strip().lower()
    query = str(req.query or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="RAG query is required.")
    _guard_query_text(cfg, query, endpoint="RAG")

    qe = _get_engine(cfg, lang, req.variant)
    if qe is None:
        raise HTTPException(status_code=400, detail="Search index not found. Run /v1/index/rebuild first.")

    hits = qe.search(
        query=query,
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
