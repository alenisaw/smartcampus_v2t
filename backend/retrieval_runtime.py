# backend/retrieval_runtime.py
"""
Retrieval runtime helpers for SmartCampus V2T backend.

Purpose:
- Resolve and cache retrieval, guard, translation, and grounded-generation services.
- Keep search/report/QA/RAG runtime glue out of `backend/api.py`.
"""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from fastapi import HTTPException

from backend.schemas import Citation, MetricsSummaryResponse, SearchHit

if TYPE_CHECKING:
    from src.guard.service import GuardService
    from src.llm.client import LLMClient
    from src.search import QueryEngine
    from src.translation.service import TranslationService


_ENGINE_CACHE: Dict[str, Dict[str, Any]] = {}
_TRANS_CACHE: Dict[str, Any] = {}
_LLM_CACHE: Dict[str, Any] = {}
_GUARD_CACHE: Dict[str, Any] = {}


def _index_version(index_dir: Path) -> float:
    p1 = index_dir / "manifest.json"
    p2 = index_dir / "meta.json"
    version = 0.0
    for path in [p1, p2]:
        try:
            version = max(version, float(path.stat().st_mtime))
        except Exception:
            pass
    return version


def resolved_index_dir(cfg: Any, language: str, variant: Optional[str] = None) -> Tuple[Path, str]:
    """Resolve the active index directory and search config fingerprint."""

    from src.search.builder import resolve_index_dir, search_config_fingerprint

    cfg_fp = search_config_fingerprint(cfg)
    resolved_variant = str(variant).strip().lower() if variant else getattr(cfg, "active_variant", None)
    return resolve_index_dir(Path(cfg.paths.indexes_dir), cfg_fp, language=language, variant=resolved_variant), cfg_fp


def resolve_request_language(cfg: Any, language: Optional[str]) -> str:
    """Resolve one request language against config defaults."""

    return (language or getattr(cfg.ui, "default_lang", None) or "en").strip().lower()


def get_engine(cfg: Any, language: str, variant: Optional[str] = None) -> Optional["QueryEngine"]:
    """Build and cache the query engine for the resolved language and variant."""

    from src.search import QueryEngine

    lang = (language or getattr(cfg.ui, "default_lang", None) or "en").strip().lower()
    resolved_variant = str(variant).strip().lower() if variant else getattr(cfg, "active_variant", None)
    idx_dir, cfg_fp = resolved_index_dir(cfg, lang, resolved_variant)
    version = _index_version(idx_dir)
    if version <= 0:
        return None
    cache_key = f"{lang}::{resolved_variant or 'base'}"
    cache = _ENGINE_CACHE.setdefault(cache_key, {"ver": None, "qe": None, "dir": None})
    if cache["qe"] is None or cache["ver"] != version or cache["dir"] != str(idx_dir):
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
        cache["ver"] = version
        cache["dir"] = str(idx_dir)
    return cache["qe"]


def get_engine_or_400(cfg: Any, language: str, variant: Optional[str] = None) -> "QueryEngine":
    """Return the resolved query engine or raise the standard index-missing API error."""

    engine = get_engine(cfg, language, variant)
    if engine is None:
        raise HTTPException(status_code=400, detail="Search index not found. Run /v1/index/rebuild first.")
    return engine


def get_translation_service(cfg: Any) -> Optional["TranslationService"]:
    """Build and cache the translation service for output/query translation."""

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


def translate_query(cfg: Any, query: str, src_lang: str, tgt_lang: str) -> str:
    """Translate search queries into fallback languages when enabled."""

    if not query:
        return query
    src_lang = (src_lang or "").strip().lower()
    tgt_lang = (tgt_lang or "").strip().lower()
    if not src_lang or not tgt_lang or src_lang == tgt_lang:
        return query
    if not bool(getattr(cfg.search, "translate_queries", False)):
        return query
    try:
        translator = get_translation_service(cfg)
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
            translated = translator.translate(
                [query],
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                batch_size=1,
                max_new_tokens=max_new_tokens,
                use_cache=False,
            )
            if translated:
                cache.put_many([query], [translated[0]])
                return translated[0]
        except Exception:
            pass
    try:
        translated = translator.translate(
            [query],
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            batch_size=1,
            max_new_tokens=max_new_tokens,
            use_cache=cache_enabled,
        )
        if translated:
            return translated[0]
    except Exception:
        return query
    return query


def search_hits(
    cfg: Any,
    *,
    query: str,
    language: str,
    variant: Optional[str],
    top_k: int,
    video_id: Optional[str] = None,
    filters: Optional[Dict[str, Any]] = None,
    dedupe: bool = True,
) -> List[Any]:
    """Run one search request against the resolved engine."""

    engine = get_engine_or_400(cfg, language, variant)
    return engine.search(
        query=str(query or "").strip(),
        top_k=max(1, int(top_k)),
        video_id=video_id,
        filters=filters,
        dedupe=bool(dedupe),
    )


def search_hits_with_fallback(
    cfg: Any,
    *,
    query: str,
    language: str,
    variant: Optional[str],
    top_k: int,
    video_id: Optional[str] = None,
    filters: Optional[Dict[str, Any]] = None,
    dedupe: bool = True,
) -> List[SearchHit]:
    """Run search with optional fallback-language retries and return deduplicated schema hits."""

    out: List[SearchHit] = []
    seen = set()
    primary_hits = search_hits(
        cfg,
        query=query,
        language=language,
        variant=variant,
        top_k=top_k,
        video_id=video_id,
        filters=filters,
        dedupe=dedupe,
    )
    append_unique_hits(out, seen, primary_hits, limit=int(top_k))

    if len(out) >= int(top_k):
        return out

    fallback_langs = [str(x).strip().lower() for x in (getattr(cfg.search, "fallback_langs", []) or [])]
    for fallback_lang in fallback_langs:
        if not fallback_lang or fallback_lang == language:
            continue
        fallback_engine = get_engine(cfg, fallback_lang, variant)
        if fallback_engine is None:
            continue
        fallback_query = translate_query(cfg, str(query or "").strip(), src_lang=language, tgt_lang=fallback_lang)
        fallback_hits = fallback_engine.search(
            query=fallback_query,
            top_k=max(1, int(top_k)),
            video_id=video_id,
            filters=filters,
            dedupe=bool(dedupe),
        )
        append_unique_hits(out, seen, fallback_hits, limit=int(top_k))
        if len(out) >= int(top_k):
            break
    return out


def annotation_hits(
    *,
    video_id: str,
    language: str,
    annotations: List[Dict[str, Any]],
    variant: Optional[str] = None,
    top_k: int = 10,
) -> List[Any]:
    """Convert stored annotations into hit-like objects for grounded report flows."""

    hits: List[Any] = []
    for ann in (annotations or [])[: max(1, int(top_k))]:
        hits.append(
            SimpleNamespace(
                video_id=str(video_id),
                language=str(language or ""),
                start_sec=float(ann.get("start_sec", 0.0) or 0.0),
                end_sec=float(ann.get("end_sec", 0.0) or 0.0),
                description=str(ann.get("normalized_caption") or ann.get("description") or ""),
                score=1.0,
                sparse_score=1.0,
                dense_score=1.0,
                segment_id=ann.get("segment_id"),
                event_type=ann.get("event_type"),
                risk_level=ann.get("risk_level"),
                tags=ann.get("tags") or [],
                objects=ann.get("objects") or [],
                people_count_bucket=ann.get("people_count_bucket"),
                motion_type=ann.get("motion_type"),
                anomaly_flag=bool(ann.get("anomaly_flag", False)),
                variant=variant,
            )
        )
    return hits


def translate_output_text(cfg: Any, text: str, *, target_lang: str, target_name: str) -> str:
    """Translate generated report/qa text to the requested language with optional post-edit."""

    content = str(text or "").strip()
    if not content:
        return content
    tgt = str(target_lang or "").strip().lower()
    src = str(getattr(cfg.translation, "source_lang", None) or getattr(cfg.model, "language", "en")).strip().lower() or "en"
    if not tgt or tgt == src:
        return content

    translator = get_translation_service(cfg)
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


def get_llm_client(cfg: Any) -> Optional["LLMClient"]:
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


def get_guard_service(cfg: Any) -> Optional["GuardService"]:
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


def guard_query_text(cfg: Any, text: str, *, endpoint: str) -> None:
    """Apply the active guard policy to user-provided text."""

    normalized = re.sub(r"\s+", " ", str(text or "").strip())
    if not normalized:
        raise HTTPException(status_code=400, detail=f"{endpoint} query is empty.")
    service = get_guard_service(cfg)
    if service is None or not bool(getattr(cfg.guard, "query_gate", False)):
        return
    decision = service.inspect(normalized, mode="query")
    if not bool(decision.get("allowed", True)):
        raise HTTPException(status_code=400, detail=f"{endpoint} query blocked by guard policy.")


def guard_output_text(cfg: Any, text: str) -> str:
    """Apply the active output guard to generated text."""

    service = get_guard_service(cfg)
    if service is None:
        return str(text or "")
    return service.sanitize_output(str(text or ""))


def build_context_block(hits: List[Any], *, max_hits: int = 6) -> str:
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


def generate_grounded_text(
    cfg: Any,
    *,
    task: str,
    user_input: str,
    hits: List[Any],
    fallback_text: str,
) -> Tuple[str, str, str]:
    """Generate grounded text with the main LLM and fall back deterministically."""

    context = build_context_block(hits)
    if not context.strip():
        return fallback_text, "deterministic", context

    client = get_llm_client(cfg)
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


def hit_to_schema(hit: Any) -> SearchHit:
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


def append_unique_hits(out: List[SearchHit], seen: set, hits: List[Any], *, limit: int) -> None:
    """Append serialized hits while deduplicating by video/time/language."""

    max_items = max(0, int(limit))
    for hit in hits:
        key = (
            str(getattr(hit, "video_id", "") or ""),
            float(getattr(hit, "start_sec", 0.0) or 0.0),
            float(getattr(hit, "end_sec", 0.0) or 0.0),
            str(getattr(hit, "language", "") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(hit_to_schema(hit))
        if len(out) >= max_items:
            break


def hit_to_citation(hit: Any) -> Citation:
    """Convert a hit into a compact citation payload."""

    return Citation(
        video_id=str(getattr(hit, "video_id", "") or ""),
        start_sec=float(getattr(hit, "start_sec", 0.0) or 0.0),
        end_sec=float(getattr(hit, "end_sec", 0.0) or 0.0),
        segment_id=getattr(hit, "segment_id", None),
        variant=getattr(hit, "variant", None),
    )


def report_text_from_hits(hits: List[Any], video_id: Optional[str]) -> str:
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


def qa_text_from_hits(_question: str, hits: List[Any]) -> str:
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


def metrics_summary_from_outputs(video_id: str, language: str, outputs: Dict[str, Any]) -> MetricsSummaryResponse:
    """Convert stored metrics and manifests into one compact summary payload."""

    metrics = outputs.get("metrics") if isinstance(outputs.get("metrics"), dict) else {}
    run_manifest = outputs.get("run_manifest") if isinstance(outputs.get("run_manifest"), dict) else {}

    timings: Dict[str, float] = {}
    for key in ("preprocess_time_sec", "model_time_sec", "postprocess_time_sec", "total_time_sec"):
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
