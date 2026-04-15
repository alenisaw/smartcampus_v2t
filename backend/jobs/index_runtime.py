# backend/jobs/index_runtime.py
"""
Index runtime helpers for SmartCampus V2T backend.

Purpose:
- Share index build, state write, and metrics update logic across backend flows.
- Keep index lifecycle handling out of API and worker monoliths.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from backend.deps import atomic_write_json, now_ts, read_json
from src.search import build_or_update_index
from src.search.builder import select_embedding_model_ref
from src.search.store import resolve_index_dir
from src.utils.video_store import metrics_path, update_metrics


def _read_index_meta(index_dir: Path) -> Dict[str, Any]:
    """Read the built index metadata when it exists."""

    meta = read_json(Path(index_dir) / "meta.json", default={}) or {}
    return meta if isinstance(meta, dict) else {}


def _build_index_payload(*, index_dir: Path, language: str, variant: Optional[str], built_at: float, time_sec: float) -> Dict[str, Any]:
    """Build one normalized index result payload from persisted metadata."""

    meta = _read_index_meta(index_dir)
    runtime = meta.get("runtime") if isinstance(meta.get("runtime"), dict) else {}
    counts = runtime.get("counts") if isinstance(runtime.get("counts"), dict) else {}
    notes = runtime.get("notes") if isinstance(runtime.get("notes"), dict) else {}
    ann = meta.get("ann") if isinstance(meta.get("ann"), dict) else {}
    return {
        "language": str(language),
        "variant": str(variant or ""),
        "built_at": float(built_at),
        "time_sec": float(time_sec),
        "index_dir": str(index_dir),
        "config_tag": str(meta.get("config_tag") or ""),
        "ann_backend": str(meta.get("ann_backend") or ann.get("backend") or "exact"),
        "ann_index_type": str(meta.get("ann_index_type") or ann.get("index_type") or "flat"),
        "ann_rows": int(ann.get("rows") or 0),
        "dense_valid_count": int(meta.get("dense_valid_count") or counts.get("dense_valid_count") or 0),
        "num_docs": int(meta.get("num_docs") or counts.get("docs_total") or 0),
        "embed_dim": int(meta.get("embed_dim") or 0),
        "incremental": bool(notes.get("incremental", False)),
        "meta": meta,
    }


def build_index_for_language(
    *,
    cfg: Any,
    cfg_fp: str,
    language: str,
) -> Dict[str, Any]:
    """Build or refresh the search index for one language and return runtime metadata."""

    started_at = time.perf_counter()
    result_path = build_or_update_index(
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
        dense_input_mode=str(getattr(cfg.search, "dense_input_mode", "text")),
        cache_dir=Path(cfg.paths.cache_dir),
        use_embed_cache=bool(getattr(cfg.search, "embed_cache", True)),
        ann_backend=str(getattr(cfg.search, "ann_backend", "auto")),
        ann_index_type=str(getattr(cfg.search, "ann_index_type", "hnsw")),
        ann_hnsw_m=int(getattr(cfg.search, "ann_hnsw_m", 32)),
        ann_ef_construction=int(getattr(cfg.search, "ann_ef_construction", 80)),
        ann_ef_search=int(getattr(cfg.search, "ann_ef_search", 64)),
    )
    elapsed = time.perf_counter() - started_at
    built_at = now_ts()
    index_dir = Path(result_path).resolve().parent
    if not index_dir.exists():
        index_dir = resolve_index_dir(
            Path(cfg.paths.indexes_dir),
            cfg_fp,
            language=str(language),
            variant=cfg.active_variant,
        )
    return _build_index_payload(
        index_dir=index_dir,
        language=str(language),
        variant=cfg.active_variant,
        built_at=built_at,
        time_sec=elapsed,
    )


def rebuild_index_status(
    *,
    cfg: Any,
    cfg_fp: str,
    languages: Iterable[str] | None = None,
) -> Dict[str, Any]:
    """Build indexes for all requested languages and return the index status payload."""

    status: Dict[str, Any] = {"languages": {}, "updated_at": now_ts(), "last_error": None}
    built_times = []
    for lang in (languages or cfg.ui.langs):
        lang_key = str(lang)
        try:
            result = build_index_for_language(cfg=cfg, cfg_fp=cfg_fp, language=lang_key)
            built_at = float(result.get("built_at") or now_ts())
            built_times.append(built_at)
            status["languages"][lang_key] = {**result, "last_error": None}
        except Exception as exc:
            status["languages"][lang_key] = {"built_at": None, "last_error": str(exc)}
            status["last_error"] = str(exc)

    status["built_at"] = max(built_times) if built_times else None
    status["version"] = now_ts()
    status["updated_at"] = now_ts()
    return status


def write_index_state(
    paths: Any,
    *,
    built: bool = False,
    last_error: str | None = None,
    language: str | None = None,
    payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Persist a normalized index state payload and return it."""

    status = read_json(paths.index_state_path, default={}) or {}
    ts = now_ts()
    status["updated_at"] = ts
    status["last_error"] = last_error
    if built:
        status["built_at"] = ts
        status["version"] = ts
    if language:
        languages = status.get("languages")
        if not isinstance(languages, dict):
            languages = {}
        entry = languages.get(str(language))
        if not isinstance(entry, dict):
            entry = {}
        if isinstance(payload, dict):
            entry.update(payload)
        entry["last_error"] = last_error
        languages[str(language)] = entry
        status["languages"] = languages
    atomic_write_json(paths.index_state_path, status)
    return status


def write_index_metrics(
    *,
    cfg: Any,
    video_id: str,
    language: str,
    variant: str | None,
    index_payload: Dict[str, Any],
) -> None:
    """Persist one index timing entry into the per-video metrics artifact."""

    built_at = float(index_payload.get("built_at") or now_ts())
    update_metrics(
        metrics_path(Path(cfg.paths.videos_dir), video_id, variant=variant),
        {
            "indexing": {
                str(language): {
                    "time_sec": float(index_payload.get("time_sec") or 0.0),
                    "built_at": built_at,
                    "ann_backend": str(index_payload.get("ann_backend") or "exact"),
                    "ann_index_type": str(index_payload.get("ann_index_type") or "flat"),
                    "ann_rows": int(index_payload.get("ann_rows") or 0),
                    "dense_valid_count": int(index_payload.get("dense_valid_count") or 0),
                    "num_docs": int(index_payload.get("num_docs") or 0),
                    "embed_dim": int(index_payload.get("embed_dim") or 0),
                    "incremental": bool(index_payload.get("incremental", False)),
                    "config_tag": str(index_payload.get("config_tag") or ""),
                }
            }
        },
    )
