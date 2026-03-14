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
from typing import Any, Dict, Iterable

from backend.deps import atomic_write_json, now_ts, read_json
from src.search import build_or_update_index
from src.search.builder import select_embedding_model_ref
from src.utils.video_store import metrics_path, update_metrics


def build_index_for_language(
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
        dense_input_mode=str(getattr(cfg.search, "dense_input_mode", "text")),
        cache_dir=Path(cfg.paths.cache_dir),
        use_embed_cache=bool(getattr(cfg.search, "embed_cache", True)),
    )
    return time.perf_counter() - started_at


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
            build_index_for_language(cfg=cfg, cfg_fp=cfg_fp, language=lang_key)
            built_at = now_ts()
            built_times.append(built_at)
            status["languages"][lang_key] = {"built_at": built_at, "last_error": None}
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
) -> Dict[str, Any]:
    """Persist a normalized index state payload and return it."""

    status = read_json(paths.index_state_path, default={}) or {}
    ts = now_ts()
    status["updated_at"] = ts
    status["last_error"] = last_error
    if built:
        status["built_at"] = ts
        status["version"] = ts
    atomic_write_json(paths.index_state_path, status)
    return status


def write_index_metrics(
    *,
    cfg: Any,
    video_id: str,
    language: str,
    variant: str | None,
    index_time: float,
) -> None:
    """Persist one index timing entry into the per-video metrics artifact."""

    update_metrics(
        metrics_path(Path(cfg.paths.videos_dir), video_id, variant=variant),
        {"indexing": {str(language): {"time_sec": float(index_time), "built_at": float(now_ts())}}},
    )
