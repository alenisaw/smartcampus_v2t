# src/search/engine.py
"""
Search query engine for SmartCampus V2T.

Purpose:
- Execute hybrid retrieval, deduplication, filtering, and reranking over built indexes.
- Provide the runtime search surface used by backend and evaluation scripts.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.runtime import load_pipeline_config

from .embed import MODEL_NAME_DEFAULT, build_text_embedder, looks_like_transformers_model as _looks_like_transformers_model
from .builder import (
    load_index,
    resolve_index_dir,
    search_config_fingerprint,
)
from .rank import (
    as_str_set as _as_str_set,
    build_reranker as _build_reranker,
    dedupe_time_hits_bucket as _dedupe_time_hits_bucket,
    dedupe_time_hits_overlap_nms as _dedupe_time_hits_overlap_nms,
    heuristic_rerank_bonus as _heuristic_rerank_bonus,
    in_sorted as _in_sorted,
    minmax_norm_from_dict as _minmax_norm_from_dict,
    minmax_norm_on_indices as _minmax_norm_on_indices,
    rrf_fuse as _rrf_fuse,
    topn_indices as _topn_indices,
)
from .text import tokenize as _tokenize
from .types import HybridIndex, SearchResult

logger = logging.getLogger(__name__)

DEFAULT_CFG_PATH = PROJECT_ROOT / "configs" / "profiles" / "main.yaml"

def _load_cfg(config_path: Optional[Path]) -> Any:
    p = Path(config_path) if config_path is not None else DEFAULT_CFG_PATH
    return load_pipeline_config(p)

class QueryEngine:
    def __init__(
        self,
        index: Optional[HybridIndex] = None,
        config_path: Optional[Path] = None,
        index_dir: Optional[Path] = None,
        config_fingerprint: Optional[str] = None,
        language: Optional[str] = None,
        variant: Optional[str] = None,
        w_bm25: Optional[float] = None,
        w_dense: Optional[float] = None,
        candidate_k_sparse: Optional[int] = None,
        candidate_k_dense: Optional[int] = None,
        embedding_backend: Optional[str] = None,
        fallback_embed_model_name: Optional[str] = None,
        rerank_enabled: Optional[bool] = None,
        rerank_top_k: Optional[int] = None,
        reranker_model_name: Optional[str] = None,
        reranker_backend: Optional[str] = None,
        fusion: Optional[str] = None,
        rrf_k: Optional[int] = None,
        dedupe_mode: Optional[str] = None,
        dedupe_tol_sec: Optional[float] = None,
        dedupe_overlap_thr: Optional[float] = None,
        normalize_text: Optional[bool] = None,
        lemmatize: Optional[bool] = None,
        device: Optional[str] = None,
        embed_model_name: Optional[str] = None,
        dense_chunk_size: int = 4096,
    ) -> None:
        cfg = _load_cfg(config_path)

        s = cfg.search
        base_index_dir = Path(index_dir) if index_dir is not None else Path(cfg.paths.indexes_dir)

        cfg_fp = config_fingerprint or search_config_fingerprint(cfg)
        self.language = (language or getattr(cfg.ui, "default_lang", None) or "en").strip().lower()
        self.variant = str(variant).strip().lower() if variant else None
        resolved_dir = resolve_index_dir(base_index_dir, cfg_fp, language=self.language, variant=self.variant)

        self.index_dir = base_index_dir
        self.config_fingerprint = str(cfg_fp)

        self.index = index or load_index(resolved_dir)

        self.w_bm25 = float(w_bm25 if w_bm25 is not None else getattr(s, "w_bm25", 0.45))
        self.w_dense = float(w_dense if w_dense is not None else getattr(s, "w_dense", 0.55))

        self.candidate_k_sparse = int(
            candidate_k_sparse if candidate_k_sparse is not None else getattr(s, "candidate_k_sparse", 200)
        )
        self.candidate_k_dense = int(
            candidate_k_dense if candidate_k_dense is not None else getattr(s, "candidate_k_dense", 200)
        )

        self.fusion = str(fusion if fusion is not None else getattr(s, "fusion", "rrf")).lower().strip()
        self.rrf_k = int(rrf_k if rrf_k is not None else getattr(s, "rrf_k", 60))

        self.dedupe_mode = str(
            dedupe_mode if dedupe_mode is not None else getattr(s, "dedupe_mode", "overlap")
        ).lower().strip()
        self.dedupe_tol_sec = float(dedupe_tol_sec if dedupe_tol_sec is not None else getattr(s, "dedupe_tol_sec", 1.0))
        self.dedupe_overlap_thr = float(
            dedupe_overlap_thr if dedupe_overlap_thr is not None else getattr(s, "dedupe_overlap_thr", 0.7)
        )

        self.dense_chunk_size = max(256, int(dense_chunk_size))
        self.normalize_text = bool(
            normalize_text if normalize_text is not None else getattr(s, "normalize_text", True)
        )
        self.lemmatize = bool(
            lemmatize if lemmatize is not None else getattr(s, "lemmatize", False)
        )
        self.rerank_enabled = bool(rerank_enabled if rerank_enabled is not None else getattr(s, "rerank_enabled", True))
        self.rerank_top_k = max(
            1,
            int(rerank_top_k if rerank_top_k is not None else getattr(s, "rerank_top_k", 30)),
        )

        meta_model = None
        try:
            meta_model = (self.index.meta or {}).get("model_name")
        except Exception:
            meta_model = None

        model_name = (
            str(embed_model_name)
            if embed_model_name is not None
            else str(meta_model or getattr(s, "embedding_model_id", None) or getattr(s, "embed_model_name", MODEL_NAME_DEFAULT))
        )
        embedding_backend_name = str(
            embedding_backend if embedding_backend is not None else getattr(s, "embedding_backend", "auto")
        ).strip().lower() or "auto"
        fallback_name = (
            str(fallback_embed_model_name)
            if fallback_embed_model_name is not None
            else str(getattr(s, "embed_model_name", MODEL_NAME_DEFAULT))
        )

        query_prefix = str(getattr(s, "query_prefix", "query: "))
        passage_prefix = str(getattr(s, "passage_prefix", "passage: "))
        self.embedder = build_text_embedder(
            model_name=str(model_name),
            device=device,
            query_prefix=query_prefix,
            passage_prefix=passage_prefix,
            backend=embedding_backend_name,
            fallback_model_name=fallback_name,
        )
        reranker_name = str(
            reranker_model_name if reranker_model_name is not None else getattr(s, "reranker_model_id", "")
        ).strip()
        reranker_backend_name = str(
            reranker_backend if reranker_backend is not None else getattr(s, "reranker_backend", "auto")
        ).strip().lower() or "auto"
        self.reranker = _build_reranker(
            reranker_name,
            reranker_backend_name,
            device=device,
            looks_like_transformers_model=_looks_like_transformers_model,
        )
        self.reranker_backend = (
            "transformers"
            if self.reranker is not None
            else ("heuristic" if self.rerank_enabled else "disabled")
        )
        self.last_stats: Dict[str, Any] = {}

        self._dense_valid_indices_cache: Optional[np.ndarray] = None

    def get_last_stats(self) -> Dict[str, Any]:
        return dict(self.last_stats or {})

    def _build_mask(self, docs, video_id: Optional[str]) -> List[bool]:
        mask = [True] * len(docs)
        if video_id is not None:
            mask = [m and (d.video_id == video_id) for m, d in zip(mask, docs)]
        return mask

    def _matches_filters(self, doc: Any, filters: Optional[Dict[str, Any]]) -> bool:
        if not filters:
            return True

        meta = getattr(doc, "extra", None)
        if not isinstance(meta, dict):
            meta = {}

        start_sec = float(getattr(doc, "start_sec", 0.0) or 0.0)
        end_sec = float(getattr(doc, "end_sec", 0.0) or 0.0)
        duration_sec = max(0.0, end_sec - start_sec)

        if filters.get("start_sec") is not None and start_sec < float(filters["start_sec"]):
            return False
        if filters.get("end_sec") is not None and end_sec > float(filters["end_sec"]):
            return False
        if filters.get("min_duration_sec") is not None and duration_sec < float(filters["min_duration_sec"]):
            return False
        if filters.get("max_duration_sec") is not None and duration_sec > float(filters["max_duration_sec"]):
            return False

        for key in ("event_type", "risk_level", "people_count_bucket", "motion_type", "variant"):
            expected = filters.get(key)
            if expected is None:
                continue
            actual = str(meta.get(key) or "").strip().lower()
            if actual != str(expected).strip().lower():
                return False

        if filters.get("anomaly_only") is True and not bool(meta.get("anomaly_flag", False)):
            return False

        expected_tags = _as_str_set(filters.get("tags"))
        if expected_tags:
            actual_tags = _as_str_set(meta.get("tags"))
            if not expected_tags.issubset(actual_tags):
                return False

        expected_objects = _as_str_set(filters.get("objects"))
        if expected_objects:
            actual_objects = _as_str_set(meta.get("objects"))
            if not expected_objects.issubset(actual_objects):
                return False

        return True

    def _dense_valid_indices(self) -> np.ndarray:
        if self._dense_valid_indices_cache is not None:
            return self._dense_valid_indices_cache

        dv = getattr(self.index, "dense_valid", None)
        n = len(self.index.docs) if self.index.docs else 0

        if dv is None or not hasattr(dv, "shape") or int(dv.shape[0]) != int(n):
            idx = np.arange(int(n), dtype=np.int64)
            self._dense_valid_indices_cache = idx
            return idx

        dv_bool = np.asarray(dv, dtype=np.bool_)
        idx = np.flatnonzero(dv_bool).astype(np.int64, copy=False)
        self._dense_valid_indices_cache = idx
        return idx

    def _empty_search_result(
        self,
        *,
        query: str,
        n_docs: int,
        reason: str,
        return_stats: bool,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Union[List[SearchResult], Tuple[List[SearchResult], Dict[str, Any]]]:
        stats = {
            "query": str(query),
            "ok": False,
            "reason": str(reason),
            "n_docs": int(n_docs),
        }
        if extra:
            stats.update(extra)
        self.last_stats = stats
        return ([], dict(stats)) if return_stats else []

    def _resolve_valid_indices(
        self,
        docs: List[Any],
        *,
        video_id: Optional[str],
        active_filters: Dict[str, Any],
    ) -> Tuple[List[int], float]:
        t_filter0 = time.perf_counter()
        mask = self._build_mask(docs, video_id=video_id)
        valid_indices = [
            i
            for i, (m, d) in enumerate(zip(mask, docs))
            if m and self._matches_filters(d, active_filters)
        ]
        return valid_indices, (time.perf_counter() - t_filter0) * 1000.0

    def _compute_sparse_stage(
        self,
        query: str,
        *,
        valid_indices: List[int],
    ) -> Tuple[List[str], Any, List[int], float]:
        t_sparse0 = time.perf_counter()
        q_tokens = _tokenize(query, lang=self.language, lemmatize=self.lemmatize, normalize=self.normalize_text)
        bm25_all = self.index.bm25.score(q_tokens)
        top_sparse = _topn_indices(bm25_all, valid_indices, self.candidate_k_sparse)
        return q_tokens, bm25_all, top_sparse, (time.perf_counter() - t_sparse0) * 1000.0

    def _compute_dense_stage(
        self,
        q_vec: np.ndarray,
        *,
        valid_indices: List[int],
    ) -> Tuple[np.ndarray, List[int], List[int], float]:
        t_dense0 = time.perf_counter()
        dense_valid_idx = self._dense_valid_indices()
        if dense_valid_idx.size == 0:
            dense_valid_indices: List[int] = []
        else:
            v = np.asarray(valid_indices, dtype=np.int64)
            in_dv = _in_sorted(dense_valid_idx, v)
            dense_valid_indices = v[in_dv].tolist()

        top_dense = self.index.ann_index.topk(
            q_vec=q_vec,
            valid_indices=dense_valid_indices,
            k=self.candidate_k_dense,
            chunk_size=self.dense_chunk_size,
        )
        return dense_valid_idx, dense_valid_indices, top_dense, (time.perf_counter() - t_dense0) * 1000.0

    def _collect_candidate_scores(
        self,
        *,
        bm25_all: Any,
        q_vec: np.ndarray,
        candidate_indices: List[int],
        dense_valid_idx: np.ndarray,
        include_dense: bool,
    ) -> Tuple[Dict[int, float], Dict[int, float], float]:
        t_dense_cand0 = time.perf_counter()
        sparse_raw: Dict[int, float] = {int(i): float(bm25_all[int(i)]) for i in candidate_indices}
        dense_raw: Dict[int, float] = {}
        if include_dense:
            cand = np.asarray(candidate_indices, dtype=np.int64)
            is_dv = _in_sorted(dense_valid_idx, cand)
            dense_cand = cand[is_dv].tolist()
            dense_raw = self.index.ann_index.scores(q_vec=q_vec, indices=dense_cand)
        return sparse_raw, dense_raw, (time.perf_counter() - t_dense_cand0) * 1000.0

    def _fuse_candidate_scores(
        self,
        *,
        bm25_all: Any,
        candidate_indices: List[int],
        sparse_raw: Dict[int, float],
        dense_raw: Dict[int, float],
    ) -> Tuple[List[int], Dict[int, float], Dict[int, float], Dict[int, float], float]:
        t_fuse0 = time.perf_counter()
        if self.fusion == "minmax":
            sparse_used = _minmax_norm_on_indices(bm25_all, candidate_indices)
            dense_used = _minmax_norm_from_dict(dense_raw, candidate_indices)
            fused = {
                int(i): (self.w_bm25 * sparse_used.get(int(i), 0.0)) + (self.w_dense * dense_used.get(int(i), 0.0))
                for i in candidate_indices
            }
        elif self.fusion == "rrf":
            fused = _rrf_fuse(
                dense_scores=dense_raw,
                sparse_scores=sparse_raw,
                candidate_indices=candidate_indices,
                k=self.rrf_k,
                w_sparse=self.w_bm25,
                w_dense=self.w_dense,
            )
            sparse_used = _minmax_norm_on_indices(bm25_all, candidate_indices)
            dense_used = _minmax_norm_from_dict(dense_raw, candidate_indices)
        else:
            raise ValueError(f"Unknown fusion mode: {self.fusion!r}. Use 'rrf' or 'minmax'.")

        ranked = sorted(candidate_indices, key=lambda i: fused.get(int(i), float("-inf")), reverse=True)
        return ranked, fused, sparse_used, dense_used, (time.perf_counter() - t_fuse0) * 1000.0

    def _pack_ranked_results(
        self,
        *,
        docs: List[Any],
        ranked: List[int],
        fused: Dict[int, float],
        sparse_used: Dict[int, float],
        dense_used: Dict[int, float],
        top_k: int,
    ) -> Tuple[List[SearchResult], float]:
        t_pack0 = time.perf_counter()
        results: List[SearchResult] = []
        take = max(int(top_k) * 5, int(top_k))
        for i in ranked[:take]:
            d = docs[int(i)]
            results.append(
                SearchResult(
                    score=float(fused.get(int(i), 0.0)),
                    sparse_score=float(sparse_used.get(int(i), 0.0)),
                    dense_score=float(dense_used.get(int(i), 0.0)),
                    video_id=d.video_id,
                    language=self.language,
                    start_sec=float(d.start_sec),
                    end_sec=float(d.end_sec),
                    description=str(getattr(d, "display_text", None) or d.text),
                    source_id=d.doc_id,
                    segment_id=str((d.extra or {}).get("segment_id") or "") or None,
                    event_type=str((d.extra or {}).get("event_type") or "") or None,
                    risk_level=str((d.extra or {}).get("risk_level") or "") or None,
                    tags=list((d.extra or {}).get("tags") or []),
                    objects=list((d.extra or {}).get("objects") or []),
                    people_count_bucket=str((d.extra or {}).get("people_count_bucket") or "") or None,
                    motion_type=str((d.extra or {}).get("motion_type") or "") or None,
                    anomaly_flag=bool((d.extra or {}).get("anomaly_flag", False)),
                    variant=str((d.extra or {}).get("variant") or "") or None,
                    extra=d.extra,
                )
            )
        return results, (time.perf_counter() - t_pack0) * 1000.0

    def _apply_dedupe(
        self,
        results: List[SearchResult],
        *,
        dedupe: bool,
    ) -> Tuple[List[SearchResult], int, float]:
        t_dedupe0 = time.perf_counter()
        pre_dedupe_n = len(results)
        if dedupe and results:
            if self.dedupe_mode == "overlap":
                results = _dedupe_time_hits_overlap_nms(results, overlap_thr=self.dedupe_overlap_thr)
            elif self.dedupe_mode == "bucket":
                results = _dedupe_time_hits_bucket(results, tol_sec=self.dedupe_tol_sec)
            else:
                raise ValueError(f"Unknown dedupe_mode: {self.dedupe_mode!r}. Use 'overlap' or 'bucket'.")
        return results, pre_dedupe_n, (time.perf_counter() - t_dedupe0) * 1000.0

    def _apply_rerank(
        self,
        results: List[SearchResult],
        *,
        query: str,
        q_tokens: List[str],
        top_k: int,
    ) -> Tuple[List[SearchResult], str, float]:
        t_rerank0 = time.perf_counter()
        rerank_mode = "disabled"
        if self.rerank_enabled and results:
            rerank_n = min(len(results), max(int(top_k), self.rerank_top_k))
            prefix = list(results[:rerank_n])
            suffix = list(results[rerank_n:])
            rerank_mode = "heuristic"
            if self.reranker is not None:
                try:
                    pair_scores = self.reranker.score_pairs(query, [hit.description for hit in prefix])
                    if len(pair_scores) == len(prefix):
                        scored_prefix = []
                        max_abs = max((abs(float(x)) for x in pair_scores), default=1.0) or 1.0
                        for hit, rerank_score in zip(prefix, pair_scores):
                            combined_score = float(hit.score) + (float(rerank_score) / max_abs) * 0.35
                            scored_prefix.append((combined_score, hit))
                        scored_prefix.sort(key=lambda item: item[0], reverse=True)
                        prefix = [hit for _, hit in scored_prefix]
                        rerank_mode = "transformers"
                    else:
                        prefix.sort(
                            key=lambda hit: float(hit.score) + _heuristic_rerank_bonus(q_tokens, hit, _tokenize),
                            reverse=True,
                        )
                except Exception:
                    prefix.sort(
                        key=lambda hit: float(hit.score) + _heuristic_rerank_bonus(q_tokens, hit, _tokenize),
                        reverse=True,
                    )
            else:
                prefix.sort(
                    key=lambda hit: float(hit.score) + _heuristic_rerank_bonus(q_tokens, hit, _tokenize),
                    reverse=True,
                )
            results = prefix + suffix
        return results[: int(top_k)], rerank_mode, (time.perf_counter() - t_rerank0) * 1000.0

    def _build_search_stats(
        self,
        *,
        query: str,
        q_tokens: List[str],
        video_id: Optional[str],
        active_filters: Dict[str, Any],
        n_docs: int,
        valid_indices: List[int],
        dense_valid_indices: List[int],
        top_sparse: List[int],
        top_dense: List[int],
        candidate_indices: List[int],
        pre_dedupe_n: int,
        results: List[SearchResult],
        emb: Any,
        rerank_mode: str,
        dedupe: bool,
        timings_ms: Dict[str, float],
        top_k: int,
    ) -> Dict[str, Any]:
        return {
            "ok": True,
            "query": query,
            "query_tokens": int(len(q_tokens)),
            "filters": {"video_id": video_id, "language": self.language, **active_filters},
            "index": {
                "index_dir": str(self.index_dir),
                "resolved_index_dir": str(
                    resolve_index_dir(
                        self.index_dir,
                        self.config_fingerprint,
                        self.language,
                        variant=(self.index.meta or {}).get("variant"),
                    )
                ),
                "config_fingerprint": self.config_fingerprint,
                "n_docs": int(n_docs),
                "n_valid": int(len(valid_indices)),
                "n_dense_valid": int(len(dense_valid_indices)),
                "embed_dim": int(getattr(self.index.embeddings, "shape", [0, 0])[1])
                if hasattr(self.index.embeddings, "shape")
                else None,
                "embeddings_dtype": str(getattr(emb, "dtype", "")),
            },
            "candidates": {
                "k_sparse": int(self.candidate_k_sparse),
                "k_dense": int(self.candidate_k_dense),
                "n_top_sparse": int(len(top_sparse)),
                "n_top_dense": int(len(top_dense)),
                "n_union": int(len(candidate_indices)),
                "dense_chunk_size": int(self.dense_chunk_size),
            },
            "dedupe": {
                "enabled": bool(dedupe),
                "mode": self.dedupe_mode,
                "pre_dedupe": int(pre_dedupe_n),
                "post_dedupe": int(len(results)),
                "overlap_thr": float(self.dedupe_overlap_thr),
                "tol_sec": float(self.dedupe_tol_sec),
            },
            "fusion": {
                "mode": self.fusion,
                "w_bm25": float(self.w_bm25),
                "w_dense": float(self.w_dense),
                "rrf_k": int(self.rrf_k),
            },
            "rerank": {
                "enabled": bool(self.rerank_enabled),
                "top_k": int(self.rerank_top_k),
                "backend": rerank_mode,
            },
            "timings_ms": timings_ms,
            "returned": {"top_k": int(top_k), "n_returned": int(len(results))},
        }

    def search(
        self,
        query: str,
        top_k: int = 10,
        video_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        dedupe: bool = True,
        return_stats: bool = False,
    ) -> Union[List[SearchResult], Tuple[List[SearchResult], Dict[str, Any]]]:
        t0 = time.perf_counter()

        docs = self.index.docs
        n_docs = len(docs) if docs else 0

        q = (query or "").strip()
        if not docs or not q:
            return self._empty_search_result(
                query=q,
                n_docs=n_docs,
                reason="empty_docs_or_query",
                return_stats=return_stats,
                extra={"total_ms": float((time.perf_counter() - t0) * 1000.0)},
            )

        active_filters = {k: v for k, v in (filters or {}).items() if v is not None and v != []}
        valid_indices, t_filter_ms = self._resolve_valid_indices(
            docs,
            video_id=video_id,
            active_filters=active_filters,
        )

        if not valid_indices:
            return self._empty_search_result(
                query=q,
                n_docs=n_docs,
                reason="no_valid_docs_after_filter",
                return_stats=return_stats,
                extra={
                    "n_valid": 0,
                    "filter_ms": float(t_filter_ms),
                    "total_ms": float((time.perf_counter() - t0) * 1000.0),
                },
            )

        q_tokens, bm25_all, top_sparse, t_sparse_ms = self._compute_sparse_stage(
            q,
            valid_indices=valid_indices,
        )

        t_qemb0 = time.perf_counter()
        q_vec = np.asarray(self.embedder.encode_query(q), dtype=np.float32)
        t_qemb_ms = (time.perf_counter() - t_qemb0) * 1000.0
        emb = self.index.embeddings
        dense_valid_idx, dense_valid_indices, top_dense, t_dense_topk_ms = self._compute_dense_stage(
            q_vec,
            valid_indices=valid_indices,
        )

        cand_set: Set[int] = set(top_sparse) | set(top_dense)
        if not cand_set:
            return self._empty_search_result(
                query=q,
                n_docs=n_docs,
                reason="no_candidates",
                return_stats=return_stats,
                extra={
                    "n_valid": int(len(valid_indices)),
                    "n_dense_valid": int(len(dense_valid_indices)),
                    "filter_ms": float(t_filter_ms),
                    "sparse_ms": float(t_sparse_ms),
                    "q_embed_ms": float(t_qemb_ms),
                    "dense_topk_ms": float(t_dense_topk_ms),
                    "total_ms": float((time.perf_counter() - t0) * 1000.0),
                },
            )

        candidate_indices = list(cand_set)
        sparse_raw, dense_raw, t_dense_cand_ms = self._collect_candidate_scores(
            bm25_all=bm25_all,
            q_vec=q_vec,
            candidate_indices=candidate_indices,
            dense_valid_idx=dense_valid_idx,
            include_dense=bool(top_dense),
        )

        ranked, fused, sparse_used, dense_used, t_fuse_ms = self._fuse_candidate_scores(
            bm25_all=bm25_all,
            candidate_indices=candidate_indices,
            sparse_raw=sparse_raw,
            dense_raw=dense_raw,
        )
        results, t_pack_ms = self._pack_ranked_results(
            docs=docs,
            ranked=ranked,
            fused=fused,
            sparse_used=sparse_used,
            dense_used=dense_used,
            top_k=top_k,
        )
        results, pre_dedupe_n, t_dedupe_ms = self._apply_dedupe(results, dedupe=dedupe)
        results, rerank_mode, t_rerank_ms = self._apply_rerank(
            results,
            query=q,
            q_tokens=q_tokens,
            top_k=top_k,
        )
        total_ms = (time.perf_counter() - t0) * 1000.0

        stats = self._build_search_stats(
            query=q,
            q_tokens=q_tokens,
            video_id=video_id,
            active_filters=active_filters,
            n_docs=n_docs,
            valid_indices=valid_indices,
            dense_valid_indices=dense_valid_indices,
            top_sparse=top_sparse,
            top_dense=top_dense,
            candidate_indices=candidate_indices,
            pre_dedupe_n=pre_dedupe_n,
            results=results,
            emb=emb,
            rerank_mode=rerank_mode,
            dedupe=dedupe,
            timings_ms={
                "filter": float(t_filter_ms),
                "sparse": float(t_sparse_ms),
                "q_embed": float(t_qemb_ms),
                "dense_topk": float(t_dense_topk_ms),
                "dense_candidates": float(t_dense_cand_ms),
                "fuse": float(t_fuse_ms),
                "pack": float(t_pack_ms),
                "dedupe": float(t_dedupe_ms),
                "rerank": float(t_rerank_ms),
                "total": float(total_ms),
            },
            top_k=top_k,
        )

        self.last_stats = stats

        logger.debug(
            "search ok=%s q=%r n_docs=%d n_valid=%d n_dense_valid=%d cand=%d ret=%d total_ms=%.2f",
            stats["ok"],
            q,
            stats["index"]["n_docs"],
            stats["index"]["n_valid"],
            stats["index"]["n_dense_valid"],
            stats["candidates"]["n_union"],
            stats["returned"]["n_returned"],
            stats["timings_ms"]["total"],
        )

        return (results, dict(stats)) if return_stats else results
