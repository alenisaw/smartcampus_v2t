# search/query_engine.py
"""
Hybrid Query Engine for SmartCampus V2T.

Loads defaults from config/pipeline.yaml:
- paths.indexes_dir
- search.embed_model_name
- search weights/candidates/fusion/dedupe
and uses search_config_fingerprint(cfg) to resolve versioned index dir.

Uses dense_valid.npy to avoid dense scoring over invalid/zero embeddings.
"""

from __future__ import annotations

import heapq
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_loader import load_pipeline_config

from .index_builder import (
    E5Embedder,
    HybridIndex,
    MODEL_NAME_DEFAULT,
    _tokenize,
    load_index,
    resolve_index_dir,
    search_config_fingerprint,
)

logger = logging.getLogger(__name__)

DEFAULT_CFG_PATH = PROJECT_ROOT / "config" / "pipeline.yaml"


@dataclass
class SearchResult:
    score: float
    sparse_score: float
    dense_score: float
    video_id: str
    run_id: str
    start_sec: float
    end_sec: float
    description: str
    source_id: str
    extra: Optional[Dict[str, Any]] = None


def _minmax_norm_from_dict(values: Dict[int, float], indices: Sequence[int]) -> Dict[int, float]:
    if not indices:
        return {}
    xs = [float(values.get(i, float("-inf"))) for i in indices]
    if not xs or all(v == float("-inf") for v in xs):
        return {int(i): 0.0 for i in indices}
    mn = min(xs)
    mx = max(xs)
    if mx - mn < 1e-9:
        return {int(i): 0.0 for i in indices}
    denom = mx - mn
    return {int(i): (float(values.get(i, mn)) - mn) / denom for i in indices}


def _minmax_norm_on_indices(values: Sequence[float], indices: Sequence[int]) -> Dict[int, float]:
    if not indices:
        return {}
    xs = [float(values[i]) for i in indices]
    mn = min(xs)
    mx = max(xs)
    if mx - mn < 1e-9:
        return {int(i): 0.0 for i in indices}
    denom = mx - mn
    return {int(i): (float(values[i]) - mn) / denom for i in indices}


def _topn_indices(scores: Sequence[float], indices: Sequence[int], n: int) -> List[int]:
    if n <= 0 or not indices:
        return []
    if n >= len(indices):
        return sorted(indices, key=lambda i: scores[i], reverse=True)
    return heapq.nlargest(n, indices, key=lambda i: scores[i])


def _rrf_fuse(
    dense_scores: Dict[int, float],
    sparse_scores: Dict[int, float],
    candidate_indices: Sequence[int],
    k: int = 60,
    w_sparse: float = 0.45,
    w_dense: float = 0.55,
) -> Dict[int, float]:
    cand = list(candidate_indices)

    dense_ranked = sorted(cand, key=lambda i: dense_scores.get(i, float("-inf")), reverse=True)
    sparse_ranked = sorted(cand, key=lambda i: sparse_scores.get(i, float("-inf")), reverse=True)

    dense_rank: Dict[int, int] = {i: r for r, i in enumerate(dense_ranked, start=1)}
    sparse_rank: Dict[int, int] = {i: r for r, i in enumerate(sparse_ranked, start=1)}

    fused: Dict[int, float] = {}
    for i in cand:
        rd = dense_rank.get(i, len(cand) + 1)
        rs = sparse_rank.get(i, len(cand) + 1)
        fused[i] = (w_dense / (k + rd)) + (w_sparse / (k + rs))
    return fused


def _interval_iou(a0: float, a1: float, b0: float, b1: float) -> float:
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    if inter <= 0:
        return 0.0
    union = max(a1, b1) - min(a0, b0)
    return inter / (union + 1e-9)


def _dedupe_time_hits_bucket(hits: List[SearchResult], tol_sec: float = 1.0) -> List[SearchResult]:
    out: List[SearchResult] = []
    seen = set()
    tol = max(1e-6, float(tol_sec))
    for h in hits:
        key = (h.video_id, h.run_id, int(round(h.start_sec / tol)), int(round(h.end_sec / tol)))
        if key in seen:
            continue
        seen.add(key)
        out.append(h)
    return out


def _dedupe_time_hits_overlap_nms(hits: List[SearchResult], overlap_thr: float = 0.7) -> List[SearchResult]:
    kept: List[SearchResult] = []
    thr = float(overlap_thr)
    for h in hits:
        ok = True
        for k in kept:
            if h.video_id != k.video_id or h.run_id != k.run_id:
                continue
            if _interval_iou(h.start_sec, h.end_sec, k.start_sec, k.end_sec) >= thr:
                ok = False
                break
        if ok:
            kept.append(h)
    return kept


def _topk_dense_over_indices(
    emb: np.ndarray,
    q_vec: np.ndarray,
    valid_indices: Sequence[int],
    k: int,
    chunk_size: int = 4096,
) -> List[int]:
    k = int(k)
    if k <= 0 or not valid_indices:
        return []
    k = min(k, len(valid_indices))

    heap: List[Tuple[float, int]] = []
    q = np.asarray(q_vec, dtype=np.float32)

    n = len(valid_indices)
    for s in range(0, n, int(chunk_size)):
        chunk = valid_indices[s : s + int(chunk_size)]
        if not chunk:
            continue
        idx = np.asarray(chunk, dtype=np.int64)
        mat = emb[idx]
        if mat.dtype != np.float32:
            mat = mat.astype(np.float32, copy=False)
        scores = mat @ q
        for gi, sc in zip(chunk, scores.tolist()):
            if len(heap) < k:
                heapq.heappush(heap, (float(sc), int(gi)))
            elif float(sc) > heap[0][0]:
                heapq.heapreplace(heap, (float(sc), int(gi)))

    heap.sort(key=lambda x: x[0], reverse=True)
    return [int(gi) for _sc, gi in heap]


def _dense_scores_for_indices(emb: np.ndarray, q_vec: np.ndarray, indices: Sequence[int]) -> Dict[int, float]:
    if not indices:
        return {}
    idx = np.asarray(list(indices), dtype=np.int64)
    mat = emb[idx]
    if mat.dtype != np.float32:
        mat = mat.astype(np.float32, copy=False)
    q = np.asarray(q_vec, dtype=np.float32)
    scores = (mat @ q).astype(np.float32, copy=False)
    return {int(i): float(s) for i, s in zip(idx.tolist(), scores.tolist())}


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
        w_bm25: Optional[float] = None,
        w_dense: Optional[float] = None,
        candidate_k_sparse: Optional[int] = None,
        candidate_k_dense: Optional[int] = None,
        fusion: Optional[str] = None,
        rrf_k: Optional[int] = None,
        dedupe_mode: Optional[str] = None,
        dedupe_tol_sec: Optional[float] = None,
        dedupe_overlap_thr: Optional[float] = None,
        device: Optional[str] = None,
        embed_model_name: Optional[str] = None,
        dense_chunk_size: int = 4096,
    ) -> None:
        cfg = _load_cfg(config_path)

        s = cfg.search
        base_index_dir = Path(index_dir) if index_dir is not None else Path(cfg.paths.indexes_dir)

        cfg_fp = config_fingerprint or search_config_fingerprint(cfg)
        resolved_dir = resolve_index_dir(base_index_dir, cfg_fp)

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

        meta_model = None
        try:
            meta_model = (self.index.meta or {}).get("model_name")
        except Exception:
            meta_model = None

        model_name = (
            str(embed_model_name)
            if embed_model_name is not None
            else str(getattr(s, "embed_model_name", None) or meta_model or MODEL_NAME_DEFAULT)
        )

        self.embedder = E5Embedder(model_name=str(model_name), device=device)
        self.last_stats: Dict[str, Any] = {}

        self._dense_valid_indices_cache: Optional[np.ndarray] = None

    def get_last_stats(self) -> Dict[str, Any]:
        return dict(self.last_stats or {})

    def _build_mask(self, docs, video_id: Optional[str], run_id: Optional[str]) -> List[bool]:
        mask = [True] * len(docs)
        if video_id is not None:
            mask = [m and (d.video_id == video_id) for m, d in zip(mask, docs)]
        if run_id is not None:
            mask = [m and (d.run_id == run_id) for m, d in zip(mask, docs)]
        return mask

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

    def search(
        self,
        query: str,
        top_k: int = 10,
        video_id: Optional[str] = None,
        run_id: Optional[str] = None,
        dedupe: bool = True,
        return_stats: bool = False,
    ) -> Union[List[SearchResult], Tuple[List[SearchResult], Dict[str, Any]]]:
        t0 = time.perf_counter()

        docs = self.index.docs
        n_docs = len(docs) if docs else 0

        q = (query or "").strip()
        if not docs or not q:
            self.last_stats = {
                "query": q,
                "ok": False,
                "reason": "empty_docs_or_query",
                "n_docs": int(n_docs),
                "total_ms": float((time.perf_counter() - t0) * 1000.0),
            }
            return ([], dict(self.last_stats)) if return_stats else []

        t_filter0 = time.perf_counter()
        mask = self._build_mask(docs, video_id=video_id, run_id=run_id)
        valid_indices = [i for i, m in enumerate(mask) if m]
        t_filter_ms = (time.perf_counter() - t_filter0) * 1000.0

        if not valid_indices:
            self.last_stats = {
                "query": q,
                "ok": False,
                "reason": "no_valid_docs_after_filter",
                "n_docs": int(n_docs),
                "n_valid": 0,
                "filter_ms": float(t_filter_ms),
                "total_ms": float((time.perf_counter() - t0) * 1000.0),
            }
            return ([], dict(self.last_stats)) if return_stats else []

        t_sparse0 = time.perf_counter()
        q_tokens = _tokenize(q)
        bm25_all = self.index.bm25.score(q_tokens)
        top_sparse = _topn_indices(bm25_all, valid_indices, self.candidate_k_sparse)
        t_sparse_ms = (time.perf_counter() - t_sparse0) * 1000.0


        t_qemb0 = time.perf_counter()
        q_vec = np.asarray(self.embedder.encode_query(q), dtype=np.float32)
        t_qemb_ms = (time.perf_counter() - t_qemb0) * 1000.0


        t_dense0 = time.perf_counter()
        emb = self.index.embeddings

        dense_valid_idx = self._dense_valid_indices()
        if dense_valid_idx.size == 0:
            dense_valid_indices: List[int] = []
        else:
            v = np.asarray(valid_indices, dtype=np.int64)
            dv_sorted = dense_valid_idx  # already sorted from flatnonzero
            pos = np.searchsorted(dv_sorted, v)
            in_dv = (pos < dv_sorted.size) & (dv_sorted[pos] == v)
            dense_valid_indices = v[in_dv].tolist()

        top_dense = _topk_dense_over_indices(
            emb=emb,
            q_vec=q_vec,
            valid_indices=dense_valid_indices,
            k=self.candidate_k_dense,
            chunk_size=self.dense_chunk_size,
        )
        t_dense_topk_ms = (time.perf_counter() - t_dense0) * 1000.0

        cand_set: Set[int] = set(top_sparse) | set(top_dense)
        if not cand_set:
            self.last_stats = {
                "query": q,
                "ok": False,
                "reason": "no_candidates",
                "n_docs": int(n_docs),
                "n_valid": int(len(valid_indices)),
                "n_dense_valid": int(len(dense_valid_indices)),
                "filter_ms": float(t_filter_ms),
                "sparse_ms": float(t_sparse_ms),
                "q_embed_ms": float(t_qemb_ms),
                "dense_topk_ms": float(t_dense_topk_ms),
                "total_ms": float((time.perf_counter() - t0) * 1000.0),
            }
            return ([], dict(self.last_stats)) if return_stats else []

        candidate_indices = list(cand_set)


        t_dense_cand0 = time.perf_counter()
        sparse_raw: Dict[int, float] = {int(i): float(bm25_all[int(i)]) for i in candidate_indices}

        dense_raw: Dict[int, float] = {}
        if top_dense:

            cand = np.asarray(candidate_indices, dtype=np.int64)
            dv_sorted = dense_valid_idx
            pos = np.searchsorted(dv_sorted, cand)
            is_dv = (pos < dv_sorted.size) & (dv_sorted[pos] == cand)
            dense_cand = cand[is_dv].tolist()
            dense_raw = _dense_scores_for_indices(emb=emb, q_vec=q_vec, indices=dense_cand)

        t_dense_cand_ms = (time.perf_counter() - t_dense_cand0) * 1000.0


        t_fuse0 = time.perf_counter()
        if self.fusion == "minmax":
            sparse_n = _minmax_norm_on_indices(bm25_all, candidate_indices)


            dense_n = _minmax_norm_from_dict(dense_raw, candidate_indices)

            fused = {
                int(i): (self.w_bm25 * sparse_n.get(int(i), 0.0)) + (self.w_dense * dense_n.get(int(i), 0.0))
                for i in candidate_indices
            }
            sparse_used = sparse_n
            dense_used = dense_n
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
        t_fuse_ms = (time.perf_counter() - t_fuse0) * 1000.0


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
                    run_id=d.run_id,
                    start_sec=float(d.start_sec),
                    end_sec=float(d.end_sec),
                    description=d.text,
                    source_id=d.doc_id,
                    extra=d.extra,
                )
            )
        t_pack_ms = (time.perf_counter() - t_pack0) * 1000.0


        t_dedupe0 = time.perf_counter()
        pre_dedupe_n = len(results)
        if dedupe and results:
            if self.dedupe_mode == "overlap":
                results = _dedupe_time_hits_overlap_nms(results, overlap_thr=self.dedupe_overlap_thr)
            elif self.dedupe_mode == "bucket":
                results = _dedupe_time_hits_bucket(results, tol_sec=self.dedupe_tol_sec)
            else:
                raise ValueError(f"Unknown dedupe_mode: {self.dedupe_mode!r}. Use 'overlap' or 'bucket'.")
        t_dedupe_ms = (time.perf_counter() - t_dedupe0) * 1000.0

        results = results[: int(top_k)]
        total_ms = (time.perf_counter() - t0) * 1000.0

        stats: Dict[str, Any] = {
            "ok": True,
            "query": q,
            "query_tokens": int(len(q_tokens)),
            "filters": {"video_id": video_id, "run_id": run_id},
            "index": {
                "index_dir": str(self.index_dir),
                "resolved_index_dir": str(resolve_index_dir(self.index_dir, self.config_fingerprint)),
                "config_fingerprint": self.config_fingerprint,
                "n_docs": int(n_docs),
                "n_valid": int(len(valid_indices)),
                "n_dense_valid": int(len(dense_valid_indices)),
                "embed_dim": int(getattr(self.index.embeddings, "shape", [0, 0])[1]) if hasattr(self.index.embeddings, "shape") else None,
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
            "timings_ms": {
                "filter": float(t_filter_ms),
                "sparse": float(t_sparse_ms),
                "q_embed": float(t_qemb_ms),
                "dense_topk": float(t_dense_topk_ms),
                "dense_candidates": float(t_dense_cand_ms),
                "fuse": float(t_fuse_ms),
                "pack": float(t_pack_ms),
                "dedupe": float(t_dedupe_ms),
                "total": float(total_ms),
            },
            "returned": {"top_k": int(top_k), "n_returned": int(len(results))},
        }

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
