# search/query_engine.py

from __future__ import annotations

import heapq
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .index_builder import E5Embedder, HybridIndex, _tokenize, load_index


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


def _minmax_norm_on_indices(values: Sequence[float], indices: Sequence[int]) -> Dict[int, float]:

    if not indices:
        return {}
    xs = [float(values[i]) for i in indices]
    mn = min(xs)
    mx = max(xs)
    if mx - mn < 1e-9:
        return {i: 0.0 for i in indices}
    denom = mx - mn
    return {i: (float(values[i]) - mn) / denom for i in indices}


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
    for h in hits:
        key = (
            h.video_id,
            h.run_id,
            int(round(h.start_sec / tol_sec)),
            int(round(h.end_sec / tol_sec)),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(h)
    return out


def _dedupe_time_hits_overlap_nms(
    hits: List[SearchResult],
    overlap_thr: float = 0.7,
) -> List[SearchResult]:

    kept: List[SearchResult] = []
    for h in hits:
        ok = True
        for k in kept:
            if h.video_id != k.video_id or h.run_id != k.run_id:
                continue
            if _interval_iou(h.start_sec, h.end_sec, k.start_sec, k.end_sec) >= overlap_thr:
                ok = False
                break
        if ok:
            kept.append(h)
    return kept


class QueryEngine:
    def __init__(
        self,
        index: Optional[HybridIndex] = None,
        index_dir: Path = Path("data/indexes"),

        w_bm25: float = 0.45,
        w_dense: float = 0.55,

        candidate_k_sparse: int = 200,
        candidate_k_dense: int = 200,
        fusion: str = "rrf",  # "rrf" | "minmax"
        rrf_k: int = 60,

        dedupe_mode: str = "overlap",  # "overlap" | "bucket"
        dedupe_tol_sec: float = 1.0,  # for bucket mode
        dedupe_overlap_thr: float = 0.7,  # for overlap mode
        device: Optional[str] = None,
    ) -> None:
        self.index = index or load_index(index_dir)

        self.w_bm25 = float(w_bm25)
        self.w_dense = float(w_dense)

        self.candidate_k_sparse = int(candidate_k_sparse)
        self.candidate_k_dense = int(candidate_k_dense)

        self.fusion = str(fusion).lower().strip()
        self.rrf_k = int(rrf_k)

        self.dedupe_mode = str(dedupe_mode).lower().strip()
        self.dedupe_tol_sec = float(dedupe_tol_sec)
        self.dedupe_overlap_thr = float(dedupe_overlap_thr)

        model_name = self.index.meta.get("model_name", "intfloat/multilingual-e5-base")
        self.embedder = E5Embedder(model_name=model_name, device=device)

    def _build_mask(
        self,
        docs,
        video_id: Optional[str],
        run_id: Optional[str],
    ) -> List[bool]:
        mask = [True] * len(docs)
        if video_id is not None:
            mask = [m and (d.video_id == video_id) for m, d in zip(mask, docs)]
        if run_id is not None:
            mask = [m and (d.run_id == run_id) for m, d in zip(mask, docs)]
        return mask

    def search(
        self,
        query: str,
        top_k: int = 10,
        video_id: Optional[str] = None,
        run_id: Optional[str] = None,
        dedupe: bool = True,
    ) -> List[SearchResult]:
        docs = self.index.docs
        if not docs:
            return []

        mask = self._build_mask(docs, video_id=video_id, run_id=run_id)
        valid_indices = [i for i, m in enumerate(mask) if m]
        if not valid_indices:
            return []

        q_tokens = _tokenize(query)
        bm25_all = self.index.bm25.score(q_tokens)  # length N


        top_sparse = _topn_indices(bm25_all, valid_indices, self.candidate_k_sparse)


        q_vec = self.embedder.encode_query(query)  # (D,)
        emb = self.index.embeddings  # (N, D)


        try:
            import numpy as np
        except Exception as e:
            raise RuntimeError("Install numpy to run dense search: pip install -U numpy") from e

        valid_emb = emb[valid_indices]  # (M, D)
        dense_valid = (valid_emb @ np.asarray(q_vec)).astype(float).tolist()  # length M

        dense_all: List[float] = [float("-inf")] * len(docs)
        for local_j, idx in enumerate(valid_indices):
            dense_all[idx] = float(dense_valid[local_j])

        top_dense = _topn_indices(dense_all, valid_indices, self.candidate_k_dense)


        cand_set: Set[int] = set(top_sparse) | set(top_dense)
        if not cand_set:

            return []

        candidate_indices = list(cand_set)


        sparse_raw = {i: float(bm25_all[i]) for i in candidate_indices}
        dense_raw = {i: float(dense_all[i]) for i in candidate_indices}

        if self.fusion == "minmax":
            sparse_n = _minmax_norm_on_indices(bm25_all, candidate_indices)
            dense_n = _minmax_norm_on_indices(dense_all, candidate_indices)
            fused = {
                i: (self.w_bm25 * sparse_n.get(i, 0.0)) + (self.w_dense * dense_n.get(i, 0.0))
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
            dense_used = _minmax_norm_on_indices(dense_all, candidate_indices)
        else:
            raise ValueError(f"Unknown fusion mode: {self.fusion!r}. Use 'rrf' or 'minmax'.")

        ranked = sorted(candidate_indices, key=lambda i: fused.get(i, float("-inf")), reverse=True)


        results: List[SearchResult] = []
        take = max(top_k * 5, top_k)
        for i in ranked[:take]:
            d = docs[i]
            results.append(
                SearchResult(
                    score=float(fused.get(i, 0.0)),
                    sparse_score=float(sparse_used.get(i, 0.0)),
                    dense_score=float(dense_used.get(i, 0.0)),
                    video_id=d.video_id,
                    run_id=d.run_id,
                    start_sec=float(d.start_sec),
                    end_sec=float(d.end_sec),
                    description=d.text,
                    source_id=d.doc_id,
                    extra=d.extra,
                )
            )

        if dedupe and results:
            if self.dedupe_mode == "overlap":
                results = _dedupe_time_hits_overlap_nms(results, overlap_thr=self.dedupe_overlap_thr)
            elif self.dedupe_mode == "bucket":
                results = _dedupe_time_hits_bucket(results, tol_sec=self.dedupe_tol_sec)
            else:
                raise ValueError(
                    f"Unknown dedupe_mode: {self.dedupe_mode!r}. Use 'overlap' or 'bucket'."
                )

        return results[:top_k]
