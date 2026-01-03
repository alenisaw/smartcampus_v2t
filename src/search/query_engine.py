# search/query_engine.py

from __future__ import annotations

import heapq
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from .index_builder import E5Embedder, HybridIndex, _tokenize, load_index

DEFAULT_INDEX_DIR = Path("data/indexes")


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
        key = (
            h.video_id,
            h.run_id,
            int(round(h.start_sec / tol)),
            int(round(h.end_sec / tol)),
        )
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
        mat = np.asarray(emb[np.asarray(chunk, dtype=np.int64)], dtype=np.float32)
        scores = mat @ q  # (B,)
        for gi, sc in zip(chunk, scores.tolist()):
            if len(heap) < k:
                heapq.heappush(heap, (float(sc), int(gi)))
            else:
                if float(sc) > heap[0][0]:
                    heapq.heapreplace(heap, (float(sc), int(gi)))

    heap.sort(key=lambda x: x[0], reverse=True)
    return [int(gi) for _sc, gi in heap]


def _dense_scores_for_indices(
    emb: np.ndarray,
    q_vec: np.ndarray,
    indices: Sequence[int],
) -> Dict[int, float]:
    if not indices:
        return {}
    idx = np.asarray(list(indices), dtype=np.int64)
    mat = np.asarray(emb[idx], dtype=np.float32)
    q = np.asarray(q_vec, dtype=np.float32)
    scores = (mat @ q).astype(np.float32)
    return {int(i): float(s) for i, s in zip(idx.tolist(), scores.tolist())}


class QueryEngine:
    def __init__(
        self,
        index: Optional[HybridIndex] = None,
        index_dir: Path = DEFAULT_INDEX_DIR,
        w_bm25: float = 0.45,
        w_dense: float = 0.55,
        candidate_k_sparse: int = 200,
        candidate_k_dense: int = 200,
        fusion: str = "rrf",
        rrf_k: int = 60,
        dedupe_mode: str = "overlap",
        dedupe_tol_sec: float = 1.0,
        dedupe_overlap_thr: float = 0.7,
        device: Optional[str] = None,
        embed_model_name: Optional[str] = None,
        dense_chunk_size: int = 4096,
    ) -> None:
        self.index = index or load_index(Path(index_dir))

        self.w_bm25 = float(w_bm25)
        self.w_dense = float(w_dense)

        self.candidate_k_sparse = int(candidate_k_sparse)
        self.candidate_k_dense = int(candidate_k_dense)

        self.fusion = str(fusion).lower().strip()
        self.rrf_k = int(rrf_k)

        self.dedupe_mode = str(dedupe_mode).lower().strip()
        self.dedupe_tol_sec = float(dedupe_tol_sec)
        self.dedupe_overlap_thr = float(dedupe_overlap_thr)

        self.dense_chunk_size = max(256, int(dense_chunk_size))

        meta_model = None
        try:
            meta_model = (self.index.meta or {}).get("model_name")
        except Exception:
            meta_model = None

        model_name = (embed_model_name or meta_model or "intfloat/multilingual-e5-base")
        self.embedder = E5Embedder(model_name=str(model_name), device=device)

    def _build_mask(self, docs, video_id: Optional[str], run_id: Optional[str]) -> List[bool]:
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

        q = (query or "").strip()
        if not q:
            return []

        mask = self._build_mask(docs, video_id=video_id, run_id=run_id)
        valid_indices = [i for i, m in enumerate(mask) if m]
        if not valid_indices:
            return []

        q_tokens = _tokenize(q)
        bm25_all = self.index.bm25.score(q_tokens)
        top_sparse = _topn_indices(bm25_all, valid_indices, self.candidate_k_sparse)

        q_vec = np.asarray(self.embedder.encode_query(q), dtype=np.float32)
        emb = self.index.embeddings

        top_dense = _topk_dense_over_indices(
            emb=emb,
            q_vec=q_vec,
            valid_indices=valid_indices,
            k=self.candidate_k_dense,
            chunk_size=self.dense_chunk_size,
        )

        cand_set: Set[int] = set(top_sparse) | set(top_dense)
        if not cand_set:
            return []

        candidate_indices = list(cand_set)

        sparse_raw: Dict[int, float] = {int(i): float(bm25_all[int(i)]) for i in candidate_indices}
        dense_raw: Dict[int, float] = _dense_scores_for_indices(emb=emb, q_vec=q_vec, indices=candidate_indices)

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

        if dedupe and results:
            if self.dedupe_mode == "overlap":
                results = _dedupe_time_hits_overlap_nms(results, overlap_thr=self.dedupe_overlap_thr)
            elif self.dedupe_mode == "bucket":
                results = _dedupe_time_hits_bucket(results, tol_sec=self.dedupe_tol_sec)
            else:
                raise ValueError(f"Unknown dedupe_mode: {self.dedupe_mode!r}. Use 'overlap' or 'bucket'.")

        return results[: int(top_k)]
