# src/search/rank.py
"""
Search ranking helpers for SmartCampus V2T.

Purpose:
- Provide dense/sparse fusion, top-k selection, deduplication, and rerank helpers.
- Keep ranking math and reranker integration separate from query-engine orchestration.
"""

from __future__ import annotations

import heapq
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np


def minmax_norm_from_dict(values: Dict[int, float], indices: Sequence[int]) -> Dict[int, float]:
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


def minmax_norm_on_indices(values: Sequence[float], indices: Sequence[int]) -> Dict[int, float]:
    if not indices:
        return {}
    xs = [float(values[i]) for i in indices]
    mn = min(xs)
    mx = max(xs)
    if mx - mn < 1e-9:
        return {int(i): 0.0 for i in indices}
    denom = mx - mn
    return {int(i): (float(values[i]) - mn) / denom for i in indices}


def topn_indices(scores: Sequence[float], indices: Sequence[int], n: int) -> List[int]:
    if n <= 0 or not indices:
        return []
    if n >= len(indices):
        return sorted(indices, key=lambda i: scores[i], reverse=True)
    return heapq.nlargest(n, indices, key=lambda i: scores[i])


def rrf_fuse(
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


def interval_iou(a0: float, a1: float, b0: float, b1: float) -> float:
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    if inter <= 0:
        return 0.0
    union = max(a1, b1) - min(a0, b0)
    return inter / (union + 1e-9)


def dedupe_time_hits_bucket(hits: List[Any], tol_sec: float = 1.0) -> List[Any]:
    out: List[Any] = []
    seen = set()
    tol = max(1e-6, float(tol_sec))
    for hit in hits:
        key = (hit.video_id, int(round(hit.start_sec / tol)), int(round(hit.end_sec / tol)))
        if key in seen:
            continue
        seen.add(key)
        out.append(hit)
    return out


def dedupe_time_hits_overlap_nms(hits: List[Any], overlap_thr: float = 0.7) -> List[Any]:
    kept: List[Any] = []
    thr = float(overlap_thr)
    for hit in hits:
        ok = True
        for kept_hit in kept:
            if hit.video_id != kept_hit.video_id:
                continue
            if interval_iou(hit.start_sec, hit.end_sec, kept_hit.start_sec, kept_hit.end_sec) >= thr:
                ok = False
                break
        if ok:
            kept.append(hit)
    return kept


def topk_dense_over_indices(
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
    for start in range(0, n, int(chunk_size)):
        chunk = valid_indices[start : start + int(chunk_size)]
        if not chunk:
            continue
        idx = np.asarray(chunk, dtype=np.int64)
        mat = emb[idx]
        if mat.dtype != np.float32:
            mat = mat.astype(np.float32, copy=False)
        scores = mat @ q
        for global_index, score in zip(chunk, scores.tolist()):
            if len(heap) < k:
                heapq.heappush(heap, (float(score), int(global_index)))
            elif float(score) > heap[0][0]:
                heapq.heapreplace(heap, (float(score), int(global_index)))

    heap.sort(key=lambda item: item[0], reverse=True)
    return [int(global_index) for _score, global_index in heap]


def dense_scores_for_indices(emb: np.ndarray, q_vec: np.ndarray, indices: Sequence[int]) -> Dict[int, float]:
    if not indices:
        return {}
    idx = np.asarray(list(indices), dtype=np.int64)
    mat = emb[idx]
    if mat.dtype != np.float32:
        mat = mat.astype(np.float32, copy=False)
    q = np.asarray(q_vec, dtype=np.float32)
    scores = (mat @ q).astype(np.float32, copy=False)
    return {int(i): float(s) for i, s in zip(idx.tolist(), scores.tolist())}


def in_sorted(sorted_unique: np.ndarray, values: np.ndarray) -> np.ndarray:
    """
    Safe membership check for sorted_unique (sorted, unique int64) against values (int64).
    Returns boolean mask aligned with values.
    """

    if sorted_unique.size == 0 or values.size == 0:
        return np.zeros((values.size,), dtype=np.bool_)
    pos = np.searchsorted(sorted_unique, values)
    in_range = pos < sorted_unique.size
    out = np.zeros((values.size,), dtype=np.bool_)
    if np.any(in_range):
        vv = values[in_range]
        pp = pos[in_range]
        out[in_range] = sorted_unique[pp] == vv
    return out


def as_str_set(values: Any) -> Set[str]:
    out: Set[str] = set()
    if isinstance(values, list):
        for item in values:
            text = str(item or "").strip().lower()
            if text:
                out.add(text)
    elif values is not None:
        text = str(values).strip().lower()
        if text:
            out.add(text)
    return out


def heuristic_rerank_bonus(query_tokens: Sequence[str], hit: Any, tokenize) -> float:
    """Compute a deterministic rerank bonus from structured metadata and text overlap."""

    if not query_tokens:
        return 0.0

    qset = {str(token).strip().lower() for token in query_tokens if str(token).strip()}
    if not qset:
        return 0.0

    bonus = 0.0
    desc_tokens = set(tokenize(str(getattr(hit, "description", "") or ""), normalize=False))
    if desc_tokens:
        bonus += 0.30 * (len(qset & desc_tokens) / max(1, len(qset)))

    for value, weight in (
        (getattr(hit, "event_type", None), 0.30),
        (getattr(hit, "risk_level", None), 0.18),
        (getattr(hit, "people_count_bucket", None), 0.12),
        (getattr(hit, "motion_type", None), 0.12),
    ):
        text = str(value or "").strip().lower()
        if text and text in qset:
            bonus += weight

    tag_hits = qset & {str(x).strip().lower() for x in (getattr(hit, "tags", None) or []) if str(x).strip()}
    if tag_hits:
        bonus += 0.24 * (len(tag_hits) / max(1, len(qset)))

    object_hits = qset & {str(x).strip().lower() for x in (getattr(hit, "objects", None) or []) if str(x).strip()}
    if object_hits:
        bonus += 0.18 * (len(object_hits) / max(1, len(qset)))

    if bool(getattr(hit, "anomaly_flag", False)) and ({"anomaly", "alert", "warning", "critical"} & qset):
        bonus += 0.14

    return float(bonus)


class TransformersReranker:
    """Score query-document pairs with a transformers sequence-classification model."""

    def __init__(self, model_name: str, device: Optional[str] = None):
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except Exception as exc:
            raise RuntimeError("transformers reranker dependencies are not installed.") from exc

        self.model_name = str(model_name)
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, trust_remote_code=True)
        self.model.eval()

        device_name = str(device or "").strip()
        if device_name:
            try:
                self.model.to(device_name)
            except Exception:
                pass

    def score_pairs(self, query: str, passages: Sequence[str]) -> List[float]:
        """Return one rerank score per passage."""

        if not passages:
            return []

        model_device = None
        try:
            model_device = next(self.model.parameters()).device
        except Exception:
            model_device = None

        encoded = self.tokenizer(
            [str(query or "")] * len(passages),
            [str(p or "") for p in passages],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        if model_device is not None:
            for key, value in encoded.items():
                try:
                    encoded[key] = value.to(model_device)
                except Exception:
                    pass

        with self._torch.inference_mode():
            outputs = self.model(**encoded)
            logits = getattr(outputs, "logits", outputs[0])
            if len(getattr(logits, "shape", ())) > 1 and int(logits.shape[-1]) > 1:
                logits = logits[:, 0]
            else:
                logits = logits.reshape(-1)
        return [float(x) for x in logits.detach().cpu().tolist()]

    def release(self) -> None:
        """Best-effort release of the transformers reranker backend."""

        model = getattr(self, "model", None)
        tokenizer = getattr(self, "tokenizer", None)
        self.model = None
        self.tokenizer = None
        try:
            if model is not None and hasattr(model, "cpu"):
                model.cpu()
        except Exception:
            pass
        del tokenizer


def build_reranker(model_name: str, backend: str, device: Optional[str], looks_like_transformers_model) -> Optional[TransformersReranker]:
    """Build a model reranker when possible, otherwise return None for heuristic fallback."""

    backend_name = str(backend or "auto").strip().lower() or "auto"
    if backend_name not in {"auto", "transformers"}:
        return None
    if backend_name == "auto" and not looks_like_transformers_model(model_name):
        return None
    try:
        return TransformersReranker(model_name=model_name, device=device)
    except Exception:
        return None
