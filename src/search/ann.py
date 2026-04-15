# src/search/ann.py
"""
ANN helpers for SmartCampus V2T search runtime.

Purpose:
- Provide one abstraction for dense retrieval backends.
- Keep FAISS integration optional while preserving an exact NumPy fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

try:
    import faiss  # type: ignore
except Exception:
    faiss = None


def ann_backend_available(name: str) -> bool:
    """Return whether the requested ANN backend is available in the current environment."""

    backend = str(name or "auto").strip().lower() or "auto"
    if backend in {"", "auto", "exact"}:
        return True
    if backend == "faiss":
        return faiss is not None
    return False


def resolve_ann_backend(name: str) -> str:
    """Resolve the effective ANN backend with a safe fallback to exact search."""

    backend = str(name or "auto").strip().lower() or "auto"
    if backend in {"", "auto"}:
        return "faiss" if faiss is not None else "exact"
    if backend == "faiss" and faiss is not None:
        return "faiss"
    return "exact"


def resolve_faiss_index_type(name: str) -> str:
    """Resolve the FAISS index type with a safe default."""

    index_type = str(name or "hnsw").strip().lower() or "hnsw"
    if index_type in {"flat", "hnsw"}:
        return index_type
    return "hnsw"


def _in_sorted(sorted_unique: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Return a membership mask aligned with `values` for sorted unique int64 arrays."""

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


def _exact_topk(
    emb: np.ndarray,
    q_vec: np.ndarray,
    valid_indices: Sequence[int],
    *,
    k: int,
    chunk_size: int,
) -> List[int]:
    """Compute exact top-k dense retrieval on a subset of embedding rows."""

    if k <= 0 or not valid_indices:
        return []
    q = np.asarray(q_vec, dtype=np.float32)
    take = min(int(k), len(valid_indices))
    best_scores: List[float] = []
    best_indices: List[int] = []
    step = max(1, int(chunk_size))

    for start in range(0, len(valid_indices), step):
        chunk = valid_indices[start : start + step]
        idx = np.asarray(chunk, dtype=np.int64)
        mat = emb[idx]
        if mat.dtype != np.float32:
            mat = mat.astype(np.float32, copy=False)
        scores = (mat @ q).astype(np.float32, copy=False)
        best_scores.extend(scores.tolist())
        best_indices.extend(idx.tolist())

    if not best_scores:
        return []
    order = np.argsort(np.asarray(best_scores, dtype=np.float32))[::-1][:take]
    return [int(best_indices[int(i)]) for i in order.tolist()]


def _exact_scores(emb: np.ndarray, q_vec: np.ndarray, indices: Sequence[int]) -> Dict[int, float]:
    """Compute exact dense scores for selected embedding rows."""

    if not indices:
        return {}
    idx = np.asarray(list(indices), dtype=np.int64)
    mat = emb[idx]
    if mat.dtype != np.float32:
        mat = mat.astype(np.float32, copy=False)
    q = np.asarray(q_vec, dtype=np.float32)
    scores = (mat @ q).astype(np.float32, copy=False)
    return {int(i): float(s) for i, s in zip(idx.tolist(), scores.tolist())}


@dataclass
class DenseANNIndex:
    """Backend-neutral dense retrieval wrapper."""

    backend: str
    embeddings: np.ndarray
    dense_valid: np.ndarray
    faiss_index: Optional[Any] = None
    faiss_rows: Optional[np.ndarray] = None

    def topk(self, q_vec: np.ndarray, valid_indices: Sequence[int], *, k: int, chunk_size: int = 4096) -> List[int]:
        """Return top-k dense hits aligned to global embedding row indices."""

        if self.backend != "faiss" or self.faiss_index is None or self.faiss_rows is None:
            return _exact_topk(self.embeddings, q_vec, valid_indices, k=k, chunk_size=chunk_size)

        valid = np.asarray(list(valid_indices), dtype=np.int64)
        if valid.size == 0 or int(k) <= 0:
            return []

        dense_rows = np.asarray(self.faiss_rows, dtype=np.int64)
        valid_sorted = np.sort(valid)
        if valid_sorted.size == dense_rows.size and np.array_equal(valid_sorted, dense_rows):
            query = np.asarray(q_vec, dtype=np.float32).reshape(1, -1)
            scores, rows = self.faiss_index.search(query, min(int(k), int(dense_rows.size)))
            out: List[int] = []
            for row in rows.reshape(-1).tolist():
                if int(row) < 0:
                    continue
                out.append(int(dense_rows[int(row)]))
            return out

        # Restrictive filters are common. For correctness, fall back to exact subset scoring
        # when the valid set differs from the full dense-valid population.
        return _exact_topk(self.embeddings, q_vec, valid_indices, k=k, chunk_size=chunk_size)

    def scores(self, q_vec: np.ndarray, indices: Sequence[int]) -> Dict[int, float]:
        """Return dense scores aligned to global embedding row indices."""

        return _exact_scores(self.embeddings, q_vec, indices)


def save_ann_bundle(
    index_dir: Path,
    *,
    embeddings: np.ndarray,
    dense_valid: np.ndarray,
    backend: str,
    index_type: str = "hnsw",
    hnsw_m: int = 32,
    ef_construction: int = 80,
    ef_search: int = 64,
) -> Dict[str, Any]:
    """Persist ANN artifacts for one index directory and return runtime metadata."""

    index_dir = Path(index_dir)
    resolved_backend = resolve_ann_backend(backend)
    resolved_index_type = resolve_faiss_index_type(index_type)
    meta: Dict[str, Any] = {
        "backend": resolved_backend,
        "index_type": resolved_index_type,
        "hnsw_m": int(hnsw_m),
        "ef_construction": int(ef_construction),
        "ef_search": int(ef_search),
    }

    if resolved_backend != "faiss" or faiss is None:
        return meta

    dense_rows = np.flatnonzero(np.asarray(dense_valid, dtype=np.bool_)).astype(np.int64, copy=False)
    ann_rows_path = index_dir / "ann_rows.npy"
    faiss_path = index_dir / "faiss.index"

    if dense_rows.size == 0:
        if faiss_path.exists():
            faiss_path.unlink(missing_ok=True)
        if ann_rows_path.exists():
            ann_rows_path.unlink(missing_ok=True)
        return meta

    mat = embeddings[dense_rows]
    if mat.dtype != np.float32:
        mat = mat.astype(np.float32, copy=False)

    if resolved_index_type == "flat":
        index = faiss.IndexFlatIP(int(mat.shape[1]))
    else:
        index = faiss.IndexHNSWFlat(int(mat.shape[1]), max(8, int(hnsw_m)), faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = max(int(ef_construction), max(8, int(hnsw_m)))
        index.hnsw.efSearch = max(int(ef_search), 1)
    index.add(mat)
    faiss.write_index(index, str(faiss_path))
    np.save(ann_rows_path, dense_rows)
    meta["rows"] = int(dense_rows.size)
    meta["dimension"] = int(mat.shape[1]) if mat.ndim == 2 else 0
    return meta


def load_ann_bundle(index_dir: Path, *, embeddings: np.ndarray, dense_valid: np.ndarray, meta: Optional[Dict[str, Any]] = None) -> DenseANNIndex:
    """Load the persisted ANN backend for one index directory with a safe fallback."""

    bundle_meta = dict(meta or {})
    backend = resolve_ann_backend(str(bundle_meta.get("backend", "exact")))
    dense_bool = np.asarray(dense_valid, dtype=np.bool_)

    if backend != "faiss" or faiss is None:
        return DenseANNIndex(backend="exact", embeddings=embeddings, dense_valid=dense_bool)

    faiss_path = Path(index_dir) / "faiss.index"
    rows_path = Path(index_dir) / "ann_rows.npy"
    if not faiss_path.exists() or not rows_path.exists():
        return DenseANNIndex(backend="exact", embeddings=embeddings, dense_valid=dense_bool)

    try:
        index = faiss.read_index(str(faiss_path))
        if hasattr(index, "hnsw"):
            try:
                index.hnsw.efSearch = max(int(bundle_meta.get("ef_search", 64) or 64), 1)
            except Exception:
                pass
        rows = np.load(rows_path, mmap_mode="r")
        return DenseANNIndex(
            backend="faiss",
            embeddings=embeddings,
            dense_valid=dense_bool,
            faiss_index=index,
            faiss_rows=np.asarray(rows, dtype=np.int64),
        )
    except Exception:
        return DenseANNIndex(backend="exact", embeddings=embeddings, dense_valid=dense_bool)
