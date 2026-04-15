# src/search/store.py
"""
Search storage helpers for SmartCampus V2T.

Purpose:
- Load and persist index manifests, corpora, embeddings, and source segment files.
- Keep on-disk index layout logic separate from builder orchestration.
"""

from __future__ import annotations

import io
import json
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import zstandard as zstd
except Exception:
    zstd = None

from .embed import MODEL_NAME_DEFAULT, sanitize_tag
from .ann import load_ann_bundle
from .types import HybridIndex, normalize_loaded_doc


def read_json(path: Path) -> Any:
    """Read one JSON file."""

    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    """Write one JSON file with UTF-8 formatting."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def read_jsonl_gz(path: Path) -> List[Dict[str, Any]]:
    """Read a gzip-compressed JSONL segment file."""

    import gzip

    rows: List[Dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def read_jsonl_zst(path: Path) -> List[Dict[str, Any]]:
    """Read a zstd-compressed JSONL segment file."""

    if zstd is None:
        raise RuntimeError("zstandard is not installed")
    rows: List[Dict[str, Any]] = []
    with path.open("rb") as handle:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(handle) as reader:
            stream = io.TextIOWrapper(reader, encoding="utf-8")
            for line in stream:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if isinstance(obj, dict):
                    rows.append(obj)
    return rows


def iter_segment_files(videos_root: Path, language: str, variant: Optional[str] = None) -> List[Path]:
    """Return all segment files for one language and optional variant."""

    videos_root = Path(videos_root)
    if not videos_root.exists():
        return []
    lang = str(language).strip().lower()
    variant = str(variant).strip().lower() if variant else None
    files: List[Path] = []
    if variant:
        if zstd is not None:
            files += list(videos_root.glob(f"*/outputs/variants/{variant}/segments/{lang}.jsonl.zst"))
        files += list(videos_root.glob(f"*/outputs/variants/{variant}/segments/{lang}.jsonl.gz"))
    else:
        if zstd is not None:
            files += list(videos_root.glob(f"*/outputs/segments/{lang}.jsonl.zst"))
        files += list(videos_root.glob(f"*/outputs/segments/{lang}.jsonl.gz"))
    return sorted(files)


def guess_source_from_path(path: Path) -> Tuple[str, Optional[str]]:
    """Infer video id and variant from one segment-file path."""

    try:
        parts = list(path.parts)
        idx = parts.index("outputs")
        video_id = parts[idx - 1] if idx >= 1 else "unknown"
        variant = None
        if len(parts) > idx + 2 and parts[idx + 1] == "variants":
            variant = parts[idx + 2]
        return str(video_id), (str(variant) if variant else None)
    except Exception:
        return "unknown", None


def file_fingerprint(path: Path) -> str:
    """Build a stable fingerprint from file size and mtime."""

    stat = path.stat()
    return f"{stat.st_size}:{stat.st_mtime_ns}"


def stable_doc_id(video_id: str, index: int, variant: Optional[str] = None) -> str:
    """Build a deterministic document id for one segment position."""

    if variant:
        return f"{video_id}/{variant}/seg_{index:04d}"
    return f"{video_id}/seg_{index:04d}"


def normalize_embed_store_dtype(embed_store_dtype: str) -> str:
    """Normalize stored embedding dtype tags."""

    dtype = (embed_store_dtype or "float32").strip().lower()
    if dtype in {"fp16", "float16", "f16"}:
        return "float16"
    return "float32"


def config_tag_from_fingerprint(config_fingerprint: Optional[str]) -> str:
    """Build a filesystem-safe short tag from the config fingerprint."""

    if not config_fingerprint:
        return "default"
    fp = str(config_fingerprint).strip()
    if not fp:
        return "default"
    return sanitize_tag(fp[:24])


def resolve_index_dir(
    base_index_dir: Path,
    config_fingerprint: Optional[str] = None,
    language: Optional[str] = None,
    variant: Optional[str] = None,
) -> Path:
    """Resolve the index directory for one config, language, and variant."""

    base = Path(base_index_dir)
    if language:
        base = base / sanitize_tag(str(language))
    if variant:
        base = base / "variants" / sanitize_tag(str(variant))
    return base / config_tag_from_fingerprint(config_fingerprint)


def load_manifest(index_dir: Path) -> Dict[str, Any]:
    """Load one index manifest with schema defaults."""

    path = index_dir / "manifest.json"
    if not path.exists():
        layout = "videos/<video_id>/outputs/segments/<lang>.jsonl.zst"
        if zstd is None:
            layout = "videos/<video_id>/outputs/segments/<lang>.jsonl.gz"
        return {
            "version": 5,
            "model_name": MODEL_NAME_DEFAULT,
            "bm25": {"k1": 1.6, "b": 0.75},
            "sources": {},
            "dense_input_mode": "text",
            "layout": layout,
            "doc_id_scheme": "stable_by_position",
            "variant": None,
            "has_dense_valid": True,
        }

    manifest = read_json(path)
    if not isinstance(manifest, dict):
        manifest = {}
    manifest.setdefault("version", 5)
    manifest.setdefault("model_name", MODEL_NAME_DEFAULT)
    manifest.setdefault("bm25", {"k1": 1.6, "b": 0.75})
    manifest.setdefault("sources", {})
    if "layout" not in manifest:
        manifest["layout"] = "videos/<video_id>/outputs/segments/<lang>.jsonl.zst"
        if zstd is None:
            manifest["layout"] = "videos/<video_id>/outputs/segments/<lang>.jsonl.gz"
    manifest.setdefault("doc_id_scheme", "stable_by_position")
    manifest.setdefault("language", None)
    manifest.setdefault("variant", None)
    manifest.setdefault("dense_input_mode", "text")
    manifest.setdefault("has_dense_valid", True)
    manifest.setdefault("ann_backend", "exact")
    return manifest


def save_manifest(index_dir: Path, manifest: Dict[str, Any]) -> None:
    """Persist one index manifest."""

    write_json(index_dir / "manifest.json", manifest)


def corpus_pkl_path(index_dir: Path) -> Path:
    """Return the persisted corpus pickle path."""

    return index_dir / "corpus.pkl"


def load_corpus(index_dir: Path, normalize_doc) -> Dict[str, Any]:
    """Load and normalize the stored corpus map."""

    pkl = corpus_pkl_path(index_dir)
    if not pkl.exists():
        return {}
    obj = pickle.loads(pkl.read_bytes())
    if not isinstance(obj, dict):
        return {}
    out: Dict[str, Any] = {}
    for key, value in obj.items():
        normalized = normalize_doc(value)
        if normalized is None:
            continue
        if not normalized.doc_id:
            normalized.doc_id = str(key)
        out[str(normalized.doc_id)] = normalized
    return out


def save_corpus(index_dir: Path, docs_by_id: Dict[str, Any]) -> None:
    """Persist the normalized corpus map."""

    pkl = corpus_pkl_path(index_dir)
    pkl.parent.mkdir(parents=True, exist_ok=True)
    pkl.write_bytes(pickle.dumps(docs_by_id))


def load_prev_embeddings(index_dir: Path) -> Tuple[List[str], Optional[Any]]:
    """Load previous embeddings and aligned doc ids when available."""

    doc_ids_path = index_dir / "doc_ids.json"
    emb_path = index_dir / "embeddings.npy"
    if not doc_ids_path.exists() or not emb_path.exists():
        return [], None
    try:
        old_doc_ids = read_json(doc_ids_path)
        # Windows keeps a stronger lock on memory-mapped files. The builder later
        # overwrites the same embeddings file during incremental updates, so avoid
        # mmap there to prevent save-time `OSError: [Errno 22] Invalid argument`.
        old_emb = np.load(emb_path, mmap_mode=None if os.name == "nt" else "r")
        if not isinstance(old_doc_ids, list):
            return [], None
        if int(old_emb.shape[0]) != len(old_doc_ids):
            return [], None
        return [str(item) for item in old_doc_ids], old_emb
    except Exception:
        return [], None


def compute_dense_valid_mask(emb_arr: Any) -> Any:
    """Compute a boolean mask for rows with non-zero dense embeddings."""

    if emb_arr is None or not hasattr(emb_arr, "shape"):
        return np.zeros((0,), dtype=bool)
    nrows = int(emb_arr.shape[0])
    if nrows == 0:
        return np.zeros((0,), dtype=bool)

    arr = emb_arr
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32, copy=False)
    norm2 = np.einsum("ij,ij->i", arr, arr)
    return norm2 > 1e-10


def load_index(index_dir: Path) -> HybridIndex:
    """Load one persisted hybrid index directory into runtime structures."""

    index_dir = Path(index_dir)
    manifest = load_manifest(index_dir)

    meta: Dict[str, Any] = {}
    meta_path = index_dir / "meta.json"
    if meta_path.exists():
        meta_obj = read_json(meta_path)
        if isinstance(meta_obj, dict):
            meta = meta_obj

    docs_by_id = load_corpus(index_dir, normalize_loaded_doc)

    doc_ids = read_json(index_dir / "doc_ids.json")
    if not isinstance(doc_ids, list):
        doc_ids = []
    doc_ids = [str(item) for item in doc_ids]
    docs = [docs_by_id[doc_id] for doc_id in doc_ids if doc_id in docs_by_id]

    bm25 = pickle.loads((index_dir / "bm25.pkl").read_bytes())
    embeddings = np.load(index_dir / "embeddings.npy", mmap_mode="r")

    dense_valid_path = index_dir / "dense_valid.npy"
    if dense_valid_path.exists():
        dense_valid = np.load(dense_valid_path, mmap_mode="r")
        if dense_valid.dtype != np.bool_:
            dense_valid = dense_valid.astype(np.bool_, copy=False)
        dense_rows = int(getattr(dense_valid, "shape", [0])[0])
        embed_rows = int(getattr(embeddings, "shape", [0, 0])[0])
        if dense_rows != embed_rows:
            dense_valid = np.ones((embed_rows,), dtype=np.bool_)
    else:
        dense_valid = np.ones((int(embeddings.shape[0]),), dtype=np.bool_)

    ann_meta = meta.get("ann") if isinstance(meta.get("ann"), dict) else {"backend": manifest.get("ann_backend", "exact")}
    ann_index = load_ann_bundle(index_dir, embeddings=embeddings, dense_valid=dense_valid, meta=ann_meta)

    return HybridIndex(
        docs=docs,
        doc_ids=doc_ids,
        bm25=bm25,
        embeddings=embeddings,
        dense_valid=dense_valid,
        ann_index=ann_index,
        meta={**manifest, **meta},
    )
