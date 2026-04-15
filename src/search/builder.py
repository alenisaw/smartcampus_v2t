# src/search/builder.py
"""
Search index builder for SmartCampus V2T.

Purpose:
- Build and update hybrid search indexes from stored video outputs.
- Assemble sparse, dense, and metadata artifacts used by retrieval.
"""

from __future__ import annotations

import json
import logging
import numpy as np
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_DATA_DIR = Path("data")
DEFAULT_VIDEOS_ROOT = DEFAULT_DATA_DIR / "videos"
DEFAULT_INDEX_DIR = DEFAULT_DATA_DIR / "indexes"

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from .corpus import (
    build_dense_text as _build_dense_text,
    build_searchable_text as _build_searchable_text,
    extract_doc_metadata as _extract_doc_metadata,
    resolve_keyframe_path as _resolve_keyframe_path,
)
from .embed import (
    MODEL_NAME_DEFAULT,
    EmbeddingCache,
    build_text_embedder,
    select_embedding_model_ref,
)
from .ann import save_ann_bundle
from .store import (
    compute_dense_valid_mask as _compute_dense_valid_mask,
    config_tag_from_fingerprint,
    file_fingerprint as _file_fingerprint,
    guess_source_from_path as _guess_source_from_path,
    iter_segment_files as _iter_segment_files,
    load_index,
    load_corpus as _load_corpus,
    load_manifest as _load_manifest,
    load_prev_embeddings as _load_prev_embeddings,
    normalize_embed_store_dtype as _normalize_embed_store_dtype,
    read_jsonl_gz as _read_jsonl_gz,
    read_jsonl_zst as _read_jsonl_zst,
    resolve_index_dir,
    save_corpus as _save_corpus,
    save_manifest as _save_manifest,
    stable_doc_id as _stable_doc_id,
    write_json as _write_json,
    zstd,
)
from .text import tokenize as _tokenize
from .types import BM25Index, Doc, normalize_loaded_doc as _normalize_loaded_doc


def _hash_text(text: str) -> str:
    import hashlib

    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _configure_manifest_layout(manifest: Dict[str, Any], *, variant: Optional[str], language: str, dense_input_mode: str) -> None:
    manifest.setdefault("sources", {})
    if variant:
        manifest["layout"] = "videos/<video_id>/outputs/variants/<variant>/segments/<lang>.jsonl.zst"
        if zstd is None:
            manifest["layout"] = "videos/<video_id>/outputs/variants/<variant>/segments/<lang>.jsonl.gz"
        manifest["doc_id_scheme"] = "stable_by_variant_and_position"
    else:
        manifest["layout"] = "videos/<video_id>/outputs/segments/<lang>.jsonl.zst"
        if zstd is None:
            manifest["layout"] = "videos/<video_id>/outputs/segments/<lang>.jsonl.gz"
        manifest["doc_id_scheme"] = "stable_by_position"
    manifest["language"] = str(language)
    manifest["variant"] = str(variant) if variant else None
    manifest["dense_input_mode"] = dense_input_mode
    manifest["version"] = max(5, int(manifest.get("version", 5) or 5))
    manifest["has_dense_valid"] = True


def _load_segment_rows(path: Path) -> List[Any]:
    if path.suffix == ".zst":
        return _read_jsonl_zst(path)
    return _read_jsonl_gz(path)


def _upsert_docs_from_segment_file(
    *,
    path: Path,
    videos_root: Path,
    language: str,
    video_id: str,
    file_variant: Optional[str],
    rows: List[Any],
    docs_by_id: Dict[str, Doc],
    changed_doc_ids: set[str],
) -> int:
    new_num_docs = 0
    for i, item in enumerate(rows):
        if not isinstance(item, dict):
            continue
        try:
            start = float(item["start_sec"])
            end = float(item["end_sec"])
            display_text = str(item.get("normalized_caption") or item.get("description", "") or "")
        except Exception:
            continue

        doc_id = _stable_doc_id(video_id, i, variant=file_variant)
        doc_extra = _extract_doc_metadata(item, file_variant)
        keyframe_raw = None
        evidence = item.get("evidence")
        if isinstance(evidence, dict):
            keyframe_raw = evidence.get("keyframe_path")
        if not keyframe_raw:
            keyframe_raw = doc_extra.get("keyframe_path")
        keyframe_abs = _resolve_keyframe_path(
            raw_path=str(keyframe_raw or ""),
            videos_root=videos_root,
            video_id=video_id,
            segment_file=path,
        )
        if keyframe_abs:
            doc_extra["keyframe_path"] = keyframe_abs
        search_text = _build_searchable_text(display_text, doc_extra)
        docs_by_id[doc_id] = Doc(
            doc_id=doc_id,
            video_id=video_id,
            language=str(language),
            start_sec=start,
            end_sec=end,
            text=search_text,
            display_text=display_text,
            extra=doc_extra,
            source_path=str(path),
        )
        changed_doc_ids.add(doc_id)
        new_num_docs += 1
    return new_num_docs


def _scan_segment_sources(
    *,
    files: List[Path],
    videos_root: Path,
    language: str,
    manifest: Dict[str, Any],
    docs_by_id: Dict[str, Doc],
) -> Dict[str, Any]:
    t_scan0 = time.perf_counter()
    changed_doc_ids: set[str] = set()
    changed_files = 0
    unchanged_files = 0
    parse_errors = 0
    docs_added_or_updated = 0
    docs_deleted = 0
    changed_any = False

    for path in files:
        path = path.resolve()
        path_key = str(path)
        fp = _file_fingerprint(path)

        prev = (manifest.get("sources") or {}).get(path_key)
        prev_fp = prev.get("fingerprint") if isinstance(prev, dict) else None
        if prev_fp == fp:
            unchanged_files += 1
            continue

        changed_files += 1
        video_id, file_variant = _guess_source_from_path(path)
        try:
            rows = _load_segment_rows(path)
        except Exception:
            parse_errors += 1
            continue

        prev_num_docs = int(prev.get("num_docs", 0)) if isinstance(prev, dict) else 0
        new_num_docs = _upsert_docs_from_segment_file(
            path=path,
            videos_root=videos_root,
            language=language,
            video_id=video_id,
            file_variant=file_variant,
            rows=rows,
            docs_by_id=docs_by_id,
            changed_doc_ids=changed_doc_ids,
        )
        docs_added_or_updated += int(new_num_docs)

        if prev_num_docs > new_num_docs:
            for j in range(new_num_docs, prev_num_docs):
                did = _stable_doc_id(video_id, j, variant=file_variant)
                if did in docs_by_id:
                    docs_by_id.pop(did, None)
                    docs_deleted += 1

        manifest["sources"][path_key] = {
            "fingerprint": fp,
            "video_id": video_id,
            "variant": file_variant,
            "language": str(language),
            "num_docs": int(new_num_docs),
        }
        changed_any = True

    return {
        "changed_any": changed_any,
        "changed_doc_ids": changed_doc_ids,
        "changed_files": changed_files,
        "unchanged_files": unchanged_files,
        "parse_errors": parse_errors,
        "docs_added_or_updated": docs_added_or_updated,
        "docs_deleted": docs_deleted,
        "scan_ms": (time.perf_counter() - t_scan0) * 1000.0,
    }


def _build_no_change_meta(
    *,
    manifest: Dict[str, Any],
    config_fingerprint: Optional[str],
    embedding_backend: str,
    embed_store_dtype: str,
    language: str,
    variant: Optional[str],
    query_prefix: str,
    passage_prefix: str,
    normalize_text: bool,
    lemmatize: bool,
    dense_input_mode: str,
    ann_backend: str,
    ann_index_type: str,
    ann_hnsw_m: int,
    ann_ef_construction: int,
    ann_ef_search: int,
    docs_by_id: Dict[str, Doc],
    files: List[Path],
    scan_stats: Dict[str, Any],
    total_ms: float,
) -> Dict[str, Any]:
    return {
        "index_schema_version": 5,
        "config_fingerprint": (str(config_fingerprint) if config_fingerprint else None),
        "config_tag": config_tag_from_fingerprint(config_fingerprint),
        "model_name": manifest.get("model_name"),
        "embedding_backend": embedding_backend,
        "num_docs": int(len(docs_by_id)),
        "bm25": dict(manifest.get("bm25") or {}),
        "embed_store_dtype": embed_store_dtype,
        "layout": manifest.get("layout"),
        "doc_id_scheme": manifest.get("doc_id_scheme"),
        "language": str(language),
        "variant": (str(variant) if variant else None),
        "query_prefix": str(query_prefix),
        "passage_prefix": str(passage_prefix),
        "normalize_text": bool(normalize_text),
        "lemmatize": bool(lemmatize),
        "dense_input_mode": dense_input_mode,
        "ann_backend": ann_backend,
        "ann_index_type": ann_index_type,
        "ann_hnsw_m": int(ann_hnsw_m),
        "ann_ef_construction": int(ann_ef_construction),
        "ann_ef_search": int(ann_ef_search),
        "has_dense_valid": True,
        "runtime": {
            "note": "No changes detected. Index not rebuilt.",
            "timings_ms": {
                "scan": float(scan_stats["scan_ms"]),
                "total": float(total_ms),
            },
            "counts": {
                "files_total": int(len(files)),
                "files_changed": int(scan_stats["changed_files"]),
                "files_unchanged": int(scan_stats["unchanged_files"]),
                "parse_errors": int(scan_stats["parse_errors"]),
                "docs_added_or_updated": int(scan_stats["docs_added_or_updated"]),
                "docs_deleted": int(scan_stats["docs_deleted"]),
            },
        },
    }


def search_config_fingerprint(cfg: Any) -> str:
    cfg_fp = getattr(cfg, "config_fingerprint", None)
    if cfg_fp:
        return str(cfg_fp)
    try:
        s = cfg.search
        payload = {
            "embed_model_name": str(getattr(s, "embed_model_name", MODEL_NAME_DEFAULT)),
            "query_prefix": str(getattr(s, "query_prefix", "query: ")),
            "passage_prefix": str(getattr(s, "passage_prefix", "passage: ")),
            "embedding_model_id": str(getattr(s, "embedding_model_id", "")),
            "embedding_backend": str(getattr(s, "embedding_backend", "auto")),
            "ann_backend": str(getattr(s, "ann_backend", "auto")),
            "ann_index_type": str(getattr(s, "ann_index_type", "hnsw")),
            "ann_hnsw_m": int(getattr(s, "ann_hnsw_m", 32)),
            "ann_ef_construction": int(getattr(s, "ann_ef_construction", 80)),
            "ann_ef_search": int(getattr(s, "ann_ef_search", 64)),
            "reranker_model_id": str(getattr(s, "reranker_model_id", "")),
            "reranker_backend": str(getattr(s, "reranker_backend", "auto")),
            "normalize_text": bool(getattr(s, "normalize_text", True)),
            "lemmatize": bool(getattr(s, "lemmatize", False)),
            "dense_input_mode": str(getattr(s, "dense_input_mode", "text")),
            "w_bm25": float(getattr(s, "w_bm25", 0.45)),
            "w_dense": float(getattr(s, "w_dense", 0.55)),
            "candidate_k_sparse": int(getattr(s, "candidate_k_sparse", 200)),
            "candidate_k_dense": int(getattr(s, "candidate_k_dense", 200)),
            "fusion": str(getattr(s, "fusion", "rrf")).strip().lower(),
            "rrf_k": int(getattr(s, "rrf_k", 60)),
            "dedupe_mode": str(getattr(s, "dedupe_mode", "overlap")).strip().lower(),
            "dedupe_tol_sec": float(getattr(s, "dedupe_tol_sec", 1.0)),
            "dedupe_overlap_thr": float(getattr(s, "dedupe_overlap_thr", 0.7)),
        }
    except Exception:
        payload = {"fallback": True}

    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    import hashlib

    return hashlib.sha1(raw).hexdigest()[:16]


def build_or_update_index(
    videos_root: Path = DEFAULT_VIDEOS_ROOT,
    index_dir: Path = DEFAULT_INDEX_DIR,
    model_name: str = MODEL_NAME_DEFAULT,
    embedding_backend: str = "auto",
    fallback_model_name: Optional[str] = None,
    device: Optional[str] = None,
    bm25_k1: float = 1.6,
    bm25_b: float = 0.75,
    batch_size: int = 64,
    config_fingerprint: Optional[str] = None,
    variant: Optional[str] = None,
    embed_store_dtype: str = "float32",
    language: str = "en",
    query_prefix: str = "query: ",
    passage_prefix: str = "passage: ",
    normalize_text: bool = True,
    lemmatize: bool = False,
    dense_input_mode: str = "text",
    cache_dir: Optional[Path] = None,
    use_embed_cache: bool = True,
    ann_backend: str = "auto",
    ann_index_type: str = "hnsw",
    ann_hnsw_m: int = 32,
    ann_ef_construction: int = 80,
    ann_ef_search: int = 64,
) -> Path:

    t_total0 = time.perf_counter()
    videos_root = Path(videos_root)

    base_index_dir = Path(index_dir)
    real_index_dir = resolve_index_dir(base_index_dir, config_fingerprint, language=language, variant=variant)
    real_index_dir.mkdir(parents=True, exist_ok=True)

    embed_store_dtype = _normalize_embed_store_dtype(embed_store_dtype)
    embedding_backend = str(embedding_backend or "auto").strip().lower() or "auto"
    dense_input_mode = str(dense_input_mode or "text").strip().lower() or "text"
    ann_backend = str(ann_backend or "auto").strip().lower() or "auto"
    ann_index_type = str(ann_index_type or "hnsw").strip().lower() or "hnsw"
    manifest = _load_manifest(real_index_dir)
    prev_manifest_version = int(manifest.get("version", 0) or 0)
    schema_changed = prev_manifest_version < 5

    old_bm25 = (manifest.get("bm25") or {})
    bm25_changed = (float(old_bm25.get("k1", 1.6)) != float(bm25_k1)) or (float(old_bm25.get("b", 0.75)) != float(bm25_b))
    model_changed = manifest.get("model_name") != model_name
    dense_mode_changed = str(manifest.get("dense_input_mode", "text")).strip().lower() != dense_input_mode

    if model_changed or schema_changed:
        manifest["model_name"] = model_name
        manifest["sources"] = {}
        docs_by_id: Dict[str, Doc] = {}
        old_doc_ids: List[str] = []
        old_emb = None
    else:
        docs_by_id = _load_corpus(real_index_dir, _normalize_loaded_doc)
        old_doc_ids, old_emb = _load_prev_embeddings(real_index_dir)

    manifest["bm25"] = {"k1": float(bm25_k1), "b": float(bm25_b)}
    _configure_manifest_layout(manifest, variant=variant, language=language, dense_input_mode=dense_input_mode)
    manifest["ann_backend"] = ann_backend

    files = _iter_segment_files(videos_root, language=language, variant=variant)
    if not files:
        suffix = f", variant={variant}" if variant else ""
        raise RuntimeError(f"No segments found under: {videos_root} (lang={language}{suffix})")

    scan_stats = _scan_segment_sources(
        files=files,
        videos_root=videos_root,
        language=language,
        manifest=manifest,
        docs_by_id=docs_by_id,
    )
    changed_any = bool(scan_stats["changed_any"])
    changed_doc_ids = set(scan_stats["changed_doc_ids"])
    changed_files = int(scan_stats["changed_files"])
    unchanged_files = int(scan_stats["unchanged_files"])
    parse_errors = int(scan_stats["parse_errors"])
    docs_added_or_updated = int(scan_stats["docs_added_or_updated"])
    docs_deleted = int(scan_stats["docs_deleted"])
    t_scan_ms = float(scan_stats["scan_ms"])

    if not changed_any and not bm25_changed and not model_changed and not dense_mode_changed:
        _save_manifest(real_index_dir, manifest)
        meta = _build_no_change_meta(
            manifest=manifest,
            config_fingerprint=config_fingerprint,
            embedding_backend=embedding_backend,
            embed_store_dtype=embed_store_dtype,
            language=language,
            variant=variant,
            query_prefix=query_prefix,
            passage_prefix=passage_prefix,
            normalize_text=normalize_text,
            lemmatize=lemmatize,
            dense_input_mode=dense_input_mode,
            ann_backend=ann_backend,
            ann_index_type=ann_index_type,
            ann_hnsw_m=ann_hnsw_m,
            ann_ef_construction=ann_ef_construction,
            ann_ef_search=ann_ef_search,
            docs_by_id=docs_by_id,
            files=files,
            scan_stats=scan_stats,
            total_ms=(time.perf_counter() - t_total0) * 1000.0,
        )
        _write_json(real_index_dir / "meta.json", meta)

        logger.info(
            "Index up-to-date (config_tag=%s) files_changed=%d docs=%d total_ms=%.2f",
            meta["config_tag"],
            changed_files,
            len(docs_by_id),
            (time.perf_counter() - t_total0) * 1000.0,
        )
        return real_index_dir / "manifest.json"

    t_corpus0 = time.perf_counter()
    _save_corpus(real_index_dir, docs_by_id)
    t_corpus_ms = (time.perf_counter() - t_corpus0) * 1000.0

    docs = [docs_by_id[k] for k in sorted(docs_by_id.keys())]
    doc_ids = [d.doc_id for d in docs]
    texts = [d.text for d in docs]
    dense_texts = [
        _build_dense_text(
            d.text,
            dense_input_mode=dense_input_mode,
            keyframe_path=str((d.extra or {}).get("keyframe_path") or ""),
        )
        for d in docs
    ]

    t_bm250 = time.perf_counter()
    tokenized = [_tokenize(t, lang=language, lemmatize=lemmatize, normalize=normalize_text) for t in texts]
    bm25 = BM25Index(tokenized, k1=bm25_k1, b=bm25_b)
    (real_index_dir / "bm25.pkl").write_bytes(pickle.dumps(bm25))
    t_bm25_ms = (time.perf_counter() - t_bm250) * 1000.0

    import numpy as np

    t_emb0 = time.perf_counter()

    need_full_reembed = model_changed or dense_mode_changed or (old_emb is None) or (len(old_doc_ids) == 0)
    n_old = int(len(old_doc_ids))
    n_total = int(len(doc_ids))

    embeddings_new = 0
    embeddings_reused = 0
    embeddings_updated = 0

    embed_cache = EmbeddingCache(Path(cache_dir), model_name) if (cache_dir and use_embed_cache) else None

    def _vec_from_cache(entry: Any) -> Optional[Any]:
        try:
            dim, dtype, blob = entry
            dt = np.float16 if str(dtype) == "float16" else np.float32
            arr = np.frombuffer(blob, dtype=dt)
            if int(dim) and int(dim) != int(arr.shape[0]):
                return None
            return arr.astype(np.float32, copy=False)
        except Exception:
            return None

    embedder_ref: Any = None

    def _ensure_embedder():
        nonlocal embedder_ref
        if embedder_ref is None:
            embedder_ref = build_text_embedder(
                model_name=model_name,
                device=device,
                query_prefix=query_prefix,
                passage_prefix=passage_prefix,
                backend=embedding_backend,
                fallback_model_name=fallback_model_name,
            )
        return embedder_ref

    if need_full_reembed:
        hashes = [_hash_text(t) for t in dense_texts] if embed_cache else []
        cached_map: Dict[str, Any] = embed_cache.get_many(hashes) if embed_cache else {}
        embedder = _ensure_embedder()
        rows: List[Any] = []
        to_encode: List[str] = []
        to_encode_idx: List[int] = []
        for i, h in enumerate(hashes if embed_cache else []):
            cached = cached_map.get(h)
            vec = _vec_from_cache(cached) if cached else None
            if vec is not None:
                rows.append(vec)
                embeddings_reused += 1
            else:
                to_encode.append(dense_texts[i])
                to_encode_idx.append(i)
                rows.append(None)

        if embed_cache:
            if to_encode:
                new_emb = embedder.encode_passages(to_encode, batch_size=batch_size)
                new_arr = np.asarray(new_emb, dtype=np.float32)
                cache_rows: Dict[str, Any] = {}
                for j, idx in enumerate(to_encode_idx):
                    rows[idx] = new_arr[j]
                    embeddings_new += 1
                    cache_rows[hashes[idx]] = (int(new_arr.shape[1]), "float16", new_arr[j].astype(np.float16).tobytes())
                embed_cache.put_many(cache_rows)
        else:
            emb = embedder.encode_passages(dense_texts, batch_size=batch_size)
            rows = list(np.asarray(emb, dtype=np.float32))
            embeddings_new = int(n_total)

        emb_arr = np.stack(rows, axis=0).astype(np.float32, copy=False) if rows else np.zeros((0, 0), dtype=np.float32)
    else:
        old_pos = {doc_id: i for i, doc_id in enumerate(old_doc_ids)}

        to_encode_ids: List[str] = []
        to_encode_texts: List[str] = []
        to_encode_hashes: List[str] = []
        for did in doc_ids:
            if did not in old_pos or did in changed_doc_ids:
                to_encode_ids.append(did)
                to_encode_texts.append(
                    _build_dense_text(
                        docs_by_id[did].text,
                        dense_input_mode=dense_input_mode,
                        keyframe_path=str(((docs_by_id[did].extra or {}).get("keyframe_path") or "")),
                    )
                )
                if embed_cache:
                    to_encode_hashes.append(_hash_text(to_encode_texts[-1]))

        new_map: Dict[str, Any] = {}
        if to_encode_ids:
            if not embed_cache:
                embedder = _ensure_embedder()
                new_emb = embedder.encode_passages(to_encode_texts, batch_size=batch_size)
                new_arr = np.asarray(new_emb, dtype=np.float32)
                new_map = {did: vec for did, vec in zip(to_encode_ids, new_arr)}
            else:
                cached_map = embed_cache.get_many(to_encode_hashes)
                to_encode_real: List[str] = []
                to_encode_real_ids: List[str] = []
                to_encode_real_hashes: List[str] = []

                for did, txt, h in zip(to_encode_ids, to_encode_texts, to_encode_hashes):
                    if h in cached_map:
                        vec = _vec_from_cache(cached_map[h])
                        if vec is not None:
                            new_map[did] = vec
                            embeddings_reused += 1
                            continue
                    to_encode_real.append(txt)
                    to_encode_real_ids.append(did)
                    to_encode_real_hashes.append(h)

                if to_encode_real:
                    embedder = _ensure_embedder()
                    new_emb = embedder.encode_passages(to_encode_real, batch_size=batch_size)
                    new_arr = np.asarray(new_emb, dtype=np.float32)
                    cache_rows: Dict[str, Any] = {}
                    for did, h, vec in zip(to_encode_real_ids, to_encode_real_hashes, new_arr):
                        new_map[did] = vec
                        cache_rows[h] = (int(vec.shape[0]), "float16", vec.astype(np.float16).tobytes())
                    if cache_rows:
                        embed_cache.put_many(cache_rows)

        if old_emb is not None and int(getattr(old_emb, "shape", [0, 0])[0]) > 0:
            dim = int(old_emb.shape[1])
        elif new_map:
            dim = int(np.asarray(next(iter(new_map.values()))).shape[0])
        else:
            dim = 0

        rows: List[Any] = []
        for did in doc_ids:
            if did in new_map:
                rows.append(new_map[did])
                if did in old_pos:
                    embeddings_updated += 1
                else:
                    embeddings_new += 1
            else:
                if did in old_pos:
                    rows.append(old_emb[old_pos[did]])
                    embeddings_reused += 1
                else:
                    rows.append(np.zeros((dim,), dtype=np.float32))
                    embeddings_new += 1

        emb_arr = np.stack(rows, axis=0).astype(np.float32, copy=False) if rows else np.zeros((0, dim), dtype=np.float32)

    if embedder_ref is not None and getattr(embedder_ref, "model_name", None):
        model_name = str(getattr(embedder_ref, "model_name"))
        manifest["model_name"] = model_name

    if embed_store_dtype == "float16":
        emb_arr = emb_arr.astype(np.float16)

    dense_valid = _compute_dense_valid_mask(emb_arr)

    t_emb_ms = (time.perf_counter() - t_emb0) * 1000.0

    t_write0 = time.perf_counter()
    _write_json(real_index_dir / "doc_ids.json", doc_ids)
    np.save(real_index_dir / "embeddings.npy", emb_arr)
    np.save(real_index_dir / "dense_valid.npy", dense_valid.astype(np.bool_, copy=False))
    ann_meta = save_ann_bundle(
        real_index_dir,
        embeddings=np.asarray(emb_arr, dtype=np.float32),
        dense_valid=dense_valid,
        backend=ann_backend,
        index_type=ann_index_type,
        hnsw_m=ann_hnsw_m,
        ef_construction=ann_ef_construction,
        ef_search=ann_ef_search,
    )
    t_write_ms = (time.perf_counter() - t_write0) * 1000.0

    embed_dim = int(emb_arr.shape[1]) if int(emb_arr.shape[0]) else 0
    dense_valid_n = int(np.count_nonzero(dense_valid)) if hasattr(dense_valid, "shape") else 0

    meta = {
        "index_schema_version": 5,
        "config_fingerprint": (str(config_fingerprint) if config_fingerprint else None),
        "config_tag": config_tag_from_fingerprint(config_fingerprint),
        "model_name": model_name,
        "embedding_backend": embedding_backend,
        "num_docs": int(len(docs)),
        "bm25": {"k1": float(bm25_k1), "b": float(bm25_b)},
        "embed_dim": embed_dim,
        "embed_store_dtype": embed_store_dtype,
        "layout": manifest.get("layout"),
        "doc_id_scheme": manifest.get("doc_id_scheme"),
        "language": str(language),
        "variant": (str(variant) if variant else None),
        "query_prefix": str(query_prefix),
        "passage_prefix": str(passage_prefix),
        "normalize_text": bool(normalize_text),
        "lemmatize": bool(lemmatize),
        "dense_input_mode": dense_input_mode,
        "ann_backend": str((ann_meta or {}).get("backend", ann_backend)),
        "ann_index_type": str((ann_meta or {}).get("index_type", ann_index_type)),
        "ann_hnsw_m": int((ann_meta or {}).get("hnsw_m", ann_hnsw_m)),
        "ann_ef_construction": int((ann_meta or {}).get("ef_construction", ann_ef_construction)),
        "ann_ef_search": int((ann_meta or {}).get("ef_search", ann_ef_search)),
        "has_dense_valid": True,
        "dense_valid_count": dense_valid_n,
        "ann": ann_meta,
        "runtime": {
            "timings_ms": {
                "scan": float(t_scan_ms),
                "save_corpus": float(t_corpus_ms),
                "bm25_build": float(t_bm25_ms),
                "embeddings_build": float(t_emb_ms),
                "write_outputs": float(t_write_ms),
                "total": float((time.perf_counter() - t_total0) * 1000.0),
            },
            "counts": {
                "files_total": int(len(files)),
                "files_changed": int(changed_files),
                "files_unchanged": int(unchanged_files),
                "parse_errors": int(parse_errors),
                "docs_total": int(len(docs)),
                "docs_changed": int(len(changed_doc_ids)),
                "docs_added_or_updated": int(docs_added_or_updated),
                "docs_deleted": int(docs_deleted),
                "bm25_changed": bool(bm25_changed),
                "model_changed": bool(model_changed),
                "dense_mode_changed": bool(dense_mode_changed),
                "embeddings_total": int(len(docs)),
                "embeddings_new": int(embeddings_new),
                "embeddings_updated": int(embeddings_updated),
                "embeddings_reused": int(embeddings_reused),
                "old_embeddings_total": int(n_old),
                "dense_valid_count": int(dense_valid_n),
            },
            "notes": {
                "incremental": bool(not need_full_reembed),
                "need_full_reembed": bool(need_full_reembed),
            },
        },
    }
    _write_json(real_index_dir / "meta.json", meta)
    _save_manifest(real_index_dir, manifest)

    logger.info(
        "Index built/updated (config_tag=%s) docs=%d changed_files=%d dense_valid=%d emb_new=%d emb_upd=%d emb_reused=%d total_ms=%.2f",
        meta["config_tag"],
        meta["num_docs"],
        changed_files,
        dense_valid_n,
        embeddings_new,
        embeddings_updated,
        embeddings_reused,
        meta["runtime"]["timings_ms"]["total"],
    )

    return real_index_dir / "meta.json"
