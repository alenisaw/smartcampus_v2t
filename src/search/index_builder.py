# search/index_builder.py
"""
Fast hybrid index builder for SmartCampus V2T.

Supports:
- Incremental updates (per annotations.json fingerprint)
- BM25 + E5 embeddings
- Efficient storage (embeddings.npy, mmap loading)
- Index versioning by config_fingerprint:
    base_index_dir/<config_tag>/

Writes per index folder:
- manifest.json
- corpus.pkl
- doc_ids.json
- bm25.pkl
- embeddings.npy
- dense_valid.npy         (bool mask: embeddings row is non-zero / valid)
- meta.json (includes runtime metrics for UI)

Layout expected:
runs_root/<video_id>/<run_id>/annotations.json
"""

from __future__ import annotations

import json
import logging
import math
import pickle
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_DATA_DIR = Path("data")
DEFAULT_RUNS_ROOT = DEFAULT_DATA_DIR / "runs"
DEFAULT_INDEX_DIR = DEFAULT_DATA_DIR / "indexes"

MODEL_NAME_DEFAULT = "intfloat/multilingual-e5-base"

_WORD_RE = re.compile(r"\w+", flags=re.UNICODE)

logger = logging.getLogger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in _WORD_RE.findall(text or "") if len(t) > 1]


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _iter_annotation_files(runs_root: Path) -> List[Path]:
    runs_root = Path(runs_root)
    if not runs_root.exists():
        return []
    return sorted(runs_root.glob("*/*/annotations.json"))


def _guess_run_id_from_path(p: Path) -> str:
    try:
        return p.parent.name
    except Exception:
        return "run_unknown"


def _guess_video_id_from_path(p: Path) -> str:
    try:
        return p.parent.parent.name
    except Exception:
        return "unknown"


def _file_fingerprint(p: Path) -> str:
    st = p.stat()
    return f"{st.st_size}:{st.st_mtime_ns}"


def _stable_doc_id(video_id: str, run_id: str, i: int) -> str:
    return f"{video_id}/{run_id}/seg_{i:04d}"


def _normalize_embed_store_dtype(embed_store_dtype: str) -> str:
    d = (embed_store_dtype or "float32").strip().lower()
    if d in {"fp16", "float16", "f16"}:
        return "float16"
    return "float32"


def _sanitize_tag(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return "default"
    s = re.sub(r"[^a-zA-Z0-9_.-]+", "_", s)
    s = s.strip("._-")
    return s or "default"


def config_tag_from_fingerprint(config_fingerprint: Optional[str]) -> str:
    if not config_fingerprint:
        return "default"
    fp = str(config_fingerprint).strip()
    if not fp:
        return "default"
    return _sanitize_tag(fp[:24])


def resolve_index_dir(base_index_dir: Path, config_fingerprint: Optional[str] = None) -> Path:
    base = Path(base_index_dir)
    return base / config_tag_from_fingerprint(config_fingerprint)


def search_config_fingerprint(cfg: Any) -> str:
    try:
        s = cfg.search
        payload = {
            "embed_model_name": str(getattr(s, "embed_model_name", MODEL_NAME_DEFAULT)),
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


@dataclass
class Doc:
    doc_id: str
    video_id: str
    run_id: str
    start_sec: float
    end_sec: float
    text: str
    extra: Optional[Dict[str, Any]] = None
    source_path: Optional[str] = None


class BM25Index:
    def __init__(self, tokenized_corpus: List[List[str]], k1: float = 1.6, b: float = 0.75):
        self.k1 = float(k1)
        self.b = float(b)

        self.corpus = tokenized_corpus
        self.N = len(tokenized_corpus)
        self.doc_lens = [len(d) for d in tokenized_corpus]
        self.avgdl = (sum(self.doc_lens) / self.N) if self.N else 0.0

        self.df: Dict[str, int] = {}
        for doc in tokenized_corpus:
            for t in set(doc):
                self.df[t] = self.df.get(t, 0) + 1

        self.idf: Dict[str, float] = {}
        for t, df in self.df.items():
            self.idf[t] = math.log(1.0 + (self.N - df + 0.5) / (df + 0.5))

        self.tf: List[Dict[str, int]] = []
        for doc in tokenized_corpus:
            d: Dict[str, int] = {}
            for t in doc:
                d[t] = d.get(t, 0) + 1
            self.tf.append(d)

    def score(self, query_tokens: List[str]) -> List[float]:
        scores = [0.0] * self.N
        if not self.N:
            return scores

        for i in range(self.N):
            dl = self.doc_lens[i]
            denom_const = self.k1 * (1.0 - self.b + self.b * (dl / (self.avgdl + 1e-9)))
            tf_i = self.tf[i]

            s = 0.0
            for q in query_tokens:
                f = tf_i.get(q)
                if not f:
                    continue
                idf = self.idf.get(q, 0.0)
                s += idf * (f * (self.k1 + 1.0)) / (f + denom_const)
            scores[i] = s

        return scores


class E5Embedder:
    def __init__(self, model_name: str = MODEL_NAME_DEFAULT, device: Optional[str] = None):
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "Install `sentence-transformers` to use E5 embeddings: pip install -U sentence-transformers"
            ) from e

        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)

    @staticmethod
    def prep_passage(text: str) -> str:
        return f"passage: {text}"

    @staticmethod
    def prep_query(text: str) -> str:
        return f"query: {text}"

    def encode_passages(self, texts: List[str], batch_size: int = 64):
        return self.model.encode(
            [self.prep_passage(t) for t in texts],
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def encode_query(self, text: str):
        return self.model.encode([self.prep_query(text)], normalize_embeddings=True, show_progress_bar=False)[0]


@dataclass
class HybridIndex:
    docs: List[Doc]
    doc_ids: List[str]
    bm25: BM25Index
    embeddings: Any
    dense_valid: Any
    meta: Dict[str, Any]


def _load_manifest(index_dir: Path) -> Dict[str, Any]:
    p = index_dir / "manifest.json"
    if not p.exists():
        return {
            "version": 4,
            "model_name": MODEL_NAME_DEFAULT,
            "bm25": {"k1": 1.6, "b": 0.75},
            "sources": {},
            "layout": "runs/<video_id>/<run_id>/annotations.json",
            "doc_id_scheme": "stable_by_position",
            "has_dense_valid": True,
        }
    m = _read_json(p)
    if not isinstance(m, dict):
        m = {}

    m.setdefault("version", 4)
    m.setdefault("model_name", MODEL_NAME_DEFAULT)
    m.setdefault("bm25", {"k1": 1.6, "b": 0.75})
    m.setdefault("sources", {})
    m.setdefault("layout", "runs/<video_id>/<run_id>/annotations.json")
    m.setdefault("doc_id_scheme", "stable_by_position")
    m.setdefault("has_dense_valid", True)
    return m


def _save_manifest(index_dir: Path, manifest: Dict[str, Any]) -> None:
    _write_json(index_dir / "manifest.json", manifest)


def _corpus_pkl_path(index_dir: Path) -> Path:
    return index_dir / "corpus.pkl"


def _load_corpus(index_dir: Path) -> Dict[str, Doc]:
    pkl = _corpus_pkl_path(index_dir)
    if pkl.exists():
        obj = pickle.loads(pkl.read_bytes())
        if isinstance(obj, dict):
            return obj
        return {}
    return {}


def _save_corpus(index_dir: Path, docs_by_id: Dict[str, Doc]) -> None:
    pkl = _corpus_pkl_path(index_dir)
    pkl.parent.mkdir(parents=True, exist_ok=True)
    pkl.write_bytes(pickle.dumps(docs_by_id))


def _load_prev_embeddings(index_dir: Path) -> Tuple[List[str], Optional[Any]]:
    doc_ids_path = index_dir / "doc_ids.json"
    emb_path = index_dir / "embeddings.npy"
    if not doc_ids_path.exists() or not emb_path.exists():
        return [], None

    import numpy as np

    try:
        old_doc_ids = _read_json(doc_ids_path)
        old_emb = np.load(emb_path, mmap_mode="r")
        if not isinstance(old_doc_ids, list):
            return [], None
        if int(old_emb.shape[0]) != len(old_doc_ids):
            return [], None
        return [str(x) for x in old_doc_ids], old_emb
    except Exception:
        return [], None


def _compute_dense_valid_mask(emb_arr: Any) -> Any:
    import numpy as np

    if emb_arr is None or not hasattr(emb_arr, "shape"):
        return np.zeros((0,), dtype=bool)
    n = int(emb_arr.shape[0])
    if n == 0:
        return np.zeros((0,), dtype=bool)

    x = emb_arr
    if x.dtype != np.float32:
        x = x.astype(np.float32, copy=False)

    # fast row norm^2
    norm2 = np.einsum("ij,ij->i", x, x)
    return norm2 > 1e-10


def build_or_update_index(
    runs_root: Path = DEFAULT_RUNS_ROOT,
    index_dir: Path = DEFAULT_INDEX_DIR,
    model_name: str = MODEL_NAME_DEFAULT,
    device: Optional[str] = None,
    bm25_k1: float = 1.6,
    bm25_b: float = 0.75,
    batch_size: int = 64,
    config_fingerprint: Optional[str] = None,
    embed_store_dtype: str = "float32",
) -> Path:

    t_total0 = time.perf_counter()

    base_index_dir = Path(index_dir)
    real_index_dir = resolve_index_dir(base_index_dir, config_fingerprint)
    real_index_dir.mkdir(parents=True, exist_ok=True)

    embed_store_dtype = _normalize_embed_store_dtype(embed_store_dtype)
    manifest = _load_manifest(real_index_dir)

    old_bm25 = (manifest.get("bm25") or {})
    bm25_changed = (float(old_bm25.get("k1", 1.6)) != float(bm25_k1)) or (float(old_bm25.get("b", 0.75)) != float(bm25_b))
    model_changed = manifest.get("model_name") != model_name

    if model_changed:
        manifest["model_name"] = model_name
        manifest["sources"] = {}
        docs_by_id: Dict[str, Doc] = {}
        old_doc_ids: List[str] = []
        old_emb = None
    else:
        docs_by_id = _load_corpus(real_index_dir)
        old_doc_ids, old_emb = _load_prev_embeddings(real_index_dir)

    manifest["bm25"] = {"k1": float(bm25_k1), "b": float(bm25_b)}
    manifest.setdefault("sources", {})
    manifest["layout"] = "runs/<video_id>/<run_id>/annotations.json"
    manifest["doc_id_scheme"] = "stable_by_position"
    manifest["version"] = int(manifest.get("version", 4) or 4)
    manifest["has_dense_valid"] = True

    files = _iter_annotation_files(Path(runs_root))
    if not files:
        raise RuntimeError(f"No annotations found under: {runs_root}")

    t_scan0 = time.perf_counter()

    changed_any = False
    changed_doc_ids: set[str] = set()
    changed_files = 0
    unchanged_files = 0
    parse_errors = 0
    docs_added_or_updated = 0
    docs_deleted = 0

    for f in files:
        f = f.resolve()
        f_key = str(f)
        fp = _file_fingerprint(f)

        prev = (manifest.get("sources") or {}).get(f_key)
        prev_fp = prev.get("fingerprint") if isinstance(prev, dict) else None

        if prev_fp == fp:
            unchanged_files += 1
            continue

        changed_files += 1
        run_id = _guess_run_id_from_path(f)
        video_id = _guess_video_id_from_path(f)

        try:
            data = _read_json(f)
        except Exception:
            parse_errors += 1
            continue
        if not isinstance(data, list):
            parse_errors += 1
            continue

        prev_num_docs = int(prev.get("num_docs", 0)) if isinstance(prev, dict) else 0

        new_num_docs = 0
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                continue
            try:
                start = float(item["start_sec"])
                end = float(item["end_sec"])
                text = str(item.get("description", "") or "")
            except Exception:
                continue

            doc_id = _stable_doc_id(video_id, run_id, i)
            docs_by_id[doc_id] = Doc(
                doc_id=doc_id,
                video_id=video_id,
                run_id=run_id,
                start_sec=start,
                end_sec=end,
                text=text,
                extra=item.get("extra"),
                source_path=f_key,
            )
            changed_doc_ids.add(doc_id)
            docs_added_or_updated += 1
            new_num_docs += 1

        if prev_num_docs > new_num_docs:
            for j in range(new_num_docs, prev_num_docs):
                did = _stable_doc_id(video_id, run_id, j)
                if did in docs_by_id:
                    docs_by_id.pop(did, None)
                    docs_deleted += 1

        manifest["sources"][f_key] = {
            "fingerprint": fp,
            "video_id": video_id,
            "run_id": run_id,
            "num_docs": int(new_num_docs),
        }
        changed_any = True

    t_scan_ms = (time.perf_counter() - t_scan0) * 1000.0

    if not changed_any and not bm25_changed and not model_changed:
        _save_manifest(real_index_dir, manifest)
        meta = {
            "index_schema_version": 4,
            "config_fingerprint": (str(config_fingerprint) if config_fingerprint else None),
            "config_tag": config_tag_from_fingerprint(config_fingerprint),
            "model_name": manifest.get("model_name"),
            "num_docs": int(len(docs_by_id)),
            "bm25": {"k1": float(bm25_k1), "b": float(bm25_b)},
            "embed_store_dtype": embed_store_dtype,
            "layout": manifest.get("layout"),
            "doc_id_scheme": manifest.get("doc_id_scheme"),
            "has_dense_valid": True,
            "runtime": {
                "note": "No changes detected. Index not rebuilt.",
                "timings_ms": {"scan": float(t_scan_ms), "total": float((time.perf_counter() - t_total0) * 1000.0)},
                "counts": {
                    "files_total": int(len(files)),
                    "files_changed": int(changed_files),
                    "files_unchanged": int(unchanged_files),
                    "parse_errors": int(parse_errors),
                    "docs_added_or_updated": int(docs_added_or_updated),
                    "docs_deleted": int(docs_deleted),
                },
            },
        }
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

    t_bm250 = time.perf_counter()
    tokenized = [_tokenize(t) for t in texts]
    bm25 = BM25Index(tokenized, k1=bm25_k1, b=bm25_b)
    (real_index_dir / "bm25.pkl").write_bytes(pickle.dumps(bm25))
    t_bm25_ms = (time.perf_counter() - t_bm250) * 1000.0

    import numpy as np

    t_emb0 = time.perf_counter()

    need_full_reembed = model_changed or (old_emb is None) or (len(old_doc_ids) == 0)
    n_old = int(len(old_doc_ids))
    n_total = int(len(doc_ids))

    embeddings_new = 0
    embeddings_reused = 0
    embeddings_updated = 0

    if need_full_reembed:
        embedder = E5Embedder(model_name=model_name, device=device)
        emb = embedder.encode_passages(texts, batch_size=batch_size)
        emb_arr = np.asarray(emb, dtype=np.float32)
        embeddings_new = int(n_total)
    else:
        old_pos = {doc_id: i for i, doc_id in enumerate(old_doc_ids)}

        to_encode_ids: List[str] = []
        to_encode_texts: List[str] = []
        for did in doc_ids:
            if did not in old_pos or did in changed_doc_ids:
                to_encode_ids.append(did)
                to_encode_texts.append(docs_by_id[did].text)

        new_map: Dict[str, Any] = {}
        if to_encode_ids:
            embedder = E5Embedder(model_name=model_name, device=device)
            new_emb = embedder.encode_passages(to_encode_texts, batch_size=batch_size)
            new_arr = np.asarray(new_emb, dtype=np.float32)
            new_map = {did: vec for did, vec in zip(to_encode_ids, new_arr)}

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

    if embed_store_dtype == "float16":
        emb_arr = emb_arr.astype(np.float16)

    dense_valid = _compute_dense_valid_mask(emb_arr)

    t_emb_ms = (time.perf_counter() - t_emb0) * 1000.0

    t_write0 = time.perf_counter()
    _write_json(real_index_dir / "doc_ids.json", doc_ids)
    np.save(real_index_dir / "embeddings.npy", emb_arr)
    np.save(real_index_dir / "dense_valid.npy", dense_valid.astype(np.bool_, copy=False))
    t_write_ms = (time.perf_counter() - t_write0) * 1000.0

    embed_dim = int(emb_arr.shape[1]) if int(emb_arr.shape[0]) else 0
    dense_valid_n = int(np.count_nonzero(dense_valid)) if hasattr(dense_valid, "shape") else 0

    meta = {
        "index_schema_version": 4,
        "config_fingerprint": (str(config_fingerprint) if config_fingerprint else None),
        "config_tag": config_tag_from_fingerprint(config_fingerprint),
        "model_name": model_name,
        "num_docs": int(len(docs)),
        "bm25": {"k1": float(bm25_k1), "b": float(bm25_b)},
        "embed_dim": embed_dim,
        "embed_store_dtype": embed_store_dtype,
        "layout": "runs/<video_id>/<run_id>/annotations.json",
        "doc_id_scheme": "stable_by_position",
        "has_dense_valid": True,
        "dense_valid_count": dense_valid_n,
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


def load_index(index_dir: Path = DEFAULT_INDEX_DIR) -> HybridIndex:

    index_dir = Path(index_dir)

    manifest = _load_manifest(index_dir)

    meta: Dict[str, Any] = {}
    meta_path = index_dir / "meta.json"
    if meta_path.exists():
        meta_obj = _read_json(meta_path)
        if isinstance(meta_obj, dict):
            meta = meta_obj

    docs_by_id = _load_corpus(index_dir)

    doc_ids = _read_json(index_dir / "doc_ids.json")
    if not isinstance(doc_ids, list):
        doc_ids = []
    doc_ids = [str(x) for x in doc_ids]
    docs = [docs_by_id[doc_id] for doc_id in doc_ids if doc_id in docs_by_id]

    bm25 = pickle.loads((index_dir / "bm25.pkl").read_bytes())

    import numpy as np

    embeddings = np.load(index_dir / "embeddings.npy", mmap_mode="r")

    dense_valid_path = index_dir / "dense_valid.npy"
    if dense_valid_path.exists():
        dense_valid = np.load(dense_valid_path, mmap_mode="r")
        if dense_valid.dtype != np.bool_:
            dense_valid = dense_valid.astype(np.bool_, copy=False)
        if int(getattr(dense_valid, "shape", [0])[0]) != int(getattr(embeddings, "shape", [0, 0])[0]):
            dense_valid = np.ones((int(embeddings.shape[0]),), dtype=np.bool_)
    else:
        dense_valid = np.ones((int(embeddings.shape[0]),), dtype=np.bool_)

    return HybridIndex(
        docs=docs,
        doc_ids=doc_ids,
        bm25=bm25,
        embeddings=embeddings,
        dense_valid=dense_valid,
        meta={**manifest, **meta},
    )
