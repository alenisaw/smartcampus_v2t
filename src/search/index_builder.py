# search/index_builder.py
"""
Fast hybrid index builder for SmartCampus V2T.
Supports incremental index updates, BM25 + E5 embeddings, and efficient storage.

Layout expected:
runs_root/<video_id>/<run_id>/annotations.json

Notes (incremental correctness):
- doc_id is stable by (video_id, run_id, segment_index) and does NOT depend on text.
- If annotations.json changes, we:
  - overwrite docs seg_0000..seg_(n-1)
  - delete tail seg_n..seg_(prev_n-1) if file got shorter
  - re-embed ONLY docs that are new or belong to changed files

Storage notes:
- embeddings.npy can be saved as float32 (default) or float16 to reduce disk usage.
- load_index() uses mmap_mode="r" for fast, low-RAM loading.
"""

import json
import math
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_DATA_DIR = Path("data")
DEFAULT_RUNS_ROOT = DEFAULT_DATA_DIR / "runs"
DEFAULT_INDEX_DIR = DEFAULT_DATA_DIR / "indexes"

MODEL_NAME_DEFAULT = "intfloat/multilingual-e5-base"

_WORD_RE = re.compile(r"\w+", flags=re.UNICODE)


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
    # stable id by position; DOES NOT depend on text
    return f"{video_id}/{run_id}/seg_{i:04d}"


def _normalize_embed_store_dtype(embed_store_dtype: str) -> str:
    d = (embed_store_dtype or "float32").strip().lower()
    if d in {"fp16", "float16", "f16"}:
        return "float16"
    return "float32"


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
    meta: Dict[str, Any]


def _load_manifest(index_dir: Path) -> Dict[str, Any]:
    p = index_dir / "manifest.json"
    if not p.exists():
        return {
            "version": 3,
            "model_name": MODEL_NAME_DEFAULT,
            "bm25": {"k1": 1.6, "b": 0.75},
            "sources": {},
            "layout": "runs/<video_id>/<run_id>/annotations.json",
            "doc_id_scheme": "stable_by_position",
        }
    m = _read_json(p)
    if not isinstance(m, dict):
        m = {}


    m.setdefault("version", 3)
    m.setdefault("sources", {})
    m.setdefault("layout", "runs/<video_id>/<run_id>/annotations.json")
    m.setdefault("doc_id_scheme", "stable_by_position")

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


def build_or_update_index(
    runs_root: Path = DEFAULT_RUNS_ROOT,
    index_dir: Path = DEFAULT_INDEX_DIR,
    model_name: str = MODEL_NAME_DEFAULT,
    device: Optional[str] = None,
    bm25_k1: float = 1.6,
    bm25_b: float = 0.75,
    batch_size: int = 64,
    config_fingerprint: Optional[str] = None,
    embed_store_dtype: str = "float32",  # "float32" | "float16"
) -> Path:
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    embed_store_dtype = _normalize_embed_store_dtype(embed_store_dtype)

    manifest = _load_manifest(index_dir)

    old_bm25 = (manifest.get("bm25") or {})
    bm25_changed = (float(old_bm25.get("k1", 1.6)) != float(bm25_k1)) or (
        float(old_bm25.get("b", 0.75)) != float(bm25_b)
    )

    model_changed = manifest.get("model_name") != model_name
    if model_changed:
        manifest["model_name"] = model_name
        manifest["sources"] = {}
        docs_by_id: Dict[str, Doc] = {}
        old_doc_ids = []
        old_emb = None
    else:
        docs_by_id = _load_corpus(index_dir)
        old_doc_ids, old_emb = _load_prev_embeddings(index_dir)

    manifest["bm25"] = {"k1": float(bm25_k1), "b": float(bm25_b)}
    manifest.setdefault("sources", {})
    manifest["layout"] = "runs/<video_id>/<run_id>/annotations.json"
    manifest["doc_id_scheme"] = "stable_by_position"
    manifest["version"] = int(manifest.get("version", 3) or 3)

    files = _iter_annotation_files(Path(runs_root))
    if not files:
        raise RuntimeError(f"No annotations found under: {runs_root}")

    changed_any = False
    changed_doc_ids: set[str] = set()

    for f in files:
        f = f.resolve()
        f_key = str(f)
        fp = _file_fingerprint(f)
        prev = (manifest.get("sources") or {}).get(f_key)
        prev_fp = prev.get("fingerprint") if isinstance(prev, dict) else None

        if prev_fp == fp:
            continue

        run_id = _guess_run_id_from_path(f)
        video_id = _guess_video_id_from_path(f)

        try:
            data = _read_json(f)
        except Exception:
            continue
        if not isinstance(data, list):
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
            new_num_docs += 1


        if prev_num_docs > new_num_docs:
            for j in range(new_num_docs, prev_num_docs):
                docs_by_id.pop(_stable_doc_id(video_id, run_id, j), None)

        manifest["sources"][f_key] = {
            "fingerprint": fp,
            "video_id": video_id,
            "run_id": run_id,
            "num_docs": int(new_num_docs),
        }
        changed_any = True

    if not changed_any and not bm25_changed and not model_changed:
        _save_manifest(index_dir, manifest)
        return index_dir / "manifest.json"

    _save_corpus(index_dir, docs_by_id)


    docs = [docs_by_id[k] for k in sorted(docs_by_id.keys())]
    doc_ids = [d.doc_id for d in docs]
    texts = [d.text for d in docs]

    tokenized = [_tokenize(t) for t in texts]
    bm25 = BM25Index(tokenized, k1=bm25_k1, b=bm25_b)
    (index_dir / "bm25.pkl").write_bytes(pickle.dumps(bm25))

    import numpy as np

    need_full_reembed = model_changed or (old_emb is None) or (len(old_doc_ids) == 0)
    if need_full_reembed:
        embedder = E5Embedder(model_name=model_name, device=device)
        emb = embedder.encode_passages(texts, batch_size=batch_size)
        emb_arr = np.asarray(emb, dtype=np.float32)
    else:
        old_pos = {doc_id: i for i, doc_id in enumerate(old_doc_ids)}

        to_encode: List[str] = []
        to_encode_texts: List[str] = []
        for d in doc_ids:
            if (d not in old_pos) or (d in changed_doc_ids):
                to_encode.append(d)
                to_encode_texts.append(docs_by_id[d].text)

        new_map: Dict[str, Any] = {}
        if to_encode:
            embedder = E5Embedder(model_name=model_name, device=device)
            new_emb = embedder.encode_passages(to_encode_texts, batch_size=batch_size)
            new_arr = np.asarray(new_emb, dtype=np.float32)
            new_map = {d: v for d, v in zip(to_encode, new_arr)}

        rows: List[Any] = []
        for d in doc_ids:
            if d in new_map:
                rows.append(new_map[d])
            elif d in old_pos:
                rows.append(old_emb[old_pos[d]])
            else:
                rows.append(new_map.get(d))

        if rows:
            emb_arr = np.stack(rows, axis=0).astype(np.float32)
        else:
            dim = int(old_emb.shape[1]) if old_emb is not None and int(old_emb.shape[0]) else 0
            emb_arr = np.zeros((0, dim), dtype=np.float32)

    if embed_store_dtype == "float16":
        emb_arr = emb_arr.astype(np.float16)

    _write_json(index_dir / "doc_ids.json", doc_ids)
    np.save(index_dir / "embeddings.npy", emb_arr)

    meta = {
        "index_schema_version": 3,
        "config_fingerprint": (str(config_fingerprint) if config_fingerprint else None),
        "model_name": model_name,
        "num_docs": len(docs),
        "bm25": {"k1": float(bm25_k1), "b": float(bm25_b)},
        "embed_dim": int(emb_arr.shape[1]) if int(emb_arr.shape[0]) else 0,
        "embed_store_dtype": embed_store_dtype,
        "layout": "runs/<video_id>/<run_id>/annotations.json",
        "doc_id_scheme": "stable_by_position",
        "note": (
            "Docs are tied to specific runs (run_id). Incremental updates are per annotations.json fingerprint. "
            "Embeddings are updated incrementally for docs that are new or belong to changed files. "
            "doc_id does not depend on text."
        ),
    }
    _write_json(index_dir / "meta.json", meta)
    _save_manifest(index_dir, manifest)

    return index_dir / "meta.json"


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
    # mmap ускоряет загрузку и почти не ест RAM
    embeddings = np.load(index_dir / "embeddings.npy", mmap_mode="r")

    return HybridIndex(
        docs=docs,
        doc_ids=doc_ids,
        bm25=bm25,
        embeddings=embeddings,
        meta={**manifest, **meta},
    )
