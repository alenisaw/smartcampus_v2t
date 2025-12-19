# search/index_builder.py

import hashlib
import json
import math
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_DATA_DIR = Path("data")
DEFAULT_ANN_ROOT = DEFAULT_DATA_DIR / "annotations"
DEFAULT_INDEX_DIR = DEFAULT_DATA_DIR / "indexes"

MODEL_NAME_DEFAULT = "intfloat/multilingual-e5-base"

_WORD_RE = re.compile(r"\w+", flags=re.UNICODE)


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in _WORD_RE.findall(text) if len(t) > 1]


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _iter_annotation_files(ann_root: Path) -> List[Path]:
    if not ann_root.exists():
        return []
    return sorted(ann_root.glob("**/run_*/annotations.json"))


def _guess_run_id_from_path(p: Path) -> str:
    for part in p.parts[::-1]:
        if part.startswith("run_"):
            return part
    return "run_unknown"


def _guess_video_id_from_path(p: Path) -> str:
    # .../annotations/<video_id>/run_003/annotations.json
    try:
        return p.parents[1].name
    except Exception:
        return "unknown"


def _file_fingerprint(p: Path) -> str:

    st = p.stat()
    return f"{st.st_size}:{st.st_mtime_ns}"


def _stable_doc_id(video_id: str, run_id: str, i: int, start: float, end: float, text: str) -> str:

    base = f"{video_id}|{run_id}|{i}|{start:.3f}|{end:.3f}|{text}"
    h = hashlib.sha1(base.encode("utf-8")).hexdigest()[:16]
    return f"{video_id}/{run_id}/seg_{i:04d}_{h}"


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
    """
    E5 expects role prefixes for best retrieval quality:
      - query: ...
      - passage: ...
    """

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
            show_progress_bar=True,
        )

    def encode_query(self, text: str):
        return self.model.encode([self.prep_query(text)], normalize_embeddings=True, show_progress_bar=False)[0]


@dataclass
class HybridIndex:
    docs: List[Doc]
    doc_ids: List[str]
    bm25: BM25Index
    embeddings  : Any  # numpy ndarray
    meta: Dict[str, Any]


def _load_manifest(index_dir: Path) -> Dict[str, Any]:
    p = index_dir / "manifest.json"
    if not p.exists():
        return {
            "version": 1,
            "model_name": MODEL_NAME_DEFAULT,
            "bm25": {"k1": 1.6, "b": 0.75},
            "sources": {},  # path -> {fingerprint, video_id, run_id, num_docs}
        }
    return _read_json(p)


def _save_manifest(index_dir: Path, manifest: Dict[str, Any]) -> None:
    _write_json(index_dir / "manifest.json", manifest)


def _load_corpus(index_dir: Path) -> Dict[str, Doc]:

    corpus_path = index_dir / "corpus.jsonl"
    if not corpus_path.exists():
        return {}

    out: Dict[str, Doc] = {}
    for line in corpus_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        d = Doc(
            doc_id=obj["doc_id"],
            video_id=obj["video_id"],
            run_id=obj["run_id"],
            start_sec=float(obj["start_sec"]),
            end_sec=float(obj["end_sec"]),
            text=str(obj["text"]),
            extra=obj.get("extra"),
            source_path=obj.get("source_path"),
        )
        out[d.doc_id] = d
    return out


def _save_corpus(index_dir: Path, docs_by_id: Dict[str, Doc]) -> None:
    corpus_path = index_dir / "corpus.jsonl"
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    for doc_id in sorted(docs_by_id.keys()):
        d = docs_by_id[doc_id]
        lines.append(
            json.dumps(
                {
                    "doc_id": d.doc_id,
                    "video_id": d.video_id,
                    "run_id": d.run_id,
                    "start_sec": d.start_sec,
                    "end_sec": d.end_sec,
                    "text": d.text,
                    "extra": d.extra,
                    "source_path": d.source_path,
                },
                ensure_ascii=False,
            )
        )
    corpus_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def build_or_update_index(
    ann_root: Path = DEFAULT_ANN_ROOT,
    index_dir: Path = DEFAULT_INDEX_DIR,
    model_name: str = MODEL_NAME_DEFAULT,
    device: Optional[str] = None,
    bm25_k1: float = 1.6,
    bm25_b: float = 0.75,
    batch_size: int = 64,
) -> Path:

    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    manifest = _load_manifest(index_dir)


    if manifest.get("model_name") != model_name:
        manifest["model_name"] = model_name
        manifest["sources"] = {}

    manifest["bm25"] = {"k1": float(bm25_k1), "b": float(bm25_b)}

    docs_by_id = _load_corpus(index_dir)

    files = _iter_annotation_files(Path(ann_root))
    if not files:
        raise RuntimeError(f"No annotations found under: {ann_root}")

    changed_any = False

    for f in files:
        f = f.resolve()
        f_key = str(f)
        fp = _file_fingerprint(f)
        prev = manifest["sources"].get(f_key)

        if prev and prev.get("fingerprint") == fp:
            continue  # unchanged


        run_id = _guess_run_id_from_path(f)
        video_id = _guess_video_id_from_path(f)
        data = _read_json(f)
        if not isinstance(data, list):
            continue


        if prev and prev.get("doc_ids"):
            for old_id in prev["doc_ids"]:
                docs_by_id.pop(old_id, None)

        new_doc_ids: List[str] = []
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                continue
            try:
                start = float(item["start_sec"])
                end = float(item["end_sec"])
                text = str(item["description"])
            except Exception:
                continue

            doc_id = _stable_doc_id(video_id, run_id, i, start, end, text)
            new_doc_ids.append(doc_id)
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

        manifest["sources"][f_key] = {
            "fingerprint": fp,
            "video_id": video_id,
            "run_id": run_id,
            "num_docs": len(new_doc_ids),
            "doc_ids": new_doc_ids,
        }
        changed_any = True

    if not changed_any:
        _save_manifest(index_dir, manifest)
        return index_dir / "manifest.json"


    _save_corpus(index_dir, docs_by_id)


    docs = [docs_by_id[k] for k in sorted(docs_by_id.keys())]
    texts = [d.text for d in docs]
    tokenized = [_tokenize(t) for t in texts]

    bm25 = BM25Index(tokenized, k1=bm25_k1, b=bm25_b)

    embedder = E5Embedder(model_name=model_name, device=device)
    embeddings = embedder.encode_passages(texts, batch_size=batch_size)


    (index_dir / "bm25.pkl").write_bytes(pickle.dumps(bm25))
    _write_json(index_dir / "doc_ids.json", [d.doc_id for d in docs])


    try:
        import numpy as np
    except Exception as e:
        raise RuntimeError("Install numpy to store embeddings: pip install -U numpy") from e
    np.save(index_dir / "embeddings.npy", np.asarray(embeddings, dtype=np.float32))


    meta = {
        "model_name": model_name,
        "num_docs": len(docs),
        "bm25": {"k1": float(bm25_k1), "b": float(bm25_b)},
        "embed_dim": int(np.asarray(embeddings).shape[1]) if len(docs) else 0,
        "note": "Docs are tied to specific runs (run_id). Incremental updates are per annotations.json fingerprint.",
    }
    _write_json(index_dir / "meta.json", meta)
    _save_manifest(index_dir, manifest)

    return index_dir / "meta.json"


def load_index(index_dir: Path = DEFAULT_INDEX_DIR) -> HybridIndex:
    index_dir = Path(index_dir)

    manifest = _load_manifest(index_dir)
    meta = {}
    meta_path = index_dir / "meta.json"
    if meta_path.exists():
        meta = _read_json(meta_path)

    docs_by_id = _load_corpus(index_dir)
    doc_ids = _read_json(index_dir / "doc_ids.json")
    docs = [docs_by_id[doc_id] for doc_id in doc_ids if doc_id in docs_by_id]

    bm25 = pickle.loads((index_dir / "bm25.pkl").read_bytes())

    import numpy as np
    embeddings = np.load(index_dir / "embeddings.npy")

    return HybridIndex(
        docs=docs,
        doc_ids=doc_ids,
        bm25=bm25,
        embeddings=embeddings,
        meta={**manifest, **meta},
    )
