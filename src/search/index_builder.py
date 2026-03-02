# src/search/index_builder.py
"""
Fast hybrid index builder for SmartCampus V2T.

Supports:
- Incremental updates (per segments.jsonl.zst fingerprint)
- BM25 + sentence-transformer embeddings
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
videos_root/<video_id>/outputs/segments/<lang>.jsonl.zst
or
videos_root/<video_id>/outputs/variants/<variant>/segments/<lang>.jsonl.zst
"""

from __future__ import annotations

import io
import json
import sqlite3
import logging
import math
import numpy as np
import pickle
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Iterable

try:
    import zstandard as zstd
except Exception:
    zstd = None
DEFAULT_DATA_DIR = Path("data")
DEFAULT_VIDEOS_ROOT = DEFAULT_DATA_DIR / "videos"
DEFAULT_INDEX_DIR = DEFAULT_DATA_DIR / "indexes"

MODEL_NAME_DEFAULT = "BAAI/bge-m3"

_WORD_RE = re.compile(r"\w+", flags=re.UNICODE)

_STOPWORDS_EN = {
    "the","a","an","and","or","is","are","was","were","to","of","in","on","for","with","at","by","from","as","that",
    "this","it","its","be","been","being","into","out","up","down","over","under","not","no","yes","but","so","if",
}
_STOPWORDS_RU = {
    "\u0438","\u0432","\u0432\u043e","\u043d\u0435","\u0447\u0442\u043e","\u043e\u043d","\u043d\u0430","\u044f","\u0441","\u0441\u043e","\u043a\u0430\u043a","\u0430","\u0442\u043e","\u0432\u0441\u0435","\u043e\u043d\u0430","\u0442\u0430\u043a","\u0435\u0433\u043e","\u043d\u043e","\u0434\u0430","\u0442\u044b","\u043a","\u0443",
    "\u0436\u0435","\u0432\u044b","\u0437\u0430","\u0431\u044b","\u043f\u043e","\u0435\u0435","\u043c\u043d\u0435","\u0435\u0441\u0442\u044c","\u043e\u043d\u0438","\u0442\u0443\u0442","\u0433\u0434\u0435","\u043c\u044b","\u0442\u0430\u043c","\u0447\u0442\u043e\u0431\u044b","\u043a\u0442\u043e","\u043a\u043e\u0433\u0434\u0430","\u0438\u0437","\u043e\u0442","\u0434\u043e",
}
_STOPWORDS_KZ = {
    "\u0436\u04d9\u043d\u0435","\u043c\u0435\u043d","\u043f\u0435\u043d","\u0431\u0435\u043d","\u0434\u0430","\u0434\u0435","\u0442\u0430","\u0442\u0435","\u0436\u043e\u049b","\u0431\u0430\u0440","\u0431\u0456\u0440","\u0435\u043a\u0456","\u04af\u0448","\u0442\u04e9\u0440\u0442","\u0431\u04b1\u043b","\u0441\u043e\u043b","\u043e\u0441\u044b","\u04af\u0448\u0456\u043d","\u043c\u0435\u043d\u0456",
    "\u0441\u0435\u043d\u0456","\u043e\u043d\u044b","\u043e\u043d\u0434\u0430","\u043c\u044b\u043d\u0434\u0430","\u0441\u043e\u043d\u0434\u0430","\u043d\u0435","\u049b\u0430\u043b\u0430\u0439","\u049b\u0430\u0439\u0434\u0430","\u049b\u0430\u0448\u0430\u043d",
}

_RU_MORPH = None

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _simple_en_lemma(t: str) -> str:
    for suf in ("ing", "ed", "es", "s"):
        if t.endswith(suf) and len(t) > len(suf) + 2:
            return t[: -len(suf)]
    return t


def _simple_ru_lemma(t: str) -> str:
    for suf in (
        "\u0430\u043c\u0438","\u044f\u043c\u0438","\u043e\u0433\u043e","\u0435\u043c\u0443","\u043e\u043c\u0443","\u044b\u043c\u0438","\u0438\u043c\u0438","\u0430\u0445","\u044f\u0445","\u043e\u0432","\u0435\u0432",
        "\u044b\u043c","\u0438\u043c","\u044b\u0439","\u0438\u0439","\u0430\u044f","\u044f\u044f","\u043e\u0435","\u0435\u0435",
    ):
        if t.endswith(suf) and len(t) > len(suf) + 2:
            return t[: -len(suf)]
    return t


def _simple_kz_lemma(t: str) -> str:
    for suf in (
        "\u043b\u0430\u0440\u044b","\u043b\u0435\u0440\u0456","\u0434\u0430\u0440\u044b","\u0434\u0435\u0440\u0456","\u0442\u0430\u0440\u044b","\u0442\u0435\u0440\u0456",
        "\u0442\u044b\u04a3","\u0442\u0456\u04a3","\u043d\u044b\u04a3","\u043d\u0456\u04a3","\u043c\u0435\u043d","\u043f\u0435\u043d","\u0431\u0435\u043d","\u0493\u0430","\u0493\u0435","\u049b\u0430","\u049b\u0435",
        "\u0434\u0430","\u0434\u0435","\u0442\u0430","\u0442\u0435",
    ):
        if t.endswith(suf) and len(t) > len(suf) + 2:
            return t[: -len(suf)]
    return t


def _normalize_tokens(tokens: Iterable[str], lang: str, lemmatize: bool) -> List[str]:
    out: List[str] = []
    lang = (lang or "en").strip().lower()
    if lang == "ru":
        stop = _STOPWORDS_RU
    elif lang == "kz":
        stop = _STOPWORDS_KZ
    else:
        stop = _STOPWORDS_EN

    for t in tokens:
        t = t.lower()
        if len(t) <= 1:
            continue
        if t in stop:
            continue
        if lemmatize:
            if lang == "ru":
                global _RU_MORPH
                if _RU_MORPH is None:
                    try:
                        import pymorphy3  # type: ignore

                        _RU_MORPH = pymorphy3.MorphAnalyzer()
                    except Exception:
                        _RU_MORPH = False
                if _RU_MORPH:
                    try:
                        t = _RU_MORPH.parse(t)[0].normal_form
                    except Exception:
                        t = _simple_ru_lemma(t)
                else:
                    t = _simple_ru_lemma(t)
            elif lang == "kz":
                t = _simple_kz_lemma(t)
            else:
                t = _simple_en_lemma(t)
        if t:
            out.append(t)
    return out


def _tokenize(text: str, lang: str = "en", *, lemmatize: bool = False, normalize: bool = True) -> List[str]:
    tokens = _WORD_RE.findall(text or "")
    if not normalize:
        return [t.lower() for t in tokens if len(t) > 1]
    return _normalize_tokens(tokens, lang=lang, lemmatize=lemmatize)


def _hash_text(text: str) -> str:
    import hashlib

    return hashlib.sha1(text.encode("utf-8")).hexdigest()


class EmbeddingCache:
    def __init__(self, cache_dir: Path, model_name: str) -> None:
        tag = _sanitize_tag(model_name.replace("/", "_"))
        db_dir = Path(cache_dir) / "embeddings"
        db_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = db_dir / f"{tag}.sqlite"
        self._init_db()

    def _init_db(self) -> None:
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS embeddings (h TEXT PRIMARY KEY, dim INTEGER, dtype TEXT, vec BLOB)"
            )
            conn.execute("PRAGMA journal_mode=WAL")
            conn.commit()
        finally:
            conn.close()

    def get_many(self, hashes: List[str]) -> Dict[str, Any]:
        if not hashes:
            return {}
        out: Dict[str, Any] = {}
        conn = sqlite3.connect(self.db_path)
        try:
            q = "SELECT h, dim, dtype, vec FROM embeddings WHERE h IN (%s)" % ",".join("?" * len(hashes))
            rows = conn.execute(q, hashes).fetchall()
            for h, dim, dtype, vec in rows:
                out[str(h)] = (int(dim), str(dtype), vec)
        finally:
            conn.close()
        return out

    def put_many(self, items: Dict[str, Any]) -> None:
        if not items:
            return
        conn = sqlite3.connect(self.db_path)
        try:
            rows = [(h, int(dim), str(dtype), vec) for h, (dim, dtype, vec) in items.items()]
            conn.executemany(
                "INSERT OR REPLACE INTO embeddings (h, dim, dtype, vec) VALUES (?, ?, ?, ?)",
                rows,
            )
            conn.commit()
        finally:
            conn.close()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _read_jsonl_gz(path: Path) -> List[Dict[str, Any]]:
    import gzip

    rows: List[Dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def _read_jsonl_zst(path: Path) -> List[Dict[str, Any]]:
    if zstd is None:
        raise RuntimeError("zstandard is not installed")
    rows: List[Dict[str, Any]] = []
    with path.open("rb") as fh:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(fh) as reader:
            stream = io.TextIOWrapper(reader, encoding="utf-8")
            for line in stream:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if isinstance(obj, dict):
                    rows.append(obj)
    return rows


def _iter_segment_files(videos_root: Path, language: str, variant: Optional[str] = None) -> List[Path]:
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


def _guess_source_from_path(p: Path) -> Tuple[str, Optional[str]]:
    try:
        parts = list(p.parts)
        idx = parts.index("outputs")
        video_id = parts[idx - 1] if idx >= 1 else "unknown"
        variant = None
        if len(parts) > idx + 2 and parts[idx + 1] == "variants":
            variant = parts[idx + 2]
        return str(video_id), (str(variant) if variant else None)
    except Exception:
        return "unknown", None


def _file_fingerprint(p: Path) -> str:
    st = p.stat()
    return f"{st.st_size}:{st.st_mtime_ns}"


def _stable_doc_id(video_id: str, i: int, variant: Optional[str] = None) -> str:
    if variant:
        return f"{video_id}/{variant}/seg_{i:04d}"
    return f"{video_id}/seg_{i:04d}"


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


def resolve_index_dir(
    base_index_dir: Path,
    config_fingerprint: Optional[str] = None,
    language: Optional[str] = None,
    variant: Optional[str] = None,
) -> Path:
    base = Path(base_index_dir)
    if language:
        base = base / _sanitize_tag(str(language))
    if variant:
        base = base / "variants" / _sanitize_tag(str(variant))
    return base / config_tag_from_fingerprint(config_fingerprint)


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
            "reranker_model_id": str(getattr(s, "reranker_model_id", "")),
            "reranker_backend": str(getattr(s, "reranker_backend", "auto")),
            "normalize_text": bool(getattr(s, "normalize_text", True)),
            "lemmatize": bool(getattr(s, "lemmatize", False)),
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
    language: str
    start_sec: float
    end_sec: float
    text: str
    display_text: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None
    source_path: Optional[str] = None


def _coerce_str_list(value: Any) -> List[str]:
    out: List[str] = []
    if isinstance(value, list):
        for item in value:
            text = str(item or "").strip()
            if text:
                out.append(text)
    elif value is not None:
        text = str(value).strip()
        if text:
            out.append(text)
    return out


def _extract_doc_metadata(item: Dict[str, Any], variant: Optional[str]) -> Dict[str, Any]:
    base_extra = item.get("extra")
    meta: Dict[str, Any] = dict(base_extra) if isinstance(base_extra, dict) else {}

    for key in ("segment_id", "event_type", "risk_level", "people_count_bucket", "motion_type"):
        value = item.get(key)
        if value is not None:
            meta[key] = str(value)

    for key in ("tags", "objects", "anomaly_notes"):
        values = _coerce_str_list(item.get(key))
        if values:
            meta[key] = values
        elif key not in meta:
            meta[key] = []

    anomaly_flag = item.get("anomaly_flag")
    if anomaly_flag is not None:
        meta["anomaly_flag"] = bool(anomaly_flag)

    if variant is not None:
        meta["variant"] = str(variant)

    return meta


def _build_searchable_text(display_text: str, meta: Dict[str, Any]) -> str:
    parts: List[str] = []
    base = str(display_text or "").strip()
    if base:
        parts.append(base)

    scalar_keys = ("event_type", "risk_level", "people_count_bucket", "motion_type")
    for key in scalar_keys:
        value = str(meta.get(key) or "").strip()
        if value:
            parts.append(value)

    for key in ("tags", "objects"):
        for value in _coerce_str_list(meta.get(key)):
            parts.append(value)

    return " \n ".join(parts)


def _normalize_loaded_doc(doc: Any) -> Optional[Doc]:
    try:
        if isinstance(doc, dict):
            getter = doc.get
        else:
            getter = lambda key, default=None: getattr(doc, key, default)

        text = str(getter("text", "") or "")
        display_text = getter("display_text", None)
        if display_text is None:
            display_text = text
        extra = getter("extra", None)
        if not isinstance(extra, dict):
            extra = {}
        return Doc(
            doc_id=str(getter("doc_id", "") or ""),
            video_id=str(getter("video_id", "") or ""),
            language=str(getter("language", "") or ""),
            start_sec=float(getter("start_sec", 0.0) or 0.0),
            end_sec=float(getter("end_sec", 0.0) or 0.0),
            text=text,
            display_text=str(display_text or ""),
            extra=extra,
            source_path=getter("source_path", None),
        )
    except Exception:
        return None


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


class SentenceTransformerEmbedder:
    """Embed text with sentence-transformers for dense retrieval."""

    def __init__(
        self,
        model_name: str = MODEL_NAME_DEFAULT,
        device: Optional[str] = None,
        query_prefix: str = "query: ",
        passage_prefix: str = "passage: ",
    ):
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "Install `sentence-transformers` to use embeddings: pip install -U sentence-transformers"
            ) from e

        self.model_name = model_name
        self.query_prefix = str(query_prefix)
        self.passage_prefix = str(passage_prefix)
        self.model = SentenceTransformer(model_name, device=device)

    def prep_passage(self, text: str) -> str:
        return f"{self.passage_prefix}{text}"

    def prep_query(self, text: str) -> str:
        return f"{self.query_prefix}{text}"

    def encode_passages(self, texts: List[str], batch_size: int = 64):
        return self.model.encode(
            [self.prep_passage(t) for t in texts],
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def encode_query(self, text: str):
        return self.model.encode([self.prep_query(text)], normalize_embeddings=True, show_progress_bar=False)[0]


class TransformersTextEmbedder:
    """Embed text with a generic transformers model using masked mean pooling."""

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        query_prefix: str = "query: ",
        passage_prefix: str = "passage: ",
    ):
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except Exception as exc:
            raise RuntimeError(
                "Install `transformers` and `torch` to use the transformers embedding backend."
            ) from exc

        self.model_name = str(model_name)
        self.query_prefix = str(query_prefix)
        self.passage_prefix = str(passage_prefix)
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True)
        self.model.eval()

        device_name = str(device or "").strip()
        if device_name:
            try:
                self.model.to(device_name)
            except Exception:
                pass

    def prep_passage(self, text: str) -> str:
        return f"{self.passage_prefix}{text}"

    def prep_query(self, text: str) -> str:
        return f"{self.query_prefix}{text}"

    def _encode(self, texts: List[str], batch_size: int = 32):
        rows: List[Any] = []
        model_device = None
        try:
            model_device = next(self.model.parameters()).device
        except Exception:
            model_device = None

        for start in range(0, len(texts), max(1, int(batch_size))):
            batch = texts[start : start + max(1, int(batch_size))]
            encoded = self.tokenizer(
                batch,
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
                hidden = getattr(outputs, "last_hidden_state", None)
                if hidden is None:
                    hidden = outputs[0]
                attn = encoded.get("attention_mask")
                if attn is None:
                    pooled = hidden.mean(dim=1)
                else:
                    mask = attn.unsqueeze(-1).expand(hidden.size()).float()
                    denom = mask.sum(dim=1).clamp(min=1.0)
                    pooled = (hidden * mask).sum(dim=1) / denom
                pooled = self._torch.nn.functional.normalize(pooled, p=2, dim=1)
                rows.append(pooled.detach().cpu().numpy())

        if not rows:
            return np.zeros((0, 0), dtype=np.float32)
        return np.concatenate(rows, axis=0).astype(np.float32, copy=False)

    def encode_passages(self, texts: List[str], batch_size: int = 32):
        return self._encode([self.prep_passage(t) for t in texts], batch_size=batch_size)

    def encode_query(self, text: str):
        arr = self._encode([self.prep_query(text)], batch_size=1)
        return arr[0] if len(arr) else np.zeros((0,), dtype=np.float32)


def _looks_like_transformers_model(model_name: str) -> bool:
    """Detect model ids or paths that should prefer the transformers embedder."""

    text = str(model_name or "").strip()
    if not text:
        return False
    if "qwen" in text.lower():
        return True
    path = Path(text)
    if path.exists():
        return any(path.glob("*.safetensors")) or any(path.glob("*.bin")) or any(path.glob("*.pt"))
    return False


def select_embedding_model_ref(search_cfg: Any, models_dir: Optional[Path] = None) -> str:
    """Choose the effective embedding model ref for the current config."""

    backend_name = str(getattr(search_cfg, "embedding_backend", "auto") or "auto").strip().lower() or "auto"
    fallback_name = str(getattr(search_cfg, "embed_model_name", MODEL_NAME_DEFAULT) or MODEL_NAME_DEFAULT)
    preferred_name = str(getattr(search_cfg, "embedding_model_id", "") or fallback_name)

    if backend_name == "transformers":
        return preferred_name
    if backend_name == "sentence_transformers":
        return fallback_name
    if models_dir:
        local_candidate = Path(models_dir) / preferred_name.split("/")[-1].strip().lower()
        if _looks_like_transformers_model(str(local_candidate)):
            return str(local_candidate)
    return fallback_name


def build_text_embedder(
    *,
    model_name: str,
    device: Optional[str],
    query_prefix: str,
    passage_prefix: str,
    backend: str = "auto",
    fallback_model_name: Optional[str] = None,
):
    """Build an embedder with backend auto-selection and a sentence-transformer fallback."""

    backend_name = str(backend or "auto").strip().lower() or "auto"

    def _build_sentence(name: str):
        return SentenceTransformerEmbedder(
            model_name=name,
            device=device,
            query_prefix=query_prefix,
            passage_prefix=passage_prefix,
        )

    def _build_transformers(name: str):
        return TransformersTextEmbedder(
            model_name=name,
            device=device,
            query_prefix=query_prefix,
            passage_prefix=passage_prefix,
        )

    if backend_name == "sentence_transformers":
        return _build_sentence(model_name)
    if backend_name == "transformers":
        try:
            return _build_transformers(model_name)
        except Exception:
            if fallback_model_name:
                return _build_sentence(fallback_model_name)
            raise

    if _looks_like_transformers_model(model_name):
        try:
            return _build_transformers(model_name)
        except Exception:
            if fallback_model_name:
                return _build_sentence(fallback_model_name)

    try:
        return _build_sentence(model_name)
    except Exception:
        if fallback_model_name and fallback_model_name != model_name:
            try:
                return _build_transformers(fallback_model_name)
            except Exception:
                pass
        raise


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
        layout = "videos/<video_id>/outputs/segments/<lang>.jsonl.zst"
        if zstd is None:
            layout = "videos/<video_id>/outputs/segments/<lang>.jsonl.gz"
        return {
            "version": 5,
            "model_name": MODEL_NAME_DEFAULT,
            "bm25": {"k1": 1.6, "b": 0.75},
            "sources": {},
            "layout": layout,
            "doc_id_scheme": "stable_by_position",
            "variant": None,
            "has_dense_valid": True,
        }
    m = _read_json(p)
    if not isinstance(m, dict):
        m = {}

    m.setdefault("version", 5)
    m.setdefault("model_name", MODEL_NAME_DEFAULT)
    m.setdefault("bm25", {"k1": 1.6, "b": 0.75})
    m.setdefault("sources", {})
    if "layout" not in m:
        m["layout"] = "videos/<video_id>/outputs/segments/<lang>.jsonl.zst"
        if zstd is None:
            m["layout"] = "videos/<video_id>/outputs/segments/<lang>.jsonl.gz"
    m.setdefault("doc_id_scheme", "stable_by_position")
    m.setdefault("language", None)
    m.setdefault("variant", None)
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
            out: Dict[str, Doc] = {}
            for key, value in obj.items():
                normalized = _normalize_loaded_doc(value)
                if normalized is None:
                    continue
                if not normalized.doc_id:
                    normalized.doc_id = str(key)
                out[str(normalized.doc_id)] = normalized
            return out
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

    norm2 = np.einsum("ij,ij->i", x, x)
    return norm2 > 1e-10


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
    cache_dir: Optional[Path] = None,
    use_embed_cache: bool = True,
) -> Path:

    t_total0 = time.perf_counter()

    base_index_dir = Path(index_dir)
    real_index_dir = resolve_index_dir(base_index_dir, config_fingerprint, language=language, variant=variant)
    real_index_dir.mkdir(parents=True, exist_ok=True)

    embed_store_dtype = _normalize_embed_store_dtype(embed_store_dtype)
    embedding_backend = str(embedding_backend or "auto").strip().lower() or "auto"
    manifest = _load_manifest(real_index_dir)
    prev_manifest_version = int(manifest.get("version", 0) or 0)
    schema_changed = prev_manifest_version < 5

    old_bm25 = (manifest.get("bm25") or {})
    bm25_changed = (float(old_bm25.get("k1", 1.6)) != float(bm25_k1)) or (float(old_bm25.get("b", 0.75)) != float(bm25_b))
    model_changed = manifest.get("model_name") != model_name

    if model_changed or schema_changed:
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
    manifest["version"] = max(5, int(manifest.get("version", 5) or 5))
    manifest["has_dense_valid"] = True

    files = _iter_segment_files(Path(videos_root), language=language, variant=variant)
    if not files:
        suffix = f", variant={variant}" if variant else ""
        raise RuntimeError(f"No segments found under: {videos_root} (lang={language}{suffix})")

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
        video_id, file_variant = _guess_source_from_path(f)

        try:
            if f.suffix == ".zst":
                data = _read_jsonl_zst(f)
            else:
                data = _read_jsonl_gz(f)
        except Exception:
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
                display_text = str(item.get("normalized_caption") or item.get("description", "") or "")
            except Exception:
                continue

            doc_id = _stable_doc_id(video_id, i, variant=file_variant)
            doc_extra = _extract_doc_metadata(item, file_variant)
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
                source_path=f_key,
            )
            changed_doc_ids.add(doc_id)
            docs_added_or_updated += 1
            new_num_docs += 1

        if prev_num_docs > new_num_docs:
            for j in range(new_num_docs, prev_num_docs):
                did = _stable_doc_id(video_id, j, variant=file_variant)
                if did in docs_by_id:
                    docs_by_id.pop(did, None)
                    docs_deleted += 1

        manifest["sources"][f_key] = {
            "fingerprint": fp,
            "video_id": video_id,
            "variant": file_variant,
            "language": str(language),
            "num_docs": int(new_num_docs),
        }
        changed_any = True

    t_scan_ms = (time.perf_counter() - t_scan0) * 1000.0

    if not changed_any and not bm25_changed and not model_changed:
        _save_manifest(real_index_dir, manifest)
        meta = {
            "index_schema_version": 5,
            "config_fingerprint": (str(config_fingerprint) if config_fingerprint else None),
            "config_tag": config_tag_from_fingerprint(config_fingerprint),
            "model_name": manifest.get("model_name"),
            "embedding_backend": embedding_backend,
            "num_docs": int(len(docs_by_id)),
            "bm25": {"k1": float(bm25_k1), "b": float(bm25_b)},
            "embed_store_dtype": embed_store_dtype,
            "layout": manifest.get("layout"),
            "doc_id_scheme": manifest.get("doc_id_scheme"),
            "language": str(language),
            "variant": (str(variant) if variant else None),
            "query_prefix": str(query_prefix),
            "passage_prefix": str(passage_prefix),
            "normalize_text": bool(normalize_text),
            "lemmatize": bool(lemmatize),
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
    tokenized = [_tokenize(t, lang=language, lemmatize=lemmatize, normalize=normalize_text) for t in texts]
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
        hashes = [_hash_text(t) for t in texts] if embed_cache else []
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
                to_encode.append(texts[i])
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
            emb = embedder.encode_passages(texts, batch_size=batch_size)
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
                to_encode_texts.append(docs_by_id[did].text)
                if embed_cache:
                    to_encode_hashes.append(_hash_text(docs_by_id[did].text))

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
