# src/search/embed.py
"""
Search embedding helpers for SmartCampus V2T.

Purpose:
- Build dense embedders for hybrid retrieval indexing and querying.
- Provide reusable embedding cache and model-selection helpers.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

MODEL_NAME_DEFAULT = "BAAI/bge-m3"


def sanitize_tag(text: str) -> str:
    """Convert a model or config string into a filesystem-safe tag."""

    text = (text or "").strip()
    if not text:
        return "default"
    text = re.sub(r"[^a-zA-Z0-9_.-]+", "_", text)
    text = text.strip("._-")
    return text or "default"


class EmbeddingCache:
    """SQLite-backed embedding cache for repeated dense indexing work."""

    def __init__(self, cache_dir: Path, model_name: str) -> None:
        tag = sanitize_tag(model_name.replace("/", "_"))
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
            query = "SELECT h, dim, dtype, vec FROM embeddings WHERE h IN (%s)" % ",".join("?" * len(hashes))
            rows = conn.execute(query, hashes).fetchall()
            for hval, dim, dtype, vec in rows:
                out[str(hval)] = (int(dim), str(dtype), vec)
        finally:
            conn.close()
        return out

    def put_many(self, items: Dict[str, Any]) -> None:
        if not items:
            return
        conn = sqlite3.connect(self.db_path)
        try:
            rows = [(hval, int(dim), str(dtype), vec) for hval, (dim, dtype, vec) in items.items()]
            conn.executemany(
                "INSERT OR REPLACE INTO embeddings (h, dim, dtype, vec) VALUES (?, ?, ?, ?)",
                rows,
            )
            conn.commit()
        finally:
            conn.close()


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
        except Exception as exc:
            raise RuntimeError(
                "Install `sentence-transformers` to use embeddings: pip install -U sentence-transformers"
            ) from exc

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
            [self.prep_passage(text) for text in texts],
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
        return self._encode([self.prep_passage(text) for text in texts], batch_size=batch_size)

    def encode_query(self, text: str):
        arr = self._encode([self.prep_query(text)], batch_size=1)
        return arr[0] if len(arr) else np.zeros((0,), dtype=np.float32)


def looks_like_transformers_model(model_name: str) -> bool:
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
        if looks_like_transformers_model(str(local_candidate)):
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

    if looks_like_transformers_model(model_name):
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
