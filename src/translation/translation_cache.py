# src/translation/translation_cache.py
"""
SQLite-backed translation cache keyed by text and effective translation context.

Purpose:
- Prevent cross-run cache leakage between different configs or model revisions.
- Preserve the previous high-level API (`get_many`, `put_many`) for existing callers.
"""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path
from typing import Dict, Iterable, List, Optional


def _hash_text(text: str) -> str:
    """Build a stable hash for the raw source text only."""

    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _sanitize_tag(text: str) -> str:
    """Convert a model ID into a filesystem-safe cache filename fragment."""

    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in (text or "").strip())
    cleaned = cleaned.strip("._-")
    return cleaned or "default"


class TranslationCache:
    """Cache translations for one translation context instance."""

    def __init__(
        self,
        cache_dir: Path,
        model_name: str,
        src_lang: str,
        tgt_lang: str,
        *,
        model_version: str = "v1",
        config_fingerprint: Optional[str] = None,
    ) -> None:
        self.model_name = str(model_name or "")
        self.src_lang = str(src_lang or "").strip().lower()
        self.tgt_lang = str(tgt_lang or "").strip().lower()
        self.model_version = str(model_version or "v1")
        self.config_fingerprint = str(config_fingerprint or "default")

        db_dir = Path(cache_dir) / "translations"
        db_dir.mkdir(parents=True, exist_ok=True)

        tag = _sanitize_tag(self.model_name)
        pair = _sanitize_tag(f"{self.src_lang}_{self.tgt_lang}")
        self.db_path = db_dir / f"{tag}__{pair}.sqlite"
        self._init_db()

    def _init_db(self) -> None:
        """Create cache table and index on first use."""

        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS translations (
                    cache_key TEXT PRIMARY KEY,
                    text_hash TEXT NOT NULL,
                    src_lang TEXT NOT NULL,
                    tgt_lang TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    config_fingerprint TEXT NOT NULL,
                    translated_text TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_translations_lookup
                ON translations (text_hash, src_lang, tgt_lang, model_name, model_version, config_fingerprint)
                """
            )
            conn.execute("PRAGMA journal_mode=WAL")
            conn.commit()
        finally:
            conn.close()

    def _cache_key(self, text: str) -> str:
        """Build the full cache key for one source text under this cache context."""

        payload = "|".join(
            [
                text,
                self.src_lang,
                self.tgt_lang,
                self.model_name,
                self.model_version,
                self.config_fingerprint,
            ]
        )
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    def get_many(self, texts: Iterable[str]) -> Dict[str, Optional[str]]:
        """Return cached translations keyed by plain text hash for compatibility."""

        items = [str(text) for text in texts]
        if not items:
            return {}

        text_hashes = [_hash_text(text) for text in items]
        cache_keys = [self._cache_key(text) for text in items]
        out: Dict[str, Optional[str]] = {text_hash: None for text_hash in text_hashes}

        conn = sqlite3.connect(self.db_path)
        try:
            query = "SELECT cache_key, translated_text FROM translations WHERE cache_key IN (%s)" % ",".join(
                "?" * len(cache_keys)
            )
            rows = conn.execute(query, cache_keys).fetchall()
            found = {str(cache_key): str(translated) for cache_key, translated in rows}
            for text, text_hash, cache_key in zip(items, text_hashes, cache_keys):
                _ = text
                if cache_key in found:
                    out[text_hash] = found[cache_key]
        finally:
            conn.close()
        return out

    def put_many(self, texts: List[str], translations: List[str]) -> None:
        """Persist translations for the current cache context."""

        if not texts:
            return

        rows = []
        for text, translated in zip(texts, translations):
            text = str(text)
            rows.append(
                (
                    self._cache_key(text),
                    _hash_text(text),
                    self.src_lang,
                    self.tgt_lang,
                    self.model_name,
                    self.model_version,
                    self.config_fingerprint,
                    str(translated),
                )
            )

        conn = sqlite3.connect(self.db_path)
        try:
            conn.executemany(
                """
                INSERT OR REPLACE INTO translations (
                    cache_key,
                    text_hash,
                    src_lang,
                    tgt_lang,
                    model_name,
                    model_version,
                    config_fingerprint,
                    translated_text
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            conn.commit()
        finally:
            conn.close()
