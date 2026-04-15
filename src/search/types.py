# src/search/types.py
"""
Search index data types for SmartCampus V2T.

Purpose:
- Define normalized document and index container types used by build and query flows.
- Keep BM25 scoring state separate from index assembly and persistence helpers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


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


def normalize_loaded_doc(doc: Any) -> Optional[Doc]:
    """Coerce one stored corpus item into the normalized Doc shape."""

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
            for token in set(doc):
                self.df[token] = self.df.get(token, 0) + 1

        self.idf: Dict[str, float] = {}
        for token, df in self.df.items():
            self.idf[token] = math.log(1.0 + (self.N - df + 0.5) / (df + 0.5))

        self.tf: List[Dict[str, int]] = []
        for doc in tokenized_corpus:
            counts: Dict[str, int] = {}
            for token in doc:
                counts[token] = counts.get(token, 0) + 1
            self.tf.append(counts)

    def score(self, query_tokens: List[str]) -> List[float]:
        scores = [0.0] * self.N
        if not self.N:
            return scores

        for index in range(self.N):
            dl = self.doc_lens[index]
            denom_const = self.k1 * (1.0 - self.b + self.b * (dl / (self.avgdl + 1e-9)))
            tf_i = self.tf[index]

            score = 0.0
            for query_token in query_tokens:
                freq = tf_i.get(query_token)
                if not freq:
                    continue
                idf = self.idf.get(query_token, 0.0)
                score += idf * (freq * (self.k1 + 1.0)) / (freq + denom_const)
            scores[index] = score

        return scores


@dataclass
class HybridIndex:
    docs: List[Doc]
    doc_ids: List[str]
    bm25: BM25Index
    embeddings: Any
    dense_valid: Any
    ann_index: Any
    meta: Dict[str, Any]


@dataclass
class SearchResult:
    score: float
    sparse_score: float
    dense_score: float
    video_id: str
    language: str
    start_sec: float
    end_sec: float
    description: str
    source_id: str
    segment_id: Optional[str] = None
    event_type: Optional[str] = None
    risk_level: Optional[str] = None
    tags: Optional[List[str]] = None
    objects: Optional[List[str]] = None
    people_count_bucket: Optional[str] = None
    motion_type: Optional[str] = None
    anomaly_flag: Optional[bool] = None
    variant: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None
