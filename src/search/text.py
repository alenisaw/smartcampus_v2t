# src/search/text.py
"""
Search text normalization helpers for SmartCampus V2T.

Purpose:
- Tokenize and normalize search text across supported languages.
- Provide lightweight stopword removal and simple lemmatization utilities.
"""

from __future__ import annotations

import re
from typing import Iterable, List

_WORD_RE = re.compile(r"\w+", flags=re.UNICODE)

_STOPWORDS_EN = {
    "the", "a", "an", "and", "or", "is", "are", "was", "were", "to", "of", "in", "on", "for", "with", "at", "by", "from", "as", "that",
    "this", "it", "its", "be", "been", "being", "into", "out", "up", "down", "over", "under", "not", "no", "yes", "but", "so", "if",
}
_STOPWORDS_RU = {
    "и", "в", "во", "не", "что", "он", "на", "я", "с", "со", "как", "а", "то", "все", "она", "так", "его", "но", "да", "ты", "к", "у",
    "же", "вы", "за", "бы", "по", "ее", "мне", "есть", "они", "тут", "где", "мы", "там", "чтобы", "кто", "когда", "из", "от", "до",
}
_STOPWORDS_KZ = {
    "және", "мен", "пен", "бен", "да", "де", "та", "те", "жоқ", "бар", "бір", "екі", "үш", "төрт", "бұл", "сол", "осы", "үшін", "мені",
    "сені", "оны", "онда", "мында", "сонда", "не", "қалай", "қайда", "қашан",
}

_RU_MORPH = None


def simple_en_lemma(token: str) -> str:
    """Apply a minimal English suffix-strip lemma fallback."""

    for suffix in ("ing", "ed", "es", "s"):
        if token.endswith(suffix) and len(token) > len(suffix) + 2:
            return token[: -len(suffix)]
    return token


def simple_ru_lemma(token: str) -> str:
    """Apply a minimal Russian suffix-strip lemma fallback."""

    for suffix in (
        "ами", "ями", "ого", "ему", "ому", "ыми", "ими", "ах", "ях", "ов", "ев",
        "ым", "им", "ый", "ий", "ая", "яя", "ое", "ее",
    ):
        if token.endswith(suffix) and len(token) > len(suffix) + 2:
            return token[: -len(suffix)]
    return token


def simple_kz_lemma(token: str) -> str:
    """Apply a minimal Kazakh suffix-strip lemma fallback."""

    for suffix in (
        "лары", "лері", "дары", "дері", "тары", "тері",
        "тың", "тің", "ның", "нің", "мен", "пен", "бен", "ға", "ге", "қа", "ке",
        "да", "де", "та", "те",
    ):
        if token.endswith(suffix) and len(token) > len(suffix) + 2:
            return token[: -len(suffix)]
    return token


def normalize_tokens(tokens: Iterable[str], lang: str, lemmatize: bool) -> List[str]:
    """Normalize and optionally lemmatize tokens for sparse search indexing."""

    out: List[str] = []
    lang = (lang or "en").strip().lower()
    if lang == "ru":
        stopwords = _STOPWORDS_RU
    elif lang == "kz":
        stopwords = _STOPWORDS_KZ
    else:
        stopwords = _STOPWORDS_EN

    for token in tokens:
        token = token.lower()
        if len(token) <= 1 or token in stopwords:
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
                        token = _RU_MORPH.parse(token)[0].normal_form
                    except Exception:
                        token = simple_ru_lemma(token)
                else:
                    token = simple_ru_lemma(token)
            elif lang == "kz":
                token = simple_kz_lemma(token)
            else:
                token = simple_en_lemma(token)
        if token:
            out.append(token)
    return out


def tokenize(text: str, lang: str = "en", *, lemmatize: bool = False, normalize: bool = True) -> List[str]:
    """Tokenize text for sparse search usage."""

    tokens = _WORD_RE.findall(text or "")
    if not normalize:
        return [token.lower() for token in tokens if len(token) > 1]
    return normalize_tokens(tokens, lang=lang, lemmatize=lemmatize)
