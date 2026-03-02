# src/translation/service.py
"""
Translation service with CTranslate2-first routing for EN/RU/KZ language pairs.

Purpose:
- Provide one translation interface for API query translation and worker translation jobs.
- Route bilingual MT models per language pair.
- Prefer CTranslate2, with on-demand model conversion from Hugging Face checkpoints.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from src.translation.translation_cache import TranslationCache


def _normalize_lang(lang: str) -> str:
    """Normalize supported language aliases into one code."""

    token = (lang or "").strip().lower()
    if token == "kk":
        return "kz"
    return token


def _sanitize_tag(text: str) -> str:
    """Convert a model reference into a filesystem-safe folder name."""

    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in (text or "").strip())
    cleaned = cleaned.strip("._-")
    return cleaned or "default"


def _is_ct2_model_dir(path: Path) -> bool:
    """Detect whether a local directory already contains CTranslate2 artifacts."""

    candidate = Path(path)
    if not candidate.exists() or not candidate.is_dir():
        return False
    markers = (
        "model.bin",
        "model.bin.index.json",
        "shared_vocabulary.json",
        "shared_vocabulary.txt",
    )
    return any((candidate / name).exists() for name in markers)


@dataclass(frozen=True)
class TranslationRoute:
    """One bilingual translation model route."""

    src_lang: str
    tgt_lang: str
    model_id: str


class TranslationService:
    """Translation router backed by CTranslate2 model instances."""

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self._routes: Dict[Tuple[str, str], TranslationRoute] = {
            ("en", "ru"): TranslationRoute("en", "ru", str(cfg.translation.en_ru_model_id)),
            ("ru", "en"): TranslationRoute("ru", "en", str(cfg.translation.ru_en_model_id)),
            ("kz", "ru"): TranslationRoute("kz", "ru", str(cfg.translation.kk_ru_model_id)),
            ("ru", "kz"): TranslationRoute("ru", "kz", str(cfg.translation.ru_kk_model_id)),
        }
        self._translator_cache: Dict[str, object] = {}
        self._tokenizer_cache: Dict[str, object] = {}

    def route_for(self, src_lang: str, tgt_lang: str) -> List[TranslationRoute]:
        """Resolve the chain of bilingual routes for one requested pair."""

        src = _normalize_lang(src_lang)
        tgt = _normalize_lang(tgt_lang)
        if src == tgt:
            return []

        direct = self._routes.get((src, tgt))
        if direct:
            return [direct]

        if src == "kz" and tgt == "en":
            return [self._routes[("kz", "ru")], self._routes[("ru", "en")]]
        if src == "en" and tgt == "kz":
            return [self._routes[("en", "ru")], self._routes[("ru", "kz")]]

        raise ValueError(f"Unsupported translation pair: {src_lang} -> {tgt_lang}")

    def translate(
        self,
        texts: Iterable[str],
        *,
        src_lang: str,
        tgt_lang: str,
        batch_size: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        use_cache: bool = True,
    ) -> List[str]:
        """Translate a batch of texts through one or two routed stages."""

        items = [str(text) for text in texts]
        if not items:
            return []

        routes = self.route_for(src_lang, tgt_lang)
        if not routes:
            return items

        out = list(items)
        for route in routes:
            out = self._translate_one_route(
                out,
                route=route,
                batch_size=batch_size or int(self.cfg.translation.batch_size),
                max_new_tokens=max_new_tokens or int(self.cfg.translation.max_new_tokens),
                use_cache=use_cache and bool(self.cfg.translation.cache_enabled),
            )
        return out

    def cache(self, *, src_lang: str, tgt_lang: str) -> TranslationCache:
        """Build a cache instance bound to the active config and selected pair."""

        route_chain = self.route_for(src_lang, tgt_lang)
        model_ref = "+".join(route.model_id for route in route_chain) if route_chain else "identity"
        return TranslationCache(
            cache_dir=Path(self.cfg.paths.cache_dir),
            model_name=model_ref,
            src_lang=_normalize_lang(src_lang),
            tgt_lang=_normalize_lang(tgt_lang),
            model_version=str(self.cfg.translation.cache_version),
            config_fingerprint=str(self.cfg.config_fingerprint),
        )

    def _translate_one_route(
        self,
        texts: List[str],
        *,
        route: TranslationRoute,
        batch_size: int,
        max_new_tokens: int,
        use_cache: bool,
    ) -> List[str]:
        """Translate texts with one bilingual model and optional cache lookup."""

        if not texts:
            return []

        cache = self.cache(src_lang=route.src_lang, tgt_lang=route.tgt_lang)
        translated: List[str] = [""] * len(texts)
        miss_idx: List[int] = []
        miss_texts: List[str] = []

        if use_cache:
            cached = cache.get_many(texts)
            for idx, text in enumerate(texts):
                key = self._text_hash(text)
                hit = cached.get(key)
                if hit:
                    translated[idx] = hit
                else:
                    miss_idx.append(idx)
                    miss_texts.append(text)
        else:
            miss_idx = list(range(len(texts)))
            miss_texts = list(texts)

        if miss_texts:
            miss_translated = self._translate_via_ct2(
                miss_texts,
                model_id=route.model_id,
                batch_size=batch_size,
                max_new_tokens=max_new_tokens,
            )
            for idx, text in zip(miss_idx, miss_translated):
                translated[idx] = text
            if use_cache:
                cache.put_many(miss_texts, miss_translated)

        return translated

    def _translate_via_ct2(
        self,
        texts: List[str],
        *,
        model_id: str,
        batch_size: int,
        max_new_tokens: int,
    ) -> List[str]:
        """Translate texts with a CTranslate2 model, converting on demand if needed."""

        translator = self._get_translator(model_id)
        tokenizer = self._get_tokenizer(model_id)

        token_batches: List[List[str]] = []
        for text in texts:
            encoded_ids = tokenizer.encode(str(text), add_special_tokens=True)
            token_batches.append(tokenizer.convert_ids_to_tokens(encoded_ids))

        results: List[str] = []
        step = max(1, int(batch_size))
        for start in range(0, len(token_batches), step):
            batch = token_batches[start : start + step]
            generated = translator.translate_batch(
                batch,
                beam_size=1,
                max_decoding_length=int(max_new_tokens),
            )
            for output in generated:
                hypothesis = output.hypotheses[0] if output.hypotheses else []
                token_ids = tokenizer.convert_tokens_to_ids(hypothesis)
                results.append(tokenizer.decode(token_ids, skip_special_tokens=True).strip())
        return results

    def _get_translator(self, model_id: str):
        """Load or create a cached CTranslate2 translator instance."""

        key = str(model_id)
        if key in self._translator_cache:
            return self._translator_cache[key]

        try:
            import ctranslate2
        except Exception as exc:
            raise RuntimeError("`ctranslate2` is required for the translation layer.") from exc

        model_dir = self._ensure_ct2_model_dir(model_id)
        device_name = str(self.cfg.translation.ctranslate2_device)
        compute_type = str(self.cfg.translation.ctranslate2_compute_type)
        compute_types = [compute_type]
        if device_name == "cpu":
            for fallback in ("int8", "int16", "float32"):
                if fallback not in compute_types:
                    compute_types.append(fallback)

        translator = None
        last_error: Optional[Exception] = None
        for candidate in compute_types:
            try:
                translator = ctranslate2.Translator(
                    str(model_dir),
                    device=device_name,
                    compute_type=str(candidate),
                    inter_threads=int(self.cfg.translation.ctranslate2_inter_threads),
                    intra_threads=int(self.cfg.translation.ctranslate2_intra_threads),
                )
                break
            except Exception as exc:
                last_error = exc
                continue
        if translator is None:
            raise RuntimeError("Could not initialize a CTranslate2 translator.") from last_error
        self._translator_cache[key] = translator
        return translator

    def _get_tokenizer(self, model_id: str):
        """Load or create the tokenizer paired with the MT model."""

        key = str(model_id)
        if key in self._tokenizer_cache:
            return self._tokenizer_cache[key]

        try:
            from transformers import AutoTokenizer
        except Exception as exc:
            raise RuntimeError("`transformers` is required to tokenize MT inputs.") from exc

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._tokenizer_cache[key] = tokenizer
        return tokenizer

    def _ensure_ct2_model_dir(self, model_id: str) -> Path:
        """Resolve a usable CTranslate2 model directory, converting if needed."""

        model_ref = str(model_id).strip()
        model_path = Path(model_ref)
        if model_path.exists() and _is_ct2_model_dir(model_path):
            return model_path

        cache_root = Path(self.cfg.paths.cache_dir) / "ct2_models"
        cache_tag = _sanitize_tag(model_path.name if model_path.exists() else model_ref)
        target_dir = cache_root / cache_tag
        if target_dir.exists():
            return target_dir

        try:
            from ctranslate2.converters import TransformersConverter
        except Exception as exc:
            raise RuntimeError(
                "CTranslate2 converters are required to auto-convert Hugging Face MT models."
            ) from exc

        target_dir.mkdir(parents=True, exist_ok=True)
        converter = TransformersConverter(str(model_path if model_path.exists() else model_ref))
        converter.convert(
            str(target_dir),
            quantization=str(self.cfg.translation.ctranslate2_compute_type),
            force=True,
        )
        return target_dir

    @staticmethod
    def _text_hash(text: str) -> str:
        """Keep cache lookups aligned with TranslationCache.get_many."""

        import hashlib

        return hashlib.sha1(str(text).encode("utf-8")).hexdigest()
