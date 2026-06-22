from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.experiments import run_v08_pipeline as runner
from src.core import vlm_backend
from src.core.runtime import resolve_config_selection
from src.llm import client as llm_client
from src.translation.service import TranslationService


def test_vlm_model_load_fails_closed_unless_mock_is_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(vlm_backend.QwenVLBackend, "_load_model", staticmethod(lambda *_a, **_k: (_ for _ in ()).throw(OSError("boom"))))
    backend = vlm_backend.QwenVLBackend("missing-model", lazy_load=True)
    with pytest.raises(RuntimeError, match="Failed to load VLM"):
        backend._ensure_loaded()

    mock_backend = vlm_backend.QwenVLBackend("missing-model", lazy_load=True, allow_mock=True)
    mock_backend._ensure_loaded()
    assert mock_backend.model == "mock"


def test_llm_model_load_fails_closed_unless_mock_is_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    import transformers

    llm_client._TRANSFORMERS_CACHE.clear()
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *_a, **_k: (_ for _ in ()).throw(OSError("boom")))
    base = dict(backend="transformers", model_id="missing", timeout_sec=1, vllm_base_url="", max_new_tokens=1,
                do_sample=False, temperature=0.0, top_p=1.0, transformers_dtype="float32",
                transformers_device_map="cpu", transformers_compile=False)
    with pytest.raises(RuntimeError, match="Failed to load LLM"):
        llm_client.LLMClient(**base)._load_transformers_backend()
    tokenizer, model = llm_client.LLMClient(**base, allow_mock=True)._load_transformers_backend()
    assert (tokenizer, model) == ("mock", "mock")
    llm_client._TRANSFORMERS_CACHE.clear()


def _translation_cfg(tmp_path: Path, allow_mock: bool) -> SimpleNamespace:
    translation = SimpleNamespace(en_ru_model_id="en-ru", ru_en_model_id="ru-en", kk_ru_model_id="kk-ru",
                                  ru_kk_model_id="ru-kk", post_edit_targets=[], post_edit_max_items=1,
                                  batch_size=1, max_new_tokens=8, cache_enabled=False, cache_version="1")
    return SimpleNamespace(translation=translation, runtime=SimpleNamespace(allow_mock_backends=allow_mock),
                           paths=SimpleNamespace(cache_dir=tmp_path), config_fingerprint="fp")


def test_translation_fails_closed_unless_mock_is_explicit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    strict = TranslationService(_translation_cfg(tmp_path, False))
    monkeypatch.setattr(strict, "_translate_via_ct2", lambda *_a, **_k: (_ for _ in ()).throw(OSError("boom")))
    with pytest.raises(RuntimeError, match="Translation failed"):
        strict.translate(["hello"], src_lang="en", tgt_lang="ru", use_cache=False)

    mock = TranslationService(_translation_cfg(tmp_path, True))
    monkeypatch.setattr(mock, "_translate_via_ct2", lambda *_a, **_k: (_ for _ in ()).throw(OSError("boom")))
    assert mock.translate(["hello"], src_lang="en", tgt_lang="ru", use_cache=False) == ["[RU] hello"]


def test_manifest_prepared_video_is_imported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "prepared.mp4"
    source.write_bytes(b"video")
    videos = tmp_path / "videos"
    monkeypatch.setattr(runner, "normalize_uploaded_video", lambda path: path)
    imported = runner.import_manifest_videos(videos, [{"video_id": "v1", "prepared_video_path": str(source)}])
    assert imported == ["v1"]
    assert (videos / "v1" / "raw" / "prepared.mp4").read_bytes() == b"video"


def test_profile_path_selection_preserves_explicit_file(tmp_path: Path) -> None:
    profile = tmp_path / "Custom.yaml"
    profile.write_text("paths: {}", encoding="utf-8")
    path, name, variant = resolve_config_selection(profile= str(profile))
    assert path == profile.resolve()
    assert name == "custom"
    assert variant is None


def test_canceled_is_terminal_failure_state() -> None:
    assert "canceled" in runner.TERMINAL_FAILURE_STATES
