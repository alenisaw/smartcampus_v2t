from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from scripts.runtime import autotune_hardware, benchmark_vlm_batch
from src.search import rank


def test_cuda_probe_is_explicitly_not_vlm_validation() -> None:
    assert benchmark_vlm_batch.PROBE_TYPE == "synthetic_cuda_memory_probe"
    assert "does not load or execute the VLM" in str(benchmark_vlm_batch.__doc__)


def test_autotune_probe_runner_does_not_claim_vlm_success(monkeypatch: pytest.MonkeyPatch) -> None:
    completed = SimpleNamespace(returncode=0, stdout="MEMORY_PROBE_PASSED (NOT VLM VALIDATION)", stderr="")
    monkeypatch.setattr(autotune_hardware.subprocess, "run", lambda *_a, **_k: completed)
    ok, output = autotune_hardware.run_memory_probe(2, 48, 640, 360, "cuda")
    assert ok is True
    assert "NOT VLM VALIDATION" in output


def test_requested_model_reranker_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*_args, **_kwargs):
        raise OSError("bad model")

    monkeypatch.setattr(rank, "TransformersReranker", fail)
    with pytest.raises(RuntimeError, match="Failed to initialize requested reranker"):
        rank.build_reranker("qwen-reranker", "transformers", None, lambda _name: True)


def test_heuristic_reranker_requires_explicit_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rank, "TransformersReranker", lambda *_a, **_k: (_ for _ in ()).throw(AssertionError()))
    assert rank.build_reranker("", "heuristic", None, lambda _name: False) is None
    with pytest.raises(ValueError, match="reranker_model_id"):
        rank.build_reranker("", "auto", None, lambda _name: False)


def test_qwen_cross_encoder_scores_pairs(monkeypatch: pytest.MonkeyPatch) -> None:
    from unittest.mock import MagicMock
    import torch

    # Mock AutoConfig
    mock_config = MagicMock()
    mock_config.model_type = "qwen3_vl"
    monkeypatch.setattr("transformers.AutoConfig.from_pretrained", lambda *a, **k: mock_config)

    # Mock AutoTokenizer
    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.tensor([[1, 1, 1]])
    }
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda *a, **k: mock_tokenizer)

    # Mock Qwen3VLForConditionalGeneration
    mock_model = MagicMock()
    mock_logits = torch.zeros(1, 3, 10000)
    mock_logits[0, -1, 9693] = 1.5
    mock_logits[0, -1, 2152] = -0.25
    
    mock_outputs = MagicMock()
    mock_outputs.logits = mock_logits
    mock_model.return_value = mock_outputs
    mock_model.to.return_value = mock_model
    
    monkeypatch.setattr("transformers.Qwen3VLForConditionalGeneration.from_pretrained", lambda *a, **k: mock_model)

    reranker = rank.TransformersReranker("models/qwen3-vl-reranker-2b", device="cpu")
    # True score (1.5) - False score (-0.25) = 1.75
    assert reranker.score_pairs("query", ["a", "b"]) == [1.75, 1.75]
