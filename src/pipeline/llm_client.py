# src/pipeline/llm_client.py
"""
LLM client for SmartCampus V2T pipeline runtime.

Purpose:
- Provide a compact text and JSON generation interface for pipeline services.
- Support local transformers and OpenAI-compatible vLLM backends.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import requests

_TRANSFORMERS_CACHE: Dict[str, Tuple[Any, Any]] = {}


def _model_device(model: Any) -> Any:
    """Return the first available parameter device for a loaded torch model."""

    try:
        return next(model.parameters()).device
    except Exception:
        return None


def _has_local_weights(path: Path) -> bool:
    """Detect whether a local model directory contains actual weight files."""

    patterns = (
        "*.safetensors",
        "*.bin",
        "*.pt",
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    )
    for pattern in patterns:
        if any(path.glob(pattern)):
            return True
    return False


@dataclass
class LLMClient:
    """Thin backend-agnostic text LLM wrapper."""

    backend: str
    model_id: str
    timeout_sec: int
    vllm_base_url: str
    max_new_tokens: int
    do_sample: bool
    temperature: float
    top_p: float
    transformers_dtype: str
    transformers_device_map: str
    transformers_compile: bool
    local_model_dir: Optional[str] = None

    @classmethod
    def from_config(cls, cfg: Any) -> "LLMClient":
        """Build an LLM client from application config."""

        local_model_dir: Optional[str] = None
        try:
            models_dir = Path(getattr(cfg.paths, "models_dir"))
            model_ref = str(getattr(cfg.llm, "model_id", "") or "").strip()
            model_path = Path(model_ref)
            if model_path.exists() and _has_local_weights(model_path):
                local_model_dir = str(model_path)
            else:
                slug = model_path.name.strip().lower()
                candidate = models_dir / slug
                if candidate.exists() and _has_local_weights(candidate):
                    local_model_dir = str(candidate)
        except Exception:
            local_model_dir = None

        return cls(
            backend=str(getattr(cfg.llm, "backend", "transformers")),
            model_id=str(getattr(cfg.llm, "model_id", "")),
            timeout_sec=int(getattr(cfg.llm, "timeout_sec", 60)),
            vllm_base_url=str(getattr(cfg.llm, "vllm_base_url", "http://127.0.0.1:8001/v1")),
            max_new_tokens=int(getattr(cfg.llm, "max_new_tokens", 512)),
            do_sample=bool(getattr(cfg.llm, "do_sample", False)),
            temperature=float(getattr(cfg.llm, "temperature", 0.0)),
            top_p=float(getattr(cfg.llm, "top_p", 1.0)),
            transformers_dtype=str(getattr(cfg.llm, "transformers_dtype", "float16")),
            transformers_device_map=str(getattr(cfg.llm, "transformers_device_map", "auto")),
            transformers_compile=bool(getattr(cfg.llm, "transformers_compile", False)),
            local_model_dir=local_model_dir,
        )

    def generate_text(self, prompt: str) -> str:
        """Generate plain text with the selected backend."""

        if self.backend == "vllm":
            return self._generate_text_vllm(prompt)
        return self._generate_text_transformers(prompt)

    def generate_json(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Generate JSON and parse it into a dict if possible."""

        text = self.generate_text(prompt)
        return self._parse_json_object(text)

    def _generate_text_vllm(self, prompt: str) -> str:
        """Call a vLLM OpenAI-compatible endpoint."""

        url = self.vllm_base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": str(prompt)}],
            "temperature": 0,
            "top_p": 1,
        }
        response = requests.post(url, json=payload, timeout=self.timeout_sec)
        response.raise_for_status()
        data = response.json()
        return str((((data.get("choices") or [{}])[0]).get("message") or {}).get("content") or "").strip()

    def _generate_text_transformers(self, prompt: str) -> str:
        """Run local text generation through Hugging Face transformers."""

        tokenizer, model = self._load_transformers_backend()

        try:
            import torch
        except Exception as exc:
            raise RuntimeError("PyTorch is required for the transformers LLM backend.") from exc

        messages = [{"role": "user", "content": str(prompt)}]
        generation_kwargs = {
            "max_new_tokens": int(self.max_new_tokens),
            "do_sample": bool(self.do_sample),
            "temperature": float(self.temperature),
            "top_p": float(self.top_p),
        }
        if not generation_kwargs["do_sample"]:
            generation_kwargs.pop("temperature", None)
            generation_kwargs.pop("top_p", None)

        input_length = 0
        if hasattr(tokenizer, "apply_chat_template"):
            encoded = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
            )
            model_device = _model_device(model)
            if hasattr(encoded, "to"):
                try:
                    if model_device is not None:
                        encoded = encoded.to(model_device)
                except Exception:
                    pass
            model_inputs = {"input_ids": encoded}
            input_length = int(encoded.shape[-1])
        else:
            encoded_map = tokenizer(str(prompt), return_tensors="pt")
            model_inputs = {}
            model_device = _model_device(model)
            for key, value in dict(encoded_map).items():
                if hasattr(value, "to"):
                    try:
                        if model_device is not None:
                            value = value.to(model_device)
                    except Exception:
                        pass
                model_inputs[key] = value
            input_ids = model_inputs.get("input_ids")
            input_length = int(input_ids.shape[-1]) if input_ids is not None else 0

        with torch.inference_mode():
            output_ids = model.generate(**model_inputs, **generation_kwargs)

        generated = output_ids[0][input_length:]
        text = tokenizer.decode(generated, skip_special_tokens=True)
        return str(text or "").strip()

    def _load_transformers_backend(self) -> Tuple[Any, Any]:
        """Load and cache tokenizer/model for the local transformers backend."""

        cache_key = self.local_model_dir or self.model_id
        cached = _TRANSFORMERS_CACHE.get(cache_key)
        if cached is not None:
            return cached

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:
            raise RuntimeError("transformers backend dependencies are not installed.") from exc

        source = self.local_model_dir or self.model_id
        load_kwargs: Dict[str, Any] = {"trust_remote_code": True}
        dtype_name = str(self.transformers_dtype or "").strip().lower()
        dtype_map = {
            "float16": getattr(torch, "float16", None),
            "fp16": getattr(torch, "float16", None),
            "bfloat16": getattr(torch, "bfloat16", None),
            "bf16": getattr(torch, "bfloat16", None),
            "float32": getattr(torch, "float32", None),
            "fp32": getattr(torch, "float32", None),
        }
        chosen_dtype = dtype_map.get(dtype_name)
        if chosen_dtype is not None:
            load_kwargs["torch_dtype"] = chosen_dtype

        device_map = str(self.transformers_device_map or "").strip()
        if device_map:
            load_kwargs["device_map"] = device_map

        tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(source, **load_kwargs)

        if bool(self.transformers_compile) and hasattr(torch, "compile"):
            try:
                model = torch.compile(model)
            except Exception:
                pass

        _TRANSFORMERS_CACHE[cache_key] = (tokenizer, model)
        return tokenizer, model

    @staticmethod
    def _parse_json_object(text: str) -> Optional[Dict[str, Any]]:
        """Parse a JSON object from plain text or a fenced code block."""

        raw = str(text or "").strip()
        if not raw:
            return None

        candidates = [raw]
        if "```" in raw:
            stripped = raw.replace("```json", "```").replace("```JSON", "```")
            parts = [part.strip() for part in stripped.split("```") if part.strip()]
            candidates.extend(parts)

        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidates.append(raw[start : end + 1])

        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
            except Exception:
                continue
            if isinstance(parsed, dict):
                return parsed
        return None
