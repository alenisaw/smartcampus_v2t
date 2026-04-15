# src/llm/client.py
"""
LLM client for SmartCampus V2T semantic runtime.

Purpose:
- Provide a compact text and JSON generation interface for semantic stages.
- Support local transformers and OpenAI-compatible vLLM backends.
"""

from __future__ import annotations

import json
import gc
from functools import lru_cache
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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


def _looks_like_local_model_ref(value: str) -> bool:
    """Detect local-path-like model refs that cannot be sent to a remote OpenAI API as-is."""

    text = str(value or "").strip()
    if not text:
        return False
    path = Path(text)
    if path.exists():
        return True
    if text.startswith(".") or text.startswith("/") or text.startswith("\\"):
        return True
    return (":" in text) or ("\\" in text)


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
    vllm_served_model_name: str = ""
    vllm_api_key: str = ""

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
            vllm_served_model_name=str(getattr(cfg.llm, "vllm_served_model_name", "") or ""),
            vllm_api_key=str(getattr(cfg.llm, "vllm_api_key", "") or ""),
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

    def cache_key(self) -> str:
        """Return the stable local cache key for the transformers backend."""

        return str(self.local_model_dir or self.model_id)

    def is_loaded(self) -> bool:
        """Return whether the local transformers backend is currently resident."""

        return self.cache_key() in _TRANSFORMERS_CACHE

    def release(self) -> None:
        """Unload the resident local transformers backend for this client."""

        cached = _TRANSFORMERS_CACHE.pop(self.cache_key(), None)
        if cached is None:
            return

        tokenizer, model = cached
        try:
            if model is not None and hasattr(model, "cpu"):
                model.cpu()
        except Exception:
            pass

        del tokenizer
        del model
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                if hasattr(torch.cuda, "ipc_collect"):
                    torch.cuda.ipc_collect()
        except Exception:
            pass

    def _generate_text_vllm(self, prompt: str) -> str:
        """Call a vLLM OpenAI-compatible endpoint."""

        url = self.vllm_base_url.rstrip("/") + "/chat/completions"
        model_name = self._resolve_vllm_model_name()
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": str(prompt)}],
            "temperature": float(self.temperature if self.do_sample else 0.0),
            "top_p": float(self.top_p if self.do_sample else 1.0),
            "max_tokens": int(self.max_new_tokens),
        }
        response = self._session().post(
            url,
            json=payload,
            headers=self._vllm_headers(),
            timeout=self.timeout_sec,
        )
        response.raise_for_status()
        data = response.json()
        return str((((data.get("choices") or [{}])[0]).get("message") or {}).get("content") or "").strip()

    def is_vllm_ready(self) -> bool:
        """Return whether the configured vLLM endpoint is reachable and exposes at least one model."""

        try:
            return bool(self._fetch_vllm_models())
        except Exception:
            return False

    def _resolve_vllm_model_name(self) -> str:
        """Resolve the model name that should be sent to the OpenAI-compatible vLLM API."""

        configured_name = str(self.vllm_served_model_name or "").strip()
        if configured_name:
            return configured_name

        model_id = str(self.model_id or "").strip()
        if model_id and not _looks_like_local_model_ref(model_id):
            return model_id

        models = self._fetch_vllm_models()
        if not models:
            raise RuntimeError(f"No models exposed by vLLM endpoint: {self.vllm_base_url}")
        return str(models[0])

    def _vllm_headers(self) -> Dict[str, str]:
        """Build request headers for the OpenAI-compatible vLLM API."""

        headers = {"Content-Type": "application/json"}
        token = str(self.vllm_api_key or "").strip()
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers

    def _session(self) -> requests.Session:
        """Return a lightweight shared HTTP session for vLLM requests."""

        return _vllm_session(self.vllm_base_url)

    def _fetch_vllm_models(self) -> List[str]:
        """Fetch the list of served model IDs from the configured vLLM endpoint."""

        return _vllm_models(
            self.vllm_base_url,
            self.timeout_sec,
            str(self.vllm_api_key or ""),
        )

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
        if float(self.timeout_sec or 0) > 0:
            generation_kwargs["max_time"] = float(self.timeout_sec)
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
                return_dict=True,
            )
            model_inputs = {}
            model_device = _model_device(model)
            encoded_map = dict(encoded) if isinstance(encoded, dict) else {}
            for key, value in encoded_map.items():
                if hasattr(value, "to"):
                    try:
                        if model_device is not None:
                            value = value.to(model_device)
                    except Exception:
                        pass
                model_inputs[key] = value
            input_ids = model_inputs.get("input_ids")
            attention_mask = model_inputs.get("attention_mask")
            if attention_mask is None and input_ids is not None:
                attention_mask = torch.ones_like(input_ids)
                model_inputs["attention_mask"] = attention_mask
            if attention_mask is not None:
                input_length = int(attention_mask[0].sum().item())
            else:
                input_length = int(input_ids.shape[-1]) if input_ids is not None else 0
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
            attention_mask = model_inputs.get("attention_mask")
            if attention_mask is None and input_ids is not None:
                attention_mask = torch.ones_like(input_ids)
                model_inputs["attention_mask"] = attention_mask
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
            load_kwargs["dtype"] = chosen_dtype

        device_map = str(self.transformers_device_map or "").strip()
        if device_map:
            load_kwargs["device_map"] = device_map

        tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=True)
        try:
            model = AutoModelForCausalLM.from_pretrained(source, **load_kwargs)
        except TypeError:
            if "dtype" not in load_kwargs:
                raise
            fallback_kwargs = dict(load_kwargs)
            fallback_kwargs["torch_dtype"] = fallback_kwargs.pop("dtype")
            model = AutoModelForCausalLM.from_pretrained(source, **fallback_kwargs)

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


@lru_cache(maxsize=8)
def _vllm_session(_base_url: str) -> requests.Session:
    """Create one reusable requests session per base URL."""

    return requests.Session()


@lru_cache(maxsize=16)
def _vllm_models(base_url: str, timeout_sec: int, api_key: str) -> List[str]:
    """Fetch and cache the served model names from a vLLM OpenAI-compatible endpoint."""

    headers: Dict[str, str] = {}
    token = str(api_key or "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"

    response = _vllm_session(base_url).get(
        base_url.rstrip("/") + "/models",
        headers=headers,
        timeout=max(3, int(timeout_sec or 30)),
    )
    response.raise_for_status()
    payload = response.json()
    data = payload.get("data") or []
    out: List[str] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        model_id = str(item.get("id") or "").strip()
        if model_id:
            out.append(model_id)
    return out
