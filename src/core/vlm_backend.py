# src/core/vlm_backend.py
"""
Vision-language backend wrapper for SmartCampus V2T.

Purpose:
- Load and drive the Qwen-VL captioning model used by the pipeline.
- Provide single-clip, batched, and prompt-based generation helpers.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
import importlib.util
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from src.core.config import PipelineConfig


@dataclass
class QwenVLGenerationConfig:
    max_new_tokens: int = 96
    temperature: float = 0.2
    top_p: float = 0.9
    top_k: int = 40
    do_sample: bool = False
    repetition_penalty: float = 1.0


class QwenVLBackend:
    def __init__(
        self,
        model_name_or_path: str,
        device: str = "auto",
        dtype: str = "auto",
        generation_config: Optional[QwenVLGenerationConfig] = None,
        autocast_infer: bool = True,
        attn_implementation: str = "auto",
        torch_compile: bool = False,
        torch_compile_mode: str = "reduce-overhead",
        torch_compile_fullgraph: bool = False,
        lazy_load: bool = True,
    ) -> None:
        if generation_config is None:
            generation_config = QwenVLGenerationConfig()

        self.model_name_or_path = model_name_or_path
        self.device = device
        self.dtype = dtype
        self.generation_config = generation_config
        self.autocast_infer = bool(autocast_infer)
        self.attn_implementation = str(attn_implementation or "auto").strip().lower()
        self.torch_compile = bool(torch_compile)
        self.torch_compile_mode = str(torch_compile_mode or "reduce-overhead")
        self.torch_compile_fullgraph = bool(torch_compile_fullgraph)
        self.lazy_load = bool(lazy_load)
        self.model = None
        self.processor = None
        self.resolved_attn_implementation = "unknown"

        if device not in {"auto", "cuda", "cpu"}:
            device = "auto"
        self._resolved_device = device
        self._model_kwargs = self._build_model_kwargs(device=device, dtype=dtype)
        if not self.lazy_load:
            self._ensure_loaded()

    @staticmethod
    def _load_model(model_name_or_path: str, model_kwargs: Dict[str, Any]) -> Any:
        """Load the VLM while tolerating old transformers dtype argument names."""

        attempts = [dict(model_kwargs)]
        if "dtype" in model_kwargs:
            fallback_kwargs = dict(model_kwargs)
            fallback_kwargs["torch_dtype"] = fallback_kwargs.pop("dtype")
            attempts.append(fallback_kwargs)

        last_error: Optional[Exception] = None
        for kwargs in attempts:
            try:
                return AutoModelForImageTextToText.from_pretrained(model_name_or_path, **kwargs)
            except (TypeError, ValueError) as exc:
                last_error = exc
                if "attn_implementation" not in kwargs:
                    continue
                trimmed = dict(kwargs)
                trimmed.pop("attn_implementation", None)
                try:
                    return AutoModelForImageTextToText.from_pretrained(model_name_or_path, **trimmed)
                except (TypeError, ValueError) as inner_exc:
                    last_error = inner_exc
                    continue
        if last_error is not None:
            raise last_error
        raise RuntimeError(f"Failed to load VLM model: {model_name_or_path}")

    def _build_model_kwargs(self, *, device: str, dtype: str) -> Dict[str, Any]:
        """Build stable HF load kwargs for the VLM bundle."""

        if dtype == "auto":
            hf_dtype: Any = None
        elif str(dtype).lower() in {"bf16", "bfloat16"}:
            hf_dtype = torch.bfloat16
        elif str(dtype).lower() in {"fp16", "float16", "half"}:
            hf_dtype = torch.float16
        else:
            hf_dtype = None

        model_kwargs: Dict[str, Any] = {}
        if hf_dtype is not None:
            model_kwargs["dtype"] = hf_dtype
        if device == "cuda":
            model_kwargs["device_map"] = "auto"

        attn_impl = self._pick_attn_implementation(self.attn_implementation)
        self.resolved_attn_implementation = str(attn_impl or "unknown")
        if attn_impl:
            model_kwargs["attn_implementation"] = attn_impl
        return model_kwargs

    def _apply_generation_defaults(self) -> None:
        """Normalize generation config to deterministic defaults when sampling is disabled."""

        generation_state = getattr(self.model, "generation_config", None)
        if generation_state is None or bool(self.generation_config.do_sample):
            return
        try:
            generation_state.do_sample = False
            if hasattr(generation_state, "temperature"):
                generation_state.temperature = 1.0
            if hasattr(generation_state, "top_p"):
                generation_state.top_p = 1.0
            if hasattr(generation_state, "top_k"):
                generation_state.top_k = 50
        except Exception:
            return

    def _ensure_loaded(self) -> None:
        """Load the VLM model and processor only when they are first needed."""

        if self.model is not None and self.processor is not None:
            return

        model = self._load_model(self.model_name_or_path, dict(self._model_kwargs))
        processor = AutoProcessor.from_pretrained(self.model_name_or_path)
        model.eval()
        self.model = model
        self.processor = processor
        self._apply_generation_defaults()
        if self._resolved_device in {"cuda", "cpu"} and not hasattr(self.model, "hf_device_map"):
            self.model.to(self._resolved_device)
        self._maybe_compile_model()

    def release(self) -> None:
        """Unload the resident VLM objects and free device memory when possible."""

        model = self.model
        processor = self.processor
        self.model = None
        self.processor = None

        try:
            if model is not None and hasattr(model, "cpu"):
                model.cpu()
        except Exception:
            pass

        del model
        del processor
        gc.collect()
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                if hasattr(torch.cuda, "ipc_collect"):
                    torch.cuda.ipc_collect()
        except Exception:
            pass

    @staticmethod
    def is_memory_pressure_error(exc: Exception) -> bool:
        """Detect common CUDA and allocator OOM failures for batch backoff."""

        text = str(exc or "").strip().lower()
        return any(
            marker in text
            for marker in (
                "out of memory",
                "cuda error: out of memory",
                "cuda out of memory",
                "cudnn_status_not_supported",
                "not enough memory",
                "allocate memory",
            )
        )

    @staticmethod
    def release_inference_cache() -> None:
        """Free transient device allocations between retries when possible."""

        gc.collect()
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                if hasattr(torch.cuda, "ipc_collect"):
                    torch.cuda.ipc_collect()
        except Exception:
            pass

    @classmethod
    def from_pipeline_config(cls, cfg: PipelineConfig) -> "QwenVLBackend":
        model_cfg = cfg.model

        model_name = model_cfg.model_name_or_path
        if model_name is None:
            raise ValueError("Model path for Qwen3-VL is not set in PipelineConfig")

        gen_cfg = QwenVLGenerationConfig(
            max_new_tokens=int(model_cfg.max_new_tokens),
            temperature=float(model_cfg.temperature),
            top_p=float(model_cfg.top_p),
            top_k=int(model_cfg.top_k),
            do_sample=bool(model_cfg.do_sample),
            repetition_penalty=float(model_cfg.repetition_penalty),
        )

        return cls(
            model_name_or_path=model_name,
            device=str(model_cfg.device),
            dtype=str(model_cfg.dtype),
            generation_config=gen_cfg,
            autocast_infer=bool(getattr(cfg.runtime, "autocast_infer", True)),
            attn_implementation=str(getattr(model_cfg, "attn_implementation", "auto")),
            torch_compile=bool(getattr(cfg.runtime, "torch_compile", False)),
            torch_compile_mode=str(getattr(cfg.runtime, "torch_compile_mode", "reduce-overhead")),
            torch_compile_fullgraph=bool(getattr(cfg.runtime, "torch_compile_fullgraph", False)),
            lazy_load=True,
        )

    @staticmethod
    def _flash_attn_available() -> bool:
        try:
            return importlib.util.find_spec("flash_attn") is not None
        except Exception:
            return False

    def _pick_attn_implementation(self, pref: str) -> Optional[str]:
        pref = (pref or "").strip().lower()
        if pref in {"flash_attention_2", "flash"}:
            return "flash_attention_2" if self._flash_attn_available() else "sdpa"
        if pref in {"sdpa", "sdp"}:
            return "sdpa"
        if pref in {"eager", "vanilla", "none"}:
            return "eager"
        if pref in {"auto", ""}:
            if self._flash_attn_available():
                return "flash_attention_2"
            return "sdpa"
        return "sdpa"

    def _maybe_compile_model(self) -> None:
        if not self.torch_compile:
            return
        if not hasattr(torch, "compile"):
            return
        if hasattr(self.model, "hf_device_map"):
            return
        try:
            self.model = torch.compile(
                self.model,
                mode=self.torch_compile_mode,
                fullgraph=self.torch_compile_fullgraph,
            )
        except Exception:
            pass

    def _build_messages_for_clip(
        self,
        frame_paths: Sequence[Path],
        prompt: str,
    ) -> List[dict]:
        if not frame_paths:
            raise ValueError("frame_paths is empty for clip description")

        content: List[dict] = []
        for p in frame_paths:
            content.append({"type": "image", "image": str(p)})

        content.append({"type": "text", "text": prompt})
        return [{"role": "user", "content": content}]

    def _build_messages_for_text(self, prompt: str) -> List[dict]:
        return [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

    def _trim_generated(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        generated_ids: torch.Tensor,
    ) -> List[torch.Tensor]:
        outs: List[torch.Tensor] = []
        bsz = int(generated_ids.shape[0])
        for i in range(bsz):
            if attention_mask is not None:
                in_len = int(attention_mask[i].sum().item())
            else:
                in_len = int(input_ids[i].shape[0])
            outs.append(generated_ids[i, in_len:])
        return outs

    def _infer_input_device(self, model: torch.nn.Module) -> torch.device:
        """
        Decide which device to put inputs on.

        - Non-sharded model: next(model.parameters()).device
        - Sharded (HF device_map="auto"): pick the first CUDA device from hf_device_map
        - Fallback: cpu
        """
        try:
            return next(model.parameters()).device
        except Exception:
            pass

        if hasattr(model, "hf_device_map"):
            try:
                for _name, dev in model.hf_device_map.items():
                    if isinstance(dev, str) and dev.startswith("cuda"):
                        return torch.device(dev)
            except Exception:
                pass

        return torch.device("cpu")

    def _move_inputs_to_device(self, inputs: Any, device: torch.device) -> Any:
        """
        Robust move for HF BatchEncoding / dict-like structures.
        """
        if inputs is None:
            return None
        if isinstance(inputs, torch.Tensor):
            return inputs.to(device, non_blocking=True)
        if isinstance(inputs, dict):
            return {k: self._move_inputs_to_device(v, device) for k, v in inputs.items()}
        # HF BatchEncoding часто умеет .to()
        if hasattr(inputs, "to"):
            try:
                return inputs.to(device, non_blocking=True)
            except TypeError:
                return inputs.to(device)
            except Exception:
                pass
        return inputs

    def _run_chat(
        self,
        messages,
        gen_cfg: Optional[QwenVLGenerationConfig] = None,
    ) -> List[str]:
        if gen_cfg is None:
            gen_cfg = self.generation_config

        processor = self.processor
        model = self.model
        self._ensure_loaded()
        processor = self.processor
        model = self.model

        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            padding=True,
        )

        # FIX: always move input tensors to the right device
        target_device = self._infer_input_device(model)
        inputs = self._move_inputs_to_device(inputs, target_device)
        if inputs.get("attention_mask") is None and inputs.get("input_ids") is not None:
            inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])

        generate_kwargs = dict(
            max_new_tokens=gen_cfg.max_new_tokens,
            do_sample=gen_cfg.do_sample,
            repetition_penalty=gen_cfg.repetition_penalty,
            use_cache=True,
        )
        if gen_cfg.do_sample:
            generate_kwargs["temperature"] = gen_cfg.temperature
            generate_kwargs["top_p"] = gen_cfg.top_p
            generate_kwargs["top_k"] = gen_cfg.top_k

        with torch.inference_mode():
            if target_device.type == "cuda" and self.autocast_infer:
                with torch.autocast(device_type="cuda"):
                    generated_ids = model.generate(**inputs, **generate_kwargs)
            else:
                generated_ids = model.generate(**inputs, **generate_kwargs)

        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")

        trimmed = self._trim_generated(input_ids=input_ids, attention_mask=attention_mask, generated_ids=generated_ids)
        output_texts = processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return output_texts

    def describe_clip(
        self,
        frame_paths: Sequence[Path],
        prompt: str,
        gen_cfg: Optional[QwenVLGenerationConfig] = None,
    ) -> str:
        messages = self._build_messages_for_clip(frame_paths, prompt)
        outputs = self._run_chat(messages, gen_cfg)
        return outputs[0] if outputs else ""

    def describe_clips_batch(
        self,
        batch_frame_paths: Sequence[Sequence[Path]],
        prompt: str,
        gen_cfg: Optional[QwenVLGenerationConfig] = None,
    ) -> List[str]:
        conversations = []
        for frame_paths in batch_frame_paths:
            conversations.append(self._build_messages_for_clip(frame_paths, prompt))
        return self._run_chat(conversations, gen_cfg)

    def generate_text(
        self,
        prompt: str,
        gen_cfg: Optional[QwenVLGenerationConfig] = None,
    ) -> str:
        messages = self._build_messages_for_text(prompt)
        outputs = self._run_chat(messages, gen_cfg)
        return outputs[0] if outputs else ""
