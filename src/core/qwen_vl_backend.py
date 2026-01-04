# src/core/qwen_vl_backend.py
"""
Backend wrapper around Qwen3-VL vision-language model.

Loads model and processor, builds chat-style inputs for clips or text prompts,
and supports single-clip and batched generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from src.pipeline.pipeline_config import PipelineConfig


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
    ) -> None:
        if generation_config is None:
            generation_config = QwenVLGenerationConfig()

        self.model_name_or_path = model_name_or_path
        self.device = device
        self.dtype = dtype
        self.generation_config = generation_config
        self.autocast_infer = bool(autocast_infer)

        if device not in {"auto", "cuda", "cpu"}:
            device = "auto"

        if dtype == "auto":
            hf_dtype: Any = None
        elif dtype.lower() in {"bf16", "bfloat16"}:
            hf_dtype = torch.bfloat16
        elif dtype.lower() in {"fp16", "float16", "half"}:
            hf_dtype = torch.float16
        else:
            hf_dtype = None

        model_kwargs: dict = {}
        if hf_dtype is not None:
            model_kwargs["torch_dtype"] = hf_dtype

        if device == "cuda":
            model_kwargs["device_map"] = "auto"

        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name_or_path,
            **model_kwargs,
        )
        self.processor = AutoProcessor.from_pretrained(model_name_or_path)
        self.model.eval()

        if device in {"cuda", "cpu"} and not hasattr(self.model, "hf_device_map"):
            self.model.to(device)

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
        )

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

    def _run_chat(
        self,
        messages,
        gen_cfg: Optional[QwenVLGenerationConfig] = None,
    ) -> List[str]:
        if gen_cfg is None:
            gen_cfg = self.generation_config

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

        if hasattr(model, "hf_device_map"):
            pass
        else:
            inputs = inputs.to(model.device, non_blocking=True)

        generate_kwargs = dict(
            max_new_tokens=gen_cfg.max_new_tokens,
            temperature=gen_cfg.temperature,
            top_p=gen_cfg.top_p,
            top_k=gen_cfg.top_k,
            do_sample=gen_cfg.do_sample,
            repetition_penalty=gen_cfg.repetition_penalty,
            use_cache=True,
        )

        with torch.inference_mode():
            if getattr(model, "device", None) is not None and model.device.type == "cuda" and self.autocast_infer:
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
