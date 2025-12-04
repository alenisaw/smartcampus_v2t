# src/core/qwen_vl_backend.py

"""
Backend wrapper around Qwen3-VL vision-language model.

This module:
- loads Qwen3-VL model and processor from Hugging Face
- prepares chat-style inputs for image sequences (clips) and text-only prompts
- runs generation and returns plain text responses
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from src.pipeline.pipeline_config import PipelineConfig


@dataclass
class QwenVLGenerationConfig:
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 40
    do_sample: bool = True
    repetition_penalty: float = 1.0


class QwenVLBackend:
    def __init__(
        self,
        model_name_or_path: str,
        device: str = "auto",
        dtype: str = "auto",
        generation_config: Optional[QwenVLGenerationConfig] = None,
    ) -> None:
        if generation_config is None:
            generation_config = QwenVLGenerationConfig()

        self.model_name_or_path = model_name_or_path
        self.device = device
        self.dtype = dtype
        self.generation_config = generation_config

        if device == "auto":
            device_map = "auto"
        elif device in {"cuda", "cpu"}:
            device_map = {device: 0} if device == "cuda" else {"": "cpu"}
        else:
            device_map = "auto"

        torch_dtype: Any
        if dtype == "auto":
            torch_dtype = "auto"
        elif dtype.lower() in {"bf16", "bfloat16"}:
            torch_dtype = torch.bfloat16
        elif dtype.lower() in {"fp16", "float16", "half"}:
            torch_dtype = torch.float16
        else:
            torch_dtype = "auto"

        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name_or_path,
            dtype=torch_dtype,
            device_map=device_map,
        )
        self.processor = AutoProcessor.from_pretrained(model_name_or_path)
        self.model.eval()

    @classmethod
    def from_pipeline_config(cls, cfg: PipelineConfig) -> "QwenVLBackend":
        model_cfg = getattr(cfg, "model", None)

        if model_cfg is not None:
            model_name = (
                getattr(model_cfg, "name_or_path", None)
                or getattr(model_cfg, "path", None)
                or getattr(model_cfg, "model_name", None)
                or getattr(model_cfg, "model_name_or_path", None)
            )
            max_new_tokens = getattr(model_cfg, "max_new_tokens", 256)
            temperature = getattr(model_cfg, "temperature", 0.7)
            top_p = getattr(model_cfg, "top_p", 0.8)
            top_k = getattr(model_cfg, "top_k", 40)
            do_sample = getattr(model_cfg, "do_sample", True)
            repetition_penalty = getattr(model_cfg, "repetition_penalty", 1.0)
            dtype = getattr(model_cfg, "dtype", "auto")
        else:
            model_name = getattr(cfg, "text_model_path", None)
            max_new_tokens = getattr(cfg, "text_max_new_tokens", 256)
            temperature = 0.7
            top_p = 0.8
            top_k = 40
            do_sample = True
            repetition_penalty = 1.0
            dtype = "auto"

        if model_name is None:
            raise ValueError("Model path for Qwen3-VL is not set in PipelineConfig")

        device = getattr(cfg, "device", "auto")

        gen_cfg = QwenVLGenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=do_sample,
            repetition_penalty=repetition_penalty,
        )

        return cls(
            model_name_or_path=model_name,
            device=device,
            dtype=dtype,
            generation_config=gen_cfg,
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
            p = Path(p)
            image_path = str(p.resolve())
            content.append(
                {
                    "type": "image",
                    "image": image_path,  # ← без file://
                }
            )

        content.append({"type": "text", "text": prompt})

        messages = [
            {
                "role": "user",
                "content": content,
            }
        ]
        return messages

    def _build_messages_for_text(self, prompt: str) -> List[dict]:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        return messages

    def _run_chat(self, messages: List[dict], gen_cfg: Optional[QwenVLGenerationConfig] = None) -> str:
        if gen_cfg is None:
            gen_cfg = self.generation_config

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=gen_cfg.max_new_tokens,
            temperature=gen_cfg.temperature,
            top_p=gen_cfg.top_p,
            top_k=gen_cfg.top_k,
            do_sample=gen_cfg.do_sample,
            repetition_penalty=gen_cfg.repetition_penalty,
        )

        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]

        output_texts = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        if not output_texts:
            return ""
        return output_texts[0]

    def describe_clip(
        self,
        frame_paths: Sequence[Path],
        prompt: str,
        gen_cfg: Optional[QwenVLGenerationConfig] = None,
    ) -> str:
        messages = self._build_messages_for_clip(frame_paths, prompt)
        return self._run_chat(messages, gen_cfg)

    def generate_text(
        self,
        prompt: str,
        gen_cfg: Optional[QwenVLGenerationConfig] = None,
    ) -> str:
        messages = self._build_messages_for_text(prompt)
        return self._run_chat(messages, gen_cfg)
