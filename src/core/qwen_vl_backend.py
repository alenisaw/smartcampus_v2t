# src/core/qwen_vl_backend.py

"""
Backend wrapper around Qwen3-VL vision-language model.

Loads model and processor, builds chat-style inputs for clips or text prompts,
and supports single-clip and batched generation.
"""

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
    ) -> None:
        if generation_config is None:
            generation_config = QwenVLGenerationConfig()

        self.model_name_or_path = model_name_or_path
        self.device = device
        self.dtype = dtype
        self.generation_config = generation_config

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
            model_kwargs["dtype"] = hf_dtype
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
            device = getattr(model_cfg, "device", "auto")
        else:
            model_name = getattr(cfg, "text_model_path", None)
            max_new_tokens = getattr(cfg, "text_max_new_tokens", 256)
            temperature = 0.7
            top_p = 0.8
            top_k = 40
            do_sample = True
            repetition_penalty = 1.0
            dtype = "auto"
            device = "auto"

        if model_name is None:
            raise ValueError("Model path for Qwen3-VL is not set in PipelineConfig")

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
            image_path = str(Path(p).resolve())
            content.append(
                {
                    "type": "image",
                    "image": image_path,
                }
            )

        content.append({"type": "text", "text": prompt})

        return [
            {
                "role": "user",
                "content": content,
            }
        ]

    def _build_messages_for_text(self, prompt: str) -> List[dict]:
        return [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                ],
            }
        ]

    def _run_chat(
            self,
            messages,
            gen_cfg: Optional[QwenVLGenerationConfig] = None,
    ) -> List[str]:
        if gen_cfg is None:
            gen_cfg = self.generation_config

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            padding=True,
        ).to(self.model.device)


        self.model.generation_config.max_new_tokens = gen_cfg.max_new_tokens
        self.model.generation_config.temperature = gen_cfg.temperature
        self.model.generation_config.top_p = gen_cfg.top_p
        self.model.generation_config.top_k = gen_cfg.top_k
        self.model.generation_config.do_sample = gen_cfg.do_sample
        self.model.generation_config.repetition_penalty = gen_cfg.repetition_penalty
        self.model.generation_config.use_cache = True

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]

        output_texts = self.processor.batch_decode(
            generated_ids_trimmed,
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
            messages = self._build_messages_for_clip(frame_paths, prompt)
            conversations.append(messages)

        outputs = self._run_chat(conversations, gen_cfg)
        return outputs

    def generate_text(
        self,
        prompt: str,
        gen_cfg: Optional[QwenVLGenerationConfig] = None,
    ) -> str:
        messages = self._build_messages_for_text(prompt)
        outputs = self._run_chat(messages, gen_cfg)
        return outputs[0] if outputs else ""
