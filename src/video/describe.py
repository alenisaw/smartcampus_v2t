# src/video/describe.py
"""
Observation-only VLM pipeline for SmartCampus V2T.

Purpose:
- Generate strict clip observations from prepared clips.
- Keep the VLM layer limited to visible description plus anomaly signals.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Dict, List, Tuple

from src.core.config import PipelineConfig
from src.core.types import Annotation, RunMetrics
from src.core.vlm_backend import QwenVLBackend
from src.video.io import preprocess_video
from src.video.prompts import build_clip_observation_prompt, parse_clip_observation, strip_prefix

_TOKEN_CLEAN_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)


def norm_tokens(text: str) -> List[str]:
    """Normalize observation text into simple searchable tokens."""

    normalized = strip_prefix((text or "").lower())
    normalized = _TOKEN_CLEAN_RE.sub(" ", normalized)
    return [token for token in normalized.split() if len(token) > 2]


def text_sim(a: str, b: str) -> float:
    """Compute lightweight overlap similarity between two descriptions."""

    tokens_a = set(norm_tokens(a))
    tokens_b = set(norm_tokens(b))
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / float(min(len(tokens_a), len(tokens_b)))


def _merge_notes(left: List[str], right: List[str]) -> List[str]:
    """Merge anomaly notes while preserving order and uniqueness."""

    merged: List[str] = []
    for note in list(left or []) + list(right or []):
        text = str(note or "").strip()
        if text and text not in merged:
            merged.append(text)
    return merged[:4]


def smooth_annotations(
    anns: List[Annotation],
    sim_threshold: float = 0.7,
    gap_tolerance: float = 1.0,
) -> List[Annotation]:
    """Merge adjacent annotations when time gap is small and text is similar."""

    if not anns:
        return []

    ordered = sorted(anns, key=lambda ann: ann.start_sec)
    first = ordered[0]
    current = Annotation(
        video_id=first.video_id,
        start_sec=float(first.start_sec),
        end_sec=float(first.end_sec),
        description=strip_prefix(first.description),
        extra={"merged_from": [0]},
        anomaly_flag=bool(getattr(first, "anomaly_flag", False)),
        anomaly_confidence=float(getattr(first, "anomaly_confidence", 0.0) or 0.0),
        anomaly_notes=list(getattr(first, "anomaly_notes", []) or []),
    )

    merged: List[Annotation] = []
    for src_index, nxt in enumerate(ordered[1:], start=1):
        gap = float(nxt.start_sec) - float(current.end_sec)
        similarity = text_sim(current.description, nxt.description)
        if gap <= gap_tolerance and similarity >= sim_threshold:
            current.end_sec = float(nxt.end_sec)
            current.extra = current.extra or {}
            current.extra.setdefault("merged_from", []).append(src_index)
            current.anomaly_flag = bool(current.anomaly_flag or getattr(nxt, "anomaly_flag", False))
            current.anomaly_confidence = max(
                float(current.anomaly_confidence or 0.0),
                float(getattr(nxt, "anomaly_confidence", 0.0) or 0.0),
            )
            current.anomaly_notes = _merge_notes(
                list(current.anomaly_notes or []),
                list(getattr(nxt, "anomaly_notes", []) or []),
            )
            continue

        merged.append(current)
        current = Annotation(
            video_id=nxt.video_id,
            start_sec=float(nxt.start_sec),
            end_sec=float(nxt.end_sec),
            description=strip_prefix(nxt.description),
            extra={"merged_from": [src_index]},
            anomaly_flag=bool(getattr(nxt, "anomaly_flag", False)),
            anomaly_confidence=float(getattr(nxt, "anomaly_confidence", 0.0) or 0.0),
            anomaly_notes=list(getattr(nxt, "anomaly_notes", []) or []),
        )

    merged.append(current)
    return merged


def _bucket_indices_by_len(nested: List[List[str]]) -> Dict[int, List[int]]:
    """Group clip indices by frame-count length for efficient batch inference."""

    buckets: Dict[int, List[int]] = {}
    for index, seq in enumerate(nested):
        buckets.setdefault(len(seq), []).append(index)
    return buckets


class VideoToTextPipeline:
    """Observation-only VLM pipeline used by the worker runtime."""

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.backend = QwenVLBackend.from_pipeline_config(cfg)

    def release(self) -> None:
        """Release the resident VLM backend when it is not needed anymore."""

        backend = getattr(self, "backend", None)
        if backend is None or not hasattr(backend, "release"):
            return
        backend.release()

    def preprocess_video(self, video_path: str | Path):
        """Expose preprocessing through the video-layer entrypoint."""

        return preprocess_video(video_path, self.cfg)

    def run(
        self,
        video_id: str,
        video_duration_sec: float,
        clips: List[List[str]],
        clip_timestamps: List[Tuple[float, float]],
        preprocess_time_sec: float = 0.0,
    ):
        """Run clip observation generation and consolidation without summary generation."""

        if not clips:
            metrics = RunMetrics(
                video_id=video_id,
                language=str(getattr(self.cfg.model, "language", "") or ""),
                video_duration_sec=float(video_duration_sec),
                num_frames=0,
                num_clips=0,
                avg_clip_duration_sec=0.0,
                preprocess_time_sec=float(preprocess_time_sec),
                model_time_sec=0.0,
                postprocess_time_sec=0.0,
                total_time_sec=float(preprocess_time_sec),
                extra={"reason": "no_clips"},
            )
            return [], [], metrics

        if len(clips) != len(clip_timestamps):
            raise ValueError("clips and clip_timestamps must match in length")

        num_clips = len(clips)
        num_frames = sum(len(clip) for clip in clips)
        avg_clip_duration = float(sum((end - start) for start, end in clip_timestamps) / len(clip_timestamps))
        clip_lengths = [len(clip) for clip in clips]

        observations: List[Annotation] = []
        prompt = build_clip_observation_prompt(self.cfg.model.language)
        batch_size = max(1, int(getattr(self.cfg.model, "batch_size", 1)))
        max_batch_frames = int(getattr(self.cfg.model, "max_batch_frames", 0) or 0)

        generation_started = time.perf_counter()
        buckets = _bucket_indices_by_len(clips)

        for clip_len in sorted(buckets.keys()):
            indices = buckets[clip_len]
            effective_batch = batch_size
            if max_batch_frames > 0:
                effective_batch = max(1, min(batch_size, max_batch_frames // max(1, int(clip_len))))
            for batch_start in range(0, len(indices), effective_batch):
                batch_indices = indices[batch_start : batch_start + effective_batch]
                batch_paths = [[Path(path) for path in clips[index]] for index in batch_indices]
                batch_ts = [clip_timestamps[index] for index in batch_indices]

                raw_outputs = self.backend.describe_clips_batch(
                    batch_frame_paths=batch_paths,
                    prompt=prompt,
                )

                for (start, end), raw_output in zip(batch_ts, raw_outputs):
                    parsed = parse_clip_observation(str(raw_output or ""))
                    observations.append(
                        Annotation(
                            video_id=video_id,
                            start_sec=float(start),
                            end_sec=float(end),
                            description=str(parsed.get("description") or ""),
                            extra={"observation_contract": "clip_observation_v1"},
                            anomaly_flag=bool(parsed.get("anomaly_flag", False)),
                            anomaly_confidence=float(parsed.get("anomaly_confidence", 0.0) or 0.0),
                            anomaly_notes=list(parsed.get("anomaly_notes") or []),
                        )
                    )

        model_time = time.perf_counter() - generation_started

        post_started = time.perf_counter()
        merged = smooth_annotations(observations)
        post_time = time.perf_counter() - post_started

        flagged_count = sum(1 for ann in observations if bool(getattr(ann, "anomaly_flag", False)))
        confidence_mean = (
            float(sum(float(getattr(ann, "anomaly_confidence", 0.0) or 0.0) for ann in observations) / len(observations))
            if observations
            else 0.0
        )

        total_time = float(preprocess_time_sec) + float(model_time) + float(post_time)
        metrics = RunMetrics(
            video_id=video_id,
            language=str(getattr(self.cfg.model, "language", "") or ""),
            video_duration_sec=float(video_duration_sec),
            num_frames=int(num_frames),
            num_clips=int(num_clips),
            avg_clip_duration_sec=float(avg_clip_duration),
            preprocess_time_sec=float(preprocess_time_sec),
            model_time_sec=float(model_time),
            postprocess_time_sec=float(post_time),
            total_time_sec=float(total_time),
            extra={
                "clip_stats": {
                    "num_clips": int(num_clips),
                    "num_frames": int(num_frames),
                    "frames_min": int(min(clip_lengths)) if clip_lengths else 0,
                    "frames_max": int(max(clip_lengths)) if clip_lengths else 0,
                    "frames_avg": float(sum(clip_lengths) / len(clip_lengths)) if clip_lengths else 0.0,
                },
                "pipeline": {
                    "batch_size": int(batch_size),
                    "max_batch_frames": int(getattr(self.cfg.model, "max_batch_frames", 0) or 0),
                    "attn_implementation": str(getattr(self.cfg.model, "attn_implementation", "auto")),
                    "torch_compile": bool(getattr(self.cfg.runtime, "torch_compile", False)),
                    "torch_compile_mode": str(getattr(self.cfg.runtime, "torch_compile_mode", "")),
                    "autocast_infer": bool(getattr(self.cfg.runtime, "autocast_infer", True)),
                    "dtype": str(getattr(self.cfg.model, "dtype", "")),
                    "base_generation_lang": "en",
                    "observation_contract": "clip_observation_v1",
                },
                "anomaly_stats": {
                    "flagged_clips": int(flagged_count),
                    "flag_rate": float(flagged_count / max(1, len(observations))),
                    "confidence_mean": float(confidence_mean),
                },
            },
        )
        return merged, observations, metrics
