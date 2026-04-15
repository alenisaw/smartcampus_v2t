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
from typing import Callable, Dict, List, Optional, Tuple

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


def _annotation_similarity_merge(
    anns: List[Annotation],
    *,
    gap_tolerance: float,
    should_merge: Callable[[Annotation, Annotation], bool],
    on_merge: Optional[Callable[[Annotation, Annotation], None]] = None,
) -> List[Annotation]:
    """Merge ordered adjacent annotations with a caller-provided merge predicate."""

    if not anns:
        return []

    ordered = sorted(anns, key=lambda ann: ann.start_sec)
    first = ordered[0]
    current = Annotation(
        video_id=first.video_id,
        start_sec=float(first.start_sec),
        end_sec=float(first.end_sec),
        description=strip_prefix(first.description),
        extra=dict(getattr(first, "extra", {}) or {}),
        anomaly_flag=bool(getattr(first, "anomaly_flag", False)),
        anomaly_confidence=float(getattr(first, "anomaly_confidence", 0.0) or 0.0),
        anomaly_notes=list(getattr(first, "anomaly_notes", []) or []),
    )
    current.extra = current.extra or {}
    current.extra["merged_from"] = list((current.extra.get("merged_from") or [0]))

    merged: List[Annotation] = []
    for src_index, nxt in enumerate(ordered[1:], start=1):
        gap = float(nxt.start_sec) - float(current.end_sec)
        if gap <= float(gap_tolerance) and should_merge(current, nxt):
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
            if on_merge is not None:
                on_merge(current, nxt)
            continue

        merged.append(current)
        current = Annotation(
            video_id=nxt.video_id,
            start_sec=float(nxt.start_sec),
            end_sec=float(nxt.end_sec),
            description=strip_prefix(nxt.description),
            extra=dict(getattr(nxt, "extra", {}) or {}),
            anomaly_flag=bool(getattr(nxt, "anomaly_flag", False)),
            anomaly_confidence=float(getattr(nxt, "anomaly_confidence", 0.0) or 0.0),
            anomaly_notes=list(getattr(nxt, "anomaly_notes", []) or []),
        )
        current.extra = current.extra or {}
        current.extra["merged_from"] = list((current.extra.get("merged_from") or [src_index]))

    merged.append(current)
    return merged


def smooth_annotations(
    anns: List[Annotation],
    sim_threshold: float = 0.7,
    gap_tolerance: float = 1.0,
) -> List[Annotation]:
    """Merge adjacent annotations when time gap is small and text is similar."""

    def _should_merge(current: Annotation, nxt: Annotation) -> bool:
        return text_sim(current.description, nxt.description) >= float(sim_threshold)

    return _annotation_similarity_merge(
        anns,
        gap_tolerance=float(gap_tolerance),
        should_merge=_should_merge,
    )


def semantic_merge_annotations(
    anns: List[Annotation],
    *,
    embeddings: List[Optional["np.ndarray"]],
    tau: float,
    gap_tolerance: float = 1.0,
) -> List[Annotation]:
    """Merge adjacent annotations using cosine similarity over caller-provided embeddings."""

    import numpy as np

    if len(anns) != len(embeddings):
        raise ValueError("anns and embeddings must have the same length")

    def _normalized(vec: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if vec is None:
            return None
        arr = np.asarray(vec, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            return None
        norm = float(np.linalg.norm(arr))
        if norm <= 0.0:
            return None
        return arr / norm

    ordered = sorted(zip(anns, embeddings), key=lambda item: item[0].start_sec)
    if not ordered:
        return []

    normalized_inputs = [_normalized(vec) for _, vec in ordered]
    ordered_anns = [ann for ann, _ in ordered]

    first = ordered_anns[0]
    current = Annotation(
        video_id=first.video_id,
        start_sec=float(first.start_sec),
        end_sec=float(first.end_sec),
        description=strip_prefix(first.description),
        extra=dict(getattr(first, "extra", {}) or {}),
        anomaly_flag=bool(getattr(first, "anomaly_flag", False)),
        anomaly_confidence=float(getattr(first, "anomaly_confidence", 0.0) or 0.0),
        anomaly_notes=list(getattr(first, "anomaly_notes", []) or []),
    )
    current.extra = current.extra or {}
    current.extra["merged_from"] = list((current.extra.get("merged_from") or [0]))
    current_vec = normalized_inputs[0]

    merged: List[Annotation] = []
    for src_index, nxt in enumerate(ordered_anns[1:], start=1):
        nxt_vec = normalized_inputs[src_index]
        gap = float(nxt.start_sec) - float(current.end_sec)
        similarity = float(np.dot(current_vec, nxt_vec)) if current_vec is not None and nxt_vec is not None else -1.0
        if gap <= float(gap_tolerance) and similarity >= float(tau):
            current.end_sec = float(nxt.end_sec)
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
            if current_vec is None:
                current_vec = nxt_vec
            elif nxt_vec is not None:
                merged_vec = current_vec + nxt_vec
                merged_norm = float(np.linalg.norm(merged_vec))
                current_vec = (merged_vec / merged_norm) if merged_norm > 0.0 else current_vec
            continue

        merged.append(current)
        current = Annotation(
            video_id=nxt.video_id,
            start_sec=float(nxt.start_sec),
            end_sec=float(nxt.end_sec),
            description=strip_prefix(nxt.description),
            extra=dict(getattr(nxt, "extra", {}) or {}),
            anomaly_flag=bool(getattr(nxt, "anomaly_flag", False)),
            anomaly_confidence=float(getattr(nxt, "anomaly_confidence", 0.0) or 0.0),
            anomaly_notes=list(getattr(nxt, "anomaly_notes", []) or []),
        )
        current.extra = current.extra or {}
        current.extra["merged_from"] = list((current.extra.get("merged_from") or [src_index]))
        current_vec = nxt_vec

    merged.append(current)
    return merged


def _clip_len(paths: List[str]) -> int:
    """Return a stable non-zero clip frame count for packing math."""

    return max(1, int(len(paths)))


def _pack_clip_batches(
    clips: List[List[str]],
    *,
    max_batch_clips: int,
    max_batch_frames: int,
    frame_tolerance: int,
) -> List[List[int]]:
    """Pack clip indices into near-homogeneous batches under a frame budget."""

    limit_clips = max(1, int(max_batch_clips))
    limit_frames = max(0, int(max_batch_frames))
    tolerance = max(0, int(frame_tolerance))

    ordered = sorted(range(len(clips)), key=lambda index: (_clip_len(clips[index]), -index), reverse=True)
    batches: List[List[int]] = []
    current: List[int] = []
    current_min = 0
    current_max = 0

    def flush() -> None:
        nonlocal current, current_min, current_max
        if current:
            batches.append(list(current))
        current = []
        current_min = 0
        current_max = 0

    for index in ordered:
        clip_frames = _clip_len(clips[index])
        if not current:
            current = [index]
            current_min = clip_frames
            current_max = clip_frames
            continue

        next_min = min(current_min, clip_frames)
        next_max = max(current_max, clip_frames)
        next_count = len(current) + 1
        padded_frames = next_max * next_count
        exceeds_tolerance = tolerance > 0 and (next_max - next_min) > tolerance
        exceeds_clip_count = next_count > limit_clips
        exceeds_frame_budget = limit_frames > 0 and padded_frames > limit_frames

        if exceeds_tolerance or exceeds_clip_count or exceeds_frame_budget:
            flush()
            current = [index]
            current_min = clip_frames
            current_max = clip_frames
            continue

        current.append(index)
        current_min = next_min
        current_max = next_max

    flush()
    return batches


def _batch_payload_summary(clips: List[List[str]], batches: List[List[int]]) -> Dict[str, float]:
    """Summarize padded-vs-real frame use for one packed batch plan."""

    if not batches:
        return {
            "planned_batches": 0,
            "planned_clips_avg": 0.0,
            "planned_clips_max": 0,
            "planned_frames_real": 0,
            "planned_frames_padded": 0,
            "planned_padding_ratio": 0.0,
        }

    real_total = 0
    padded_total = 0
    clip_counts = [len(batch) for batch in batches]
    for batch in batches:
        lengths = [_clip_len(clips[index]) for index in batch]
        real_total += int(sum(lengths))
        padded_total += int((max(lengths) if lengths else 0) * len(lengths))

    return {
        "planned_batches": int(len(batches)),
        "planned_clips_avg": float(sum(clip_counts) / len(clip_counts)),
        "planned_clips_max": int(max(clip_counts) if clip_counts else 0),
        "planned_frames_real": int(real_total),
        "planned_frames_padded": int(padded_total),
        "planned_padding_ratio": float((padded_total - real_total) / padded_total) if padded_total > 0 else 0.0,
    }


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

    def _describe_batch_with_backoff(
        self,
        *,
        batch_indices: List[int],
        clips: List[List[str]],
        clip_timestamps: List[Tuple[float, float]],
        prompt: str,
        outputs_by_index: List[Optional[Annotation]],
        batch_runtime: Dict[str, float],
    ) -> None:
        """Run one planned batch, splitting it recursively when the device runs out of memory."""

        batch_paths = [[Path(path) for path in clips[index]] for index in batch_indices]
        batch_ts = [clip_timestamps[index] for index in batch_indices]
        lengths = [_clip_len(clips[index]) for index in batch_indices]

        try:
            raw_outputs = self.backend.describe_clips_batch(
                batch_frame_paths=batch_paths,
                prompt=prompt,
            )
        except Exception as exc:
            if len(batch_indices) <= 1 or not self.backend.is_memory_pressure_error(exc):
                raise
            midpoint = max(1, len(batch_indices) // 2)
            batch_runtime["oom_retry_count"] += 1
            self.backend.release_inference_cache()
            self._describe_batch_with_backoff(
                batch_indices=batch_indices[:midpoint],
                clips=clips,
                clip_timestamps=clip_timestamps,
                prompt=prompt,
                outputs_by_index=outputs_by_index,
                batch_runtime=batch_runtime,
            )
            self._describe_batch_with_backoff(
                batch_indices=batch_indices[midpoint:],
                clips=clips,
                clip_timestamps=clip_timestamps,
                prompt=prompt,
                outputs_by_index=outputs_by_index,
                batch_runtime=batch_runtime,
            )
            return

        batch_runtime["actual_batches"] += 1
        batch_runtime["actual_batch_clips_total"] += len(batch_indices)
        batch_runtime["actual_batch_clips_max"] = max(batch_runtime["actual_batch_clips_max"], len(batch_indices))
        batch_runtime["actual_frames_real"] += int(sum(lengths))
        batch_runtime["actual_frames_padded"] += int((max(lengths) if lengths else 0) * len(lengths))

        for index, (start, end), raw_output in zip(batch_indices, batch_ts, raw_outputs):
            parsed = parse_clip_observation(str(raw_output or ""))
            outputs_by_index[index] = Annotation(
                video_id="",
                start_sec=float(start),
                end_sec=float(end),
                description=str(parsed.get("description") or ""),
                extra={"observation_contract": "clip_observation_v1"},
                anomaly_flag=bool(parsed.get("anomaly_flag", False)),
                anomaly_confidence=float(parsed.get("anomaly_confidence", 0.0) or 0.0),
                anomaly_notes=list(parsed.get("anomaly_notes") or []),
            )

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

        prompt = build_clip_observation_prompt(self.cfg.model.language)
        batch_size = max(1, int(getattr(self.cfg.model, "batch_size", 1)))
        max_batch_clips = max(1, int(getattr(self.cfg.model, "max_batch_clips", batch_size) or batch_size))
        max_batch_frames = int(getattr(self.cfg.model, "max_batch_frames", 0) or 0)
        batch_frame_tolerance = max(0, int(getattr(self.cfg.model, "batch_frame_tolerance", 2) or 0))

        generation_started = time.perf_counter()
        planned_batches = _pack_clip_batches(
            clips,
            max_batch_clips=min(batch_size, max_batch_clips),
            max_batch_frames=max_batch_frames,
            frame_tolerance=batch_frame_tolerance,
        )
        batch_plan_summary = _batch_payload_summary(clips, planned_batches)
        outputs_by_index: List[Optional[Annotation]] = [None] * len(clips)
        batch_runtime: Dict[str, float] = {
            "actual_batches": 0,
            "actual_batch_clips_total": 0,
            "actual_batch_clips_max": 0,
            "actual_frames_real": 0,
            "actual_frames_padded": 0,
            "oom_retry_count": 0,
        }

        for batch_indices in planned_batches:
            self._describe_batch_with_backoff(
                batch_indices=list(batch_indices),
                clips=clips,
                clip_timestamps=clip_timestamps,
                prompt=prompt,
                outputs_by_index=outputs_by_index,
                batch_runtime=batch_runtime,
            )

        observations: List[Annotation] = []
        for index, item in enumerate(outputs_by_index):
            if item is None:
                raise RuntimeError(f"Missing VLM output for clip index {index}")
            item.video_id = video_id
            observations.append(item)

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
                    "max_batch_clips": int(max_batch_clips),
                    "max_batch_frames": int(getattr(self.cfg.model, "max_batch_frames", 0) or 0),
                    "batch_frame_tolerance": int(batch_frame_tolerance),
                    "attn_implementation": str(getattr(self.cfg.model, "attn_implementation", "auto")),
                    "resolved_attn_implementation": str(getattr(self.backend, "resolved_attn_implementation", "unknown")),
                    "torch_compile": bool(getattr(self.cfg.runtime, "torch_compile", False)),
                    "torch_compile_mode": str(getattr(self.cfg.runtime, "torch_compile_mode", "")),
                    "autocast_infer": bool(getattr(self.cfg.runtime, "autocast_infer", True)),
                    "dtype": str(getattr(self.cfg.model, "dtype", "")),
                    "target_fps": float(getattr(self.cfg.video, "target_fps", 0.0) or 0.0),
                    "analysis_fps": float(getattr(self.cfg.video, "analysis_fps", 0.0) or 0.0),
                    "base_generation_lang": "en",
                    "observation_contract": "clip_observation_v1",
                },
                "batching": {
                    **batch_plan_summary,
                    "actual_batches": int(batch_runtime["actual_batches"]),
                    "actual_clips_avg": float(batch_runtime["actual_batch_clips_total"] / max(1, batch_runtime["actual_batches"])),
                    "actual_clips_max": int(batch_runtime["actual_batch_clips_max"]),
                    "actual_frames_real": int(batch_runtime["actual_frames_real"]),
                    "actual_frames_padded": int(batch_runtime["actual_frames_padded"]),
                    "actual_padding_ratio": float(
                        (batch_runtime["actual_frames_padded"] - batch_runtime["actual_frames_real"])
                        / batch_runtime["actual_frames_padded"]
                    )
                    if batch_runtime["actual_frames_padded"] > 0
                    else 0.0,
                    "oom_retry_count": int(batch_runtime["oom_retry_count"]),
                    "model_frames_per_sec": float(num_frames / model_time) if model_time > 0 else 0.0,
                    "model_clips_per_sec": float(num_clips / model_time) if model_time > 0 else 0.0,
                },
                "anomaly_stats": {
                    "flagged_clips": int(flagged_count),
                    "flag_rate": float(flagged_count / max(1, len(observations))),
                    "confidence_mean": float(confidence_mean),
                },
            },
        )
        return merged, observations, metrics
