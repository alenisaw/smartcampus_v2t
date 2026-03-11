# src/pipeline/video_to_text.py
"""
Video-to-text pipeline for SmartCampus V2T.

Purpose:
- Generate canonical English clip captions from preprocessed frames.
- Smooth adjacent segments and produce optional grounded summary text.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import cv2
except Exception:
    cv2 = None

from src.core.vlm_backend import QwenVLBackend
from src.core.types import Annotation, RunMetrics
from src.pipeline.config import PipelineConfig

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_TIMED_ITEM_RE = re.compile(r"\b(.+?)\s*\(\s*(\d+:\d{2})\s*-\s*(\d+:\d{2})\s*\)\b")
_WS_RE = re.compile(r"\s+")
_LIST_PREFIX_RE = re.compile(r"^\s*[-*]+\s*")
_ENUM_PREFIX_RE = re.compile(r"^\s*\(?\d+\)?[.)]\s*")
_GENERIC_OPENERS_RE = re.compile(
    r"^(in the frame|in the image|in the video|we can see|there is|there are|visible is|shown is)\s*[:,\-]?\s*",
    flags=re.IGNORECASE,
)
_SUMMARY_KEYS: List[Tuple[str, str]] = [
    ("- Scene type:", "unknown"),
    ("- People density:", "low"),
    ("- Motion type:", "stable"),
    ("- Anomalies:", "none"),
    ("- Risk class:", "normal"),
]


def build_prompt(lang: str) -> str:
    """Return the canonical clip-description prompt.

    `lang` is kept for compatibility with old callers, but base generation is now
    always English. RU/KZ remain translation outputs, not direct VLM generations.
    """

    _ = lang
    return (
        "Describe this CCTV segment in 1-2 short natural English sentences. "
        "Start directly with the observed action. Avoid openers like 'In the frame' or 'We can see'. "
        "Use only visible facts about people, vehicles, objects, motion, and scene changes. "
        "Do not guess location type, roles, identity, intent, emotion, or cause. "
        "Do not use classification labels, safety categories, or metric words. "
        "If nothing unusual happens, describe ordinary activity plainly."
    )


def build_global_summary_prompt(
    lang: str,
    video_id: str,
    duration_sec: float,
    merged_anns: List[Annotation],
) -> str:
    """Build the grounded summary prompt for the merged English annotations."""

    _ = lang
    header = [
        "You are a CCTV analyst.",
        "Use only the segment descriptions below. Do not invent facts.",
        "First write 1-2 natural English sentences describing what happens overall and how activity changes over time.",
        "Then output exactly 5 lines, each starting with '- ', using this template:",
        "- Scene type: 1-3 words or 'unknown'",
        "- People density: none / low / medium / high",
        "- Motion type: none / weak / stable / increasing / chaotic",
        "- Anomalies: 1-3 timed items like 'event (0:01-0:10)' or 'none'",
        "- Risk class: normal / suspicious / dangerous",
        "",
        f"Video: {video_id}",
        f"Duration: ~{int(duration_sec)} seconds",
        "",
        "Segments:",
    ]
    body = [
        f"[{format_ts(ann.start_sec)} - {format_ts(ann.end_sec)}] {strip_prefix(ann.description)}"
        for ann in merged_anns
    ]
    return "\n".join(header + body)


def format_ts(sec: float) -> str:
    """Format a second offset as M:SS."""

    minutes = int(sec // 60)
    seconds = int(sec % 60)
    return f"{minutes}:{seconds:02d}"


def strip_prefix(text: str) -> str:
    """Drop timeline prefixes like `[0:00 - 0:04]` from captions."""

    return re.sub(r"^\[[0-9:\s\-]+\]\s*", "", text or "")


def _collapse_spaces(text: str) -> str:
    """Collapse all whitespace runs into single spaces."""

    normalized = (text or "").replace("\n", " ").replace("\r", " ").replace("\t", " ")
    return _WS_RE.sub(" ", normalized).strip()


def _remove_list_prefixes(text: str) -> str:
    """Drop bullet and numbered-list prefixes from one line."""

    out = _LIST_PREFIX_RE.sub("", text or "")
    out = _ENUM_PREFIX_RE.sub("", out)
    return out.strip()


def _strip_generic_openers(text: str) -> str:
    """Remove generic, non-informative lead-ins from model output."""

    return _GENERIC_OPENERS_RE.sub("", text or "").strip()


def _force_max_sentences(text: str, max_sentences: int = 2) -> str:
    """Clamp free text to at most N sentences."""

    raw = (text or "").strip()
    if not raw:
        return raw
    parts = [part.strip() for part in _SENT_SPLIT_RE.split(raw) if part.strip()]
    kept = parts[: max(1, int(max_sentences))] if parts else [raw]
    out = _collapse_spaces(" ".join(kept))
    if out and out[-1] not in ".!?":
        out += "."
    return out


def _dedupe_repeated_words(text: str, max_repeat: int = 2) -> str:
    """Remove pathological local word repetition from VLM output."""

    words = (text or "").split()
    out: List[str] = []
    run_word: Optional[str] = None
    run_len = 0
    for word in words:
        key = word.lower()
        if key == run_word:
            run_len += 1
            if run_len >= max_repeat:
                continue
        else:
            run_word = key
            run_len = 0
        out.append(word)
    merged = " ".join(out).strip()
    if merged and merged[-1] not in ".!?":
        merged += "."
    return merged


def sanitize_clip_text(text: str, lang: str) -> str:
    """Normalize one raw clip caption into canonical English text."""

    _ = lang
    normalized = strip_prefix(text)
    normalized = _collapse_spaces(normalized)
    normalized = _strip_generic_openers(normalized)
    normalized = _remove_list_prefixes(normalized)
    normalized = _force_max_sentences(normalized, max_sentences=2)
    normalized = _dedupe_repeated_words(normalized, max_repeat=2)
    return normalized


def _extract_first_sentences(text: str, count: int) -> str:
    """Extract the first N sentences from a text block."""

    normalized = _collapse_spaces(text)
    parts = [part.strip() for part in _SENT_SPLIT_RE.split(normalized) if part.strip()]
    out = " ".join(parts[:count]).strip() if parts else normalized.strip()
    if out and out[-1] not in ".!?":
        out += "."
    return out


def _pick_lines_starting(lines: List[str], prefix: str) -> Optional[str]:
    """Find the first line that starts with the requested prefix."""

    needle = prefix.lower()
    for line in lines:
        if line.lower().startswith(needle):
            return line
    return None


def _extract_after_prefix(line: str, prefix: str) -> str:
    """Return the suffix of a line after a known prefix."""

    if line.lower().startswith(prefix.lower()):
        return line[len(prefix) :].strip()
    return line.strip()


def _normalize_timed_anomalies(raw: str) -> str:
    """Normalize anomaly line content into compact timed entries."""

    if not raw:
        return "none"
    low = raw.strip().lower()
    if low in {"none", "none.", "no"}:
        return "none"

    items = _TIMED_ITEM_RE.findall(raw)
    if not items:
        return "none"

    cleaned: List[str] = []
    for label, start, end in items[:3]:
        compact_label = _collapse_spaces(label)
        compact_label = re.sub(r"[;]+$", "", compact_label).strip()
        compact_label = re.sub(r"^[-*]+\s*", "", compact_label).strip()
        if compact_label:
            cleaned.append(f"{compact_label} ({start}-{end})")
    return "; ".join(cleaned) if cleaned else "none"


def _strip_metric_fragments(short_text: str) -> str:
    """Remove metric-line fragments that leak into the short summary text."""

    normalized = _collapse_spaces(short_text or "")
    if not normalized:
        return normalized

    low = normalized.lower()
    cut_pos: Optional[int] = None
    for prefix, _default in _SUMMARY_KEYS:
        prefix_low = prefix.lower()
        for candidate in (prefix_low, prefix_low.replace("- ", ""), f" {prefix_low}"):
            pos = low.find(candidate)
            if pos != -1:
                cut_pos = pos if cut_pos is None else min(cut_pos, pos)
    if cut_pos is not None:
        normalized = normalized[:cut_pos].strip()

    normalized = re.sub(r"(short summary[:\-]?\s*)", "", normalized, flags=re.IGNORECASE).strip()
    normalized = _strip_generic_openers(normalized)
    normalized = _force_max_sentences(normalized, max_sentences=2)
    return normalized


def sanitize_global_summary(text: str, lang: str) -> str:
    """Normalize global summary output into one short summary plus 5 metric lines."""

    _ = lang
    raw = text or ""
    lines = [line.strip() for line in raw.splitlines() if line.strip()]

    short_label = "Short summary:"
    short_line = _pick_lines_starting(lines, short_label)
    short_text = _extract_after_prefix(short_line, short_label) if short_line else ""
    short_text = _extract_first_sentences(raw, 2) if not short_text else _collapse_spaces(short_text)
    short_text = _strip_metric_fragments(short_text)

    found: Dict[str, str] = {}
    for line in lines:
        if not line.startswith("-"):
            continue
        for key, default in _SUMMARY_KEYS:
            if line.lower().startswith(key.lower()):
                found[key] = _collapse_spaces(line[len(key) :].strip()) or default

    def pick_density(value: str) -> str:
        low = (value or "").lower()
        if "none" in low or low == "no":
            return "none"
        if "high" in low or "many" in low or "crowd" in low:
            return "high"
        if "med" in low:
            return "medium"
        return "low"

    def pick_motion(value: str) -> str:
        low = (value or "").lower()
        if "chaot" in low:
            return "chaotic"
        if "increas" in low or "escalat" in low:
            return "increasing"
        if "weak" in low or "light" in low:
            return "weak"
        if "none" in low or low == "no":
            return "none"
        return "stable"

    def pick_risk(value: str) -> str:
        low = (value or "").lower()
        if "danger" in low or "critical" in low:
            return "dangerous"
        if "susp" in low or "attention" in low or "warn" in low:
            return "suspicious"
        return "normal"

    def pick_anomalies(value: str) -> str:
        compact = _collapse_spaces(value or "")
        if not compact:
            return "none"
        low = compact.lower()
        if low in {"none", "none.", "no"}:
            return "none"
        compact = re.sub(r"^\s*[-*]+\s*", "", compact).strip()
        return compact[:200] if compact else "none"

    timed_line = _pick_lines_starting(lines, "Timed anomalies:")
    timed_raw = _extract_after_prefix(timed_line, "Timed anomalies:") if timed_line else ""
    timed_norm = _normalize_timed_anomalies(timed_raw)

    out_lines = [f"{short_label} {short_text}".strip()]
    for index, (key, default) in enumerate(_SUMMARY_KEYS):
        value = found.get(key, default)
        if index == 1:
            value = pick_density(value)
        elif index == 2:
            value = pick_motion(value)
        elif index == 3:
            value = pick_anomalies(value)
            if value == "none" and timed_norm != "none":
                value = timed_norm
        elif index == 4:
            value = pick_risk(value)
        out_lines.append(f"{key} {value}".strip())
    return "\n".join(out_lines).strip()


def norm_tokens(text: str) -> List[str]:
    """Normalize caption text into simple searchable tokens."""

    normalized = strip_prefix((text or "").lower())
    normalized = re.sub(r"[^\w\s]", " ", normalized, flags=re.UNICODE)
    return [token for token in normalized.split() if len(token) > 2]


def text_sim(a: str, b: str) -> float:
    """Compute lightweight overlap similarity between two captions."""

    tokens_a = set(norm_tokens(a))
    tokens_b = set(norm_tokens(b))
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / float(min(len(tokens_a), len(tokens_b)))


def smooth_annotations(
    anns: List[Annotation],
    sim_threshold: float = 0.7,
    gap_tolerance: float = 1.0,
) -> List[Annotation]:
    """Merge adjacent annotations when the time gap is small and text is similar."""

    if not anns:
        return []

    ordered = sorted(anns, key=lambda ann: ann.start_sec)
    merged: List[Annotation] = []
    current = Annotation(
        video_id=ordered[0].video_id,
        start_sec=float(ordered[0].start_sec),
        end_sec=float(ordered[0].end_sec),
        description=strip_prefix(ordered[0].description),
        extra={"merged_from": [0]},
    )

    for src_index, nxt in enumerate(ordered[1:], start=1):
        gap = float(nxt.start_sec) - float(current.end_sec)
        similarity = text_sim(current.description, nxt.description)
        if gap <= gap_tolerance and similarity >= sim_threshold:
            current.end_sec = float(nxt.end_sec)
            current.extra = current.extra or {}
            current.extra.setdefault("merged_from", []).append(src_index)
            continue

        merged.append(current)
        current = Annotation(
            video_id=nxt.video_id,
            start_sec=float(nxt.start_sec),
            end_sec=float(nxt.end_sec),
            description=strip_prefix(nxt.description),
            extra={"merged_from": [src_index]},
        )

    merged.append(current)
    return merged


def _bucket_indices_by_len(nested: List[List[str]]) -> Dict[int, List[int]]:
    """Group clip indices by frame-count length for efficient batch inference."""

    buckets: Dict[int, List[int]] = {}
    for index, seq in enumerate(nested):
        buckets.setdefault(len(seq), []).append(index)
    return buckets


@dataclass(frozen=True)
class _FrameLike:
    """Compact frame view used by clip-building helpers."""

    path: str
    timestamp_sec: float


def build_clips_from_video_meta(
    video_meta: Any,
    window_sec: float,
    stride_sec: float,
    min_clip_frames: int,
    max_clip_frames: int,
    keyframe_policy: str = "middle",
    return_keyframes: bool = False,
):
    """Build sliding-window clips from preprocessed frame metadata."""

    def _pick_keyframe_path(paths_list: List[str], policy: str) -> Optional[str]:
        if not paths_list:
            return None
        mode = str(policy or "middle").strip().lower()
        if mode in {"first", "start"}:
            return str(paths_list[0])
        if mode in {"last", "end"}:
            return str(paths_list[-1])
        if mode in {"middle", "mid", ""}:
            return str(paths_list[len(paths_list) // 2])
        if mode in {"sharpest", "max_sharpness"} and cv2 is not None:
            best_index = len(paths_list) // 2
            best_score = -1.0
            for index, frame_path in enumerate(paths_list):
                try:
                    image = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
                    if image is None:
                        continue
                    score = float(cv2.Laplacian(image, cv2.CV_64F).var())
                    if score > best_score:
                        best_score = score
                        best_index = index
                except Exception:
                    continue
            return str(paths_list[best_index])
        return str(paths_list[len(paths_list) // 2])

    frames_raw = getattr(video_meta, "frames", None) or []
    if not frames_raw:
        if return_keyframes:
            return [], [], []
        return [], []

    frames = [
        _FrameLike(
            path=str(getattr(frame, "path", "")),
            timestamp_sec=float(getattr(frame, "timestamp_sec", 0.0)),
        )
        for frame in frames_raw
    ]
    frames = sorted(frames, key=lambda item: item.timestamp_sec)
    timestamps = [float(item.timestamp_sec) for item in frames]
    paths = [str(item.path) for item in frames]
    duration = float(getattr(video_meta, "duration_sec", 0.0) or 0.0)

    clips: List[List[str]] = []
    clip_timestamps: List[Tuple[float, float]] = []
    clip_keyframes: List[Optional[str]] = []

    if window_sec <= 0 or stride_sec <= 0 or duration <= 0:
        if return_keyframes:
            return clips, clip_timestamps, clip_keyframes
        return clips, clip_timestamps

    total_frames = len(frames)
    left = 0
    right = 0
    current_time = 0.0

    while current_time < duration + 1e-6:
        window_end = min(current_time + float(window_sec), duration)

        while left < total_frames and timestamps[left] < current_time:
            left += 1
        if right < left:
            right = left
        while right < total_frames and timestamps[right] <= window_end:
            right += 1

        count = right - left
        if count >= int(min_clip_frames):
            window_paths = paths[left:right]
            if len(window_paths) > int(max_clip_frames):
                step = len(window_paths) / float(max_clip_frames)
                indices = [min(len(window_paths) - 1, int(index * step)) for index in range(int(max_clip_frames))]
                window_paths = [window_paths[index] for index in indices]
            last_ts = timestamps[right - 1] if right - 1 >= left else window_end
            clips.append(window_paths)
            clip_timestamps.append((float(current_time), float(last_ts)))
            clip_keyframes.append(_pick_keyframe_path(window_paths, keyframe_policy))

        current_time += float(stride_sec)

    if return_keyframes:
        return clips, clip_timestamps, clip_keyframes
    return clips, clip_timestamps


class VideoToTextPipeline:
    """Canonical English clip-caption pipeline used by the worker runtime."""

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.backend = QwenVLBackend.from_pipeline_config(cfg)

    def run(
        self,
        video_id: str,
        video_duration_sec: float,
        clips: List[List[str]],
        clip_timestamps: List[Tuple[float, float]],
        preprocess_time_sec: float = 0.0,
    ):
        """Run clip captioning, smoothing, and optional global summary generation."""

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
            return [], metrics

        if len(clips) != len(clip_timestamps):
            raise ValueError("clips and clip_timestamps must match in length")

        num_clips = len(clips)
        num_frames = sum(len(clip) for clip in clips)
        avg_clip_duration = float(sum((end - start) for start, end in clip_timestamps) / len(clip_timestamps))
        clip_lengths = [len(clip) for clip in clips]

        annotations: List[Annotation] = []
        clip_prompt = build_prompt(self.cfg.model.language)
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
                batch_paths: List[List[Path]] = [[Path(path) for path in clips[index]] for index in batch_indices]
                batch_ts = [clip_timestamps[index] for index in batch_indices]

                texts = self.backend.describe_clips_batch(
                    batch_frame_paths=batch_paths,
                    prompt=clip_prompt,
                )
                texts = [sanitize_clip_text(text, lang="en") for text in texts]

                for (start, end), text in zip(batch_ts, texts):
                    annotations.append(
                        Annotation(
                            video_id=video_id,
                            start_sec=float(start),
                            end_sec=float(end),
                            description=text,
                            extra=None,
                        )
                    )

        model_time = time.perf_counter() - generation_started

        post_started = time.perf_counter()
        merged = smooth_annotations(annotations)
        post_time = time.perf_counter() - post_started

        global_summary: Optional[str] = None
        try:
            if merged:
                summary_prompt = build_global_summary_prompt(
                    lang="en",
                    video_id=video_id,
                    duration_sec=float(video_duration_sec),
                    merged_anns=merged[:10],
                )
                global_summary = self.backend.generate_text(summary_prompt)
                if global_summary:
                    global_summary = sanitize_global_summary(global_summary, lang="en")
        except Exception:
            global_summary = None

        extra: Dict[str, Any] = {
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
            },
        }
        if global_summary:
            extra["global_summary"] = global_summary

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
            extra=extra or None,
        )
        return merged, metrics
