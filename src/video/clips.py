# src/video/clips.py
"""
Clip-window construction helpers for SmartCampus V2T.

Purpose:
- Build sliding-window clip batches from prepared frame metadata.
- Keep clip and keyframe selection separate from VLM caption generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

try:
    import cv2
except Exception:
    cv2 = None


@dataclass(frozen=True)
class _FrameLike:
    """Compact frame view used by clip-building helpers."""

    path: str
    timestamp_sec: float


def _pick_keyframe_path(paths_list: List[str], policy: str) -> Optional[str]:
    """Pick one representative keyframe path for a clip window."""

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


def _downsample_frames_by_fps(frames: List[_FrameLike], analysis_fps: float) -> List[_FrameLike]:
    """Reduce clip frame density to the target analysis FPS while preserving endpoints."""

    if len(frames) <= 1 or float(analysis_fps) <= 0.0:
        return list(frames)

    min_interval = 1.0 / float(analysis_fps)
    selected: List[_FrameLike] = [frames[0]]
    last_kept_ts = float(frames[0].timestamp_sec)

    for frame in frames[1:-1]:
        if float(frame.timestamp_sec) - last_kept_ts + 1e-9 >= min_interval:
            selected.append(frame)
            last_kept_ts = float(frame.timestamp_sec)

    tail = frames[-1]
    if str(tail.path) == str(selected[-1].path):
        return selected
    if float(tail.timestamp_sec) - last_kept_ts + 1e-9 >= min_interval:
        selected.append(tail)
    else:
        selected[-1] = tail
    return selected


def build_clips_from_video_meta(
    video_meta: Any,
    window_sec: float,
    stride_sec: float,
    min_clip_frames: int,
    max_clip_frames: int,
    analysis_fps: float = 0.0,
    keyframe_policy: str = "middle",
    return_keyframes: bool = False,
):
    """Build sliding-window clips from preprocessed frame metadata."""

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

        window_frames = frames[left:right]
        if analysis_fps > 0:
            window_frames = _downsample_frames_by_fps(window_frames, float(analysis_fps))

        count = len(window_frames)
        if count >= int(min_clip_frames):
            window_paths = [str(item.path) for item in window_frames]
            if len(window_paths) > int(max_clip_frames):
                step = len(window_paths) / float(max_clip_frames)
                indices = [min(len(window_paths) - 1, int(index * step)) for index in range(int(max_clip_frames))]
                window_paths = [window_paths[index] for index in indices]
            last_ts = float(window_frames[-1].timestamp_sec) if window_frames else window_end
            clips.append(window_paths)
            clip_timestamps.append((float(current_time), float(last_ts)))
            clip_keyframes.append(_pick_keyframe_path(window_paths, keyframe_policy))

        current_time += float(stride_sec)

    if return_keyframes:
        return clips, clip_timestamps, clip_keyframes
    return clips, clip_timestamps
