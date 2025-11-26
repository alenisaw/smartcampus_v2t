# src/inference.py
"""
Video-to-text inference pipeline for smartcampus_v2t.

This module provides:
- Loading preprocessed video frames from data/processed/frames/<video_name>/.
- Splitting videos into sliding windows (window_size / stride).
- Batched caption generation for all windows:
    * multiple model backends (dummy, InternVideo2, TimeSformer, VideoMAE, etc.)
    * optional feature caching (saving / loading precomputed visual embeddings)
    * ability to run in features-only mode for fast text-only inference
- Adding timestamps for each window (based on FPS).
- Saving all results into a single JSON file (one entry per window).
- Post-processing per video:
    * late fusion (concat / keep_first over fixed groups of windows)
    * caption smoothing (e.g., merging repeated captions)

This file defines both the high-level inference loop and the model abstraction
(CaptionModelBase) that allows plugging in different video transformers and LLM
heads without changing the pipeline logic.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .config import PROCESSED_DIR, CAPTIONS_PATH, ensure_directories
from .utils import setup_logging, relative_to_project

logger = logging.getLogger(__name__)

CaptionEntry = Dict[str, Any]


class CaptionModelBase:
    def __init__(
        self,
        device: str = "cuda",
        backend: str = "dummy",
        use_feature_cache: bool = False,
        features_root: Optional[Path] = None,
        features_only: bool = False,
    ) -> None:
        self.device = device
        self.backend = backend
        self.use_feature_cache = use_feature_cache
        self.features_root = features_root
        self.features_only = features_only

    def caption_batch(
        self,
        video_name: str,
        frames: Optional[Sequence[np.ndarray]],
        windows: Sequence[Tuple[int, int]],
        fps: float,
    ) -> List[Dict[str, Any]]:
        raise NotImplementedError


class DummyCaptionModel(CaptionModelBase):
    def _features_path(self, video_name: str) -> Path:
        root = self.features_root or (PROCESSED_DIR / "features")
        return root / f"{video_name}.npy"

    def _load_or_compute_features(
        self,
        video_name: str,
        frames: Optional[Sequence[np.ndarray]],
        num_frames: int,
    ) -> np.ndarray:
        path = self._features_path(video_name)
        if self.use_feature_cache and path.exists():
            logger.info("Loading cached features for %s", video_name)
            return np.load(path)

        if frames is None:
            raise RuntimeError(
                f"features_only=True but cache for {video_name} not found at {path}"
            )

        feats = np.arange(num_frames, dtype=np.float32)[:, None]

        if self.use_feature_cache:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, feats)
            logger.info("Saved features cache for %s to %s", video_name, path)

        return feats

    def caption_batch(
        self,
        video_name: str,
        frames: Optional[Sequence[np.ndarray]],
        windows: Sequence[Tuple[int, int]],
        fps: float,
    ) -> List[Dict[str, Any]]:
        num_frames = len(frames) if frames is not None else 0
        if self.features_only:
            feats = self._load_or_compute_features(video_name, None, num_frames)
        else:
            feats = self._load_or_compute_features(video_name, frames, num_frames)

        outputs: List[Dict[str, Any]] = []
        for idx, (start, end) in enumerate(windows):
            window_feats = feats[start:end]
            score = float(window_feats.mean()) if window_feats.size > 0 else 0.0
            caption = f"dummy caption {idx} for {video_name} ({end - start} frames)"
            outputs.append({"caption": caption, "score": score})
        return outputs


def create_model(
    backend: str,
    device: str,
    use_feature_cache: bool,
    features_root: Optional[Path],
    features_only: bool,
) -> CaptionModelBase:
    backend = backend.lower()
    if backend in {"dummy", "internvideo2", "timesformer", "videomae"}:
        logger.info(
            "Initializing %s backend on %s (dummy implementation for now)",
            backend,
            device,
        )
        return DummyCaptionModel(
            device=device,
            backend=backend,
            use_feature_cache=use_feature_cache,
            features_root=features_root,
            features_only=features_only,
        )
    raise ValueError(f"Unknown backend: {backend}")


def make_windows(num_frames: int, window_size: int, stride: int) -> List[Tuple[int, int]]:
    if num_frames == 0:
        return []
    windows: List[Tuple[int, int]] = []
    start = 0
    while start < num_frames:
        end = min(start + window_size, num_frames)
        windows.append((start, end))
        if end == num_frames:
            break
        start += stride
    return windows


def load_frames_from_dir(video_dir: Path) -> List[np.ndarray]:
    if not video_dir.exists():
        logger.warning("Frame directory missing: %s", video_dir)
        return []

    jpgs = sorted(video_dir.glob("frame_*.jpg"))
    npys = sorted(video_dir.glob("frame_*.npy"))
    frames: List[np.ndarray] = []

    if npys:
        for p in npys:
            frames.append(np.load(p))
    elif jpgs:
        for p in jpgs:
            img = cv2.imread(str(p))
            if img is None:
                continue
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames.append(rgb.astype(np.float32) / 255.0)
    else:
        logger.warning("No frames found in %s", video_dir)

    return frames


def collect_video_dirs(frames_root: Path, max_videos: int = 0) -> List[Path]:
    if not frames_root.exists():
        logger.warning("Frames root missing: %s", frames_root)
        return []
    dirs = sorted([p for p in frames_root.iterdir() if p.is_dir()])
    if max_videos > 0:
        dirs = dirs[:max_videos]
    return dirs


def apply_late_fusion(
    entries: List[CaptionEntry],
    strategy: str,
    group_size: int,
) -> List[CaptionEntry]:
    if strategy == "none" or group_size <= 1:
        return entries

    fused: List[CaptionEntry] = []
    n = len(entries)
    i = 0
    while i < n:
        chunk = entries[i : i + group_size]
        start_frame = chunk[0]["start_frame"]
        end_frame = chunk[-1]["end_frame"]
        start_sec = chunk[0]["start_sec"]
        end_sec = chunk[-1]["end_sec"]

        if strategy == "concat":
            caption = " ".join(c["caption"] for c in chunk)
            score_vals = [c.get("score", 0.0) for c in chunk]
            score = float(np.mean(score_vals)) if score_vals else 0.0
        elif strategy == "keep_first":
            caption = chunk[0]["caption"]
            score = chunk[0].get("score", 0.0)
        else:
            raise ValueError(f"Unknown fusion strategy: {strategy}")

        fused.append(
            {
                **chunk[0],
                "start_frame": start_frame,
                "end_frame": end_frame,
                "num_frames": end_frame - start_frame,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "caption": caption,
                "score": score,
            }
        )
        i += group_size

    return fused


def smooth_captions(
    entries: List[CaptionEntry],
    smooth_mode: str,
) -> List[CaptionEntry]:
    if smooth_mode == "none":
        return entries

    if smooth_mode == "merge_repeated":
        if not entries:
            return entries
        smoothed: List[CaptionEntry] = []
        current = dict(entries[0])
        for nxt in entries[1:]:
            if nxt["caption"] == current["caption"]:
                current["end_frame"] = nxt["end_frame"]
                current["end_sec"] = nxt["end_sec"]
                current["num_frames"] = current["end_frame"] - current["start_frame"]
                current["score"] = max(
                    float(current.get("score", 0.0)),
                    float(nxt.get("score", 0.0)),
                )
            else:
                smoothed.append(current)
                current = dict(nxt)
        smoothed.append(current)
        return smoothed

    raise ValueError(f"Unknown smooth mode: {smooth_mode}")


def run_inference_on_video(
    video_dir: Path,
    model: CaptionModelBase,
    window_size: int,
    stride: int,
    fps: float,
    max_windows: int = 0,
    fusion_strategy: str = "none",
    fusion_group_size: int = 1,
    smooth_mode: str = "none",
    features_only: bool = False,
) -> List[CaptionEntry]:
    if features_only:
        frames: Optional[List[np.ndarray]] = []
    else:
        frames = load_frames_from_dir(video_dir)

    n = len(frames)
    if n == 0:
        logger.warning("No frames for %s", video_dir)
        return []

    windows = make_windows(n, window_size, stride)
    if max_windows > 0:
        windows = windows[:max_windows]

    video_name = video_dir.name
    logger.info(
        "Video %s: %d frames, %d windows",
        video_name,
        n,
        len(windows),
    )

    batch_outputs = model.caption_batch(video_name, frames, windows, fps)

    entries: List[CaptionEntry] = []
    for idx, ((start, end), out) in enumerate(zip(windows, batch_outputs)):
        caption = out.get("caption", "")
        score = out.get("score", None)
        entry: CaptionEntry = {
            "video": video_name,
            "window_index": idx,
            "start_frame": start,
            "end_frame": end,
            "num_frames": end - start,
            "start_sec": start / fps,
            "end_sec": end / fps,
            "caption": caption,
        }
        if score is not None:
            entry["score"] = float(score)
        entries.append(entry)

    entries = apply_late_fusion(entries, fusion_strategy, fusion_group_size)
    entries = smooth_captions(entries, smooth_mode)

    return entries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Video→text inference for smartcampus_v2t.")

    parser.add_argument(
        "--frames-root",
        type=str,
        default=str(PROCESSED_DIR / "frames"),
        help="Root directory containing per-video frame folders.",
    )

    parser.add_argument(
        "--features-root",
        type=str,
        default=str(PROCESSED_DIR / "features"),
        help="Root directory for cached visual features (.npy per video).",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=str(CAPTIONS_PATH),
        help="Where to write the final JSON file.",
    )

    parser.add_argument("--window-size", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)

    parser.add_argument(
        "--max-windows-per-video",
        type=int,
        default=0,
        help="Limit number of windows per video (0 = no limit).",
    )

    parser.add_argument(
        "--max-videos",
        type=int,
        default=0,
        help="Limit number of videos to process (0 = no limit).",
    )

    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Frame rate used to convert frame indices to timestamps.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for the caption model.",
    )

    parser.add_argument(
        "--backend",
        type=str,
        default="dummy",
        choices=["dummy", "internvideo2", "timesformer", "videomae"],
        help="Caption model backend.",
    )

    parser.add_argument(
        "--use-feature-cache",
        action="store_true",
        help="Enable loading/saving cached visual features.",
    )

    parser.add_argument(
        "--features-only",
        action="store_true",
        help="Run using only cached features (no frame loading).",
    )

    parser.add_argument(
        "--fusion-strategy",
        type=str,
        default="none",
        choices=["none", "concat", "keep_first"],
        help="Late fusion strategy over fixed-size window groups.",
    )

    parser.add_argument(
        "--fusion-group-size",
        type=int,
        default=1,
        help="Number of consecutive windows to fuse together.",
    )

    parser.add_argument(
        "--smooth-mode",
        type=str,
        default="none",
        choices=["none", "merge_repeated"],
        help="Caption smoothing mode across time.",
    )

    return parser.parse_args()


def main() -> None:
    setup_logging()
    ensure_directories()
    args = parse_args()

    frames_root = Path(args.frames_root).resolve()
    features_root = Path(args.features_root).resolve()
    output_path = Path(args.output).resolve()

    logger.info("Frames root: %s", relative_to_project(frames_root))
    logger.info("Features root: %s", relative_to_project(features_root))
    logger.info("Output JSON: %s", relative_to_project(output_path))

    video_dirs = collect_video_dirs(frames_root, max_videos=args.max_videos)
    if not video_dirs:
        logger.warning("No frame folders found.")
        return

    model = create_model(
        backend=args.backend,
        device=args.device,
        use_feature_cache=args.use_feature_cache,
        features_root=features_root,
        features_only=args.features_only,
    )

    all_results: List[CaptionEntry] = []

    for video_dir in video_dirs:
        logger.info("Running inference for %s", relative_to_project(video_dir))
        video_results = run_inference_on_video(
            video_dir=video_dir,
            model=model,
            window_size=args.window_size,
            stride=args.stride,
            fps=args.fps,
            max_windows=args.max_windows_per_video,
            fusion_strategy=args.fusion_strategy,
            fusion_group_size=args.fusion_group_size,
            smooth_mode=args.smooth_mode,
            features_only=args.features_only,
        )
        all_results.extend(video_results)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    logger.info(
        "Saved %d caption entries for %d videos",
        len(all_results),
        len(video_dirs),
    )


if __name__ == "__main__":
    main()
