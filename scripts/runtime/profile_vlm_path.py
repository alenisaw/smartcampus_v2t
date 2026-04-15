"""
Profile the preprocess -> clip build -> VLM observation path for a single video.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.runtime import load_pipeline_config
from src.video.clips import build_clips_from_video_meta
from src.video.describe import VideoToTextPipeline


def _clip_stats(clips: list[list[str]]) -> Dict[str, Any]:
    lengths = [len(item) for item in clips]
    total = sum(lengths)
    return {
        "num_clips": len(clips),
        "num_frames": int(total),
        "frames_min": int(min(lengths)) if lengths else 0,
        "frames_max": int(max(lengths)) if lengths else 0,
        "frames_avg": float(total / len(lengths)) if lengths else 0.0,
    }


def _metrics_to_dict(metrics: Any) -> Dict[str, Any]:
    if isinstance(metrics, dict):
        return dict(metrics)
    dump = getattr(metrics, "model_dump", None)
    if callable(dump):
        payload = dump()
        if isinstance(payload, dict):
            return payload
    return dict(getattr(metrics, "__dict__", {}) or {})


def _top_level_batching(metrics: Dict[str, Any]) -> Dict[str, Any]:
    extra = metrics.get("extra") if isinstance(metrics.get("extra"), dict) else {}
    return dict(extra.get("batching") or {}) if isinstance(extra.get("batching"), dict) else {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile SmartCampus VLM path on a local video.")
    parser.add_argument("--video", required=True, help="Path to a local video file.")
    parser.add_argument("--profile", default="main", help="Runtime profile name.")
    parser.add_argument("--variant", default="", help="Optional runtime variant.")
    parser.add_argument("--device", default="", help="Optional model device override.")
    parser.add_argument("--limit-clips", type=int, default=0, help="Optional cap for processed clips.")
    parser.add_argument("--output", default="", help="Optional path to write the JSON report.")
    args = parser.parse_args()

    video_path = Path(args.video).expanduser().resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cfg = load_pipeline_config(profile=args.profile, variant=(args.variant or None))
    if str(args.device or "").strip():
        cfg.model.device = str(args.device).strip()

    pipeline = VideoToTextPipeline(cfg)
    try:
        preprocess_started = time.perf_counter()
        video_meta = pipeline.preprocess_video(video_path)
        preprocess_elapsed = time.perf_counter() - preprocess_started

        clip_build_started = time.perf_counter()
        clips, clip_timestamps, _clip_keyframes = build_clips_from_video_meta(
            video_meta=video_meta,
            window_sec=cfg.clips.window_sec,
            stride_sec=cfg.clips.stride_sec,
            min_clip_frames=cfg.clips.min_clip_frames,
            max_clip_frames=cfg.clips.max_clip_frames,
            analysis_fps=float(getattr(cfg.video, "analysis_fps", 0.0) or 0.0),
            keyframe_policy=str(getattr(cfg.clips, "keyframe_policy", "middle")),
            return_keyframes=True,
        )
        clip_build_elapsed = time.perf_counter() - clip_build_started

        if int(args.limit_clips) > 0:
            limit = int(args.limit_clips)
            clips = list(clips[:limit])
            clip_timestamps = list(clip_timestamps[:limit])

        inference_started = time.perf_counter()
        merged, observations, metrics = pipeline.run(
            video_id=str(video_meta.video_id),
            video_duration_sec=float(video_meta.duration_sec),
            clips=list(clips),
            clip_timestamps=list(clip_timestamps),
            preprocess_time_sec=float(preprocess_elapsed),
        )
        inference_elapsed = time.perf_counter() - inference_started
        metrics_dict = _metrics_to_dict(metrics)

        report = {
            "video": {
                "path": str(video_path),
                "video_id": str(video_meta.video_id),
                "duration_sec": float(getattr(video_meta, "duration_sec", 0.0) or 0.0),
            },
            "runtime": {
                "profile": str(cfg.active_profile),
                "variant": str(cfg.active_variant or ""),
                "device": str(getattr(cfg.model, "device", "") or ""),
            },
            "config": {
                "target_fps": float(getattr(cfg.video, "target_fps", 0.0) or 0.0),
                "analysis_fps": float(getattr(cfg.video, "analysis_fps", 0.0) or 0.0),
                "window_sec": float(getattr(cfg.clips, "window_sec", 0.0) or 0.0),
                "stride_sec": float(getattr(cfg.clips, "stride_sec", 0.0) or 0.0),
                "min_clip_frames": int(getattr(cfg.clips, "min_clip_frames", 0) or 0),
                "max_clip_frames": int(getattr(cfg.clips, "max_clip_frames", 0) or 0),
                "batch_size": int(getattr(cfg.model, "batch_size", 1) or 1),
                "max_batch_clips": int(getattr(cfg.model, "max_batch_clips", 1) or 1),
                "max_batch_frames": int(getattr(cfg.model, "max_batch_frames", 0) or 0),
                "batch_frame_tolerance": int(getattr(cfg.model, "batch_frame_tolerance", 0) or 0),
            },
            "timings_sec": {
                "preprocess": float(preprocess_elapsed),
                "build_clips": float(clip_build_elapsed),
                "run_pipeline": float(inference_elapsed),
                "total": float(preprocess_elapsed + clip_build_elapsed + inference_elapsed),
            },
            "clip_stats": _clip_stats(list(clips)),
            "batching": _top_level_batching(metrics_dict),
            "outputs": {
                "merged_annotations": int(len(merged or [])),
                "raw_observations": int(len(observations or [])),
            },
            "metrics": metrics_dict,
        }

        payload = json.dumps(report, ensure_ascii=False, indent=2)
        print(payload)

        output_path = Path(str(args.output)).expanduser().resolve() if str(args.output or "").strip() else None
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(payload + "\n", encoding="utf-8")
    finally:
        pipeline.release()


if __name__ == "__main__":
    main()
