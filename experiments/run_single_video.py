# experiments/run_single_video.py

"""
Run the full SmartCampus V2T pipeline on a single video (Qwen3-VL only).

Behavior:
- If launched with --video <path>, the specified video is used.
- If launched without args, the first file from data/raw/ is selected.

Pipeline:
1) Load PipelineConfig from config/pipeline.yaml
2) Run preprocess_video() with this config → VideoMeta (frames + stats)
3) Build temporal clips from frames using ClipsConfig (window_sec / stride_sec)
4) Run VideoToTextPipeline (Qwen3-VL backend)
5) Print first 5 clip descriptions and timing metrics
6) Save annotations.json and metrics.json into experiments/results/<video_id>/
"""

import argparse
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_pipeline_config
from src.pipeline.video_to_text import VideoToTextPipeline
from src.preprocessing.video_io import preprocess_video
from src.core.types import VideoMeta, FrameInfo, Annotation, RunMetrics

RAW_DIR = PROJECT_ROOT / "data" / "raw"
RESULTS_ROOT = PROJECT_ROOT / "experiments" / "results"


def dataclass_to_dict(obj) -> Dict[str, Any]:
    if is_dataclass(obj):
        return asdict(obj)
    if hasattr(obj, "__dict__"):
        return dict(obj.__dict__)
    raise TypeError("Object is not a dataclass or simple class instance")


def build_clips_from_video_meta(
    video_meta: VideoMeta,
    window_sec: float,
    stride_sec: float,
    min_clip_frames: int,
    max_clip_frames: int,
) -> Tuple[List[List[str]], List[Tuple[float, float]]]:

    if not video_meta.frames:
        return [], []

    frames: List[FrameInfo] = sorted(
        video_meta.frames,
        key=lambda f: f.timestamp_sec,
    )

    duration = float(video_meta.duration_sec)
    clips: List[List[str]] = []
    clip_timestamps: List[Tuple[float, float]] = []

    t = 0.0
    while t < duration:
        t_end = min(t + window_sec, duration)
        window_frames = [f for f in frames if t <= f.timestamp_sec <= t_end]

        if len(window_frames) >= min_clip_frames:
            paths = [str(f.path) for f in window_frames]

            if len(paths) > max_clip_frames:
                step = len(paths) / max_clip_frames
                indices = [int(i * step) for i in range(max_clip_frames)]
                paths = [paths[i] for i in indices]

            last_ts = float(window_frames[-1].timestamp_sec)
            clips.append(paths)
            clip_timestamps.append((float(t), last_ts))

        t += stride_sec
        if stride_sec <= 0:
            break

    return clips, clip_timestamps


def format_time_mmss(t: float) -> str:
    if t < 0:
        t = 0.0
    total_seconds = int(round(t))
    m = total_seconds // 60
    s = total_seconds % 60
    return f"{m}:{s:02d}"


def run_single_video(
    video_path: Path,
    device: str = "cuda",
    save_json: bool = True,
):
    video_path = Path(video_path).resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    video_id = video_path.stem
    print(f"\n[run_single_video] Video path: {video_path}")
    print(f"[run_single_video] Video id:   {video_id}")


    cfg = load_pipeline_config(PROJECT_ROOT / "config" / "pipeline.yaml")
    cfg.model.device = device


    print("\n=== STEP 1 — Preprocessing video ===")
    video_meta: VideoMeta = preprocess_video(video_path, cfg)

    duration_sec = float(video_meta.duration_sec)
    preprocess_time_sec = float(
        (video_meta.extra or {}).get("preprocess_time_sec", 0.0)
    )

    print(f"Duration:   {duration_sec:.2f} sec")
    print(f"Frames:     {video_meta.num_frames}")
    print(f"Preprocess: {preprocess_time_sec:.2f} sec")


    print("\n=== STEP 2 — Building clips ===")
    clips, clip_timestamps = build_clips_from_video_meta(
        video_meta=video_meta,
        window_sec=cfg.clips.window_sec,
        stride_sec=cfg.clips.stride_sec,
        min_clip_frames=cfg.clips.min_clip_frames,
        max_clip_frames=cfg.clips.max_clip_frames,
    )

    print(f"Clips built: {len(clips)}")

    if not clips:
        print("❌ No clips built. Check clip config or preprocessing.")
        metrics = RunMetrics(
            video_id=video_id,
            video_duration_sec=duration_sec,
            num_frames=video_meta.num_frames,
            num_clips=0,
            avg_clip_duration_sec=0.0,
            preprocess_time_sec=preprocess_time_sec,
            model_time_sec=0.0,
            postprocess_time_sec=0.0,
            total_time_sec=preprocess_time_sec,
            extra={"reason": "no_clips"},
        )
        annotations: List[Annotation] = []


    else:
        print("\n=== STEP 3 — Running Qwen3-VL pipeline ===")
        pipeline = VideoToTextPipeline(cfg)
        annotations, metrics = pipeline.run(
            video_id=video_id,
            video_duration_sec=duration_sec,
            clips=clips,
            clip_timestamps=clip_timestamps,
            preprocess_time_sec=preprocess_time_sec,
        )


    print("\n=== SEMANTIC OUTPUT (first 5 clips) ===")
    if not annotations:
        print("❌ No annotations returned.")
    else:
        print(f"Total annotations: {len(annotations)}\n")
        for a in annotations[:5]:
            start_str = format_time_mmss(a.start_sec)
            end_str = format_time_mmss(a.end_sec)
            print(
                f"[clip {a.clip_index:03d}] "
                f"[{start_str} - {end_str}] → {a.description}"
            )


    print("\n=== TIMING METRICS (sec) ===")
    print(f"preprocess:  {metrics.preprocess_time_sec:.3f}")
    print(f"model:       {metrics.model_time_sec:.3f}")
    print(f"postprocess: {metrics.postprocess_time_sec:.3f}")
    print(f"total:       {metrics.total_time_sec:.3f}")

    if save_json:
        base_dir = RESULTS_ROOT / video_id
        base_dir.mkdir(parents=True, exist_ok=True)

        existing = [
            p for p in base_dir.iterdir()
            if p.is_dir() and p.name.startswith("run_")
        ]
        if existing:
            nums = []
            for p in existing:
                name = p.name.replace("run_", "")
                if name.isdigit():
                    nums.append(int(name))
            next_id = (max(nums) + 1) if nums else 1
        else:
            next_id = 1

        out_dir = base_dir / f"run_{next_id:03d}"
        out_dir.mkdir(parents=True, exist_ok=True)

        ann_dicts = [
            {
                "video_id": a.video_id,
                "clip_index": a.clip_index,
                "start_sec": a.start_sec,
                "end_sec": a.end_sec,
                "description": a.description,
                "extra": a.extra,
            }
            for a in annotations
        ]
        metrics_dict = dataclass_to_dict(metrics)

        with (out_dir / "annotations.json").open("w", encoding="utf-8") as f:
            json.dump(ann_dicts, f, indent=2, ensure_ascii=False)

        with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
            json.dump(metrics_dict, f, indent=2, ensure_ascii=False)

        print(f"\nResults saved to: {out_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run SmartCampus V2T on a single video")
    parser.add_argument("--video", type=str, help="Video file path")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument("--no-save", action="store_true", help="Do not save JSON results")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.video:
        run_single_video(
            video_path=Path(args.video),
            device=args.device,
            save_json=not args.no_save,
        )
    else:
        video_files = [p for p in RAW_DIR.iterdir() if p.is_file()]
        if not video_files:
            raise FileNotFoundError(f"No videos in: {RAW_DIR}")

        test_video = video_files[0]
        print("\nRunning fallback mode:")
        print(f"Using video: {test_video}")

        run_single_video(
            video_path=test_video,
            device=args.device,
            save_json=True,
        )
