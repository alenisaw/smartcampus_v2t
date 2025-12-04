# src/utils/config.py

"""
Helpers for loading pipeline configuration from config/pipeline.yaml
and resolving all paths relative to project root.
"""

from pathlib import Path
from typing import Any, Dict

import yaml

from src.pipeline.pipeline_config import (
    PathsConfig,
    VideoConfig,
    ClipsConfig,
    ModelConfig,
    RuntimeConfig,
    PipelineConfig,
)


def _to_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.lower() in {"1", "true", "yes", "y"}
    return bool(v)


def load_pipeline_config(config_path: Path) -> PipelineConfig:

    config_path = Path(config_path).resolve()
    with config_path.open("r", encoding="utf-8") as f:
        raw: Dict[str, Any] = yaml.safe_load(f)


    paths_raw = raw["paths"]


    root_dir = (config_path.parent / paths_raw.get("root_dir", ".")).resolve()

    def rel(p: str) -> Path:
        return (root_dir / p).resolve()

    paths = PathsConfig(
        root_dir=root_dir,
        raw_dir=rel(paths_raw["raw_dir"]),
        prepared_dir=rel(paths_raw["prepared_dir"]),
        results_dir=rel(paths_raw["results_dir"]),
        models_dir=rel(paths_raw["models_dir"]),
        qwen_vl_dir=rel(paths_raw["qwen_vl_dir"]),
        dnn_face_proto=rel(paths_raw["dnn_face_proto"]),
        dnn_face_model=rel(paths_raw["dnn_face_model"]),
    )


    video_raw = raw["video"]
    video = VideoConfig(
        target_fps=float(video_raw["target_fps"]),
        model_input_size=tuple(video_raw["model_input_size"]),
        min_change_threshold=float(video_raw["min_change_threshold"]),
        dark_threshold=float(video_raw["dark_threshold"]),
        save_small_frames=_to_bool(video_raw["save_small_frames"]),
        face_blur=_to_bool(video_raw["face_blur"]),
        face_detector_type=str(video_raw["face_detector_type"]),
        max_frames=video_raw.get("max_frames"),
    )


    clips_raw = raw["clips"]
    clips = ClipsConfig(
        window_sec=float(clips_raw["window_sec"]),
        stride_sec=float(clips_raw["stride_sec"]),
        min_clip_frames=int(clips_raw["min_clip_frames"]),
        max_clip_frames=int(clips_raw["max_clip_frames"]),
    )


    model_raw = raw["model"]
    model = ModelConfig(

        model_name_or_path=str(paths.qwen_vl_dir),
        device=str(model_raw.get("device", "cuda")),
        dtype=str(model_raw.get("dtype", "fp8")),
        language=str(model_raw.get("language", "en")),
        max_new_tokens=int(model_raw["max_new_tokens"]),
        temperature=float(model_raw["temperature"]),
        top_p=float(model_raw["top_p"]),
        repetition_penalty=float(model_raw["repetition_penalty"]),
        timeout_sec=int(model_raw["timeout_sec"]),
    )


    runtime_raw = raw["runtime"]
    runtime = RuntimeConfig(
        seed=int(runtime_raw["seed"]),
        num_workers=int(runtime_raw["num_workers"]),
        log_level=str(runtime_raw["log_level"]),
        overwrite_existing=_to_bool(runtime_raw["overwrite_existing"]),
    )

    return PipelineConfig(
        paths=paths,
        video=video,
        clips=clips,
        model=model,
        runtime=runtime,
    )
