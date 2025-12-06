# src/pipeline/pipeline_config.py

"""
Unified configuration dataclasses for the SmartCampus video-to-text pipeline.
This config matches the new architecture based on Qwen3-VL only.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple


@dataclass
class PathsConfig:
    root_dir: Path
    raw_dir: Path
    prepared_dir: Path
    results_dir: Path

    models_dir: Path
    qwen_vl_dir: Path

    dnn_face_proto: Path
    dnn_face_model: Path


@dataclass
class VideoConfig:
    target_fps: float
    model_input_size: Tuple[int, int]

    min_change_threshold: float
    dark_threshold: float

    save_small_frames: bool

    face_blur: bool
    face_detector_type: str
    max_frames: Optional[int]


@dataclass
class ClipsConfig:
    window_sec: float
    stride_sec: float
    min_clip_frames: int
    max_clip_frames: int


@dataclass
class ModelConfig:
    model_name_or_path: str
    device: str
    dtype: str

    language: str
    batch_size: int

    max_new_tokens: int
    temperature: float
    top_p: float
    repetition_penalty: float

    timeout_sec: int


@dataclass
class RuntimeConfig:
    seed: int
    num_workers: int
    log_level: str
    overwrite_existing: bool


@dataclass
class PipelineConfig:
    paths: PathsConfig
    video: VideoConfig
    clips: ClipsConfig
    model: ModelConfig
    runtime: RuntimeConfig

