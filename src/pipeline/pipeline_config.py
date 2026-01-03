# src/pipeline/pipeline_config.py

"""
Unified configuration dataclasses for the SmartCampus video-to-text pipeline.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass
class PathsConfig:
    root_dir: Path
    data_dir: Path

    raw_dir: Path
    prepared_dir: Path
    runs_dir: Path

    indexes_dir: Path
    thumbs_dir: Path

    models_dir: Path
    qwen_vl_dir: Path

    dnn_face_proto: Path
    dnn_face_model: Path


@dataclass
class UiConfig:
    langs: List[str]
    default_lang: str
    cache_ttl_sec: int

    assets_dir: Path
    ui_text_path: Path
    styles_path: Path
    logo_path: Path


@dataclass
class SearchConfig:
    embed_model_name: str

    w_bm25: float
    w_dense: float

    candidate_k_sparse: int
    candidate_k_dense: int

    fusion: str
    rrf_k: int

    dedupe_mode: str
    dedupe_tol_sec: float
    dedupe_overlap_thr: float


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

    jpeg_quality_frames: int
    jpeg_quality_small: int

    face_detect_every_saved: int
    face_detect_force_on_scenecut: bool
    face_blur_strength: int
    face_conf_threshold: float


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
    top_k: int
    repetition_penalty: float
    do_sample: bool

    timeout_sec: int


@dataclass
class RuntimeConfig:
    seed: int
    num_workers: int
    log_level: str

    overwrite_existing: bool

    strict_paths: bool

    torch_threads: int
    cuda_tf32: bool
    matmul_precision: str
    autocast_infer: bool


@dataclass
class PipelineConfig:
    paths: PathsConfig
    ui: UiConfig
    search: SearchConfig
    video: VideoConfig
    clips: ClipsConfig
    model: ModelConfig
    runtime: RuntimeConfig
