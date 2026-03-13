# src/core/config.py
"""
Typed runtime configuration for SmartCampus V2T.

Purpose:
- Define the effective runtime config contract after profile and variant resolution.
- Keep strongly typed settings under the new compact `src.core` layout.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass
class PathsConfig:
    """Filesystem paths used across UI, API, workers, caches, and artifacts."""

    root_dir: Path
    data_dir: Path
    videos_dir: Path
    cache_dir: Path
    indexes_dir: Path
    thumbs_dir: Path
    models_dir: Path
    qwen_vl_dir: Path
    dnn_face_proto: Path
    dnn_face_model: Path
    config_dir: Path
    profiles_dir: Path
    variants_dir: Path


@dataclass
class UiConfig:
    """UI-facing config for language, assets, and client-side cache behavior."""

    langs: List[str]
    default_lang: str
    cache_ttl_sec: int
    assets_dir: Path
    ui_text_path: Path
    styles_path: Path
    logo_path: Path


@dataclass
class BackendConfig:
    """HTTP settings for FastAPI backend endpoints."""

    scheme: str
    host: str
    port: int


@dataclass
class SearchConfig:
    """Retrieval, fusion, and index-building parameters."""

    embed_model_name: str
    embedding_model_id: str
    embedding_backend: str
    reranker_model_id: str
    reranker_backend: str
    query_prefix: str
    passage_prefix: str
    normalize_text: bool
    lemmatize: bool
    fallback_langs: List[str]
    translate_queries: bool
    embed_cache: bool
    w_bm25: float
    w_dense: float
    candidate_k_sparse: int
    candidate_k_dense: int
    fusion: str
    rrf_k: int
    rerank_enabled: bool
    rerank_top_k: int
    dedupe_mode: str
    dedupe_tol_sec: float
    dedupe_overlap_thr: float
    embedding_dim: int
    dense_input_mode: str


@dataclass
class VideoConfig:
    """Preprocessing and decode policy for preparing video frames."""

    target_fps: float
    analysis_fps: float
    model_input_size: Tuple[int, int]
    decode_resolution: Tuple[int, int]
    pixel_format: str
    min_change_threshold: float
    dark_threshold: float
    blur_threshold: float
    lazy_motion_threshold: float
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
    anonymization_enabled: bool


@dataclass
class ClipsConfig:
    """Temporal slicing policy for generating clip windows."""

    window_sec: float
    stride_sec: float
    min_clip_frames: int
    max_clip_frames: int
    keyframe_policy: str


@dataclass
class ModelConfig:
    """Vision-language model inference settings for clip captioning."""

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
    attn_implementation: str
    max_batch_frames: int
    timeout_sec: int


@dataclass
class LlmConfig:
    """Text LLM settings for structuring, summaries, and optional post-edit."""

    backend: str
    model_id: str
    max_new_tokens: int
    do_sample: bool
    temperature: float
    top_p: float
    timeout_sec: int
    transformers_dtype: str
    transformers_device_map: str
    transformers_compile: bool
    vllm_base_url: str
    vllm_timeout_sec: int


@dataclass
class GuardConfig:
    """Policy gates for inbound queries and generated outputs."""

    enabled: bool
    query_gate: bool
    output_gate: bool
    model_id: str


@dataclass
class RuntimeConfig:
    """Runtime stability and performance flags."""

    seed: int
    num_workers: int
    log_level: str
    overwrite_existing: bool
    strict_paths: bool
    torch_threads: int
    cuda_tf32: bool
    matmul_precision: str
    autocast_infer: bool
    torch_compile: bool
    torch_compile_mode: str
    torch_compile_fullgraph: bool
    metrics_repeats: int
    metrics_store_samples: bool


@dataclass
class TranslationConfig:
    """Translation backend, cache, routing, and post-edit policy."""

    backend: str
    model_name_or_path: str
    device: str
    dtype: str
    batch_size: int
    max_new_tokens: int
    source_lang: str
    target_langs: List[str]
    cache_enabled: bool
    cache_version: str
    query_time_translation: bool
    offline_translation: bool
    post_edit_targets: List[str]
    post_edit_max_items: int
    ctranslate2_device: str
    ctranslate2_compute_type: str
    ctranslate2_inter_threads: int
    ctranslate2_intra_threads: int
    en_ru_model_id: str
    ru_en_model_id: str
    kk_ru_model_id: str
    ru_kk_model_id: str


@dataclass
class JobsConfig:
    """Filesystem locations and defaults for persisted jobs."""

    dir: Path


@dataclass
class QueueConfig:
    """Filesystem queue configuration."""

    dir: Path


@dataclass
class LocksConfig:
    """Worker lock directory."""

    dir: Path


@dataclass
class WorkerConfig:
    """Worker poll and leasing policy."""

    poll_interval_sec: float
    lease_sec: float
    heartbeat_sec: float
    short_first: bool
    max_concurrent_jobs: int


@dataclass
class IndexRuntimeConfig:
    """Index orchestration flags."""

    auto_update_on_done: bool


@dataclass
class WebhookConfig:
    """Outbound webhook notifications."""

    enabled: bool
    url: str
    timeout_sec: float


@dataclass
class ExperimentConfig:
    """Profile-level experiment selection and multi-variant execution settings."""

    mode: str
    compare_on_single_video: bool
    variant_ids: List[str] = field(default_factory=list)


@dataclass
class PipelineConfig:
    """The fully merged application config resolved for one profile and optional variant."""

    paths: PathsConfig
    ui: UiConfig
    backend: BackendConfig
    search: SearchConfig
    video: VideoConfig
    clips: ClipsConfig
    model: ModelConfig
    llm: LlmConfig
    guard: GuardConfig
    runtime: RuntimeConfig
    translation: TranslationConfig
    jobs: JobsConfig
    queue: QueueConfig
    locks: LocksConfig
    worker: WorkerConfig
    index: IndexRuntimeConfig
    webhook: WebhookConfig
    experiment: ExperimentConfig
    config_path: Path
    active_profile: str
    active_variant: Optional[str]
    config_fingerprint: str
