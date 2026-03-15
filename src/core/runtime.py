# src/core/runtime.py
"""
Runtime config loading for SmartCampus V2T.

Purpose:
- Resolve profile and variant YAML inputs into typed effective runtime config.
- Keep raw config merging, path resolution, and validation under `src.core`.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml

from src.core.config import (
    BackendConfig,
    ClipsConfig,
    ExperimentConfig,
    GuardConfig,
    IndexRuntimeConfig,
    JobsConfig,
    LlmConfig,
    LocksConfig,
    ModelConfig,
    PathsConfig,
    PipelineConfig,
    QueueConfig,
    RuntimeConfig,
    SearchConfig,
    TranslationConfig,
    UiConfig,
    VideoConfig,
    WebhookConfig,
    WorkerConfig,
)

DEFAULT_CONFIG_REL = Path("configs") / "profiles" / "main.yaml"
PROFILE_ENV_NAME = "SMARTCAMPUS_PROFILE"
VARIANT_ENV_NAME = "SMARTCAMPUS_VARIANT"


def _to_bool(value: Any) -> bool:
    """Normalize common YAML/ENV boolean values."""

    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _to_list(value: Any, default: Optional[Iterable[str]] = None) -> List[str]:
    """Normalize string or list values into a clean string list."""

    if value is None:
        value = list(default or [])
    if isinstance(value, str):
        items = [chunk.strip() for chunk in value.split(",")]
    elif isinstance(value, (list, tuple)):
        items = [str(chunk).strip() for chunk in value]
    else:
        items = [str(value).strip()]
    return [item for item in items if item]


def _looks_like_hf_id(value: str) -> bool:
    """Detect Hugging Face model IDs so they are not treated as local paths."""

    text = (value or "").strip()
    return ("/" in text) and (not text.startswith(".")) and (not text.startswith("/")) and (":" not in text)


def _deep_merge(base: Any, override: Any) -> Any:
    """Recursively merge YAML dictionaries, replacing non-dicts wholesale."""

    if not isinstance(base, dict) or not isinstance(override, dict):
        return copy.deepcopy(override)

    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _read_yaml_dict(path: Path) -> Dict[str, Any]:
    """Load a YAML file and ensure it resolves to a dictionary."""

    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid config: {path} (expected YAML dict)")
    return raw


def _load_with_extends(path: Path, visited: Optional[set[Path]] = None) -> Tuple[Dict[str, Any], List[Path]]:
    """Load a config file, resolving its optional `extends` chain."""

    resolved = path.resolve()
    visited = visited or set()
    if resolved in visited:
        raise ValueError(f"Recursive config extends detected: {resolved}")
    visited.add(resolved)

    raw = _read_yaml_dict(resolved)
    deps = [resolved]

    parent_ref = raw.pop("extends", None)
    if parent_ref:
        parent_path = (resolved.parent / str(parent_ref)).resolve()
        parent_raw, parent_deps = _load_with_extends(parent_path, visited=visited)
        deps = parent_deps + deps
        raw = _deep_merge(parent_raw, raw)

    return raw, deps


def resolve_config_selection(
    config_path: Optional[Path] = None,
    *,
    profile: Optional[str] = None,
    variant: Optional[str] = None,
) -> Tuple[Path, str, Optional[str]]:
    """Resolve the selected profile path and optional variant from args/env/defaults."""

    default_path = (Path(__file__).resolve().parents[2] / DEFAULT_CONFIG_REL).resolve()
    provided_path = Path(config_path).resolve() if config_path is not None else default_path
    profiles_dir = provided_path.parent if provided_path.parent.name == "profiles" else default_path.parent

    inferred_profile = provided_path.stem if provided_path.suffix.lower() in {".yaml", ".yml"} else "main"
    resolved_profile = (profile or os.environ.get(PROFILE_ENV_NAME) or inferred_profile or "main")
    resolved_profile = str(resolved_profile).strip().lower() or "main"

    resolved_variant = variant if variant is not None else os.environ.get(VARIANT_ENV_NAME)
    resolved_variant = (str(resolved_variant).strip().lower() or None) if resolved_variant else None

    profile_path = (profiles_dir / f"{resolved_profile}.yaml").resolve()
    if not profile_path.exists() and provided_path.exists():
        profile_path = provided_path
    return profile_path, resolved_profile, resolved_variant


def _resolve_rel(base_dir: Path, value: Any, default_rel: str) -> Path:
    """Resolve an absolute or relative path against the project root."""

    text = str(value if value is not None else default_rel)
    path = Path(text)
    return path if path.is_absolute() else (base_dir / path).resolve()


def _resolve_model_name_or_path(root_dir: Path, raw_value: Optional[str], fallback_path: Path) -> str:
    """Resolve a model reference to either a HF ID or a project-relative path."""

    if raw_value is None or not str(raw_value).strip():
        return str(fallback_path)

    text = str(raw_value).strip()
    candidate = Path(text)
    if candidate.is_absolute() and candidate.exists():
        return str(candidate)
    rooted_candidate = (root_dir / candidate).resolve()
    if rooted_candidate.exists():
        return str(rooted_candidate)
    if _looks_like_hf_id(text):
        return text
    return str(_resolve_rel(root_dir, text, str(fallback_path)))


def _resolve_reference_or_path(root_dir: Path, raw_value: Optional[str], fallback_value: str) -> str:
    """Resolve a model/reference string to either an HF id or an absolute local path."""

    if raw_value is None or not str(raw_value).strip():
        text = str(fallback_value)
    else:
        text = str(raw_value).strip()
    candidate = Path(text)
    if candidate.is_absolute() and candidate.exists():
        return str(candidate)
    rooted_candidate = (root_dir / candidate).resolve()
    if rooted_candidate.exists():
        return str(rooted_candidate)
    if _looks_like_hf_id(text):
        return text
    return str(_resolve_rel(root_dir, text, fallback_value))


def _config_fingerprint(raw: Dict[str, Any], profile: str, variant: Optional[str]) -> str:
    """Build a stable SHA1 fingerprint for the merged effective config."""

    payload = {
        "active_profile": profile,
        "active_variant": variant,
        "config": raw,
    }
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:16]


def _dependency_files(
    config_path: Path,
    profile: str,
    variant: Optional[str],
) -> List[Path]:
    """Collect all files that influence the effective config for cache invalidation."""

    _raw, deps = _load_with_extends(config_path)
    variants_dir = config_path.resolve().parents[1] / "variants"

    if variant:
        variant_path = (variants_dir / f"{variant}.yaml").resolve()
        if variant_path.exists():
            deps.append(variant_path)

    unique: List[Path] = []
    seen: set[Path] = set()
    for item in deps:
        if item not in seen:
            unique.append(item)
            seen.add(item)
    return unique


def config_cache_token(
    config_path: Optional[Path] = None,
    *,
    profile: Optional[str] = None,
    variant: Optional[str] = None,
) -> str:
    """Return a cache token that changes when any config dependency changes."""

    root_path, active_profile, active_variant = resolve_config_selection(
        config_path=config_path,
        profile=profile,
        variant=variant,
    )
    files = _dependency_files(root_path, active_profile, active_variant)
    chunks: List[str] = [active_profile, active_variant or "-"]
    for item in files:
        try:
            stat = item.stat()
            chunks.append(f"{item}:{stat.st_mtime_ns}:{stat.st_size}")
        except FileNotFoundError:
            chunks.append(f"{item}:missing")
    return "|".join(chunks)


def _merge_effective_raw(profile_path: Path, active_variant: Optional[str]) -> Dict[str, Any]:
    """Load the effective raw config for one profile and optional variant."""

    root_raw, _ = _load_with_extends(profile_path)
    if active_variant:
        variants_dir = profile_path.parents[1] / "variants"
        variant_path = (variants_dir / f"{active_variant}.yaml").resolve()
        if variant_path.exists():
            variant_raw, _ = _load_with_extends(variant_path)
            root_raw = _deep_merge(root_raw, variant_raw)
    return copy.deepcopy(root_raw)


def _resolve_hw_pair(raw_value: Any, default: Tuple[int, int]) -> Tuple[int, int]:
    """Resolve a width-height pair from raw config data."""

    values = raw_value or [default[0], default[1]]
    if not isinstance(values, (list, tuple)) or len(values) != 2:
        values = [default[0], default[1]]
    return int(values[0]), int(values[1])


def _build_paths_config(profile_path: Path, paths_raw: Dict[str, Any]) -> tuple[Path, PathsConfig]:
    """Build the filesystem path section of the typed config."""

    root_dir = _resolve_rel(profile_path.parent, paths_raw.get("root_dir"), "../..")
    data_dir = _resolve_rel(root_dir, paths_raw.get("data_dir"), "data")
    return root_dir, PathsConfig(
        root_dir=root_dir,
        data_dir=data_dir,
        videos_dir=_resolve_rel(root_dir, paths_raw.get("videos_dir"), "data/videos"),
        cache_dir=_resolve_rel(root_dir, paths_raw.get("cache_dir"), "data/cache"),
        indexes_dir=_resolve_rel(root_dir, paths_raw.get("indexes_dir"), "data/indexes"),
        thumbs_dir=_resolve_rel(root_dir, paths_raw.get("thumbs_dir"), "data/thumbs"),
        models_dir=_resolve_rel(root_dir, paths_raw.get("models_dir"), "models"),
        qwen_vl_dir=_resolve_rel(root_dir, paths_raw.get("qwen_vl_dir"), "models/qwen3-vl-2b-instruct"),
        dnn_face_proto=_resolve_rel(root_dir, paths_raw.get("dnn_face_proto"), "models/face_detector/deploy.prototxt"),
        dnn_face_model=_resolve_rel(
            root_dir,
            paths_raw.get("dnn_face_model"),
            "models/face_detector/res10_300x300_ssd_iter_140000.caffemodel",
        ),
        config_dir=profile_path.parents[1].resolve(),
        profiles_dir=profile_path.parent.resolve(),
        variants_dir=(profile_path.parents[1] / "variants").resolve(),
    )


def _build_ui_config(root_dir: Path, ui_raw: Dict[str, Any]) -> UiConfig:
    """Build the UI-facing section of the typed config."""

    langs = _to_list(ui_raw.get("langs"), default=["ru", "kz", "en"]) or ["ru", "kz", "en"]
    default_lang = str(ui_raw.get("default_lang", langs[0])).strip().lower() or langs[0]
    if default_lang not in langs:
        default_lang = langs[0]
    return UiConfig(
        langs=langs,
        default_lang=default_lang,
        cache_ttl_sec=int(ui_raw.get("cache_ttl_sec", 2)),
        assets_dir=_resolve_rel(root_dir, ui_raw.get("assets_dir"), "app/assets"),
        ui_text_path=_resolve_rel(root_dir, ui_raw.get("ui_text_path"), "app/assets/ui_text.json"),
        styles_path=_resolve_rel(root_dir, ui_raw.get("styles_path"), "app/assets/styles.css"),
        logo_path=_resolve_rel(root_dir, ui_raw.get("logo_path"), "app/assets/logo.png"),
    )


def _build_backend_config(backend_raw: Dict[str, Any]) -> BackendConfig:
    """Build the backend HTTP section of the typed config."""

    return BackendConfig(
        scheme=str(backend_raw.get("scheme", "http")).strip().lower() or "http",
        host=str(backend_raw.get("host", "127.0.0.1")).strip() or "127.0.0.1",
        port=int(backend_raw.get("port", 8000)),
    )


def _build_search_config(root_dir: Path, search_raw: Dict[str, Any]) -> SearchConfig:
    """Build the retrieval section of the typed config."""

    embedding_model_raw = search_raw.get("embedding_model_id")
    embedding_model_id = ""
    if str(embedding_model_raw or "").strip():
        embedding_model_id = _resolve_reference_or_path(root_dir, embedding_model_raw, str(embedding_model_raw))

    return SearchConfig(
        embed_model_name=str(search_raw.get("embed_model_name", "BAAI/bge-m3")),
        embedding_model_id=embedding_model_id,
        embedding_backend=str(search_raw.get("embedding_backend", "auto")).strip().lower() or "auto",
        reranker_model_id=_resolve_reference_or_path(root_dir, search_raw.get("reranker_model_id"), "Qwen/Qwen3-VL-Reranker-2B"),
        reranker_backend=str(search_raw.get("reranker_backend", "auto")).strip().lower() or "auto",
        query_prefix=str(search_raw.get("query_prefix", "query: ")),
        passage_prefix=str(search_raw.get("passage_prefix", "passage: ")),
        normalize_text=_to_bool(search_raw.get("normalize_text", True)),
        lemmatize=_to_bool(search_raw.get("lemmatize", False)),
        fallback_langs=_to_list(search_raw.get("fallback_langs"), default=["en"]),
        translate_queries=_to_bool(search_raw.get("translate_queries", True)),
        embed_cache=_to_bool(search_raw.get("embed_cache", True)),
        w_bm25=float(search_raw.get("w_bm25", 0.45)),
        w_dense=float(search_raw.get("w_dense", 0.55)),
        candidate_k_sparse=int(search_raw.get("candidate_k_sparse", 200)),
        candidate_k_dense=int(search_raw.get("candidate_k_dense", 200)),
        fusion=str(search_raw.get("fusion", "rrf")).strip().lower() or "rrf",
        rrf_k=int(search_raw.get("rrf_k", 60)),
        rerank_enabled=_to_bool(search_raw.get("rerank_enabled", True)),
        rerank_top_k=int(search_raw.get("rerank_top_k", 30)),
        dedupe_mode=str(search_raw.get("dedupe_mode", "overlap")).strip().lower() or "overlap",
        dedupe_tol_sec=float(search_raw.get("dedupe_tol_sec", 1.0)),
        dedupe_overlap_thr=float(search_raw.get("dedupe_overlap_thr", 0.7)),
        embedding_dim=int(search_raw.get("embedding_dim", 1024)),
        dense_input_mode=str(search_raw.get("dense_input_mode", "text")).strip().lower() or "text",
    )


def _video_io_value(video_raw: Dict[str, Any], video_io_raw: Dict[str, Any], key: str, default: Any) -> Any:
    """Resolve one video IO setting with `video_io` taking precedence over `video`."""

    return video_io_raw[key] if key in video_io_raw else video_raw.get(key, default)


def _build_video_config(video_raw: Dict[str, Any], video_io_raw: Dict[str, Any]) -> VideoConfig:
    """Build the video preprocessing section of the typed config."""

    decode_resolution = _resolve_hw_pair(video_raw.get("decode_resolution"), (1280, 720))
    model_input_size = _resolve_hw_pair(video_raw.get("model_input_size"), (256, 256))
    return VideoConfig(
        target_fps=float(video_raw.get("target_fps", video_raw.get("analysis_fps", 3.0))),
        analysis_fps=float(video_raw.get("analysis_fps", video_raw.get("target_fps", 3.0))),
        model_input_size=model_input_size,
        decode_resolution=decode_resolution,
        pixel_format=str(video_raw.get("pixel_format", "yuv420p")),
        min_change_threshold=float(video_raw.get("min_change_threshold", 0.03)),
        dark_threshold=float(video_raw.get("dark_threshold", 30.0)),
        blur_threshold=float(video_raw.get("blur_threshold", 35.0)),
        lazy_motion_threshold=float(video_raw.get("lazy_motion_threshold", 0.02)),
        save_small_frames=_to_bool(video_raw.get("save_small_frames", False)),
        face_blur=_to_bool(video_raw.get("face_blur", True)),
        face_detector_type=str(video_raw.get("face_detector_type", "dnn")),
        max_frames=video_raw.get("max_frames"),
        jpeg_quality_frames=int(_video_io_value(video_raw, video_io_raw, "jpeg_quality_frames", 82)),
        jpeg_quality_small=int(_video_io_value(video_raw, video_io_raw, "jpeg_quality_small", 75)),
        face_detect_every_saved=int(_video_io_value(video_raw, video_io_raw, "face_detect_every_saved", 6)),
        face_detect_force_on_scenecut=_to_bool(_video_io_value(video_raw, video_io_raw, "face_detect_force_on_scenecut", True)),
        face_blur_strength=int(_video_io_value(video_raw, video_io_raw, "face_blur_strength", 31)),
        face_conf_threshold=float(_video_io_value(video_raw, video_io_raw, "face_conf_threshold", 0.5)),
        anonymization_enabled=_to_bool(video_raw.get("anonymization_enabled", video_raw.get("face_blur", True))),
    )


def _build_model_config(root_dir: Path, paths: PathsConfig, model_raw: Dict[str, Any]) -> ModelConfig:
    """Build the vision-language model section of the typed config."""

    return ModelConfig(
        model_name_or_path=_resolve_model_name_or_path(root_dir, model_raw.get("model_name_or_path"), paths.qwen_vl_dir),
        device=str(model_raw.get("device", "cuda")),
        dtype=str(model_raw.get("dtype", "fp16")),
        language=str(model_raw.get("language", "en")).strip().lower() or "en",
        batch_size=int(model_raw.get("batch_size", 1)),
        max_new_tokens=int(model_raw.get("max_new_tokens", 128)),
        temperature=float(model_raw.get("temperature", 0.0)),
        top_p=float(model_raw.get("top_p", 1.0)),
        top_k=int(model_raw.get("top_k", 40)),
        repetition_penalty=float(model_raw.get("repetition_penalty", 1.0)),
        do_sample=_to_bool(model_raw.get("do_sample", False)),
        attn_implementation=str(model_raw.get("attn_implementation", "auto")),
        max_batch_frames=int(model_raw.get("max_batch_frames", 64)),
        timeout_sec=int(model_raw.get("timeout_sec", 60)),
    )


def _build_llm_config(root_dir: Path, llm_raw: Dict[str, Any]) -> LlmConfig:
    """Build the semantic LLM section of the typed config."""

    transformers_raw = llm_raw.get("transformers") or {}
    vllm_raw = llm_raw.get("vllm") or {}
    return LlmConfig(
        backend=str(llm_raw.get("backend", "transformers")).strip().lower() or "transformers",
        model_id=_resolve_reference_or_path(root_dir, llm_raw.get("model_id"), "Qwen/Qwen3-4B-Instruct-2507"),
        max_new_tokens=int(llm_raw.get("max_new_tokens", 512)),
        do_sample=_to_bool(llm_raw.get("do_sample", False)),
        temperature=float(llm_raw.get("temperature", 0.0)),
        top_p=float(llm_raw.get("top_p", 1.0)),
        timeout_sec=int(llm_raw.get("timeout_sec", 60)),
        transformers_dtype=str(transformers_raw.get("dtype", "float16")),
        transformers_device_map=str(transformers_raw.get("device_map", "auto")),
        transformers_compile=_to_bool(transformers_raw.get("compile", False)),
        vllm_base_url=str(vllm_raw.get("base_url", "http://127.0.0.1:8001/v1")),
        vllm_timeout_sec=int(vllm_raw.get("timeout_sec", 30)),
    )


def _build_guard_config(root_dir: Path, guard_raw: Dict[str, Any]) -> GuardConfig:
    """Build the guard/policy section of the typed config."""

    return GuardConfig(
        enabled=_to_bool(guard_raw.get("enabled", False)),
        query_gate=_to_bool(guard_raw.get("query_gate", False)),
        output_gate=_to_bool(guard_raw.get("output_gate", False)),
        model_id=_resolve_reference_or_path(root_dir, guard_raw.get("model_id"), "Qwen/Qwen3Guard-Gen-0.6B"),
    )


def _build_runtime_config(runtime_raw: Dict[str, Any]) -> RuntimeConfig:
    """Build the runtime/performance section of the typed config."""

    return RuntimeConfig(
        seed=int(runtime_raw.get("seed", 42)),
        num_workers=int(runtime_raw.get("num_workers", 4)),
        log_level=str(runtime_raw.get("log_level", "INFO")),
        overwrite_existing=_to_bool(runtime_raw.get("overwrite_existing", False)),
        strict_paths=_to_bool(runtime_raw.get("strict_paths", True)),
        torch_threads=int(runtime_raw.get("torch_threads", 0)),
        cuda_tf32=_to_bool(runtime_raw.get("cuda_tf32", True)),
        matmul_precision=str(runtime_raw.get("matmul_precision", "high")),
        autocast_infer=_to_bool(runtime_raw.get("autocast_infer", True)),
        torch_compile=_to_bool(runtime_raw.get("torch_compile", False)),
        torch_compile_mode=str(runtime_raw.get("torch_compile_mode", "reduce-overhead")),
        torch_compile_fullgraph=_to_bool(runtime_raw.get("torch_compile_fullgraph", False)),
        metrics_repeats=max(1, int(runtime_raw.get("metrics_repeats", 1))),
        metrics_store_samples=_to_bool(runtime_raw.get("metrics_store_samples", True)),
        structuring_mode=str(runtime_raw.get("structuring_mode", "rules")).strip().lower() or "rules",
        summary_mode=str(runtime_raw.get("summary_mode", "deterministic")).strip().lower() or "deterministic",
        summary_polish_enabled=_to_bool(runtime_raw.get("summary_polish_enabled", False)),
        summary_polish_priority=str(runtime_raw.get("summary_polish_priority", "025")).strip() or "025",
        worker_output_guard_backend=str(runtime_raw.get("worker_output_guard_backend", "rules")).strip().lower() or "rules",
    )


def _build_translation_config(root_dir: Path, translation_raw: Dict[str, Any]) -> TranslationConfig:
    """Build the translation section of the typed config."""

    ct2_raw = translation_raw.get("ctranslate2") or {}
    routes_raw = translation_raw.get("routes") or {}
    return TranslationConfig(
        backend=str(translation_raw.get("backend", "ctranslate2")).strip().lower() or "ctranslate2",
        model_name_or_path=str(translation_raw.get("model_name_or_path", "ct2://router")),
        device=str(translation_raw.get("device", "cpu")),
        dtype=str(translation_raw.get("dtype", "float32")),
        batch_size=int(translation_raw.get("batch_size", 8)),
        max_new_tokens=int(translation_raw.get("max_new_tokens", 128)),
        source_lang=str(translation_raw.get("source_lang", "en")).strip().lower() or "en",
        target_langs=_to_list(translation_raw.get("target_langs"), default=["ru", "kz"]),
        cache_enabled=_to_bool(translation_raw.get("cache_enabled", True)),
        cache_version=str(translation_raw.get("cache_version", "v2")),
        query_time_translation=_to_bool(translation_raw.get("query_time_translation", True)),
        offline_translation=_to_bool(translation_raw.get("offline_translation", True)),
        post_edit_targets=_to_list(
            translation_raw.get("post_edit_targets"),
            default=["summary", "reports", "qa", "selected_segments"],
        ),
        post_edit_max_items=max(1, int(translation_raw.get("post_edit_max_items", 64))),
        ctranslate2_device=str(ct2_raw.get("device", "cpu")),
        ctranslate2_compute_type=str(ct2_raw.get("compute_type", "int8_float16")),
        ctranslate2_inter_threads=int(ct2_raw.get("inter_threads", 2)),
        ctranslate2_intra_threads=int(ct2_raw.get("intra_threads", 0)),
        en_ru_model_id=_resolve_reference_or_path(root_dir, routes_raw.get("en_ru_model_id"), "Helsinki-NLP/opus-mt-en-ru"),
        ru_en_model_id=_resolve_reference_or_path(root_dir, routes_raw.get("ru_en_model_id"), "Helsinki-NLP/opus-mt-ru-en"),
        kk_ru_model_id=_resolve_reference_or_path(root_dir, routes_raw.get("kk_ru_model_id"), "deepvk/kazRush-kk-ru"),
        ru_kk_model_id=_resolve_reference_or_path(root_dir, routes_raw.get("ru_kk_model_id"), "deepvk/kazRush-ru-kk"),
    )


def _build_execution_configs(
    root_dir: Path,
    *,
    jobs_raw: Dict[str, Any],
    queue_raw: Dict[str, Any],
    locks_raw: Dict[str, Any],
    worker_raw: Dict[str, Any],
    index_raw: Dict[str, Any],
    webhook_raw: Dict[str, Any],
    experiment_raw: Dict[str, Any],
    active_profile: str,
) -> tuple[JobsConfig, QueueConfig, LocksConfig, WorkerConfig, IndexRuntimeConfig, WebhookConfig, ExperimentConfig]:
    """Build jobs, queue, worker, index, webhook, and experiment config sections."""

    jobs = JobsConfig(dir=_resolve_rel(root_dir, jobs_raw.get("dir"), "data/jobs"))
    queue = QueueConfig(dir=_resolve_rel(root_dir, queue_raw.get("dir"), "data/queue"))
    locks = LocksConfig(dir=_resolve_rel(root_dir, locks_raw.get("dir"), "data/locks"))
    worker = WorkerConfig(
        poll_interval_sec=float(worker_raw.get("poll_interval_sec", 1.0)),
        lease_sec=float(worker_raw.get("lease_sec", 30.0)),
        heartbeat_sec=float(worker_raw.get("heartbeat_sec", 5.0)),
        short_first=_to_bool(worker_raw.get("short_first", True)),
        max_concurrent_jobs=int(worker_raw.get("max_concurrent_jobs", 1)),
    )
    index_cfg = IndexRuntimeConfig(auto_update_on_done=_to_bool(index_raw.get("auto_update_on_done", True)))
    webhook = WebhookConfig(
        enabled=_to_bool(webhook_raw.get("enabled", False)),
        url=str(webhook_raw.get("url", "")),
        timeout_sec=float(webhook_raw.get("timeout_sec", 5.0)),
    )
    experiment = ExperimentConfig(
        mode=str(experiment_raw.get("mode", active_profile)).strip().lower() or active_profile,
        compare_on_single_video=_to_bool(experiment_raw.get("compare_on_single_video", active_profile == "experimental")),
        variant_ids=_to_list(experiment_raw.get("variant_ids"), default=["fast", "throughput", "max_quality"]),
    )
    return jobs, queue, locks, worker, index_cfg, webhook, experiment


def _attach_runtime_context(effective_raw: Dict[str, Any], cfg: PipelineConfig) -> Dict[str, Any]:
    """Attach resolved runtime metadata to the raw config payload."""

    effective_raw.setdefault("runtime_context", {})
    effective_raw["runtime_context"].update(
        {
            "active_profile": cfg.active_profile,
            "active_variant": cfg.active_variant,
            "config_fingerprint": cfg.config_fingerprint,
            "config_path": str(cfg.config_path),
        }
    )
    return effective_raw


def load_pipeline_bundle(
    config_path: Optional[Path] = None,
    *,
    profile: Optional[str] = None,
    variant: Optional[str] = None,
) -> Tuple[PipelineConfig, Dict[str, Any]]:
    """Load the effective config and return both typed and raw merged forms."""

    profile_path, active_profile, active_variant = resolve_config_selection(
        config_path=config_path,
        profile=profile,
        variant=variant,
    )
    effective_raw = _merge_effective_raw(profile_path, active_variant)
    fingerprint = _config_fingerprint(effective_raw, active_profile, active_variant)

    paths_raw = effective_raw.get("paths") or {}
    if not isinstance(paths_raw, dict):
        raise ValueError(f"Invalid config: {profile_path} (paths must be a dict)")

    root_dir, paths = _build_paths_config(profile_path, paths_raw)

    ui_raw = effective_raw.get("ui") or {}
    ui = _build_ui_config(root_dir, ui_raw)

    backend_raw = effective_raw.get("backend") or {}
    backend = _build_backend_config(backend_raw)

    search_raw = effective_raw.get("search") or {}
    search = _build_search_config(root_dir, search_raw)

    video_raw = effective_raw.get("video") or {}
    video_io_raw = effective_raw.get("video_io") or {}
    video = _build_video_config(video_raw, video_io_raw)

    clips_raw = effective_raw.get("clips") or {}
    clips = ClipsConfig(
        window_sec=float(clips_raw.get("window_sec", 4.0)),
        stride_sec=float(clips_raw.get("stride_sec", 2.0)),
        min_clip_frames=int(clips_raw.get("min_clip_frames", 2)),
        max_clip_frames=int(clips_raw.get("max_clip_frames", 16)),
        keyframe_policy=str(clips_raw.get("keyframe_policy", "middle")),
    )

    model_raw = effective_raw.get("model") or {}
    model = _build_model_config(root_dir, paths, model_raw)

    llm_raw = effective_raw.get("llm") or {}
    llm = _build_llm_config(root_dir, llm_raw)

    guard_raw = effective_raw.get("guard") or {}
    guard = _build_guard_config(root_dir, guard_raw)

    runtime_raw = effective_raw.get("runtime") or {}
    runtime = _build_runtime_config(runtime_raw)

    translation_raw = effective_raw.get("translation") or {}
    translation = _build_translation_config(root_dir, translation_raw)

    jobs_raw = effective_raw.get("jobs") or {}
    queue_raw = effective_raw.get("queue") or {}
    locks_raw = effective_raw.get("locks") or {}
    worker_raw = effective_raw.get("worker") or {}
    index_raw = effective_raw.get("index") or {}
    webhook_raw = effective_raw.get("webhook") or {}
    experiment_raw = effective_raw.get("experiment") or {}

    jobs, queue, locks, worker, index_cfg, webhook, experiment = _build_execution_configs(
        root_dir,
        jobs_raw=jobs_raw,
        queue_raw=queue_raw,
        locks_raw=locks_raw,
        worker_raw=worker_raw,
        index_raw=index_raw,
        webhook_raw=webhook_raw,
        experiment_raw=experiment_raw,
        active_profile=active_profile,
    )

    cfg = PipelineConfig(
        paths=paths,
        ui=ui,
        backend=backend,
        search=search,
        video=video,
        clips=clips,
        model=model,
        llm=llm,
        guard=guard,
        runtime=runtime,
        translation=translation,
        jobs=jobs,
        queue=queue,
        locks=locks,
        worker=worker,
        index=index_cfg,
        webhook=webhook,
        experiment=experiment,
        config_path=profile_path,
        active_profile=active_profile,
        active_variant=active_variant,
        config_fingerprint=fingerprint,
    )

    ensure_dirs(cfg.paths, strict=cfg.runtime.strict_paths)
    _validate_paths(cfg)
    _apply_runtime_perf_flags(cfg.runtime)
    return cfg, _attach_runtime_context(effective_raw, cfg)


def load_pipeline_config(
    config_path: Optional[Path] = None,
    *,
    profile: Optional[str] = None,
    variant: Optional[str] = None,
) -> PipelineConfig:
    """Public loader used by the rest of the application."""

    cfg, _ = load_pipeline_bundle(
        config_path=config_path,
        profile=profile,
        variant=variant,
    )
    return cfg


def _apply_runtime_perf_flags(runtime: RuntimeConfig) -> None:
    """Apply optional torch performance flags from the runtime config."""

    try:
        import torch
    except Exception:
        return

    try:
        threads = int(getattr(runtime, "torch_threads", 0) or 0)
        if threads > 0:
            torch.set_num_threads(threads)
    except Exception:
        pass

    try:
        if not torch.cuda.is_available():
            return

        try:
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass

        precision = str(getattr(runtime, "matmul_precision", "") or "").strip().lower()
        use_tf32 = bool(getattr(runtime, "cuda_tf32", True))

        # PyTorch 2.9 deprecates the older allow_tf32/set_float32_matmul_precision
        # path in favor of backend-local fp32_precision controls.
        matmul_mode = "ieee"
        if use_tf32 and precision not in {"highest", "ieee"}:
            matmul_mode = "tf32"

        try:
            if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
                torch.backends.cuda.matmul.fp32_precision = matmul_mode
            elif precision in {"highest", "high", "medium"}:
                torch.set_float32_matmul_precision(precision)
        except Exception:
            pass

        cudnn_mode = "tf32" if use_tf32 else "ieee"
        try:
            if hasattr(torch.backends.cudnn, "fp32_precision"):
                torch.backends.cudnn.fp32_precision = cudnn_mode
            if hasattr(torch.backends.cudnn, "conv") and hasattr(torch.backends.cudnn.conv, "fp32_precision"):
                torch.backends.cudnn.conv.fp32_precision = cudnn_mode
            if hasattr(torch.backends.cudnn, "rnn") and hasattr(torch.backends.cudnn.rnn, "fp32_precision"):
                torch.backends.cudnn.rnn.fp32_precision = cudnn_mode
            elif hasattr(torch.backends.cudnn, "allow_tf32"):
                torch.backends.cudnn.allow_tf32 = use_tf32
        except Exception:
            pass

        try:
            if not hasattr(torch.backends.cuda.matmul, "fp32_precision") and hasattr(torch.backends.cuda.matmul, "allow_tf32"):
                torch.backends.cuda.matmul.allow_tf32 = use_tf32
        except Exception:
            pass
    except Exception:
        return


def ensure_dirs(paths: PathsConfig, strict: bool = False) -> None:
    """Create required writable directories before the app starts."""

    required = [
        paths.data_dir,
        paths.videos_dir,
        paths.cache_dir,
        paths.indexes_dir,
        paths.thumbs_dir,
        paths.models_dir,
        paths.profiles_dir,
        paths.variants_dir,
    ]
    for item in required:
        try:
            item.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            if strict:
                raise RuntimeError(f"Could not create directory: {item}") from exc


def _validate_paths(cfg: PipelineConfig) -> None:
    """Validate only the paths that must exist for the current config to be usable."""

    strict = bool(getattr(cfg.runtime, "strict_paths", False))

    model_ref = str(cfg.model.model_name_or_path or "").strip()
    if strict and model_ref and (not _looks_like_hf_id(model_ref)):
        model_path = Path(model_ref)
        if not model_path.exists():
            raise FileNotFoundError(f"Model path not found: {model_path}")

    if strict:
        for name, item in (
            ("ui.ui_text_path", cfg.ui.ui_text_path),
            ("ui.styles_path", cfg.ui.styles_path),
            ("ui.logo_path", cfg.ui.logo_path),
        ):
            if item and not Path(item).exists():
                raise FileNotFoundError(f"{name} not found: {item}")

        if cfg.video.face_blur:
            if not cfg.paths.dnn_face_proto.exists():
                raise FileNotFoundError(f"dnn_face_proto not found: {cfg.paths.dnn_face_proto}")
            if not cfg.paths.dnn_face_model.exists():
                raise FileNotFoundError(f"dnn_face_model not found: {cfg.paths.dnn_face_model}")
