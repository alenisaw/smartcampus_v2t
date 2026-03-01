# src/utils/config_loader.py
"""
Helpers for loading pipeline configuration from configs/pipeline.yaml
and resolving all paths relative to project root.

New storage layout:
- data/videos (per-video storage: raw/cache/outputs)
- data/cache  (global caches)
- data/indexes
- data/thumbs
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from src.pipeline.pipeline_config import (
    PathsConfig,
    UiConfig,
    BackendConfig,
    SearchConfig,
    VideoConfig,
    ClipsConfig,
    ModelConfig,
    RuntimeConfig,
    TranslationConfig,
    PipelineConfig,
)


def _to_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.lower() in {"1", "true", "yes", "y"}
    return bool(v)


def _looks_like_hf_id(s: str) -> bool:
    s = (s or "").strip()
    return ("/" in s) and (not s.startswith(".")) and (not s.startswith("/")) and (":" not in s)


def _resolve_maybe_relative(root_dir: Path, p: Any) -> Path:
    p = Path(str(p))
    return p if p.is_absolute() else (root_dir / p).resolve()


def _resolve_model_name_or_path(root_dir: Path, raw_value: Optional[str], fallback_path: Path) -> str:
    if raw_value is None or not str(raw_value).strip():
        return str(fallback_path)

    s = str(raw_value).strip()
    if _looks_like_hf_id(s):
        return s

    return str(_resolve_maybe_relative(root_dir, s))


def _validate_raw_config(raw: Dict[str, Any], config_path: Path) -> None:
    errors: list[str] = []

    def req_section(name: str) -> Dict[str, Any]:
        sec = raw.get(name)
        if not isinstance(sec, dict):
            errors.append(f"Missing or invalid section: {name}")
            return {}
        return sec

    paths_raw = req_section("paths")
    for key in ("root_dir", "data_dir"):
        if key not in paths_raw:
            errors.append(f"paths.{key} is required")

    ui_raw = req_section("ui")
    if "langs" in ui_raw and not isinstance(ui_raw.get("langs"), list):
        errors.append("ui.langs must be a list")

    search_raw = req_section("search")
    for key in ("embed_model_name", "w_bm25", "w_dense"):
        if key not in search_raw:
            errors.append(f"search.{key} is required")

    video_raw = req_section("video")
    for key in ("target_fps", "model_input_size", "min_change_threshold", "dark_threshold"):
        if key not in video_raw:
            errors.append(f"video.{key} is required")
    mis = video_raw.get("model_input_size")
    if isinstance(mis, (list, tuple)) and len(mis) != 2:
        errors.append("video.model_input_size must have 2 elements")

    clips_raw = req_section("clips")
    for key in ("window_sec", "stride_sec", "min_clip_frames", "max_clip_frames"):
        if key not in clips_raw:
            errors.append(f"clips.{key} is required")

    model_raw = req_section("model")
    for key in ("model_name_or_path", "device", "dtype", "language", "batch_size", "max_new_tokens"):
        if key not in model_raw:
            errors.append(f"model.{key} is required")

    runtime_raw = req_section("runtime")
    for key in ("seed", "num_workers", "log_level"):
        if key not in runtime_raw:
            errors.append(f"runtime.{key} is required")

    translation_raw = raw.get("translation")
    if translation_raw is not None and not isinstance(translation_raw, dict):
        errors.append("translation must be a dict if provided")

    if errors:
        msg = "\n".join(f"- {e}" for e in errors)
        raise ValueError(f"Invalid config: {config_path}\n{msg}")


def _apply_runtime_perf_flags(runtime: RuntimeConfig) -> None:
    try:
        import torch
    except Exception:
        return


    try:
        n = int(getattr(runtime, "torch_threads", 0) or 0)
        if n > 0:
            torch.set_num_threads(n)
    except Exception:
        pass

    try:
        if torch.cuda.is_available():
            try:
                torch.backends.cudnn.benchmark = True
            except Exception:
                pass

            prec = str(getattr(runtime, "matmul_precision", "") or "").strip().lower()
            try:
                if prec in {"highest", "high", "medium"}:
                    torch.set_float32_matmul_precision(prec)
            except Exception:
                pass

            if prec in {"highest", "high", "medium"}:
                if prec == "highest":
                    matmul = "ieee"
                    conv = "ieee"
                else:
                    matmul = "tf32"
                    conv = "tf32"
            else:
                use_tf32 = bool(getattr(runtime, "cuda_tf32", True))
                matmul = "tf32" if use_tf32 else "ieee"
                conv = "tf32" if use_tf32 else "ieee"

            torch.backends.cuda.matmul.fp32_precision = matmul
            torch.backends.cudnn.conv.fp32_precision = conv

            try:
                use_tf32 = bool(getattr(runtime, "cuda_tf32", True))
                torch.backends.cuda.matmul.allow_tf32 = use_tf32
                torch.backends.cudnn.allow_tf32 = use_tf32
            except Exception:
                pass

            try:
                torch.backends.cuda.enable_flash_sdp(True)
                torch.backends.cuda.enable_mem_efficient_sdp(True)
                torch.backends.cuda.enable_math_sdp(True)
            except Exception:
                pass
    except Exception:
        pass


def ensure_dirs(paths: PathsConfig, strict: bool = False) -> None:
    for d in (
        paths.data_dir,
        paths.videos_dir,
        paths.cache_dir,
        paths.indexes_dir,
        paths.thumbs_dir,
        paths.models_dir,
    ):
        try:
            d.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            if strict:
                raise RuntimeError(f"Could not create directory: {d}") from e


def _validate_paths(cfg: PipelineConfig) -> None:
    strict = bool(getattr(cfg.runtime, "strict_paths", False))

    m = str(cfg.model.model_name_or_path or "").strip()
    if m and (not _looks_like_hf_id(m)):
        mp = Path(m)
        if not mp.is_absolute():
            mp = (cfg.paths.root_dir / mp).resolve()
        if strict and not mp.exists():
            raise FileNotFoundError(f"Model path not found: {mp} (cfg.model.model_name_or_path={m!r})")

    if strict:
        for name, p in (
            ("ui.ui_text_path", cfg.ui.ui_text_path),
            ("ui.styles_path", cfg.ui.styles_path),
            ("ui.logo_path", cfg.ui.logo_path),
        ):
            if p and not Path(p).exists():
                raise FileNotFoundError(f"{name} not found: {p}")

    if strict and bool(getattr(cfg.video, "face_blur", False)):
        if not cfg.paths.dnn_face_proto.exists():
            raise FileNotFoundError(f"dnn_face_proto not found: {cfg.paths.dnn_face_proto}")
        if not cfg.paths.dnn_face_model.exists():
            raise FileNotFoundError(f"dnn_face_model not found: {cfg.paths.dnn_face_model}")


def load_pipeline_config(config_path: Path) -> PipelineConfig:
    config_path = Path(config_path).resolve()
    with config_path.open("r", encoding="utf-8") as f:
        raw: Dict[str, Any] = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Invalid config: {config_path} (expected YAML dict)")

    _validate_raw_config(raw, config_path)

    paths_raw = raw["paths"]
    root_dir = (config_path.parent / paths_raw.get("root_dir", ".")).resolve()

    def rel(p: str) -> Path:
        return (root_dir / p).resolve()

    data_dir = rel(paths_raw.get("data_dir", "data"))

    def rel_default(key: str, default_rel: str) -> Path:
        v = paths_raw.get(key)
        if v is None or str(v).strip() == "":
            return rel(default_rel)
        return rel(str(v))

    videos_dir = rel_default("videos_dir", "data/videos")
    cache_dir = rel_default("cache_dir", "data/cache")
    indexes_dir = rel_default("indexes_dir", "data/indexes")
    thumbs_dir = rel_default("thumbs_dir", "data/thumbs")

    paths = PathsConfig(
        root_dir=root_dir,
        data_dir=data_dir,
        videos_dir=videos_dir,
        cache_dir=cache_dir,
        indexes_dir=indexes_dir,
        thumbs_dir=thumbs_dir,
        models_dir=rel_default("models_dir", "models"),
        qwen_vl_dir=rel_default("qwen_vl_dir", "models/qwen3-vl-2b-instruct"),
        dnn_face_proto=rel_default("dnn_face_proto", "models/face_detector/deploy.prototxt"),
        dnn_face_model=rel_default("dnn_face_model", "models/face_detector/res10_300x300_ssd_iter_140000.caffemodel"),
    )

    ui_raw = raw.get("ui") or {}
    langs = ui_raw.get("langs") or ["ru", "kz", "en"]
    if not isinstance(langs, list) or not langs:
        langs = ["ru", "kz", "en"]

    default_lang = str(ui_raw.get("default_lang", "ru")).strip().lower() or "ru"
    if default_lang not in [str(x).strip().lower() for x in langs]:
        default_lang = str(langs[0]).strip().lower()

    ui = UiConfig(
        langs=[str(x).strip().lower() for x in langs if str(x).strip()],
        default_lang=default_lang,
        cache_ttl_sec=int(ui_raw.get("cache_ttl_sec", 2)),
        assets_dir=rel(str(ui_raw.get("assets_dir", "app/assets"))),
        ui_text_path=rel(str(ui_raw.get("ui_text_path", "app/assets/ui_text.json"))),
        styles_path=rel(str(ui_raw.get("styles_path", "app/assets/styles.css"))),
        logo_path=rel(str(ui_raw.get("logo_path", "app/assets/logo.png"))),
    )

    backend_raw = raw.get("backend") or {}
    backend = BackendConfig(
        scheme=str(backend_raw.get("scheme", "http")).strip().lower() or "http",
        host=str(backend_raw.get("host", "127.0.0.1")).strip() or "127.0.0.1",
        port=int(backend_raw.get("port", 8000)),
    )

    search_raw = raw.get("search") or {}
    search = SearchConfig(
        embed_model_name=str(search_raw.get("embed_model_name", "intfloat/multilingual-e5-base")),
        query_prefix=str(search_raw.get("query_prefix", "query: ")),
        passage_prefix=str(search_raw.get("passage_prefix", "passage: ")),
        normalize_text=_to_bool(search_raw.get("normalize_text", True)),
        lemmatize=_to_bool(search_raw.get("lemmatize", False)),
        fallback_langs=[str(x).strip().lower() for x in (search_raw.get("fallback_langs") or ["en"]) if str(x).strip()],
        translate_queries=_to_bool(search_raw.get("translate_queries", False)),
        embed_cache=_to_bool(search_raw.get("embed_cache", True)),
        w_bm25=float(search_raw.get("w_bm25", 0.45)),
        w_dense=float(search_raw.get("w_dense", 0.55)),
        candidate_k_sparse=int(search_raw.get("candidate_k_sparse", 200)),
        candidate_k_dense=int(search_raw.get("candidate_k_dense", 200)),
        fusion=str(search_raw.get("fusion", "rrf")),
        rrf_k=int(search_raw.get("rrf_k", 60)),
        dedupe_mode=str(search_raw.get("dedupe_mode", "overlap")),
        dedupe_tol_sec=float(search_raw.get("dedupe_tol_sec", 1.0)),
        dedupe_overlap_thr=float(search_raw.get("dedupe_overlap_thr", 0.7)),
    )

    video_raw = raw["video"]
    video_io_raw = raw.get("video_io") or {}

    def _video_io_get(key: str, default: Any) -> Any:
        if key in video_io_raw:
            return video_io_raw.get(key, default)
        return video_raw.get(key, default)

    video = VideoConfig(
        target_fps=float(video_raw["target_fps"]),
        model_input_size=tuple(video_raw["model_input_size"]),
        min_change_threshold=float(video_raw["min_change_threshold"]),
        dark_threshold=float(video_raw["dark_threshold"]),
        save_small_frames=_to_bool(video_raw.get("save_small_frames", False)),
        face_blur=_to_bool(video_raw.get("face_blur", False)),
        face_detector_type=str(video_raw.get("face_detector_type", "dnn")),
        max_frames=video_raw.get("max_frames"),

        # moved from video_io.py constants
        jpeg_quality_frames=int(_video_io_get("jpeg_quality_frames", 82)),
        jpeg_quality_small=int(_video_io_get("jpeg_quality_small", 75)),
        face_detect_every_saved=int(_video_io_get("face_detect_every_saved", 6)),
        face_detect_force_on_scenecut=_to_bool(_video_io_get("face_detect_force_on_scenecut", True)),
        face_blur_strength=int(_video_io_get("face_blur_strength", 31)),
        face_conf_threshold=float(_video_io_get("face_conf_threshold", 0.5)),
    )

    clips_raw = raw["clips"]
    clips = ClipsConfig(
        window_sec=float(clips_raw["window_sec"]),
        stride_sec=float(clips_raw["stride_sec"]),
        min_clip_frames=int(clips_raw["min_clip_frames"]),
        max_clip_frames=int(clips_raw["max_clip_frames"]),
    )

    model_raw = raw["model"]
    model_name_or_path = _resolve_model_name_or_path(
        root_dir=root_dir,
        raw_value=model_raw.get("model_name_or_path"),
        fallback_path=paths.qwen_vl_dir,
    )

    model = ModelConfig(
        model_name_or_path=str(model_name_or_path),
        device=str(model_raw.get("device", "cuda")),
        dtype=str(model_raw.get("dtype", "fp16")),
        language=str(model_raw.get("language", "ru")),
        batch_size=int(model_raw.get("batch_size", 1)),
        max_new_tokens=int(model_raw.get("max_new_tokens", 128)),
        temperature=float(model_raw.get("temperature", 0.2)),
        top_p=float(model_raw.get("top_p", 0.9)),
        top_k=int(model_raw.get("top_k", 40)),
        repetition_penalty=float(model_raw.get("repetition_penalty", 1.0)),
        do_sample=_to_bool(model_raw.get("do_sample", False)),
        attn_implementation=str(model_raw.get("attn_implementation", "auto")),
        max_batch_frames=int(model_raw.get("max_batch_frames", 0)),
        timeout_sec=int(model_raw.get("timeout_sec", 60)),
    )

    runtime_raw = raw["runtime"]
    runtime = RuntimeConfig(
        seed=int(runtime_raw.get("seed", 42)),
        num_workers=int(runtime_raw.get("num_workers", 4)),
        log_level=str(runtime_raw.get("log_level", "INFO")),
        overwrite_existing=_to_bool(runtime_raw.get("overwrite_existing", False)),
        strict_paths=_to_bool(runtime_raw.get("strict_paths", False)),
        torch_threads=int(runtime_raw.get("torch_threads", 0)),
        cuda_tf32=_to_bool(runtime_raw.get("cuda_tf32", True)),
        matmul_precision=str(runtime_raw.get("matmul_precision", "high")),
        autocast_infer=_to_bool(runtime_raw.get("autocast_infer", True)),
        torch_compile=_to_bool(runtime_raw.get("torch_compile", False)),
        torch_compile_mode=str(runtime_raw.get("torch_compile_mode", "reduce-overhead")),
        torch_compile_fullgraph=_to_bool(runtime_raw.get("torch_compile_fullgraph", False)),
    )

    translation_raw = raw.get("translation") or {}
    translation = TranslationConfig(
        model_name_or_path=str(translation_raw.get("model_name_or_path", "facebook/nllb-200-distilled-1.3B")),
        device=str(translation_raw.get("device", model.device)),
        dtype=str(translation_raw.get("dtype", model.dtype)),
        batch_size=int(translation_raw.get("batch_size", 8)),
        max_new_tokens=int(translation_raw.get("max_new_tokens", 128)),
        source_lang=str(translation_raw.get("source_lang", "en")),
        target_langs=[str(x) for x in (translation_raw.get("target_langs") or ["ru", "kz"])],
        cache_enabled=_to_bool(translation_raw.get("cache_enabled", True)),
    )

    cfg = PipelineConfig(
        paths=paths,
        ui=ui,
        backend=backend,
        search=search,
        video=video,
        clips=clips,
        model=model,
        runtime=runtime,
        translation=translation,
    )

    ensure_dirs(cfg.paths, strict=cfg.runtime.strict_paths)
    _validate_paths(cfg)
    _apply_runtime_perf_flags(cfg.runtime)

    return cfg
