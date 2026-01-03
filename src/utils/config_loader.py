# src/utils/config_loader.py
"""
Helpers for loading pipeline configuration from config/pipeline.yaml
and resolving all paths relative to project root.

New storage layout:
- data/raw
- data/prepared
- data/runs   (single source of truth for run outputs)
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
    SearchConfig,
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
    except Exception:
        pass


def ensure_dirs(paths: PathsConfig, strict: bool = False) -> None:
    for d in (
        paths.data_dir,
        paths.raw_dir,
        paths.prepared_dir,
        paths.runs_dir,
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

    raw_dir = rel_default("raw_dir", "data/raw")
    prepared_dir = rel_default("prepared_dir", "data/prepared")
    runs_dir = rel_default("runs_dir", "data/runs")

    indexes_dir = rel_default("indexes_dir", "data/indexes")
    thumbs_dir = rel_default("thumbs_dir", "data/thumbs")

    paths = PathsConfig(
        root_dir=root_dir,
        data_dir=data_dir,
        raw_dir=raw_dir,
        prepared_dir=prepared_dir,
        runs_dir=runs_dir,
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

    search_raw = raw.get("search") or {}
    search = SearchConfig(
        embed_model_name=str(search_raw.get("embed_model_name", "intfloat/multilingual-e5-base")),
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
        jpeg_quality_frames=int(video_raw.get("jpeg_quality_frames", 82)),
        jpeg_quality_small=int(video_raw.get("jpeg_quality_small", 75)),
        face_detect_every_saved=int(video_raw.get("face_detect_every_saved", 6)),
        face_detect_force_on_scenecut=_to_bool(video_raw.get("face_detect_force_on_scenecut", True)),
        face_blur_strength=int(video_raw.get("face_blur_strength", 31)),
        face_conf_threshold=float(video_raw.get("face_conf_threshold", 0.5)),
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
    )

    cfg = PipelineConfig(
        paths=paths,
        ui=ui,
        search=search,
        video=video,
        clips=clips,
        model=model,
        runtime=runtime,
    )

    ensure_dirs(cfg.paths, strict=cfg.runtime.strict_paths)
    _validate_paths(cfg)
    _apply_runtime_perf_flags(cfg.runtime)

    return cfg
