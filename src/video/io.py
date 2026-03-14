# src/video/io.py
"""
Video preprocessing for SmartCampus V2T.

Purpose:
- Decode, normalize, filter, and persist frames for downstream stages.
- Handle FPS normalization, anonymization, quality filtering, and cached video metadata.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.core.config import PipelineConfig
from src.core.types import FrameInfo, VideoMeta
from src.utils.video_store import cache_dir as video_cache_dir
from src.utils.video_store import find_video_file

SMALL_SIZE: Tuple[int, int] = (64, 64)
FaceBox = Tuple[int, int, int, int]


@dataclass
class _VideoIoSettings:
    jpeg_q_frames: int
    jpeg_q_small: int
    face_detect_every_saved: int
    face_detect_force_on_scenecut: bool
    face_blur_strength: int
    face_conf_threshold: float


@dataclass
class _VideoProcessingState:
    frame_idx: int = 0
    prev_small_kept: Optional[np.ndarray] = None
    warmup_diffs: List[float] = field(default_factory=list)
    warmup_done: bool = False
    adaptive_threshold: float = 0.0
    last_face_boxes: List[FaceBox] = field(default_factory=list)
    last_face_detect_saved_idx: int = -10_000


@dataclass
class _VideoProcessingStats:
    num_raw_frames: int
    num_sampled_frames: int = 0
    num_saved_frames: int = 0
    num_dark_skipped: int = 0
    num_lazy_skipped: int = 0
    num_blur_flagged: int = 0
    num_blur_skipped: int = 0
    num_scene_cuts: int = 0
    raw_read_frames: int = 0


@dataclass
class _VideoPreparationContext:
    video_id: str
    video_path: Path
    prepared_root: Path
    frames_dir: Path
    small_dir: Optional[Path]
    decode_source_path: Path
    decode_meta: Dict[str, Any]
    source_fps: float
    fps: float
    target_fps: float
    step_frames: int
    duration_sec: float
    num_raw_frames: int
    model_input_size: Tuple[int, int]
    base_threshold: float
    dark_threshold: float
    blur_threshold: float
    max_saved: Optional[int]
    face_detector: Any
    face_detector_type: str
    clahe: Any
    io_settings: _VideoIoSettings


def _cfg_video_io_get(config: PipelineConfig, key: str, default: Any) -> Any:
    try:
        vio = getattr(config, "video_io", None)
        if vio is None:
            vio = getattr(config, "videoio", None)
        if vio is None:
            vio = getattr(config, "video_io_cfg", None)
        if vio is None:
            vio = getattr(config, "video", None)
        if vio is None:
            return default
        return getattr(vio, key, default)
    except Exception:
        return default


def preprocess_video(video_path: str | Path, config: PipelineConfig) -> VideoMeta:
    """Preprocess one video path into cached frame metadata."""

    video_path = Path(video_path).resolve()
    video_id = video_path.stem
    return prepare_video(video_id=video_id, config=config, video_path=video_path)


def prepare_video(
    video_id: str,
    config: PipelineConfig,
    video_path: Optional[Path] = None,
) -> VideoMeta:
    """Prepare cached frame metadata for one video id."""

    paths_cfg = config.paths
    video_cfg = config.video
    video_path = _resolve_video_path(video_id, Path(paths_cfg.videos_dir), video_path)
    io_settings = _load_video_io_settings(config)

    video_fp = _file_fingerprint(video_path)
    cfg_fp = _video_cfg_fingerprint(config)

    cache_hit = _try_load_cached_video_meta(
        prepared_base=video_cache_dir(Path(paths_cfg.videos_dir), video_id),
        video_fp=video_fp,
        cfg_fp=cfg_fp,
    )
    if cache_hit is not None:
        try:
            if cache_hit.extra is None:
                cache_hit.extra = {}
            cache_hit.extra.setdefault("cache", {})
            cache_hit.extra["cache"]["hit"] = True
        except Exception:
            pass
        return cache_hit

    prepared_root, frames_dir, small_dir = _ensure_cache_dirs(
        prepared_base=video_cache_dir(Path(paths_cfg.videos_dir), video_id),
        video_fp=video_fp,
        cfg_fp=cfg_fp,
        save_small=bool(video_cfg.save_small_frames),
    )

    preprocess_started = time.perf_counter()
    context = _build_video_preparation_context(
        video_id=video_id,
        video_path=video_path,
        config=config,
        prepared_root=prepared_root,
        frames_dir=frames_dir,
        small_dir=small_dir,
        io_settings=io_settings,
    )
    frames, state, stats = _process_video_frames(context)
    preprocess_time_sec = float(time.perf_counter() - preprocess_started)
    early_stop = _warn_if_early_stop(video_path, stats)

    video_meta = _build_video_meta(
        context=context,
        frames=frames,
        state=state,
        stats=stats,
        preprocess_time_sec=preprocess_time_sec,
        early_stop=early_stop,
        video_fp=video_fp,
        cfg_fp=cfg_fp,
    )

    _write_meta_json(video_meta)
    return video_meta


def _resolve_video_path(video_id: str, videos_dir: Path, video_path: Optional[Path]) -> Path:
    if video_path is None:
        video_path = find_video_file(videos_dir, video_id)
        if video_path is None:
            video_path = videos_dir / video_id / "raw" / f"{video_id}.mp4"
    resolved = Path(video_path).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Video not found: {resolved}")
    return resolved


def _load_video_io_settings(config: PipelineConfig) -> _VideoIoSettings:
    return _VideoIoSettings(
        jpeg_q_frames=int(_cfg_video_io_get(config, "jpeg_quality_frames", 82)),
        jpeg_q_small=int(_cfg_video_io_get(config, "jpeg_quality_small", 75)),
        face_detect_every_saved=int(_cfg_video_io_get(config, "face_detect_every_saved", 6)),
        face_detect_force_on_scenecut=bool(_cfg_video_io_get(config, "face_detect_force_on_scenecut", True)),
        face_blur_strength=int(_cfg_video_io_get(config, "face_blur_strength", 31)),
        face_conf_threshold=float(_cfg_video_io_get(config, "face_conf_threshold", 0.5)),
    )


def _build_video_preparation_context(
    *,
    video_id: str,
    video_path: Path,
    config: PipelineConfig,
    prepared_root: Path,
    frames_dir: Path,
    small_dir: Optional[Path],
    io_settings: _VideoIoSettings,
) -> _VideoPreparationContext:
    paths_cfg = config.paths
    video_cfg = config.video

    decode_started = time.perf_counter()
    decode_source_path, decode_meta = _prepare_decode_source(
        source_path=video_path,
        out_dir=prepared_root,
        target_fps=float(getattr(video_cfg, "target_fps", 0.0) or 0.0),
        resolution=tuple(getattr(video_cfg, "decode_resolution", (1280, 720)) or (1280, 720)),
        pixel_format=str(getattr(video_cfg, "pixel_format", "yuv420p") or "yuv420p"),
    )
    decode_meta["time_sec"] = float(time.perf_counter() - decode_started)

    video_info = _get_video_info(str(decode_source_path))
    source_fps = float(video_info["fps"])
    fps = _normalize_video_fps(video_path, source_fps)
    target_fps = float(video_cfg.target_fps)
    step_frames = max(1, int(round(fps / target_fps))) if target_fps > 0 else 1
    face_detector, face_detector_type = _build_face_detector(video_cfg, paths_cfg)

    return _VideoPreparationContext(
        video_id=video_id,
        video_path=video_path,
        prepared_root=prepared_root,
        frames_dir=frames_dir,
        small_dir=small_dir,
        decode_source_path=decode_source_path,
        decode_meta=decode_meta,
        source_fps=source_fps,
        fps=fps,
        target_fps=target_fps,
        step_frames=step_frames,
        duration_sec=float(video_info["duration_sec"]),
        num_raw_frames=int(video_info["total_frames"]),
        model_input_size=tuple(video_cfg.model_input_size),
        base_threshold=float(video_cfg.min_change_threshold),
        dark_threshold=float(video_cfg.dark_threshold),
        blur_threshold=float(video_cfg.blur_threshold),
        max_saved=int(video_cfg.max_frames) if video_cfg.max_frames is not None else None,
        face_detector=face_detector,
        face_detector_type=face_detector_type,
        clahe=cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)),
        io_settings=io_settings,
    )


def _normalize_video_fps(video_path: Path, source_fps: float) -> float:
    fps = source_fps
    clamped = False
    if fps < 1:
        fps = 25.0
        clamped = True
    elif fps > 120:
        fps = 30.0
        clamped = True
    fps = float(round(fps))
    if clamped:
        print(
            f"[prepare_video] Unstable FPS {source_fps:.3f} for {video_path}, "
            f"using fallback {fps:.1f}"
        )
    return fps


def _build_face_detector(video_cfg: Any, paths_cfg: Any) -> Tuple[Any, str]:
    if not bool(video_cfg.face_blur):
        return None, "none"
    detector = _load_face_detector_dnn(
        paths_cfg.dnn_face_proto,
        paths_cfg.dnn_face_model,
    )
    return detector, "dnn"


def _process_video_frames(
    context: _VideoPreparationContext,
) -> Tuple[List[FrameInfo], _VideoProcessingState, _VideoProcessingStats]:
    cap = cv2.VideoCapture(str(context.decode_source_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video {context.decode_source_path}")

    frames: List[FrameInfo] = []
    state = _VideoProcessingState(adaptive_threshold=context.base_threshold * 0.5)
    stats = _VideoProcessingStats(num_raw_frames=context.num_raw_frames)

    try:
        while True:
            ok = cap.grab()
            if not ok:
                break

            stats.raw_read_frames += 1

            if state.frame_idx % context.step_frames != 0:
                state.frame_idx += 1
                continue

            ok, frame = cap.retrieve()
            if not ok:
                state.frame_idx += 1
                continue

            timestamp = state.frame_idx / context.fps
            stats.num_sampled_frames += 1

            small_gray = _downscale_for_analysis(frame)
            blur_score, blur_flag, blur_skip = _assess_blur_frame(
                small_gray,
                context.blur_threshold,
            )
            if blur_flag:
                stats.num_blur_flagged += 1
            if blur_skip:
                stats.num_blur_skipped += 1
                state.frame_idx += 1
                continue

            frame, is_dark, dark_skip = _restore_dark_frame(
                frame,
                small_gray,
                context.dark_threshold,
                context.clahe,
            )
            if is_dark:
                stats.num_dark_skipped += 1
            if dark_skip:
                state.frame_idx += 1
                continue

            if _consume_warmup_frame(state, small_gray, context.base_threshold):
                state.frame_idx += 1
                continue

            lazy_ok, is_scene_cut = _evaluate_motion_gate(
                prev_small_kept=state.prev_small_kept,
                small_gray=small_gray,
                adaptive_threshold=state.adaptive_threshold,
                stats=stats,
            )
            if not lazy_ok:
                stats.num_lazy_skipped += 1
                state.frame_idx += 1
                continue

            frame = _maybe_blur_faces(
                frame=frame,
                context=context,
                state=state,
                stats=stats,
                is_scene_cut=is_scene_cut,
            )

            frames.append(
                _save_prepared_frame(
                    context=context,
                    saved_index=len(frames),
                    frame_idx=state.frame_idx,
                    timestamp=timestamp,
                    frame=frame,
                    small_gray=small_gray,
                    blur_score=blur_score,
                    blur_flag=blur_flag,
                    is_scene_cut=is_scene_cut,
                )
            )

            state.prev_small_kept = small_gray
            stats.num_saved_frames += 1

            if context.max_saved is not None and stats.num_saved_frames >= context.max_saved:
                break

            state.frame_idx += 1
    finally:
        cap.release()

    return frames, state, stats


def _assess_blur_frame(small_gray: np.ndarray, blur_threshold: float) -> Tuple[float, bool, bool]:
    blur_score = _blur_variance_score(small_gray)
    blur_flag = blur_score < blur_threshold
    blur_skip = blur_flag and blur_score < max(1.0, blur_threshold * 0.35)
    return blur_score, blur_flag, blur_skip


def _restore_dark_frame(
    frame: np.ndarray,
    small_gray: np.ndarray,
    dark_threshold: float,
    clahe: Any,
) -> Tuple[np.ndarray, bool, bool]:
    is_dark = _is_dark_frame(small_gray, brightness_threshold=dark_threshold)
    if not is_dark:
        return frame, False, False

    mean_brightness = float(np.mean(small_gray))
    if mean_brightness < dark_threshold * 0.4:
        return frame, True, True

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    brightened = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return brightened, True, False


def _consume_warmup_frame(
    state: _VideoProcessingState,
    small_gray: np.ndarray,
    base_threshold: float,
) -> bool:
    if state.warmup_done:
        return False

    if state.prev_small_kept is not None:
        state.warmup_diffs.append(_compute_frame_change_cpu(state.prev_small_kept, small_gray))
    state.prev_small_kept = small_gray

    if len(state.warmup_diffs) >= 8:
        scene_motion = float(np.mean(state.warmup_diffs)) if state.warmup_diffs else 0.0
        state.adaptive_threshold = max(
            base_threshold * 0.25,
            min(base_threshold * 0.7, scene_motion * 1.2),
        )
        state.warmup_done = True

    return True


def _evaluate_motion_gate(
    *,
    prev_small_kept: Optional[np.ndarray],
    small_gray: np.ndarray,
    adaptive_threshold: float,
    stats: _VideoProcessingStats,
) -> Tuple[bool, bool]:
    lazy_ok = True
    is_scene_cut = False
    if prev_small_kept is None:
        return lazy_ok, is_scene_cut

    diff = _compute_frame_change_cpu(prev_small_kept, small_gray)
    if diff < adaptive_threshold:
        lazy_ok = False
    if diff > adaptive_threshold * 8.0:
        stats.num_scene_cuts += 1
        is_scene_cut = True
    return lazy_ok, is_scene_cut


def _maybe_blur_faces(
    *,
    frame: np.ndarray,
    context: _VideoPreparationContext,
    state: _VideoProcessingState,
    stats: _VideoProcessingStats,
    is_scene_cut: bool,
) -> np.ndarray:
    if context.face_detector is None:
        return frame

    need_detect = (stats.num_saved_frames - state.last_face_detect_saved_idx) >= context.io_settings.face_detect_every_saved
    if context.io_settings.face_detect_force_on_scenecut and is_scene_cut:
        need_detect = True

    if need_detect:
        try:
            state.last_face_boxes = _detect_faces_dnn(
                context.face_detector,
                frame,
                conf_threshold=context.io_settings.face_conf_threshold,
            )
            state.last_face_detect_saved_idx = stats.num_saved_frames
        except Exception:
            state.last_face_boxes = []

    if not state.last_face_boxes:
        return frame

    return _blur_boxes(
        frame,
        state.last_face_boxes,
        blur_strength=context.io_settings.face_blur_strength,
    )


def _save_prepared_frame(
    *,
    context: _VideoPreparationContext,
    saved_index: int,
    frame_idx: int,
    timestamp: float,
    frame: np.ndarray,
    small_gray: np.ndarray,
    blur_score: float,
    blur_flag: bool,
    is_scene_cut: bool,
) -> FrameInfo:
    resized = _letterbox_resize(frame, target_size=context.model_input_size)
    frame_name = f"frame_{frame_idx:06d}.jpg"
    frame_path = context.frames_dir / frame_name
    _save_frame(resized, frame_path, quality=context.io_settings.jpeg_q_frames)

    small_path: Optional[Path] = None
    if context.small_dir is not None:
        small_path = context.small_dir / frame_name
        _save_small_frame(small_gray, small_path, quality=context.io_settings.jpeg_q_small)

    return FrameInfo(
        video_id=context.video_id,
        frame_index=saved_index,
        timestamp_sec=float(timestamp),
        path=frame_path,
        small_path=small_path,
        extra={
            "blur_score": float(blur_score),
            "blur_flag": bool(blur_flag),
            "scene_cut": bool(is_scene_cut),
        },
    )


def _warn_if_early_stop(video_path: Path, stats: _VideoProcessingStats) -> bool:
    if stats.num_raw_frames <= 0:
        return False
    if stats.raw_read_frames >= int(0.9 * stats.num_raw_frames):
        return False

    print(
        f"[prepare_video] Warning: read only {stats.raw_read_frames}/{stats.num_raw_frames} frames "
        f"for {video_path} (~{100.0 * stats.raw_read_frames / max(1, stats.num_raw_frames):.1f}%)."
    )
    return True


def _build_video_meta(
    *,
    context: _VideoPreparationContext,
    frames: List[FrameInfo],
    state: _VideoProcessingState,
    stats: _VideoProcessingStats,
    preprocess_time_sec: float,
    early_stop: bool,
    video_fp: str,
    cfg_fp: str,
) -> VideoMeta:
    return VideoMeta(
        video_id=context.video_id,
        original_path=context.video_path,
        original_fps=float(context.fps),
        duration_sec=float(context.duration_sec),
        processed_fps=context.target_fps,
        num_frames=len(frames),
        prepared_dir=context.prepared_root,
        frame_dir=context.frames_dir,
        small_frame_dir=context.small_dir,
        frames=frames,
        extra={
            "frame_stats": {
                "source_fps": context.source_fps,
                "num_raw_frames": stats.num_raw_frames,
                "raw_read_frames": stats.raw_read_frames,
                "num_sampled_frames": stats.num_sampled_frames,
                "num_saved_frames": stats.num_saved_frames,
                "num_dark_skipped": stats.num_dark_skipped,
                "num_lazy_skipped": stats.num_lazy_skipped,
                "num_blur_flagged": stats.num_blur_flagged,
                "num_blur_skipped": stats.num_blur_skipped,
                "num_scene_cuts": stats.num_scene_cuts,
                "early_stop": early_stop,
            },
            "sampling": {"step_frames": int(context.step_frames)},
            "adaptive_threshold": float(state.adaptive_threshold),
            "preprocess_time_sec": float(preprocess_time_sec),
            "face_detector_type": context.face_detector_type,
            "cache": {
                "hit": False,
                "video_fingerprint": video_fp,
                "video_cfg_fingerprint": cfg_fp,
            },
            "decode": context.decode_meta,
            "video_io": {
                "jpeg_quality_frames": context.io_settings.jpeg_q_frames,
                "jpeg_quality_small": context.io_settings.jpeg_q_small,
                "face_detect_every_saved": context.io_settings.face_detect_every_saved,
                "face_detect_force_on_scenecut": context.io_settings.face_detect_force_on_scenecut,
                "face_blur_strength": context.io_settings.face_blur_strength,
                "face_conf_threshold": context.io_settings.face_conf_threshold,
                "blur_threshold": context.blur_threshold,
            },
        },
    )


def _file_fingerprint(p: Path) -> str:
    st = p.stat()
    return f"{st.st_size}:{st.st_mtime_ns}"


def _short_hash(s: str, n: int = 12) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:n]


def _video_cfg_fingerprint(config: PipelineConfig) -> str:
    try:
        v = config.video
        paths = config.paths
        vio = getattr(config, "video_io", None)

        payload = {
            "target_fps": getattr(v, "target_fps", None),
            "analysis_fps": getattr(v, "analysis_fps", None),
            "decode_resolution": tuple(getattr(v, "decode_resolution", ()) or ()),
            "pixel_format": str(getattr(v, "pixel_format", "")),
            "min_change_threshold": getattr(v, "min_change_threshold", None),
            "dark_threshold": getattr(v, "dark_threshold", None),
            "max_frames": getattr(v, "max_frames", None),
            "model_input_size": tuple(getattr(v, "model_input_size", ()) or ()),
            "save_small_frames": bool(getattr(v, "save_small_frames", False)),
            "face_blur": bool(getattr(v, "face_blur", False)),
            "dnn_face_proto": str(getattr(paths, "dnn_face_proto", "")),
            "dnn_face_model": str(getattr(paths, "dnn_face_model", "")),
            "video_io": {
                "jpeg_quality_frames": getattr(vio, "jpeg_quality_frames", None) if vio else None,
                "jpeg_quality_small": getattr(vio, "jpeg_quality_small", None) if vio else None,
                "face_detect_every_saved": getattr(vio, "face_detect_every_saved", None) if vio else None,
                "face_detect_force_on_scenecut": getattr(vio, "face_detect_force_on_scenecut", None) if vio else None,
                "face_blur_strength": getattr(vio, "face_blur_strength", None) if vio else None,
                "face_conf_threshold": getattr(vio, "face_conf_threshold", None) if vio else None,
            },
        }
    except Exception:
        payload = {"fallback": True}

    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16]


def _cache_dir(prepared_base: Path, video_fp: str, cfg_fp: str) -> Path:
    vtag = _short_hash(video_fp, 12)
    return Path(prepared_base) / f"v{vtag}" / f"c{cfg_fp}"


def _try_load_cached_video_meta(
    prepared_base: Path,
    video_fp: str,
    cfg_fp: str,
) -> Optional[VideoMeta]:
    root = _cache_dir(prepared_base, video_fp, cfg_fp)
    meta_path = root / "meta.json"
    if not meta_path.exists():
        return None
    try:
        meta_obj = json.loads(meta_path.read_text(encoding="utf-8"))
        vm = _video_meta_from_json_obj(meta_obj)
        return vm
    except Exception:
        return None


def _video_meta_from_json_obj(obj: Dict[str, Any]) -> VideoMeta:
    def P(x: Any) -> Optional[Path]:
        if x is None:
            return None
        try:
            return Path(x)
        except Exception:
            return None

    frames_list: List[FrameInfo] = []
    for i, fr in enumerate(obj.get("frames") or []):
        if not isinstance(fr, dict):
            continue
        frames_list.append(
            FrameInfo(
                video_id=str(fr.get("video_id") or obj.get("video_id") or ""),
                frame_index=int(fr.get("frame_index", i)),
                timestamp_sec=float(fr.get("timestamp_sec", 0.0)),
                path=P(fr.get("path")) or Path(""),
                small_path=P(fr.get("small_path")),
            )
        )

    return VideoMeta(
        video_id=str(obj.get("video_id") or ""),
        original_path=P(obj.get("original_path")) or Path(""),
        original_fps=float(obj.get("original_fps", 0.0)),
        duration_sec=float(obj.get("duration_sec", 0.0)),
        processed_fps=float(obj.get("processed_fps", 0.0)),
        num_frames=int(obj.get("num_frames", len(frames_list))),
        prepared_dir=P(obj.get("prepared_dir")) or Path(""),
        frame_dir=P(obj.get("frame_dir")) or Path(""),
        small_frame_dir=P(obj.get("small_frame_dir")),
        frames=frames_list,
        extra=obj.get("extra") or {},
    )


def _ensure_cache_dirs(
    prepared_base: Path,
    video_fp: str,
    cfg_fp: str,
    save_small: bool,
) -> Tuple[Path, Path, Optional[Path]]:
    root = _cache_dir(prepared_base, video_fp, cfg_fp)
    frames_dir = root / "frames"
    small_dir = root / "small" if save_small else None

    frames_dir.mkdir(parents=True, exist_ok=True)
    if small_dir is not None:
        small_dir.mkdir(parents=True, exist_ok=True)

    return root, frames_dir, small_dir


def _write_meta_json(video_meta: VideoMeta) -> None:
    meta_dict = asdict(video_meta)

    def _convert_paths(obj: Any) -> Any:
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, list):
            return [_convert_paths(x) for x in obj]
        if isinstance(obj, dict):
            return {k: _convert_paths(v) for k, v in obj.items()}
        return obj

    meta_dict = _convert_paths(meta_dict)
    meta_path = video_meta.prepared_dir / "meta.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta_dict, f, indent=2, ensure_ascii=False)


def _get_video_info(video_path: str) -> Dict[str, float]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    if fps <= 0 or fps > 240:
        fps_for_duration = 25.0 if total_frames > 0 else 0.0
    else:
        fps_for_duration = fps

    duration = total_frames / fps_for_duration if fps_for_duration > 0 else 0.0
    cap.release()

    return {
        "fps": float(fps),
        "total_frames": int(total_frames),
        "duration_sec": float(duration),
    }


def _downscale_for_analysis(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, SMALL_SIZE, interpolation=cv2.INTER_AREA)


def _compute_frame_change_cpu(prev_small: np.ndarray, curr_small: np.ndarray) -> float:
    diff = cv2.absdiff(prev_small, curr_small)
    return float(diff.mean()) / 255.0


def _is_dark_frame(small_gray_frame: np.ndarray, brightness_threshold: float) -> bool:
    return float(np.mean(small_gray_frame)) < brightness_threshold


def _blur_variance_score(gray_frame: np.ndarray) -> float:
    """Estimate frame sharpness using variance of the Laplacian."""

    return float(cv2.Laplacian(gray_frame, cv2.CV_64F).var())


def _load_face_detector_dnn(proto_path: Path, model_path: Path):
    proto_path = proto_path.resolve()
    model_path = model_path.resolve()
    if not proto_path.exists():
        raise FileNotFoundError(f"DNN proto not found: {proto_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"DNN model not found: {model_path}")
    return cv2.dnn.readNetFromCaffe(str(proto_path), str(model_path))


def _detect_faces_dnn(net, frame: np.ndarray, conf_threshold: float = 0.5) -> List[Tuple[int, int, int, int]]:
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(
        frame,
        scalefactor=1.0,
        size=(300, 300),
        mean=(104.0, 177.0, 123.0),
    )
    net.setInput(blob)
    detections = net.forward()
    boxes: List[Tuple[int, int, int, int]] = []

    for i in range(detections.shape[2]):
        confidence = float(detections[0, 0, i, 2])
        if confidence < conf_threshold:
            continue
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h], dtype=np.float32)
        x1, y1, x2, y2 = box.astype(int)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        if x2 > x1 and y2 > y1:
            boxes.append((x1, y1, x2 - x1, y2 - y1))

    return boxes


def _blur_boxes(frame: np.ndarray, boxes: List[Tuple[int, int, int, int]], blur_strength: int = 31) -> np.ndarray:
    anon = frame.copy()
    if blur_strength % 2 == 0:
        blur_strength += 1
    if blur_strength < 3:
        blur_strength = 3

    for x, y, w, h in boxes:
        roi = anon[y:y + h, x:x + w]
        if roi.size > 0:
            anon[y:y + h, x:x + w] = cv2.GaussianBlur(
                roi,
                (blur_strength, blur_strength),
                0,
            )
    return anon


def _letterbox_resize(frame: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    th = int(target_size[0])
    tw = int(target_size[1])
    h, w = frame.shape[:2]
    scale = min(tw / float(w), th / float(h))
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
    result = np.full((th, tw, 3), 0, dtype=np.uint8)
    top = (th - nh) // 2
    left = (tw - nw) // 2
    result[top:top + nh, left:left + nw] = resized
    return result


def _save_frame(frame: np.ndarray, out_path: Path, quality: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    cv2.imwrite(str(out_path), frame, params)


def _save_small_frame(small_frame: np.ndarray, out_path: Path, quality: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    cv2.imwrite(str(out_path), small_frame, params)


def _prepare_decode_source(
    *,
    source_path: Path,
    out_dir: Path,
    target_fps: float,
    resolution: Tuple[int, int],
    pixel_format: str,
) -> Tuple[Path, Dict[str, Any]]:
    """Normalize decode parameters with FFmpeg when available."""

    target = Path(source_path)
    width = int(resolution[0]) if resolution and len(resolution) > 0 else 1280
    height = int(resolution[1]) if resolution and len(resolution) > 1 else 720
    pix_fmt = str(pixel_format or "yuv420p").strip() or "yuv420p"
    ffmpeg_bin = shutil.which("ffmpeg")
    meta: Dict[str, Any] = {
        "enabled": bool(ffmpeg_bin),
        "applied": False,
        "backend": "opencv",
        "source_path": str(source_path),
        "decoded_path": str(source_path),
        "resolution": f"{width}x{height}",
        "pixel_format": pix_fmt,
        "target_fps": float(target_fps),
        "error": None,
    }
    if not ffmpeg_bin:
        return target, meta

    out_dir.mkdir(parents=True, exist_ok=True)
    normalized_path = out_dir / "decoded_normalized.mp4"
    vf_parts = [f"scale={width}:{height}:flags=lanczos"]
    if float(target_fps) > 0:
        vf_parts.insert(0, f"fps={float(target_fps):.3f}")
    vf_expr = ",".join(vf_parts)

    command = [
        str(ffmpeg_bin),
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source_path),
        "-vf",
        vf_expr,
        "-pix_fmt",
        pix_fmt,
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        str(normalized_path),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode == 0 and normalized_path.exists() and normalized_path.stat().st_size > 0:
            meta["applied"] = True
            meta["backend"] = "ffmpeg"
            meta["decoded_path"] = str(normalized_path)
            return normalized_path, meta
        stderr = str(result.stderr or "").strip()
        meta["error"] = stderr[:500] if stderr else "ffmpeg_failed"
    except Exception as exc:
        meta["error"] = str(exc)
    return target, meta
