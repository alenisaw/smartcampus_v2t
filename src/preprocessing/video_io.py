# src/preprocessing/video_io.py
"""
Video preprocessing module for SmartCampus V2T.

Features:
- robust FPS normalization for unstable or variable-FPS videos
- adaptive motion threshold based on real scene dynamics
- frame sampling and dark-frame filtering with optional CLAHE brighten
- detection and skipping of near-identical (lazy) frames
- DNN-based face anonymization (optimized: detect faces not every frame + reuse boxes)
- letterbox resize for stable aspect ratio
- prepared cache: reuse prepared frames if video+video-config fingerprints match
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.core.types import FrameInfo, VideoMeta
from src.pipeline.pipeline_config import PipelineConfig

SMALL_SIZE: Tuple[int, int] = (64, 64)


def _cfg_video_io_get(config: PipelineConfig, key: str, default: Any) -> Any:
    """
    Read video_io.* from PipelineConfig if present, otherwise fallback to default.
    This keeps backward compatibility if YAML is missing the section.
    """
    try:
        vიო = getattr(config, "video_io", None)
        if vio is None:
            vio = getattr(config, "videoio", None)
        if vio is None:
            vio = getattr(config, "video_io_cfg", None)
        if vio is None:
            return default
        return getattr(vio, key, default)
    except Exception:
        return default


def preprocess_video(video_path: str | Path, config: PipelineConfig) -> VideoMeta:
    video_path = Path(video_path).resolve()
    video_id = video_path.stem
    return prepare_video(video_id=video_id, config=config, video_path=video_path)


def prepare_video(
    video_id: str,
    config: PipelineConfig,
    video_path: Optional[Path] = None,
) -> VideoMeta:
    paths_cfg = config.paths
    video_cfg = config.video

    if video_path is None:
        video_path = paths_cfg.raw_dir / f"{video_id}.mp4"
    video_path = video_path.resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    jpeg_q_frames = int(_cfg_video_io_get(config, "jpeg_quality_frames", 82))
    jpeg_q_small = int(_cfg_video_io_get(config, "jpeg_quality_small", 75))

    face_detect_every_saved = int(_cfg_video_io_get(config, "face_detect_every_saved", 6))
    face_detect_force_on_scenecut = bool(_cfg_video_io_get(config, "face_detect_force_on_scenecut", True))
    face_blur_strength = int(_cfg_video_io_get(config, "face_blur_strength", 31))
    face_conf_threshold = float(_cfg_video_io_get(config, "face_conf_threshold", 0.5))

    video_fp = _file_fingerprint(video_path)
    cfg_fp = _video_cfg_fingerprint(config)

    cache_hit = _try_load_cached_video_meta(
        prepared_base=paths_cfg.prepared_dir,
        video_id=video_id,
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
        video_id=video_id,
        prepared_base=paths_cfg.prepared_dir,
        video_fp=video_fp,
        cfg_fp=cfg_fp,
        save_small=bool(video_cfg.save_small_frames),
    )

    start_time = time.perf_counter()
    video_info = _get_video_info(str(video_path))

    source_fps = float(video_info["fps"])
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

    target_fps = float(video_cfg.target_fps)
    step_frames = max(1, int(round(fps / target_fps))) if target_fps > 0 else 1

    if bool(video_cfg.face_blur):
        face_detector = _load_face_detector_dnn(
            paths_cfg.dnn_face_proto,
            paths_cfg.dnn_face_model,
        )
        face_detector_type = "dnn"
    else:
        face_detector = None
        face_detector_type = "none"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video {video_path}")

    frames: List[FrameInfo] = []
    num_raw_frames = int(video_info["total_frames"])
    num_sampled_frames = 0
    num_saved_frames = 0
    num_dark_skipped = 0
    num_lazy_skipped = 0
    num_scene_cuts = 0
    raw_read_frames = 0

    prev_small_kept: Optional[np.ndarray] = None
    frame_idx = 0

    base_threshold = float(video_cfg.min_change_threshold)
    dark_threshold = float(video_cfg.dark_threshold)
    max_saved = int(video_cfg.max_frames) if video_cfg.max_frames is not None else None
    model_input_size = tuple(video_cfg.model_input_size)

    warmup_diffs: List[float] = []
    warmup_done = False
    adaptive_threshold = base_threshold * 0.5

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    last_face_boxes: List[Tuple[int, int, int, int]] = []
    last_face_detect_saved_idx = -10_000

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        raw_read_frames += 1

        if frame_idx % step_frames != 0:
            frame_idx += 1
            continue

        timestamp = frame_idx / fps
        num_sampled_frames += 1

        small_gray = _downscale_for_analysis(frame)

        is_dark = _is_dark_frame(small_gray, brightness_threshold=dark_threshold)
        if is_dark:
            num_dark_skipped += 1
            mean_brightness = float(np.mean(small_gray))
            if mean_brightness < dark_threshold * 0.4:
                frame_idx += 1
                continue

            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l = clahe.apply(l)
            lab = cv2.merge((l, a, b))
            frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        if not warmup_done:
            if prev_small_kept is not None:
                warmup_diffs.append(_compute_frame_change_cpu(prev_small_kept, small_gray))
            prev_small_kept = small_gray

            if len(warmup_diffs) >= 8:
                scene_motion = float(np.mean(warmup_diffs)) if warmup_diffs else 0.0
                adaptive_threshold = max(
                    base_threshold * 0.25,
                    min(base_threshold * 0.7, scene_motion * 1.2),
                )
                warmup_done = True

            frame_idx += 1
            continue

        lazy_ok = True
        is_scene_cut = False
        if prev_small_kept is not None:
            diff = _compute_frame_change_cpu(prev_small_kept, small_gray)
            if diff < adaptive_threshold:
                lazy_ok = False
            if diff > adaptive_threshold * 8.0:
                num_scene_cuts += 1
                is_scene_cut = True

        if not lazy_ok:
            num_lazy_skipped += 1
            frame_idx += 1
            continue

        if face_detector is not None:
            need_detect = (num_saved_frames - last_face_detect_saved_idx) >= face_detect_every_saved
            if face_detect_force_on_scenecut and is_scene_cut:
                need_detect = True

            if need_detect:
                try:
                    last_face_boxes = _detect_faces_dnn(face_detector, frame, conf_threshold=face_conf_threshold)
                    last_face_detect_saved_idx = num_saved_frames
                except Exception:
                    last_face_boxes = []

            if last_face_boxes:
                frame = _blur_boxes(frame, last_face_boxes, blur_strength=face_blur_strength)

        resized = _letterbox_resize(frame, target_size=model_input_size)

        frame_name = f"frame_{frame_idx:06d}.jpg"
        frame_path = frames_dir / frame_name
        _save_frame(resized, frame_path, quality=jpeg_q_frames)

        small_path: Optional[Path] = None
        if small_dir is not None:
            small_path = small_dir / frame_name
            _save_small_frame(small_gray, small_path, quality=jpeg_q_small)

        frames.append(
            FrameInfo(
                video_id=video_id,
                frame_index=len(frames),
                timestamp_sec=float(timestamp),
                path=frame_path,
                small_path=small_path,
            )
        )

        prev_small_kept = small_gray
        num_saved_frames += 1

        if max_saved is not None and num_saved_frames >= max_saved:
            break

        frame_idx += 1

    cap.release()
    preprocess_time_sec = time.perf_counter() - start_time

    early_stop = False
    if num_raw_frames > 0 and raw_read_frames < int(0.9 * num_raw_frames):
        early_stop = True
        print(
            f"[prepare_video] Warning: read only {raw_read_frames}/{num_raw_frames} frames "
            f"for {video_path} (~{100.0 * raw_read_frames / max(1, num_raw_frames):.1f}%)."
        )

    video_meta = VideoMeta(
        video_id=video_id,
        original_path=video_path,
        original_fps=float(fps),
        duration_sec=float(video_info["duration_sec"]),
        processed_fps=target_fps,
        num_frames=len(frames),
        prepared_dir=prepared_root,
        frame_dir=frames_dir,
        small_frame_dir=small_dir,
        frames=frames,
        extra={
            "frame_stats": {
                "source_fps": source_fps,
                "num_raw_frames": num_raw_frames,
                "raw_read_frames": raw_read_frames,
                "num_sampled_frames": num_sampled_frames,
                "num_saved_frames": num_saved_frames,
                "num_dark_skipped": num_dark_skipped,
                "num_lazy_skipped": num_lazy_skipped,
                "num_scene_cuts": num_scene_cuts,
                "early_stop": early_stop,
            },
            "sampling": {"step_frames": int(step_frames)},
            "adaptive_threshold": float(adaptive_threshold),
            "preprocess_time_sec": float(preprocess_time_sec),
            "face_detector_type": face_detector_type,
            "cache": {
                "hit": False,
                "video_fingerprint": video_fp,
                "video_cfg_fingerprint": cfg_fp,
            },
            "video_io": {
                "jpeg_quality_frames": jpeg_q_frames,
                "jpeg_quality_small": jpeg_q_small,
                "face_detect_every_saved": face_detect_every_saved,
                "face_detect_force_on_scenecut": face_detect_force_on_scenecut,
                "face_blur_strength": face_blur_strength,
                "face_conf_threshold": face_conf_threshold,
            },
        },
    )

    _write_meta_json(video_meta)
    return video_meta


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


def _cache_dir(prepared_base: Path, video_id: str, video_fp: str, cfg_fp: str) -> Path:
    vtag = _short_hash(video_fp, 12)
    return Path(prepared_base) / video_id / "cache" / f"v{vtag}" / f"c{cfg_fp}"


def _try_load_cached_video_meta(
    prepared_base: Path,
    video_id: str,
    video_fp: str,
    cfg_fp: str,
) -> Optional[VideoMeta]:
    root = _cache_dir(prepared_base, video_id, video_fp, cfg_fp)
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
    video_id: str,
    prepared_base: Path,
    video_fp: str,
    cfg_fp: str,
    save_small: bool,
) -> Tuple[Path, Path, Optional[Path]]:
    root = _cache_dir(prepared_base, video_id, video_fp, cfg_fp)
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
