# src/preprocess.py
"""
Preprocessing pipeline for smartcampus_v2t.

This module provides:
- Walking through raw campus videos (default: data/raw/) and extracting frames.
- Storing processed frames under data/processed/frames/<video_name>/.
- Configurable frame sampling:
    * extract every N-th frame (frame_step)
    * optional FPS-based sampling
- Frame transformations:
    * resizing to a fixed resolution
    * removal of extremely dark / empty frames
    * lightweight motion-based filtering to drop near-duplicate frames
- Privacy protection (enabled by default):
    * face detection and strong anonymizing blur on all detected faces
- Output formats:
    * .jpg for visual inspection and lightweight storage
    * .npy for fast loading into video transformer models
- Parallel processing:
    * optional multiprocessing to accelerate preprocessing over many videos

This file defines the full preprocessing workflow: video loading, per-frame
operations, anonymization, filtering, and saving frames into a consistent
directory structure used later by the inference pipeline.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

from .config import RAW_VIDEO_DIR, PROCESSED_DIR, MODELS_DIR, ensure_directories
from .utils import setup_logging, list_video_files, relative_to_project

logger = logging.getLogger(__name__)

# always skip totally dark frames; this is safe and usually desired
SKIP_DARK = True
DARK_THRESHOLD = 10.0  # [0..255], anything below this is basically black

# light motion filtering by default
DEFAULT_MIN_MOTION = 1.5  # 0 disables motion filter, >0 starts filtering

# Haar cascade for face detection (you need haarcascade_frontalface_default.xml here)
FACE_CASCADE_PATH = MODELS_DIR / "haarcascade_frontalface_default.xml"
_face_cascade = None  # lazy-loaded


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract frames from campus videos.")

    parser.add_argument("--input", type=str, default=str(RAW_VIDEO_DIR))
    parser.add_argument("--output", type=str, default=str(PROCESSED_DIR / "frames"))
    parser.add_argument("--frame-step", type=int, default=1)

    parser.add_argument(
        "--resize",
        type=int,
        nargs=2,
        default=(224, 224),
        metavar=("W", "H"),
        help="Resize frames to (W, H). Default: 224 224.",
    )
    parser.add_argument(
        "--no-resize",
        action="store_true",
        help="If set, do not resize frames at all.",
    )

    # anonymization is ON by default; this flag turns it OFF
    parser.add_argument(
        "--no-anonymize",
        action="store_true",
        help="Disable anonymization. By default strong face blur is applied.",
    )

    parser.add_argument(
        "--as-numpy",
        action="store_true",
        help="Save frames as .npy (float32 RGB in [0,1]) instead of .jpg.",
    )

    parser.add_argument(
        "--min-motion",
        type=float,
        default=DEFAULT_MIN_MOTION,
        help="Mean abs diff threshold [0..255] for motion filtering. "
             "0 disables motion filtering. Default: 1.5.",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of processes for parallel video processing.",
    )

    return parser.parse_args()


def _get_face_cascade():
    global _face_cascade
    if _face_cascade is not None:
        return _face_cascade

    if not FACE_CASCADE_PATH.exists():
        logger.warning(
            "Face cascade not found at %s, anonymization will be a no-op.",
            relative_to_project(FACE_CASCADE_PATH),
        )
        _face_cascade = None
        return None

    cascade = cv2.CascadeClassifier(str(FACE_CASCADE_PATH))
    if cascade.empty():
        logger.warning(
            "Failed to load face cascade from %s, anonymization disabled.",
            relative_to_project(FACE_CASCADE_PATH),
        )
        _face_cascade = None
    else:
        _face_cascade = cascade

    return _face_cascade


def anonymize_frame(frame):
    # strong blur on detected faces; if cascade is missing, just return frame
    cascade = _get_face_cascade()
    if cascade is None:
        return frame

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) == 0:
        return frame

    for (x, y, w, h) in faces:
        face_roi = frame[y:y + h, x:x + w]
        # big kernel so the face is really unreadable
        face_roi = cv2.GaussianBlur(face_roi, (51, 51), 0)
        frame[y:y + h, x:x + w] = face_roi

    return frame


def _frame_is_dark(gray: np.ndarray) -> bool:
    # average grayscale < threshold → consider it dark
    return gray.mean() < DARK_THRESHOLD


def process_video(
    video_path: Path,
    out_root: Path,
    frame_step: int,
    resize: Optional[Tuple[int, int]],
    anonymize: bool,
    min_motion: float,
    as_numpy: bool,
) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning("Can't open video %s", relative_to_project(video_path))
        return 0

    out_dir = out_root / video_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    total_saved = 0
    frame_idx = 0
    last_kept_gray: Optional[np.ndarray] = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # only keep every N-th frame
        if frame_idx % frame_step != 0:
            frame_idx += 1
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if SKIP_DARK and _frame_is_dark(gray):
            frame_idx += 1
            continue

        if min_motion > 0.0 and last_kept_gray is not None:
            diff = np.mean(
                np.abs(
                    gray.astype(np.float32) - last_kept_gray.astype(np.float32)
                )
            )
            if diff < min_motion:
                frame_idx += 1
                continue

        if anonymize:
            frame = anonymize_frame(frame)

        if resize is not None:
            w, h = resize
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)

        # update last kept gray only when we actually keep the frame
        last_kept_gray = gray

        if as_numpy:
            # BGR → RGB, normalize to [0,1]
            rgb = frame[:, :, ::-1].astype(np.float32) / 255.0
            out_path = out_dir / f"frame_{frame_idx:06d}.npy"
            np.save(out_path, rgb)
            ok = True
        else:
            out_path = out_dir / f"frame_{frame_idx:06d}.jpg"
            ok = cv2.imwrite(str(out_path), frame)

        if ok:
            total_saved += 1

        frame_idx += 1

    cap.release()
    logger.info("Video %s → %d frames saved", relative_to_project(video_path), total_saved)
    return total_saved


def _process_video_wrapper(args_tuple) -> int:
    # small helper so ProcessPoolExecutor can pass args as one tuple
    return process_video(*args_tuple)


def main():
    setup_logging()
    ensure_directories()
    args = parse_args()

    input_dir = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_step = max(1, args.frame_step)
    min_motion = max(0.0, args.min_motion)
    workers = max(1, args.workers)

    if args.no_resize:
        resize: Optional[Tuple[int, int]] = None
    else:
        resize = tuple(args.resize)  # type: ignore

    anonymize = not args.no_anonymize  # default: True

    logger.info("Input:       %s", relative_to_project(input_dir))
    logger.info("Output:      %s", relative_to_project(output_dir))
    logger.info("Frame step:  %d", frame_step)
    logger.info("Resize:      %s", "OFF" if resize is None else f"{resize[0]}x{resize[1]}")
    logger.info("Anonymize:   %s", "ON" if anonymize else "OFF")
    logger.info("Skip dark:   %s (thr=%.1f)", "ON" if SKIP_DARK else "OFF", DARK_THRESHOLD)
    logger.info("Min motion:  %.2f", min_motion)
    logger.info("As numpy:    %s", "ON" if args.as_numpy else "OFF")
    logger.info("Workers:     %d", workers)

    videos = list_video_files(input_dir)
    if not videos:
        logger.warning("No videos found.")
        return

    jobs = [
        (
            vid,
            output_dir,
            frame_step,
            resize,
            anonymize,
            min_motion,
            args.as_numpy,
        )
        for vid in videos
    ]

    total = 0

    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_process_video_wrapper, job) for job in jobs]
            for f in tqdm(as_completed(futures), total=len(futures), desc="processing"):
                total += f.result()
    else:
        for job in tqdm(jobs, desc="processing"):
            total += _process_video_wrapper(job)

    logger.info("Total frames saved: %d", total)


if __name__ == "__main__":
    main()
