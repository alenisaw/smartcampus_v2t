# src/core/types.py

"""
Simple shared data structures used across the project.

These dataclasses define:
- how preprocessed videos are represented (frames, meta)
- how model outputs are stored (caption segments)
- how annotations/search hits look for RAG/search
- how runtime metrics are tracked for each run
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class FrameInfo:
    video_id: str
    frame_index: int
    timestamp_sec: float
    path: Path
    small_path: Optional[Path] = None
    extra: Optional[Dict[str, Any]] = None


@dataclass
class Clip:
    video_id: str
    start_sec: float
    end_sec: float
    frames: List[FrameInfo] = field(default_factory=list)
    extra: Optional[Dict[str, Any]] = None


@dataclass
class VideoMeta:
    video_id: str

    original_path: Path
    original_fps: float
    duration_sec: float

    processed_fps: float
    num_frames: int

    prepared_dir: Path
    frame_dir: Path
    small_frame_dir: Optional[Path] = None

    frames: List[FrameInfo] = field(default_factory=list)

    extra: Optional[Dict[str, Any]] = None


@dataclass
class CaptionSegment:
    video_id: str
    start_sec: float
    end_sec: float
    text: str

    raw_output: Optional[Dict[str, Any]] = None
    extra: Optional[Dict[str, Any]] = None


@dataclass
class Annotation:

    video_id: str
    start_sec: float
    end_sec: float
    description: str
    extra: Optional[Dict[str, Any]] = None


@dataclass
class SearchHit:
    score: float
    video_id: str
    start_sec: float
    end_sec: float
    description: str
    source_id: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None


@dataclass
class RunMetrics:
    video_id: str
    language: str = "unknown"

    video_duration_sec: float = 0.0
    num_frames: int = 0
    num_clips: int = 0
    avg_clip_duration_sec: float = 0.0

    preprocess_time_sec: float = 0.0
    model_time_sec: float = 0.0
    postprocess_time_sec: float = 0.0
    total_time_sec: float = 0.0

    extra: Dict[str, Any] = field(default_factory=dict)
