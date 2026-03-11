# backend/schemas.py
"""
Backend schemas for SmartCampus V2T.

Purpose:
- Define request and response payloads for API routes.
- Keep backend contracts explicit for videos, jobs, queue, search, and reports.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class VideoItem(BaseModel):
    video_id: str
    filename: str
    path: str
    size_bytes: Optional[int] = None
    mtime: Optional[float] = None
    languages: List[str] = Field(default_factory=list)
    variants: Dict[str, Any] = Field(default_factory=dict)

class VideoOutputs(BaseModel):
    video_id: str
    language: str
    variant: Optional[str] = None
    manifest: Optional[Dict[str, Any]] = None
    run_manifest: Optional[Dict[str, Any]] = None
    batch_manifest: Optional[Dict[str, Any]] = None
    annotations: List[Dict[str, Any]] = Field(default_factory=list)
    metrics: Optional[Dict[str, Any]] = None
    global_summary: Optional[str] = None


class JobCreateRequest(BaseModel):
    video_id: str
    profile: Optional[str] = None
    variant: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None


class JobCreateResponse(BaseModel):
    job_id: str
    state: str
    stage: Optional[str] = None


class JobStatus(BaseModel):
    job_id: str
    video_id: Optional[str] = None
    job_type: Optional[str] = None
    profile: Optional[str] = None
    variant: Optional[str] = None
    language: Optional[str] = None
    source_language: Optional[str] = None
    state: str
    stage: Optional[str] = None
    progress: Optional[float] = None
    message: Optional[str] = None
    error: Optional[Dict[str, Any]] = None
    created_at: Optional[float] = None
    started_at: Optional[float] = None
    updated_at: Optional[float] = None
    finished_at: Optional[float] = None


class JobCancelResponse(BaseModel):
    job_id: str
    state: str


class QueueStatus(BaseModel):
    paused: bool = False
    updated_at: Optional[float] = None


class QueueItem(BaseModel):
    job_id: str
    video_id: Optional[str] = None
    job_type: Optional[str] = None
    profile: Optional[str] = None
    variant: Optional[str] = None
    language: Optional[str] = None
    state: Optional[str] = None
    created_at: Optional[float] = None


class QueueRunningItem(BaseModel):
    job_id: str
    video_id: Optional[str] = None
    job_type: Optional[str] = None
    profile: Optional[str] = None
    variant: Optional[str] = None
    language: Optional[str] = None
    state: str
    stage: Optional[str] = None
    progress: Optional[float] = None
    message: Optional[str] = None
    created_at: Optional[float] = None
    started_at: Optional[float] = None
    updated_at: Optional[float] = None


class QueueListResponse(BaseModel):
    status: QueueStatus
    running: Optional[QueueRunningItem] = None
    queued: List[QueueItem] = Field(default_factory=list)


class QueueMoveRequest(BaseModel):
    job_id: str
    direction: str = Field(..., description="one of: up, down, top, bottom")
    steps: int = 1


class QueueMoveResponse(BaseModel):
    ok: bool
    job_id: str
    old_index: int
    new_index: int
    queued_count: int


class IndexStatus(BaseModel):
    built_at: Optional[float] = None
    updated_at: Optional[float] = None
    version: Optional[float] = None
    last_error: Optional[str] = None
    languages: Optional[Dict[str, Dict[str, Any]]] = None


class IndexRebuildResponse(BaseModel):
    ok: bool
    status: IndexStatus


class SearchRequest(BaseModel):
    query: str
    top_k: int = 10
    video_id: Optional[str] = None
    language: Optional[str] = None
    variant: Optional[str] = None
    dedupe: bool = True
    start_sec: Optional[float] = None
    end_sec: Optional[float] = None
    min_duration_sec: Optional[float] = None
    max_duration_sec: Optional[float] = None
    event_type: Optional[str] = None
    risk_level: Optional[str] = None
    tags: Optional[List[str]] = None
    objects: Optional[List[str]] = None
    people_count_bucket: Optional[str] = None
    motion_type: Optional[str] = None
    anomaly_only: bool = False


class SearchHit(BaseModel):
    video_id: str
    language: str
    start_sec: float
    end_sec: float
    description: str
    score: float
    sparse_score: float
    dense_score: float
    segment_id: Optional[str] = None
    event_type: Optional[str] = None
    risk_level: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    objects: List[str] = Field(default_factory=list)
    people_count_bucket: Optional[str] = None
    motion_type: Optional[str] = None
    anomaly_flag: bool = False
    variant: Optional[str] = None


class SearchResponse(BaseModel):
    hits: List[SearchHit]


class Citation(BaseModel):
    video_id: str
    start_sec: float
    end_sec: float
    segment_id: Optional[str] = None
    variant: Optional[str] = None


class ReportRequest(BaseModel):
    video_id: Optional[str] = None
    language: Optional[str] = None
    variant: Optional[str] = None
    query: Optional[str] = None
    top_k: int = 8


class ReportResponse(BaseModel):
    language: str
    variant: Optional[str] = None
    report: str
    mode: str = "deterministic"
    latency_sec: float = 0.0
    hit_count: int = 0
    citations: List[Citation] = Field(default_factory=list)
    supporting_hits: List[SearchHit] = Field(default_factory=list)


class QaRequest(BaseModel):
    question: str
    language: Optional[str] = None
    variant: Optional[str] = None
    video_id: Optional[str] = None
    top_k: int = 5


class QaResponse(BaseModel):
    language: str
    variant: Optional[str] = None
    answer: str
    mode: str = "deterministic"
    context: Optional[str] = None
    latency_sec: float = 0.0
    hit_count: int = 0
    citations: List[Citation] = Field(default_factory=list)
    supporting_hits: List[SearchHit] = Field(default_factory=list)


class RagRequest(BaseModel):
    query: str
    language: Optional[str] = None
    variant: Optional[str] = None
    video_id: Optional[str] = None
    top_k: int = 6


class RagResponse(BaseModel):
    language: str
    variant: Optional[str] = None
    answer: str
    mode: str = "deterministic"
    context: str
    latency_sec: float = 0.0
    hit_count: int = 0
    citations: List[Citation] = Field(default_factory=list)
    supporting_hits: List[SearchHit] = Field(default_factory=list)


class MetricsSummaryResponse(BaseModel):
    video_id: str
    language: str
    variant: Optional[str] = None
    profile: Optional[str] = None
    config_fingerprint: Optional[str] = None
    timings_sec: Dict[str, float] = Field(default_factory=dict)
    counters: Dict[str, Any] = Field(default_factory=dict)
