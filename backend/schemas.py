# backend/schemas.py
"""
Pydantic schemas for SmartCampus V2T backend.

Purpose:
- Define request/response payloads for videos, runs, jobs, queue, search, and index.
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


class RunsMap(BaseModel):
    runs_map: Dict[str, List[str]]


class RunOutputs(BaseModel):
    video_id: str
    run_id: str
    manifest: Optional[Dict[str, Any]] = None
    annotations: List[Dict[str, Any]] = Field(default_factory=list)
    metrics: Optional[Dict[str, Any]] = None
    global_summary: Optional[str] = None
    language: Optional[str] = None
    device: Optional[str] = None


class JobCreateRequest(BaseModel):
    video_id: str
    extra: Optional[Dict[str, Any]] = None


class JobCreateResponse(BaseModel):
    job_id: str
    state: str
    stage: Optional[str] = None


class JobStatus(BaseModel):
    job_id: str
    video_id: Optional[str] = None
    state: str
    stage: Optional[str] = None
    progress: Optional[float] = None
    message: Optional[str] = None
    run_id: Optional[str] = None
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
    state: Optional[str] = None
    created_at: Optional[float] = None


class QueueListResponse(BaseModel):
    status: QueueStatus
    queued: List[QueueItem] = Field(default_factory=list)


class IndexStatus(BaseModel):
    built_at: Optional[float] = None
    updated_at: Optional[float] = None
    version: Optional[float] = None
    last_error: Optional[str] = None


class IndexRebuildResponse(BaseModel):
    ok: bool
    status: IndexStatus


class SearchRequest(BaseModel):
    query: str
    top_k: int = 10
    video_id: Optional[str] = None
    run_id: Optional[str] = None
    dedupe: bool = True


class SearchHit(BaseModel):
    video_id: str
    run_id: str
    start_sec: float
    end_sec: float
    description: str
    score: float
    sparse_score: float
    dense_score: float


class SearchResponse(BaseModel):
    hits: List[SearchHit]
