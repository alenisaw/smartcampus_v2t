# backend/http/grounded.py
"""
Grounded-response helpers for SmartCampus V2T backend routes.

Purpose:
- Centralize report, QA, and RAG response construction over shared retrieval/generation flow.
- Keep backend route handlers thin without changing API behavior or response payloads.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, List

from fastapi import HTTPException

from backend.http.common import ApiContext, grounded_response_payload, grounded_text_result, variant_filters
from backend.deps import read_video_outputs
from backend.retrieval_runtime import (
    annotation_hits as _annotation_hits,
    guard_query_text as _guard_query_text,
    qa_text_from_hits as _qa_text_from_hits,
    report_text_from_hits as _report_text_from_hits,
    resolve_request_language as _resolve_request_language,
    search_hits as _search_hits,
)
from backend.schemas import QaRequest, QaResponse, RagRequest, RagResponse, ReportRequest, ReportResponse


def build_report_response(req: ReportRequest, context: ApiContext) -> ReportResponse:
    """Execute the grounded report flow and return the API response model."""

    started_at = time.perf_counter()
    cfg = context.cfg
    paths = context.paths
    lang = _resolve_request_language(cfg, req.language)

    query = str(req.query or "").strip()
    video_id = str(req.video_id or "").strip()
    if not (query or video_id):
        raise HTTPException(status_code=400, detail="Report request requires a query or video_id.")
    if query:
        _guard_query_text(cfg, query, endpoint="Reports")

    supporting_hits: List[Any] = []
    if query:
        supporting_hits = _search_hits(
            cfg,
            query=query,
            language=lang,
            variant=req.variant,
            top_k=max(1, int(req.top_k)),
            video_id=req.video_id,
            filters=variant_filters(req.variant),
            dedupe=True,
        )
    elif video_id:
        outputs = read_video_outputs(Path(paths.videos_dir), video_id, lang, variant=req.variant)
        supporting_hits = _annotation_hits(
            video_id=video_id,
            language=lang,
            annotations=list(outputs.get("annotations") or []),
            variant=req.variant,
            top_k=max(1, int(req.top_k)),
        )

    fallback_report = _report_text_from_hits(supporting_hits, req.video_id)
    report_text, mode, _context = grounded_text_result(
        cfg,
        task="grounded report",
        user_input=str(req.query or req.video_id or "report"),
        hits=supporting_hits,
        fallback_text=fallback_report,
        target_lang=lang,
        target_name="reports",
    )
    hit_payload = grounded_response_payload(supporting_hits)

    return ReportResponse(
        language=lang,
        variant=req.variant,
        report=report_text,
        mode=mode,
        latency_sec=float(time.perf_counter() - started_at),
        hit_count=int(len(supporting_hits)),
        citations=hit_payload["citations"],
        supporting_hits=hit_payload["supporting_hits"],
    )


def build_qa_response(req: QaRequest, context: ApiContext) -> QaResponse:
    """Execute the grounded QA flow and return the API response model."""

    started_at = time.perf_counter()
    cfg = context.cfg
    lang = _resolve_request_language(cfg, req.language)
    question = str(req.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is required.")
    _guard_query_text(cfg, question, endpoint="QA")
    if bool(getattr(cfg.guard, "enabled", False)) and bool(getattr(cfg.guard, "query_gate", False)) and len(question) < 3:
        raise HTTPException(status_code=400, detail="Question is too short for QA.")

    hits = _search_hits(
        cfg,
        query=question,
        language=lang,
        variant=req.variant,
        top_k=max(1, int(req.top_k)),
        video_id=req.video_id,
        filters=variant_filters(req.variant),
        dedupe=True,
    )

    fallback_answer = _qa_text_from_hits(question, hits)
    answer_text, mode, grounded_context = grounded_text_result(
        cfg,
        task="grounded question answering",
        user_input=question,
        hits=hits,
        fallback_text=fallback_answer,
        target_lang=lang,
        target_name="qa",
    )
    hit_payload = grounded_response_payload(hits)

    return QaResponse(
        language=lang,
        variant=req.variant,
        answer=answer_text,
        mode=mode,
        context=grounded_context,
        latency_sec=float(time.perf_counter() - started_at),
        hit_count=int(len(hits)),
        citations=hit_payload["citations"],
        supporting_hits=hit_payload["supporting_hits"],
    )


def build_rag_response(req: RagRequest, context: ApiContext) -> RagResponse:
    """Execute the grounded RAG flow and return the API response model."""

    started_at = time.perf_counter()
    cfg = context.cfg
    lang = _resolve_request_language(cfg, req.language)
    query = str(req.query or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="RAG query is required.")
    _guard_query_text(cfg, query, endpoint="RAG")

    hits = _search_hits(
        cfg,
        query=query,
        language=lang,
        variant=req.variant,
        top_k=max(1, int(req.top_k)),
        video_id=req.video_id,
        filters=variant_filters(req.variant),
        dedupe=True,
    )

    fallback_answer = _qa_text_from_hits(query, hits)
    answer_text, mode, grounded_context = grounded_text_result(
        cfg,
        task="grounded retrieval augmented answer",
        user_input=query,
        hits=hits,
        fallback_text=fallback_answer,
        target_lang=lang,
        target_name="qa",
    )
    hit_payload = grounded_response_payload(hits)

    return RagResponse(
        language=lang,
        variant=req.variant,
        answer=answer_text,
        mode=mode,
        context=grounded_context,
        latency_sec=float(time.perf_counter() - started_at),
        hit_count=int(len(hits)),
        citations=hit_payload["citations"],
        supporting_hits=hit_payload["supporting_hits"],
    )
