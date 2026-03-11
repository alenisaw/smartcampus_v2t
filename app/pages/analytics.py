# app/pages/analytics.py
"""
Search page for SmartCampus V2T Streamlit UI.

Purpose:
- Render retrieval inspection together with grounded report, QA, and RAG flows.
- Keep search and assistant-facing workflows in one operator page.
"""

from __future__ import annotations

from typing import Any, Dict, List

import streamlit as st

from app.api_client import BackendClient
from app.components.chrome import soft_note
from app.components.common import (
    render_hit_inspector,
    render_metrics_summary,
    render_outputs_overview,
    render_search_results,
    render_supporting_hits,
)
from app.lib.formatters import clip_text, collect_available_languages, variant_from_token, variant_label, video_variant_tokens
from app.lib.i18n import Tget, get_T


def _render_response_block(title: str, text: str, latency_sec: float, hit_count: int, mode: str, *, alt: bool = False) -> None:
    """Render one generated answer block."""

    if not text.strip():
        return
    css = "answer-block alt" if alt else "answer-block"
    st.markdown(f"<div class='{css}'>{clip_text(text, 420)}</div>", unsafe_allow_html=True)
    st.caption(
        f"{title} | mode={mode or 'deterministic'} | latency={float(latency_sec or 0.0):.2f}s | hits={int(hit_count or 0)}"
    )


def render_page(client: BackendClient, cfg: Any, ui_text: Dict[str, Dict[str, Any]]) -> None:
    """Render the unified search, report, QA, and RAG page."""

    T = get_T(ui_text, str(st.session_state.get("ui_lang") or cfg.ui.default_lang or "en"))
    videos = client.list_videos()
    if not videos:
        soft_note(Tget(T, "empty_library", "The library is empty. Upload a video to begin."), kind="info")
        return

    st.markdown(f"<div class='section-title'>{Tget(T, 'search_page_title', 'Search')}</div>", unsafe_allow_html=True)

    video_ids = [""] + [str(item.get("video_id") or "") for item in videos if str(item.get("video_id") or "")]
    selected_video_id = str(st.session_state.get("search_video_id") or "")
    if selected_video_id not in video_ids:
        selected_video_id = ""
    selected_item = next((item for item in videos if str(item.get("video_id") or "") == selected_video_id), None)

    variant_tokens = video_variant_tokens(selected_item)
    current_variant_token = str(st.session_state.get("search_variant") or variant_tokens[0])
    if current_variant_token not in variant_tokens:
        current_variant_token = variant_tokens[0]
    st.session_state["search_variant"] = current_variant_token

    base_lang = str(st.session_state.get("ui_lang") or cfg.ui.default_lang or "en")
    selected_langs = collect_available_languages(selected_item or {}, variant_from_token(st.session_state.get("search_variant")))
    current_lang = str(st.session_state.get("search_lang") or (selected_langs[0] if selected_langs else base_lang))
    if current_lang not in (selected_langs or [base_lang]):
        current_lang = selected_langs[0] if selected_langs else base_lang
    st.session_state["search_lang"] = current_lang

    top = st.columns([2.2, 1.15, 1.05, 0.9], gap="small")
    with top[0]:
        st.selectbox(
            Tget(T, "video_filter", "Video"),
            options=video_ids,
            key="search_video_id",
            format_func=lambda x: x or Tget(T, "all_videos", "All videos"),
        )
    with top[1]:
        st.selectbox(Tget(T, "variant", "Variant"), options=variant_tokens, key="search_variant", format_func=variant_label)
    with top[2]:
        st.selectbox(Tget(T, "output_language", "Output language"), options=selected_langs or [base_lang], key="search_lang")
    with top[3]:
        st.number_input(Tget(T, "topk", "Top-K"), min_value=1, max_value=20, value=int(st.session_state.get("search_topk") or 8), key="search_topk")

    try:
        outputs = client.get_video_outputs(
            str(st.session_state.get("search_video_id") or ""),
            str(st.session_state.get("search_lang") or base_lang),
            variant=variant_from_token(st.session_state.get("search_variant")),
        ) if st.session_state.get("search_video_id") else {}
    except Exception:
        outputs = {}

    search_tab, report_tab, qa_tab, rag_tab, metrics_tab = st.tabs(
        [
            Tget(T, "tab_search_inner", "Search"),
            Tget(T, "tab_report_inner", "Report"),
            Tget(T, "tab_qa_inner", "QA"),
            Tget(T, "tab_rag_inner", "RAG"),
            Tget(T, "tab_metrics_inner", "Metrics"),
        ]
    )

    with search_tab:
        query_cols = st.columns([2.6, 1.1, 1.0, 1.0], gap="small")
        with query_cols[0]:
            st.text_input(Tget(T, "query", "Query"), key="search_query_box", placeholder=Tget(T, "search_placeholder", "Find a person, object, group, motion..."))
        with query_cols[1]:
            event_type = st.text_input(Tget(T, "event_filter", "Event type"), key="search_event_type")
        with query_cols[2]:
            risk_level = st.selectbox(
                Tget(T, "risk_filter", "Risk"),
                options=["", "normal", "attention", "warning", "critical"],
                key="search_risk_level",
                format_func=lambda x: x or Tget(T, "any_value", "Any"),
            )
        with query_cols[3]:
            anomaly_only = st.checkbox(Tget(T, "anomaly_only", "Anomalies only"), key="search_anomaly_only")

        action_cols = st.columns([4, 1], gap="small")
        with action_cols[0]:
            if st.button(Tget(T, "search_action", "Run search"), key="search_run_btn", use_container_width=True):
                try:
                    hits = client.search(
                        query=str(st.session_state.get("search_query_box") or "").strip(),
                        top_k=int(st.session_state.get("search_topk") or 8),
                        video_id=str(st.session_state.get("search_video_id") or "") or None,
                        language=str(st.session_state.get("search_lang") or base_lang),
                        variant=variant_from_token(st.session_state.get("search_variant")),
                        event_type=str(event_type or "").strip() or None,
                        risk_level=str(risk_level or "").strip() or None,
                        anomaly_only=bool(anomaly_only),
                    )
                    st.session_state["search_hits"] = hits
                except Exception as exc:
                    st.session_state["search_hits"] = []
                    soft_note(f"{Tget(T, 'error_prefix', 'Error')}: {exc}", kind="warn")
        with action_cols[1]:
            if st.button(Tget(T, "clear", "Clear"), key="search_clear_btn", use_container_width=True):
                st.session_state["search_hits"] = []
                st.session_state["search_query_box"] = ""
                st.rerun()

        hits: List[Dict[str, Any]] = st.session_state.get("search_hits") or []
        left, right = st.columns([1.25, 1.75], gap="large")
        with left:
            selected_hit = render_search_results(T, hits, open_prefix="search_open")
        with right:
            render_hit_inspector(T, selected_hit)

    with report_tab:
        st.text_input(
            Tget(T, "report_prompt", "Report focus"),
            key="assistant_report_query",
            placeholder=Tget(T, "report_placeholder", "Describe the activity near the entrance"),
        )
        if st.button(Tget(T, "build_report", "Build report"), key="assistant_build_report", use_container_width=True):
            try:
                st.session_state["assistant_report_payload"] = client.build_report(
                    video_id=str(st.session_state.get("search_video_id") or "") or None,
                    language=str(st.session_state.get("search_lang") or base_lang),
                    variant=variant_from_token(st.session_state.get("search_variant")),
                    query=str(st.session_state.get("assistant_report_query") or "").strip() or None,
                    top_k=int(st.session_state.get("search_topk") or 8),
                )
            except Exception as exc:
                soft_note(f"{Tget(T, 'error_prefix', 'Error')}: {exc}", kind="warn")
        report_payload = st.session_state.get("assistant_report_payload") or {}
        report_text = str(report_payload.get("report") or "").strip()
        _render_response_block(
            Tget(T, "reports_title", "Grounded report"),
            report_text,
            float(report_payload.get("latency_sec", 0.0) or 0.0),
            int(report_payload.get("hit_count", 0) or 0),
            str(report_payload.get("mode") or "deterministic"),
        )
        if report_text:
            with st.expander(Tget(T, "full_report", "Full report"), expanded=False):
                st.write(report_text)
        render_supporting_hits(Tget(T, "supporting_hits", "Supporting hits"), list(report_payload.get("supporting_hits") or []))

    with qa_tab:
        st.text_area(
            Tget(T, "assistant_prompt", "Question about the video"),
            key="assistant_question",
            height=120,
            placeholder=Tget(T, "assistant_placeholder", "What is happening near the entrance?"),
        )
        if st.button(Tget(T, "ask", "Ask"), key="assistant_ask_btn", use_container_width=True):
            try:
                st.session_state["assistant_qa_payload"] = client.ask_qa(
                    question=str(st.session_state.get("assistant_question") or ""),
                    language=str(st.session_state.get("search_lang") or base_lang),
                    variant=variant_from_token(st.session_state.get("search_variant")),
                    video_id=str(st.session_state.get("search_video_id") or "") or None,
                    top_k=int(st.session_state.get("search_topk") or 8),
                )
            except Exception as exc:
                soft_note(f"{Tget(T, 'error_prefix', 'Error')}: {exc}", kind="warn")
        qa_payload = st.session_state.get("assistant_qa_payload") or {}
        qa_text = str(qa_payload.get("answer") or "").strip()
        _render_response_block(
            Tget(T, "tab_qa_inner", "QA"),
            qa_text,
            float(qa_payload.get("latency_sec", 0.0) or 0.0),
            int(qa_payload.get("hit_count", 0) or 0),
            str(qa_payload.get("mode") or "deterministic"),
        )
        context = str(qa_payload.get("context") or "").strip()
        if context:
            with st.expander(Tget(T, "qa_context", "QA context"), expanded=False):
                st.code(context)
        render_supporting_hits(Tget(T, "qa_evidence", "QA evidence"), list(qa_payload.get("supporting_hits") or []))

    with rag_tab:
        st.text_area(
            Tget(T, "assistant_prompt", "Question about the video"),
            key="assistant_rag_query",
            height=120,
            placeholder=Tget(T, "assistant_placeholder", "What is happening near the entrance?"),
        )
        if st.button(Tget(T, "rag", "RAG"), key="assistant_rag_btn", use_container_width=True):
            try:
                st.session_state["assistant_rag_payload"] = client.ask_rag(
                    query=str(st.session_state.get("assistant_rag_query") or ""),
                    language=str(st.session_state.get("search_lang") or base_lang),
                    variant=variant_from_token(st.session_state.get("search_variant")),
                    video_id=str(st.session_state.get("search_video_id") or "") or None,
                    top_k=int(st.session_state.get("search_topk") or 8),
                )
            except Exception as exc:
                soft_note(f"{Tget(T, 'error_prefix', 'Error')}: {exc}", kind="warn")
        rag_payload = st.session_state.get("assistant_rag_payload") or {}
        rag_text = str(rag_payload.get("answer") or "").strip()
        _render_response_block(
            Tget(T, "tab_rag_inner", "RAG"),
            rag_text,
            float(rag_payload.get("latency_sec", 0.0) or 0.0),
            int(rag_payload.get("hit_count", 0) or 0),
            str(rag_payload.get("mode") or "deterministic"),
            alt=True,
        )
        context = str(rag_payload.get("context") or "").strip()
        if context:
            with st.expander(Tget(T, "rag_context", "RAG context"), expanded=False):
                st.code(context)
        render_supporting_hits(Tget(T, "rag_evidence", "RAG evidence"), list(rag_payload.get("supporting_hits") or []))

    with metrics_tab:
        if outputs:
            render_outputs_overview(T, outputs, str(st.session_state.get("search_lang") or base_lang))
            st.markdown(f"<div class='section-title'>{Tget(T, 'metrics', 'Metrics')}</div>", unsafe_allow_html=True)
            render_metrics_summary(T, outputs)
            run_manifest = outputs.get("run_manifest") if isinstance(outputs.get("run_manifest"), dict) else {}
            if run_manifest:
                with st.expander(Tget(T, "run_manifest", "Run manifest"), expanded=False):
                    st.json(run_manifest)
        else:
            st.caption(Tget(T, "pick_video_for_metrics", "Select one video to inspect run outputs and metrics."))
