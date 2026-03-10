"""Assistant page for grounded reports, QA, RAG, and runtime inspection."""

from __future__ import annotations

from typing import Any, Dict, List

import streamlit as st

from app.api_client import BackendClient
from app.components.chrome import render_language_switcher, soft_note
from app.components.common import render_metrics_summary, render_outputs_overview, render_status_overview, render_supporting_hits
from app.lib.formatters import clip_text, collect_available_languages, variant_from_token, variant_label, video_variant_tokens
from app.lib.i18n import get_T, Tget


def _render_response_block(title: str, text: str, latency_sec: float, hit_count: int, mode: str, *, alt: bool = False) -> None:
    """Render one generated answer block."""

    if not text.strip():
        return
    css = "answer-block alt" if alt else "answer-block"
    st.markdown(f"<div class='{css}'>{clip_text(text, 420)}</div>", unsafe_allow_html=True)
    st.caption(f"{title} · mode={mode or 'deterministic'} · latency={float(latency_sec or 0.0):.2f}s · hits={int(hit_count or 0)}")


def render_page(client: BackendClient, cfg: Any, ui_text: Dict[str, Dict[str, Any]]) -> None:
    """Render the grounded report / QA / RAG page."""

    T = get_T(ui_text, str(st.session_state.get("ui_lang") or cfg.ui.default_lang or "en"))
    render_language_switcher(T, cfg)

    videos = client.list_videos()
    if not videos:
        soft_note(Tget(T, "empty_library", "The library is empty. Upload a video to begin."), kind="info")
        return

    try:
        queue = client.queue_list()
    except Exception:
        queue = {}
    try:
        index_status = client.index_status()
    except Exception:
        index_status = {}
    render_status_overview(T, queue, index_status)

    video_ids = [str(item.get("video_id") or "") for item in videos if str(item.get("video_id") or "")]
    selected = str(st.session_state.get("assistant_video_id") or (video_ids[0] if video_ids else ""))
    if selected not in video_ids and video_ids:
        selected = video_ids[0]
    st.session_state["assistant_video_id"] = selected

    selected_item = next((item for item in videos if str(item.get("video_id") or "") == selected), None)
    variant_tokens = video_variant_tokens(selected_item)
    current_variant_token = str(st.session_state.get("assistant_variant") or variant_tokens[0])
    if current_variant_token not in variant_tokens:
        current_variant_token = variant_tokens[0]
    st.session_state["assistant_variant"] = current_variant_token

    selected_langs = collect_available_languages(selected_item or {}, variant_from_token(current_variant_token))
    assistant_lang = str(st.session_state.get("assistant_lang") or (selected_langs[0] if selected_langs else "en"))
    if assistant_lang not in selected_langs:
        assistant_lang = selected_langs[0] if selected_langs else "en"

    top = st.columns([2, 1, 1, 1], gap="small")
    with top[0]:
        st.selectbox(Tget(T, "selected_video", "Selected video"), options=video_ids, key="assistant_video_id")
    with top[1]:
        st.selectbox(Tget(T, "variant", "Variant"), options=variant_tokens, key="assistant_variant", format_func=variant_label)
    with top[2]:
        st.selectbox(Tget(T, "output_language", "Output language"), options=selected_langs or ["en"], key="assistant_lang")
    with top[3]:
        st.number_input(Tget(T, "topk", "Top-K"), min_value=1, max_value=20, value=int(st.session_state.get("assistant_top_k") or 6), key="assistant_top_k")

    try:
        outputs = client.get_video_outputs(
            str(st.session_state.get("assistant_video_id") or ""),
            str(st.session_state.get("assistant_lang") or "en"),
            variant=variant_from_token(st.session_state.get("assistant_variant")),
        )
    except Exception:
        outputs = {}

    report_tab, qa_tab, rag_tab, metrics_tab = st.tabs(["Report", "QA", "RAG", "Pipeline"])

    with report_tab:
        st.markdown(f"<div class='section-title'>{Tget(T, 'reports_title', 'Grounded report')}</div>", unsafe_allow_html=True)
        st.text_input(Tget(T, "report_prompt", "Report focus"), key="assistant_report_query", placeholder=Tget(T, "report_placeholder", "Summarize the visible activity near the entrance"))
        if st.button(Tget(T, "build_report", "Build report"), key="assistant_build_report", use_container_width=True):
            try:
                st.session_state["assistant_report_payload"] = client.build_report(
                    video_id=str(st.session_state.get("assistant_video_id") or "") or None,
                    language=str(st.session_state.get("assistant_lang") or "en"),
                    variant=variant_from_token(st.session_state.get("assistant_variant")),
                    query=str(st.session_state.get("assistant_report_query") or "").strip() or None,
                    top_k=int(st.session_state.get("assistant_top_k") or 6),
                )
            except Exception as exc:
                soft_note(f"{Tget(T, 'error_prefix', 'Error')}: {exc}", kind="warn")
        report_payload = st.session_state.get("assistant_report_payload") or {}
        report_text = str(report_payload.get("report") or "").strip()
        _render_response_block(
            "Report",
            report_text,
            float(report_payload.get("latency_sec", 0.0) or 0.0),
            int(report_payload.get("hit_count", 0) or 0),
            str(report_payload.get("mode") or "deterministic"),
        )
        if report_text:
            with st.expander(Tget(T, "full_report", "Full report"), expanded=False):
                st.write(report_text)
        render_supporting_hits("Supporting hits", list(report_payload.get("supporting_hits") or []))

    with qa_tab:
        st.markdown(f"<div class='section-title'>{Tget(T, 'qa_title', 'Grounded QA')}</div>", unsafe_allow_html=True)
        st.text_area(
            Tget(T, "assistant_prompt", "Ask about the video"),
            key="assistant_question",
            height=120,
            placeholder=Tget(T, "assistant_placeholder", "What happens near the entrance?"),
        )
        if st.button(Tget(T, "ask", "Ask"), key="assistant_ask_btn", use_container_width=True):
            try:
                st.session_state["assistant_qa_payload"] = client.ask_qa(
                    question=str(st.session_state.get("assistant_question") or ""),
                    language=str(st.session_state.get("assistant_lang") or "en"),
                    variant=variant_from_token(st.session_state.get("assistant_variant")),
                    video_id=str(st.session_state.get("assistant_video_id") or "") or None,
                    top_k=int(st.session_state.get("assistant_top_k") or 6),
                )
            except Exception as exc:
                soft_note(f"{Tget(T, 'error_prefix', 'Error')}: {exc}", kind="warn")
        qa_payload = st.session_state.get("assistant_qa_payload") or {}
        qa_text = str(qa_payload.get("answer") or "").strip()
        _render_response_block(
            "QA",
            qa_text,
            float(qa_payload.get("latency_sec", 0.0) or 0.0),
            int(qa_payload.get("hit_count", 0) or 0),
            str(qa_payload.get("mode") or "deterministic"),
        )
        context = str(qa_payload.get("context") or "").strip()
        if context:
            with st.expander(Tget(T, "qa_context", "QA context"), expanded=False):
                st.code(context)
        render_supporting_hits("QA evidence", list(qa_payload.get("supporting_hits") or []))

    with rag_tab:
        st.markdown(f"<div class='section-title'>RAG</div>", unsafe_allow_html=True)
        st.text_area(
            Tget(T, "assistant_prompt", "Ask about the video"),
            key="assistant_rag_query",
            height=120,
            placeholder=Tget(T, "assistant_placeholder", "What happens near the entrance?"),
        )
        if st.button(Tget(T, "rag", "RAG"), key="assistant_rag_btn", use_container_width=True):
            try:
                st.session_state["assistant_rag_payload"] = client.ask_rag(
                    query=str(st.session_state.get("assistant_rag_query") or ""),
                    language=str(st.session_state.get("assistant_lang") or "en"),
                    variant=variant_from_token(st.session_state.get("assistant_variant")),
                    video_id=str(st.session_state.get("assistant_video_id") or "") or None,
                    top_k=int(st.session_state.get("assistant_top_k") or 6),
                )
            except Exception as exc:
                soft_note(f"{Tget(T, 'error_prefix', 'Error')}: {exc}", kind="warn")
        rag_payload = st.session_state.get("assistant_rag_payload") or {}
        rag_text = str(rag_payload.get("answer") or "").strip()
        _render_response_block(
            "RAG",
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
        render_supporting_hits("RAG evidence", list(rag_payload.get("supporting_hits") or []))

    with metrics_tab:
        render_outputs_overview(T, outputs, str(st.session_state.get("assistant_lang") or "en"))
        st.markdown(f"<div class='section-title'>{Tget(T, 'metrics', 'Metrics')}</div>", unsafe_allow_html=True)
        render_metrics_summary(T, outputs)
        run_manifest = outputs.get("run_manifest") if isinstance(outputs.get("run_manifest"), dict) else {}
        if run_manifest:
            with st.expander("Run manifest", expanded=False):
                st.json(run_manifest)
