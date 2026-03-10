"""Analytics page for retrieval and result inspection."""

from __future__ import annotations

from typing import Any, Dict, List

import streamlit as st

from app.api_client import BackendClient
from app.components.chrome import render_language_switcher, soft_note
from app.components.common import render_hit_inspector, render_search_results, render_status_overview
from app.lib.formatters import collect_available_languages, variant_from_token, variant_label, video_variant_tokens
from app.lib.i18n import get_T, Tget


def render_page(client: BackendClient, cfg: Any, ui_text: Dict[str, Dict[str, Any]]) -> None:
    """Render the analytics and retrieval view."""

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
    lang = str(st.session_state.get("ui_lang") or cfg.ui.default_lang or "en")

    top = st.columns([2.2, 1, 1, 1], gap="small")
    with top[0]:
        st.text_input(Tget(T, "query", "Query"), key="search_query_box", placeholder=Tget(T, "search_placeholder", "Find a person, vehicle, crowd, object..."))
    with top[1]:
        st.selectbox(Tget(T, "video_filter", "Video"), options=video_ids, key="search_video_id", format_func=lambda x: x or Tget(T, "all_videos", "All videos"))
    with top[2]:
        st.selectbox(Tget(T, "variant", "Variant"), options=variant_tokens, key="search_variant", format_func=variant_label)
    with top[3]:
        st.number_input(Tget(T, "topk", "Top-K"), min_value=1, max_value=20, value=int(st.session_state.get("search_topk") or 8), key="search_topk")

    filter_cols = st.columns(4, gap="small")
    with filter_cols[0]:
        event_type = st.text_input(Tget(T, "event_filter", "Event type"), key="search_event_type")
    with filter_cols[1]:
        risk_level = st.selectbox(
            Tget(T, "risk_filter", "Risk"),
            options=["", "normal", "attention", "warning", "critical"],
            key="search_risk_level",
            format_func=lambda x: x or Tget(T, "any_value", "Any"),
        )
    with filter_cols[2]:
        anomaly_only = st.checkbox(Tget(T, "anomaly_only", "Anomalies only"), key="search_anomaly_only")
    with filter_cols[3]:
        selected_langs = collect_available_languages(selected_item or {}, variant_from_token(st.session_state.get("search_variant")))
        search_lang = str(st.session_state.get("search_lang") or (selected_langs[0] if selected_langs else lang))
        if search_lang not in selected_langs:
            search_lang = selected_langs[0] if selected_langs else lang
        st.selectbox("Search language", options=selected_langs or [lang], index=(selected_langs or [lang]).index(search_lang), key="search_lang")

    action_cols = st.columns([4, 1], gap="small")
    with action_cols[0]:
        if st.button(Tget(T, "search_action", "Run search"), key="search_run_btn", use_container_width=True):
            try:
                hits = client.search(
                    query=str(st.session_state.get("search_query_box") or "").strip(),
                    top_k=int(st.session_state.get("search_topk") or 8),
                    video_id=str(st.session_state.get("search_video_id") or "") or None,
                    language=str(st.session_state.get("search_lang") or lang),
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
        if st.button("Clear", key="search_clear_btn", use_container_width=True):
            st.session_state["search_hits"] = []
            st.session_state["search_query_box"] = ""
            st.rerun()

    hits: List[Dict[str, Any]] = st.session_state.get("search_hits") or []
    left, right = st.columns([1.25, 1.75], gap="large")
    with left:
        selected_hit = render_search_results(T, hits, open_prefix="search_open")
    with right:
        render_hit_inspector(T, selected_hit)
