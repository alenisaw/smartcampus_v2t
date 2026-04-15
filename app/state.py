# app/state.py
"""
Session-backed UI state for SmartCampus V2T.

Purpose:
- Define the Streamlit session state contract used by the operator UI.
- Keep transient UI state handling separate from rendering logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import streamlit as st


@dataclass
class UIState:
    defaults: Dict[str, Any] = None

    def bind_defaults(self) -> None:
        if self.defaults is None:
            self.defaults = {
                # home
                "home_search": "",
                "carousel_page": 0,
                "overview_stage_index": 0,
                "selected_video_id": None,
                "confirm_delete_video_id": None,
                "preview_seek_sec": 0,
                "outputs_lang_sel": None,
                "active_job_id": None,
                "active_job_video_id": None,
                "active_job_started_at": None,
                "_run_request": False,
                # processing
                "device": "cuda",
                "device_proc": "cuda",
                "force_overwrite_outputs": False,
                # queue ui
                "queue_panel_open": False,
                "queue_selected_remove": None,
                # search
                "search_query": "",
                "search_topk": 5,
                "search_video_filter": "(all)",
                "search_selected_hit_index": 0,
                "selected_hit": None,
                "search_query_box": "",
                "search_video_id": "",
                "search_lang": "",
                "search_variant": "__base__",
                "search_event_type": "",
                "search_risk_level": "",
                "search_motion_type": "",
                "search_people_count_bucket": "",
                "search_anomaly_only": False,
                "search_dedupe": True,
                "search_hits": [],
                "search_rebuild_status": "",
                "search_rag_input": "",
                "search_rag_messages": [],
                "search_time_filter": False,
                "search_time_start": "",
                "search_time_end": "",
                "search_duration_filter": False,
                "search_dur_min": "",
                "search_dur_max": "",
                "saved_queries": [],
                "save_query_name": "",
                "saved_query_sel": None,
                "search_type": "segments",
                "assistant_open": False,
                "queue_open": False,
                # reports
                "reports_video_id": "",
                "reports_lang": "",
                "reports_variant": "__base__",
                "reports_focus": "",
                "reports_payload": {},
                # gallery
                "gallery_search": "",
                "gallery_view": "list",
                "gallery_filter_tags": "",
                "gallery_filter_date": "",
                "gallery_filter_duration": "",
                "gallery_sort": "date_desc",
                # dashboard
                "show_uploader": False,
                # scroll helper
                "scroll_to_anchor": None,
            }

        for k, v in self.defaults.items():
            if k not in st.session_state:
                st.session_state[k] = v

    @property
    def scroll_to_anchor(self) -> Optional[str]:
        return st.session_state.get("scroll_to_anchor")

    @scroll_to_anchor.setter
    def scroll_to_anchor(self, v: Optional[str]) -> None:
        st.session_state["scroll_to_anchor"] = v
