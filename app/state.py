# app/state.py
"""
Session-backed UI state for SmartCampus V2T Streamlit app.
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
                "selected_video_id": None,
                "confirm_delete_video_id": None,
                "preview_seek_sec": 0,
                "active_job_id": None,
                "active_job_video_id": None,
                "active_job_started_at": None,
                "_run_request": False,
                # processing
                "pipeline_lang": "ru",
                "device": "cuda",
                "pipeline_lang_proc": "ru",
                "device_proc": "cuda",
                "force_overwrite_run": False,
                "overwrite_run_id": None,
                "selected_run_id": None,
                "confirm_delete_run": False,
                # queue ui
                "queue_panel_open": False,
                "queue_selected_remove": None,
                # search
                "search_query": "",
                "search_topk": 10,
                "search_video_filter": "(all)",
                "search_run_filter": "(all)",
                "selected_hit": None,
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
