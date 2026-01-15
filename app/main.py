# app/main.py
"""
SmartCampus V2T — Streamlit entrypoint.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_loader import load_pipeline_config
from app.api_client import BackendClient
from app.state import UIState
from app import ui as U

TAB_IDS = ["home", "search"]
CFG_PATH = PROJECT_ROOT / "configs" / "pipeline.yaml"


def _mtime(p: Path) -> float:
    try:
        return float(p.stat().st_mtime)
    except Exception:
        return 0.0


@st.cache_resource(show_spinner=False)
def _cfg_cached(cfg_path_str: str, mtime: float):
    _ = mtime
    return load_pipeline_config(Path(cfg_path_str))


def get_cfg():
    return _cfg_cached(str(CFG_PATH), _mtime(CFG_PATH))


def _get_tab_from_query(default_tab: str = "home") -> str:
    try:
        qp = st.query_params
        tab = qp.get("tab", default_tab)
        if isinstance(tab, list):
            tab = tab[0] if tab else default_tab
        tab = (tab or default_tab).strip().lower()
    except Exception:
        qp = st.experimental_get_query_params()
        tab = (qp.get("tab", [default_tab])[0] or default_tab).strip().lower()

    return tab if tab in TAB_IDS else default_tab


def main() -> None:
    st.set_page_config(page_title="SmartCampus V2T", layout="wide")

    cfg = get_cfg()

    U.load_and_apply_css(Path(cfg.ui.styles_path))

    langs = cfg.ui.langs or ["ru", "kz", "en"]
    ui_text_path = Path(cfg.ui.ui_text_path)
    ui_text = U.load_ui_text(str(ui_text_path), _mtime(ui_text_path), ",".join(langs))

    UIState().bind_defaults()

    default_lang = (getattr(cfg.ui, "default_lang", None) or (langs[0] if langs else "ru"))
    if "ui_lang" not in st.session_state:
        st.session_state["ui_lang"] = default_lang
    if st.session_state.get("ui_lang") not in (langs or ["ru", "kz", "en"]):
        st.session_state["ui_lang"] = default_lang

    base_url = str(getattr(cfg, "backend_url", "") or "http://127.0.0.1:8000")
    client = BackendClient(base_url=base_url)

    tab = _get_tab_from_query(default_tab="home")

    T = U.get_T(ui_text, st.session_state.get("ui_lang", default_lang))
    labels = T.get("tabs") or ["Home", "Search"]
    if not isinstance(labels, list) or len(labels) != len(TAB_IDS):
        labels = ["Home", "Search"]

    U.render_header(
        T=T,
        labels=labels,
        ids=TAB_IDS,
        current_tab=tab,
        logo_path=Path(cfg.ui.logo_path),
    )

    if hasattr(client, "health") and callable(client.health):
        h = client.health()
        if not (isinstance(h, dict) and h.get("ok")):
            U.soft_note(f"Backend is not reachable at {base_url}", kind="warn")
            st.stop()

    if tab == "home":
        U.home_tab(client, cfg, ui_text)
    else:
        U.search_tab(client, cfg, ui_text)

    U.footer(U.get_T(ui_text, st.session_state.get("ui_lang", default_lang)))
    U.render_i18n_metrics()


if __name__ == "__main__":
    main()
