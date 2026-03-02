# app/main.py
"""
SmartCampus V2T Streamlit entrypoint.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import ui as U
from app.api_client import BackendClient
from app.state import UIState
from src.utils.config_loader import config_cache_token, load_pipeline_config

TAB_IDS = ["storage", "analytics", "assistant"]
CFG_PATH = PROJECT_ROOT / "configs" / "profiles" / "main.yaml"


@st.cache_resource(show_spinner=False)
def _cfg_cached(cfg_path_str: str, cache_token: str):
    """Load the typed config once per config fingerprint."""

    _ = cache_token
    return load_pipeline_config(Path(cfg_path_str))


def get_cfg():
    """Return the cached runtime config."""

    return _cfg_cached(str(CFG_PATH), config_cache_token(CFG_PATH))


def _get_tab_from_query(default_tab: str = "storage") -> str:
    """Resolve the current UI tab from query params."""

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
    """Render the demo UI and route to the selected page."""

    st.set_page_config(page_title="SmartCampus V2T", layout="wide")

    cfg = get_cfg()
    U.load_and_apply_css(Path(cfg.ui.styles_path))

    langs = cfg.ui.langs or ["ru", "kz", "en"]
    ui_text_path = Path(cfg.ui.ui_text_path)
    try:
        ui_text_mtime = float(ui_text_path.stat().st_mtime_ns)
    except Exception:
        ui_text_mtime = 0.0
    ui_text = U.load_ui_text(str(ui_text_path), ui_text_mtime, ",".join(langs))

    UIState().bind_defaults()

    default_lang = getattr(cfg.ui, "default_lang", None) or (langs[0] if langs else "ru")
    if "ui_lang" not in st.session_state:
        st.session_state["ui_lang"] = default_lang
    if st.session_state.get("ui_lang") not in (langs or ["ru", "kz", "en"]):
        st.session_state["ui_lang"] = default_lang

    try:
        backend_cfg = cfg.backend
        scheme = str(getattr(backend_cfg, "scheme", "http") or "http").strip().lower()
        host = str(getattr(backend_cfg, "host", "127.0.0.1") or "127.0.0.1").strip()
        port = int(getattr(backend_cfg, "port", 8000) or 8000)
        base_url = f"{scheme}://{host}:{port}"
    except Exception:
        base_url = "http://127.0.0.1:8000"

    client = BackendClient(base_url=base_url)
    tab = _get_tab_from_query(default_tab="storage")

    T = U.get_T(ui_text, st.session_state.get("ui_lang", default_lang))
    labels = T.get("tabs") or ["Videos", "Analytics", "Assistant"]
    if not isinstance(labels, list) or len(labels) != len(TAB_IDS):
        labels = ["Videos", "Analytics", "Assistant"]

    U.render_header(
        T=T,
        labels=labels,
        ids=TAB_IDS,
        current_tab=tab,
        logo_path=Path(cfg.ui.logo_path),
    )

    health_payload = client.health()
    if not (isinstance(health_payload, dict) and health_payload.get("ok")):
        U.soft_note(f"Backend is not reachable at {base_url}", kind="warn")
        st.stop()

    if tab == "storage":
        U.gallery_tab(client, cfg, ui_text)
    elif tab == "analytics":
        U.search_tab(client, cfg, ui_text)
    else:
        U.assistant_tab(client, cfg, ui_text)

    U.footer(U.get_T(ui_text, st.session_state.get("ui_lang", default_lang)))
    U.render_i18n_metrics()


if __name__ == "__main__":
    main()
