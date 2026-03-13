"""
Streamlit entrypoint for SmartCampus V2T.

Purpose:
- Bootstrap config loading and UI startup for the local operator console.
- Expose one stable Streamlit launch surface for the application.
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
from src.core.runtime import config_cache_token, load_pipeline_config

TAB_IDS = ["home", "processing", "video", "search", "reports"]
CFG_PATH = PROJECT_ROOT / "configs" / "profiles" / "main.yaml"
EARLY_SHELL_CSS = """
<style>
[data-testid="stSidebar"],
[data-testid="stSidebarNav"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"]{
  display:none !important;
}
</style>
"""


def _tab_labels(lang: str) -> list[str]:
    """Return localized top-level tab labels."""

    return {
        "ru": [
            "\u041e\u0431\u0437\u043e\u0440",
            "\u0425\u0440\u0430\u043d\u0438\u043b\u0438\u0449\u0435",
            "\u0412\u0438\u0434\u0435\u043e\u0430\u043d\u0430\u043b\u0438\u0442\u0438\u043a\u0430",
            "\u041f\u043e\u0438\u0441\u043a",
            "\u041e\u0442\u0447\u0451\u0442\u044b",
        ],
        "kz": [
            "\u0428\u043e\u043b\u0443",
            "\u049a\u043e\u0439\u043c\u0430",
            "\u0411\u0435\u0439\u043d\u0435 \u0430\u043d\u0430\u043b\u0438\u0442\u0438\u043a\u0430\u0441\u044b",
            "\u0406\u0437\u0434\u0435\u0443",
            "\u0415\u0441\u0435\u043f\u0442\u0435\u0440",
        ],
        "en": ["Overview", "Storage", "Video Analytics", "Search", "Reports"],
    }.get(str(lang or "en"), ["Overview", "Storage", "Video Analytics", "Search", "Reports"])


@st.cache_resource(show_spinner=False)
def _cfg_cached(cfg_path_str: str, cache_token: str):
    """Load the typed config once per config fingerprint."""

    _ = cache_token
    return load_pipeline_config(Path(cfg_path_str))


def get_cfg():
    """Return the cached runtime config."""

    return _cfg_cached(str(CFG_PATH), config_cache_token(CFG_PATH))


def _get_tab_from_query(default_tab: str = "home") -> str:
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

    aliases = {
        "operations": "processing",
        "process": "processing",
        "storage": "processing",
        "library": "processing",
        "videos": "video",
        "gallery": "video",
        "video-analytics": "video",
        "analytics": "video",
        "assistant": "search",
        "reports": "reports",
        "metrics": "reports",
        "summaries": "reports",
    }
    tab = aliases.get(tab, tab)
    return tab if tab in TAB_IDS else default_tab


def main() -> None:
    """Render the demo UI and route to the selected page."""

    st.set_page_config(page_title="SmartCampus V2T", layout="wide", initial_sidebar_state="collapsed")
    st.markdown(EARLY_SHELL_CSS, unsafe_allow_html=True)

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
    tab = _get_tab_from_query(default_tab="home")

    T = U.get_T(ui_text, st.session_state.get("ui_lang", default_lang))
    labels = _tab_labels(str(st.session_state.get("ui_lang", default_lang)))
    if not isinstance(labels, list) or len(labels) != len(TAB_IDS):
        labels = ["Overview", "Storage", "Video Analytics", "Search", "Reports"]

    tab = U.render_header(
        T=T,
        labels=labels,
        ids=TAB_IDS,
        current_tab=tab,
        logo_path=Path(cfg.ui.logo_path),
        cfg=cfg,
    )

    health_payload = client.health()
    if not (isinstance(health_payload, dict) and health_payload.get("ok")):
        U.soft_note(f"Backend is not reachable at {base_url}", kind="warn")
        st.stop()

    if tab == "home":
        U.overview_tab(client, cfg, ui_text)
    elif tab == "processing":
        U.storage_tab(client, cfg, ui_text)
    elif tab == "video":
        U.video_analytics_tab(client, cfg, ui_text)
    elif tab == "reports":
        U.reports_metrics_tab(client, cfg, ui_text)
    else:
        U.search_tab(client, cfg, ui_text)

    U.footer(U.get_T(ui_text, st.session_state.get("ui_lang", default_lang)), str(st.session_state.get("ui_lang", default_lang)))
    U.render_i18n_metrics()


if __name__ == "__main__":
    main()
