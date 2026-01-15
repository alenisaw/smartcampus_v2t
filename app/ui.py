# app/ui.py
"""
UI building blocks for SmartCampus V2T Streamlit app.

Contains:
- HTML/CSS helpers
- i18n helpers (ui_text.json)
- Header + page head rendering
- Home/Search tabs (backend-driven)
"""

from __future__ import annotations

import base64
import html
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import streamlit as st
import streamlit.components.v1 as components

from app.api_client import BackendClient
from app.state import UIState


def E(x: Any) -> str:
    return html.escape("" if x is None else str(x), quote=True)


def _mtime(p: Path) -> float:
    try:
        return float(p.stat().st_mtime)
    except Exception:
        return 0.0


def mmss(sec: float) -> str:
    sec = max(0.0, float(sec))
    total = int(round(sec))
    m = total // 60
    s = total % 60
    return f"{m}:{s:02d}"


def hms(sec: float) -> str:
    sec = max(0, int(round(float(sec))))
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def soft_note(text: str, kind: str = "info") -> None:
    cls = {"info": "soft-note", "warn": "soft-warn", "ok": "soft-ok"}.get(kind, "soft-note")
    st.markdown(f"<div class='{cls}'>{E(text)}</div>", unsafe_allow_html=True)


def thin_rule() -> None:
    st.markdown("<div class='thin-rule'></div>", unsafe_allow_html=True)


def load_and_apply_css(styles_path: Path) -> None:
    css = styles_path.read_text(encoding="utf-8") if styles_path.exists() else ""
    if css.strip():
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def _ensure_i18n_state() -> None:
    if "_i18n_missing_set" not in st.session_state:
        st.session_state["_i18n_missing_set"] = set()
    if "_i18n_missing_total" not in st.session_state:
        st.session_state["_i18n_missing_total"] = 0
    if "_i18n_missing_by_lang" not in st.session_state:
        st.session_state["_i18n_missing_by_lang"] = {}


def get_T(ui_text: Dict[str, Dict[str, Any]], lang: str) -> Dict[str, Any]:
    lang = (lang or "ru").strip().lower()
    return ui_text.get(lang) or ui_text.get("ru") or {}


def Tget(T: Dict[str, Any], key: str, fallback: str) -> str:
    v = T.get(key)
    if v is None:
        _ensure_i18n_state()
        lang = st.session_state.get("ui_lang", "ru")
        miss = f"{lang}:{key}"
        miss_set = st.session_state["_i18n_missing_set"]
        if miss not in miss_set:
            miss_set.add(miss)
            st.session_state["_i18n_missing_total"] += 1
            by_lang = st.session_state["_i18n_missing_by_lang"]
            by_lang[lang] = int(by_lang.get(lang, 0)) + 1
        return fallback
    return str(v)


@st.cache_data(show_spinner=False)
def load_ui_text(path_str: str, mtime: float, langs_key: str) -> Dict[str, Dict[str, Any]]:
    _ = mtime
    data = json.loads(Path(path_str).read_text(encoding="utf-8"))
    langs = [s.strip().lower() for s in (langs_key or "").split(",") if s.strip()]
    for lang in langs:
        if lang not in data:
            raise ValueError(f"ui_text.json missing language: {lang}")
        tabs = data[lang].get("tabs")
        if not isinstance(tabs, list) or len(tabs) != 2:
            raise ValueError(f"ui_text.json: '{lang}.tabs' must be a list of 2 labels")
    return data


def render_i18n_metrics() -> None:
    _ensure_i18n_state()
    total = int(st.session_state.get("_i18n_missing_total") or 0)
    if total <= 0:
        return
    with st.expander("i18n metrics", expanded=False):
        st.write(
            {
                "missing_total": total,
                "missing_by_lang": dict(st.session_state.get("_i18n_missing_by_lang") or {}),
            }
        )
        miss_set = st.session_state.get("_i18n_missing_set") or set()
        st.code("\n".join(sorted(miss_set)))


def _img_to_data_uri(p: Path) -> Optional[str]:
    if not p.exists():
        return None
    try:
        b = p.read_bytes()
        return "data:image/png;base64," + base64.b64encode(b).decode("utf-8")
    except Exception:
        return None


def render_header(T: Dict[str, Any], labels: List[str], ids: List[str], current_tab: str, logo_path: Path) -> None:
    links = []
    for lab, tid in zip(labels, ids):
        cls = "nav-link active" if tid == current_tab else "nav-link"
        links.append(f"<a class='{cls}' href='?tab={E(tid)}' target='_self'>{E(lab)}</a>")

    logo = _img_to_data_uri(logo_path)
    logo_html = f"<div class='hero-logo'><img src='{logo}' alt='logo' /></div>" if logo else ""

    st.markdown(
        f"""
        <div class="hero">
            {logo_html}
            <div class="hero-title">{E(Tget(T, "app_title", "SmartCampus V2T"))}</div>
            <div class="hero-sub">{E(Tget(T, "app_subtitle", ""))}</div>
            <div class="hero-flow">{E(Tget(T, "app_flow", ""))}</div>
            <div class="nav-wrap">
                <div class="nav-row">{''.join(links)}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_page_head(
    title: str,
    ui_text: Dict[str, Dict[str, Any]],
    section_links: Optional[List[Tuple[str, str]]] = None,
    langs: Optional[List[str]] = None,
    default_lang: str = "ru",
) -> Dict[str, Any]:
    langs = langs or ["ru", "kz", "en"]

    if "ui_lang" not in st.session_state:
        st.session_state["ui_lang"] = default_lang
    if st.session_state["ui_lang"] not in langs:
        st.session_state["ui_lang"] = default_lang

    T0 = get_T(ui_text, st.session_state["ui_lang"])

    left, right = st.columns([6, 2], gap="large", vertical_alignment="bottom")
    with left:
        st.markdown(
            f"<div class='page-head'><div class='page-title-text'>{E(title)}</div></div>",
            unsafe_allow_html=True,
        )
        if section_links:
            parts = [f"<a class='sec-link' href='{E(a)}' target='_self'>{E(l)}</a>" for l, a in section_links]
            st.markdown(f"<div class='section-nav'>{''.join(parts)}</div>", unsafe_allow_html=True)
        st.markdown("<div class='page-rule'></div>", unsafe_allow_html=True)

    with right:
        lab = Tget(T0, "ui_lang", "UI language")
        st.markdown("<div class='ui-lang-row'>", unsafe_allow_html=True)
        st.markdown(f"<div class='ui-lang-label'>{E(lab)}</div>", unsafe_allow_html=True)
        st.selectbox(
            lab,
            options=langs,
            index=langs.index(st.session_state["ui_lang"]),
            key="ui_lang",
            label_visibility="collapsed",
        )
        st.markdown("</div>", unsafe_allow_html=True)

    return get_T(ui_text, st.session_state["ui_lang"])


def _safe_dom_id(s: str) -> str:
    return "".join(ch for ch in (s or "") if ch.isalnum() or ch in {"_", "-"})


def anchor(anchor_id: str) -> None:
    st.markdown(f"<div id='{E(anchor_id)}' class='anchor'></div>", unsafe_allow_html=True)


def request_scroll_to(u: UIState, anchor_id: str) -> None:
    u.scroll_to_anchor = _safe_dom_id(anchor_id)


def scroll_if_requested(u: UIState) -> None:
    anchor_id = u.scroll_to_anchor
    if not anchor_id:
        return
    u.scroll_to_anchor = None
    anchor_id = _safe_dom_id(str(anchor_id))
    if not anchor_id:
        return

    components.html(
        f"""
        <script>
        (function() {{
          const id = "{anchor_id}";
          let tries = 0;
          function go() {{
            tries += 1;
            const el = parent.document.getElementById(id);
            if (el) {{
              el.scrollIntoView({{behavior: "smooth", block: "start"}});
              return;
            }}
            if (tries < 25) setTimeout(go, 120);
          }}
          setTimeout(go, 60);
        }})();
        </script>
        """,
        height=0,
        width=0,
    )


def ffmpeg_available() -> bool:
    exe = shutil.which("ffmpeg")
    if exe:
        return True
    try:
        p = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
        return p.returncode == 0
    except Exception:
        return False


def convert_to_mp4(src: Path) -> Optional[Path]:
    """Convert a video to MP4 (H.264/AAC).

    Returns the path to the created MP4 on success, otherwise None.

    Behavior:
    - Output is created next to the source (same stem, .mp4).
    - If conversion succeeds and the source is not already .mp4, we try to delete the source
      to avoid duplicates like .avi + .mp4 confusing preview / processing.
    """
    out = src.with_suffix(".mp4")
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        str(out),
    ]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode != 0:
            return None
        if not out.exists():
            return None

        # Remove source container after successful conversion
        try:
            if src.exists() and src.suffix.lower() != ".mp4":
                src.unlink()
        except Exception:
            pass

        return out
    except Exception:
        return None


def maybe_playback_warning(path: Path, T: dict) -> None:
    ext = path.suffix.lower()
    if ext in {".avi", ".mkv"}:
        soft_note(Tget(T, "playback_warn", "Playback may be unavailable due to codec/container."), kind="warn")
        st.caption(Tget(T, "convert_hint", "Recommended format: MP4 (H.264/AAC)."))


@st.cache_data(show_spinner=False)
def make_thumbnail_bytes(video_path_str: str, mtime: float, max_w: int = 520) -> Optional[bytes]:
    _ = mtime
    p = Path(video_path_str)
    if not p.exists():
        return None
    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        return None
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    h, w = frame.shape[:2]
    if w > max_w:
        scale = max_w / float(w)
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok2, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    if not ok2:
        return None
    return buf.tobytes()


def get_thumbnail(video_path: Path, thumbs_dir: Path) -> Optional[bytes]:
    thumbs_dir.mkdir(parents=True, exist_ok=True)
    tp = thumbs_dir / f"{video_path.stem}.jpg"
    v_m = _mtime(video_path)
    t_m = _mtime(tp)
    if tp.exists() and t_m >= v_m and tp.stat().st_size > 0:
        try:
            return tp.read_bytes()
        except Exception:
            pass
    b = make_thumbnail_bytes(str(video_path), v_m, max_w=520)
    if b:
        try:
            tp.write_bytes(b)
        except Exception:
            pass
    return b


def render_video_player(video_path: Path, start_sec: int) -> None:
    try:
        st.video(str(video_path), start_time=int(max(0, start_sec)))
    except TypeError:
        st.video(str(video_path))
        if start_sec > 0:
            st.caption(f"{int(start_sec)} sec")


def poll_job(client: BackendClient, job_id: str, timeout_sec: int = 60 * 60) -> Dict[str, Any]:
    t0 = time.time()
    while True:
        job = client.get_job(job_id)
        stt = str(job.get("state"))
        if stt in {"done", "failed", "canceled"}:
            return job
        if time.time() - t0 > timeout_sec:
            return job
        time.sleep(0.6)


def _ensure_defaults_for_widgets() -> None:
    # search widgets (remove Streamlit yellow warning about key+value)
    if "search_query" not in st.session_state:
        st.session_state["search_query"] = ""
    if "search_topk" not in st.session_state:
        st.session_state["search_topk"] = 10

    # home widgets
    if "home_search" not in st.session_state:
        st.session_state["home_search"] = ""


def home_tab(client: BackendClient, cfg, ui_text: Dict[str, Dict[str, Any]]) -> None:
    u = UIState()
    u.bind_defaults()
    _ensure_defaults_for_widgets()

    langs = cfg.ui.langs or ["ru", "kz", "en"]
    Ttmp = get_T(ui_text, st.session_state["ui_lang"])
    section_links = [
        (Tget(Ttmp, "videos_panel_title", "Videos"), "#sec_videos"),
        (Tget(Ttmp, "preview", "Preview"), "#sec_preview"),
        (Tget(Ttmp, "processing_in_home", "Processing"), "#sec_processing"),
        (Tget(Ttmp, "choose_run", "Select run"), "#sec_runs"),
    ]
    T = render_page_head(
        Tget(Ttmp, "home_title", "Home"),
        ui_text,
        section_links=section_links,
        langs=langs,
        default_lang=cfg.ui.default_lang,
    )

    scroll_if_requested(u)

    videos = client.list_videos()
    runs_map = client.list_runs()

    ids = sorted([v["video_id"] for v in videos])
    if ids and (st.session_state.get("selected_video_id") not in ids):
        st.session_state["selected_video_id"] = ids[0]

    vid_to_path = {v["video_id"]: Path(v["path"]) for v in videos}

    anchor("sec_videos")
    with st.container(border=True):
        st.markdown(
            f"<div class='section-title'>{E(Tget(T, 'videos_panel_title', 'Videos'))}</div>",
            unsafe_allow_html=True,
        )

        st.text_input(
            Tget(T, "search_video_label", "Search by video name"),
            key="home_search",
        )
        home_search = (st.session_state.get("home_search") or "").strip().lower()
        filtered = [vid for vid in ids if home_search in vid.lower()] if home_search else ids

        st.markdown(f"<div class='mini-title'>{E(Tget(T, 'quick_upload', 'Quick upload'))}</div>", unsafe_allow_html=True)
        uploaded = st.file_uploader(
            Tget(T, "drop_here", "Drop a video file here"),
            type=["mp4", "mov", "mkv", "avi"],
            accept_multiple_files=False,
            key="home_uploader",
        )

        if uploaded is not None:
            st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
            st.markdown("<div class='btn-open btn-ghost'>", unsafe_allow_html=True)
            if st.button(Tget(T, "save_raw", "Save"), width="stretch", key="upload_save"):
                try:
                    client.upload_video(uploaded.name, uploaded.getbuffer().tobytes())
                    soft_note(Tget(T, "saved", "Saved"), kind="ok")
                    st.rerun()
                except Exception as e:
                    soft_note(f"{Tget(T, 'run_err_prefix', 'Error')}: {e}", kind="warn")
            st.markdown("</div>", unsafe_allow_html=True)

        thin_rule()

        if not ids:
            soft_note(Tget(T, "no_videos", "No videos available. Please upload a file above."), kind="info")
        elif not filtered:
            soft_note(Tget(T, "no_matches", "No matches found"), kind="info")
        else:
            # Show only 3 videos at a time, paginate left/right
            page_size = 3
            total_pages = max(1, (len(filtered) + page_size - 1) // page_size)

            st.session_state["carousel_page"] = min(
                max(0, int(st.session_state.get("carousel_page", 0))),
                total_pages - 1,
            )

            st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
            nav_l, nav_mid, nav_r = st.columns([1, 3, 1], gap="small", vertical_alignment="center")
            with nav_l:
                st.markdown("<div class='btn-open btn-ghost icon-only btn-small'>", unsafe_allow_html=True)
                prev = st.button(
                    "◀",
                    key="car_prev",
                    help=Tget(T, "prev", "Prev"),
                    width="stretch",
                    disabled=(total_pages <= 1),
                )
                st.markdown("</div>", unsafe_allow_html=True)
            with nav_mid:
                st.markdown(
                    f"<div class='carousel-meta'>{E(Tget(T, 'page', 'Page'))}: {st.session_state['carousel_page'] + 1}/{total_pages}</div>",
                    unsafe_allow_html=True,
                )
            with nav_r:
                st.markdown("<div class='btn-open btn-ghost icon-only btn-small'>", unsafe_allow_html=True)
                nxt = st.button(
                    "▶",
                    key="car_next",
                    help=Tget(T, "next", "Next"),
                    width="stretch",
                    disabled=(total_pages <= 1),
                )
                st.markdown("</div>", unsafe_allow_html=True)

            if prev:
                st.session_state["carousel_page"] = (st.session_state["carousel_page"] - 1) % total_pages
                st.rerun()
            if nxt:
                st.session_state["carousel_page"] = (st.session_state["carousel_page"] + 1) % total_pages
                st.rerun()

            start = st.session_state["carousel_page"] * page_size
            slice_ids = filtered[start : start + page_size]

            cols = st.columns(3, gap="small")
            for i in range(3):
                if i >= len(slice_ids):
                    cols[i].empty()
                    continue

                vid = slice_ids[i]
                path = vid_to_path.get(vid)
                run_count = len(runs_map.get(vid, []) or [])
                thumb = get_thumbnail(path, Path(cfg.paths.thumbs_dir)) if path else None

                card_uid = f"{st.session_state['carousel_page']}_{start + i}_{vid}"

                with cols[i]:
                    with st.container(border=True):
                        if thumb:
                            b64 = base64.b64encode(thumb).decode("utf-8")
                            st.markdown(
                                f"<img class='thumb-img' src='data:image/jpeg;base64,{b64}' />",
                                unsafe_allow_html=True,
                            )
                        else:
                            st.markdown(
                                f"<div class='thumb-empty'>{E(Tget(T, 'no_thumbnail', 'No thumbnail'))}</div>",
                                unsafe_allow_html=True,
                            )

                        st.markdown(f"<div class='video-card-title'>{E(vid)}</div>", unsafe_allow_html=True)
                        st.markdown(
                            f"<div class='video-card-sub'>{E(path.name if path else '')} · {run_count} {E(Tget(T, 'runs_label', 'runs'))}</div>",
                            unsafe_allow_html=True,
                        )

                        if st.session_state.get("confirm_delete_video_id") == vid:
                            st.markdown("<div class='delete-confirm'>", unsafe_allow_html=True)
                            st.markdown(
                                f"<div class='delete-confirm-text'>{E(Tget(T, 'delete_confirm', 'Delete this video?'))}</div>",
                                unsafe_allow_html=True,
                            )
                            cc1, cc2 = st.columns([1, 1], gap="small")
                            with cc1:
                                st.markdown("<div class='btn-open icon-only btn-small'>", unsafe_allow_html=True)
                                if st.button(
                                    "←",
                                    key=f"del_cancel_{card_uid}",
                                    help=Tget(T, "delete_no", "Cancel"),
                                    width="stretch",
                                ):
                                    st.session_state["confirm_delete_video_id"] = None
                                    st.rerun()
                                st.markdown("</div>", unsafe_allow_html=True)
                            with cc2:
                                st.markdown("<div class='btn-del icon-only btn-small'>", unsafe_allow_html=True)
                                if st.button(
                                    "🗑",
                                    key=f"del_ok_{card_uid}",
                                    help=Tget(T, "delete_yes", "Delete"),
                                    width="stretch",
                                ):
                                    try:
                                        client.delete_video(vid)
                                        st.session_state["confirm_delete_video_id"] = None
                                        if st.session_state.get("selected_video_id") == vid:
                                            st.session_state["selected_video_id"] = None
                                        soft_note(Tget(T, "deleted", "Deleted"), kind="ok")
                                        st.rerun()
                                    except Exception as e:
                                        soft_note(f"{Tget(T, 'run_err_prefix', 'Error')}: {e}", kind="warn")
                                st.markdown("</div>", unsafe_allow_html=True)
                            st.markdown("</div>", unsafe_allow_html=True)
                        else:
                            c1, c2, c3 = st.columns([1, 1, 1], gap="small", vertical_alignment="center")
                            with c1:
                                st.markdown("<div class='btn-open icon-only btn-small'>", unsafe_allow_html=True)
                                if st.button(
                                    "⤢",
                                    key=f"open_{card_uid}",
                                    help=Tget(T, "open", "Open"),
                                    width="stretch",
                                ):
                                    st.session_state["selected_video_id"] = vid
                                    st.session_state["preview_seek_sec"] = 0
                                    request_scroll_to(u, "sec_preview")
                                    st.rerun()
                                st.markdown("</div>", unsafe_allow_html=True)
                            with c2:
                                st.markdown("<div class='btn-run icon-only btn-small'>", unsafe_allow_html=True)
                                if st.button(
                                    "▶",
                                    key=f"run_{card_uid}",
                                    help=Tget(T, "run_pipeline", "Run"),
                                    width="stretch",
                                ):
                                    st.session_state["selected_video_id"] = vid
                                    st.session_state["_run_request"] = True
                                    request_scroll_to(u, "sec_processing")
                                    st.rerun()
                                st.markdown("</div>", unsafe_allow_html=True)
                            with c3:
                                st.markdown("<div class='btn-del icon-only btn-small'>", unsafe_allow_html=True)
                                if st.button(
                                    "🗑",
                                    key=f"del_{card_uid}",
                                    help=Tget(T, "delete_video", "Delete"),
                                    width="stretch",
                                ):
                                    st.session_state["confirm_delete_video_id"] = vid
                                    st.rerun()
                                st.markdown("</div>", unsafe_allow_html=True)
    anchor("sec_preview")
    with st.container(border=True):
        st.markdown(f"<div class='section-title'>{E(Tget(T, 'preview', 'Preview'))}</div>", unsafe_allow_html=True)

        if not st.session_state.get("selected_video_id"):
            soft_note(Tget(T, "pick_video", "Select a video first."), kind="info")
        else:
            vid = st.session_state["selected_video_id"]
            v = next((x for x in videos if x["video_id"] == vid), None)
            if not v:
                soft_note(Tget(T, "raw_missing", "Source video not found"), kind="warn")
            else:
                p = Path(v["path"])
                st.markdown(
                    f"<div class='selected-caption'>{E(Tget(T, 'selected_video', 'Selected video'))}: <b>{E(vid)}</b></div>",
                    unsafe_allow_html=True,
                )
                maybe_playback_warning(p, T)
                render_video_player(p, int(st.session_state.get("preview_seek_sec") or 0))

                if p.suffix.lower() in {".avi", ".mkv"}:
                    c1, c2 = st.columns([1.2, 2.2], gap="medium")
                    with c1:
                        st.markdown("<div class='btn-open btn-ghost'>", unsafe_allow_html=True)
                        if st.button(Tget(T, "convert", "Convert to MP4"), key="convert_preview", width="stretch"):
                            if not ffmpeg_available():
                                soft_note(Tget(T, "ffmpeg_missing", "ffmpeg not found"), kind="warn")
                            else:
                                with st.spinner(Tget(T, "converting", "Conversion in progress...")):
                                    out = convert_to_mp4(p)
                                if out is None:
                                    soft_note(Tget(T, "conversion_failed", "Conversion failed"), kind="warn")
                                else:
                                    soft_note(f"{Tget(T, 'conversion_done', 'Done')}: {out.name}", kind="ok")
                        st.markdown("</div>", unsafe_allow_html=True)
                    with c2:
                        st.caption(Tget(T, "convert_hint", "Recommended format: MP4 (H.264/AAC)."))

    anchor("sec_processing")
    with st.container(border=True):
        st.markdown(f"<div class='section-title'>{E(Tget(T, 'processing_in_home', 'Processing'))}</div>", unsafe_allow_html=True)

        if not st.session_state.get("selected_video_id"):
            soft_note(Tget(T, "pick_video", "Select a video first."), kind="info")
        else:
            c1, c2 = st.columns([1, 1], gap="medium")
            with c1:
                st.markdown(f"<div class='field-label'>{E(Tget(T, 'model_lang', 'Model output language'))}</div>", unsafe_allow_html=True)
                st.selectbox(
                    Tget(T, "model_lang", "Model output language"),
                    options=langs,
                    index=langs.index(st.session_state.get("pipeline_lang_proc", st.session_state.get("pipeline_lang", "ru")))
                    if st.session_state.get("pipeline_lang_proc", "ru") in langs
                    else 0,
                    key="pipeline_lang_proc",
                    label_visibility="collapsed",
                )
            with c2:
                st.markdown(f"<div class='field-label'>{E(Tget(T, 'device', 'Device'))}</div>", unsafe_allow_html=True)
                st.selectbox(
                    Tget(T, "device", "Device"),
                    options=["cuda", "cpu"],
                    index=0 if st.session_state.get("device_proc", st.session_state.get("device", "cuda")) == "cuda" else 1,
                    key="device_proc",
                    label_visibility="collapsed",
                )

            existing_runs = runs_map.get(st.session_state["selected_video_id"], []) or []

            c3, c4 = st.columns([2.2, 2.8], gap="medium", vertical_alignment="top")
            with c3:
                st.checkbox(
                    Tget(T, "force_overwrite", "Force overwrite an existing run"),
                    key="force_overwrite_run",
                )
            with c4:
                if st.session_state.get("force_overwrite_run") and existing_runs:
                    default_overwrite = existing_runs[-1]
                    cur = st.session_state.get("overwrite_run_id") or default_overwrite
                    if cur not in existing_runs:
                        cur = default_overwrite
                    st.selectbox(
                        Tget(T, "overwrite_run_id", "Overwrite run"),
                        options=existing_runs,
                        index=existing_runs.index(cur),
                        key="overwrite_run_id",
                    )
                else:
                    st.session_state["overwrite_run_id"] = None

            st.markdown("<div class='btn-run btn-main icon-only icon-bright'>", unsafe_allow_html=True)
            run_clicked = st.button("▶", key="run_main", help=Tget(T, "run_pipeline", "Run"), width="stretch")
            st.markdown("</div>", unsafe_allow_html=True)

            should_run = bool(run_clicked) or bool(st.session_state.pop("_run_request", False))
            if should_run:
                force_ow = bool(st.session_state.get("force_overwrite_run", False))
                ow_id = st.session_state.get("overwrite_run_id") if force_ow else None
                extra = {
                    "language": str(st.session_state.get("pipeline_lang_proc", "ru")),
                    "device": str(st.session_state.get("device_proc", "cuda")),
                    "force_overwrite": bool(force_ow and ow_id),
                    "overwrite_run_id": ow_id if (force_ow and ow_id) else None,
                }
                try:
                    job = client.create_job(st.session_state["selected_video_id"], extra=extra)
                    st.session_state["active_job_id"] = job.get("job_id")
                    st.session_state["active_job_video_id"] = st.session_state["selected_video_id"]
                    st.rerun()
                except Exception as e:
                    soft_note(f"{Tget(T, 'run_err_prefix', 'Error')}: {e}", kind="warn")

            if st.session_state.get("active_job_id"):
                job_id = st.session_state["active_job_id"]
                st.info(f"{Tget(T, 'job', 'Job')}: {job_id}")
                prog = st.progress(0.0, text=Tget(T, "working", "Working..."))

                st.markdown("<div class='btn-open btn-ghost'>", unsafe_allow_html=True)
                cancel = st.button(Tget(T, "cancel_job", "Cancel job"), width="stretch", key="cancel_job")
                st.markdown("</div>", unsafe_allow_html=True)

                if cancel:
                    try:
                        client.cancel_job(job_id)
                    except Exception:
                        pass

                job = poll_job(client, job_id, timeout_sec=24 * 3600)
                state = str(job.get("state"))
                stage = str(job.get("stage") or "")
                progress = float(job.get("progress") or 0.0)
                msg = str(job.get("message") or "")
                prog.progress(min(max(progress, 0.0), 1.0), text=f"{state} · {stage} · {msg}")

                if state == "done":
                    st.session_state["active_job_id"] = None
                    st.session_state["selected_run_id"] = job.get("run_id")
                    soft_note(Tget(T, "run_done", "Done ✅"), kind="ok")
                    st.rerun()
                elif state == "failed":
                    st.session_state["active_job_id"] = None
                    soft_note(f"{Tget(T, 'failed', 'Failed')}: {job.get('error') or job.get('message')}", kind="warn")
                elif state == "canceled":
                    st.session_state["active_job_id"] = None
                    soft_note(Tget(T, "canceled", "Canceled"), kind="warn")

    anchor("sec_runs")
    with st.container(border=True):
        st.markdown(f"<div class='section-title'>{E(Tget(T, 'choose_run', 'Select run'))}</div>", unsafe_allow_html=True)

        vid = st.session_state.get("selected_video_id")
        if not vid:
            soft_note(Tget(T, "pick_video", "Select a video first."), kind="info")
            return

        runs = runs_map.get(vid, []) or []
        if not runs:
            soft_note(Tget(T, "no_runs_home", "No runs available yet. Please start processing above."), kind="info")
            return

        if st.session_state.get("selected_run_id") not in runs:
            st.session_state["selected_run_id"] = runs[-1]

        st.selectbox(
            Tget(T, "run_filter", "Run"),
            options=runs,
            index=runs.index(st.session_state["selected_run_id"]),
            key="selected_run_id",
        )

        out = client.get_run(vid, st.session_state["selected_run_id"])
        met = out.get("metrics") or {}
        preprocess_sec = float(met.get("preprocess_time_sec", 0.0) or 0.0)
        model_sec = float(met.get("model_time_sec", 0.0) or 0.0)
        total_sec = float(met.get("total_time_sec", 0.0) or 0.0)

        pills = [
            f"{Tget(T,'metrics_preprocess','Preprocess')}: {hms(preprocess_sec)}",
            f"{Tget(T,'metrics_model','Inference')}: {hms(model_sec)}",
            f"{Tget(T,'metrics_total','Total')}: {hms(total_sec)}",
        ]
        html_pills = "".join([f"<span class='pill'>{E(x)}</span>" for x in pills if x.strip()])
        st.markdown(f"<div class='pill-row'>{html_pills}</div>", unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown(f"<div class='inner-title'>{E(Tget(T, 'global_summary_box', 'Video summary'))}</div>", unsafe_allow_html=True)
            if out.get("global_summary"):
                st.write(out["global_summary"])
            else:
                st.caption(Tget(T, "no_global_summary", "No summary available."))

        with st.container(border=True):
            st.markdown(f"<div class='inner-title'>{E(Tget(T, 'segments_timeline', 'Timeline'))}</div>", unsafe_allow_html=True)
            anns = out.get("annotations") or []
            if not anns:
                soft_note(Tget(T, "no_annotations", "No annotations available."), kind="info")
            else:
                for i, a in enumerate(anns):
                    start2 = float(a.get("start_sec", 0.0) or 0.0)
                    end2 = float(a.get("end_sec", 0.0) or 0.0)
                    desc = str(a.get("description", "") or "")
                    b1, b2 = st.columns([1.1, 6], gap="small", vertical_alignment="center")
                    with b1:
                        st.markdown("<div class='btn-open btn-ghost btn-small'>", unsafe_allow_html=True)
                        if st.button(f"{mmss(start2)}–{mmss(end2)}", key=f"seg_seek_{vid}_{st.session_state['selected_run_id']}_{i}", width="stretch"):
                            st.session_state["preview_seek_sec"] = int(max(0.0, start2))
                            request_scroll_to(u, "sec_preview")
                            st.rerun()
                        st.markdown("</div>", unsafe_allow_html=True)
                    with b2:
                        st.write(desc)


def search_tab(client: BackendClient, cfg, ui_text: Dict[str, Dict[str, Any]]) -> None:
    u = UIState()
    u.bind_defaults()
    _ensure_defaults_for_widgets()

    langs = cfg.ui.langs or ["ru", "kz", "en"]
    Ttmp = get_T(ui_text, st.session_state["ui_lang"])
    section_links = [
        (Tget(Ttmp, "search", "Search"), "#sec_search"),
        (Tget(Ttmp, "results", "Results"), "#sec_results"),
        (Tget(Ttmp, "player", "Player"), "#sec_player"),
    ]
    T = render_page_head(
        Tget(Ttmp, "search_desc_title", "Search by event"),
        ui_text,
        section_links=section_links,
        langs=langs,
        default_lang=cfg.ui.default_lang,
    )

    videos = client.list_videos()
    runs_map = client.list_runs()
    vid_to_path = {v["video_id"]: Path(v["path"]) for v in videos}

    anchor("sec_search")
    with st.container(border=True):
        st.markdown(f"<div class='section-title'>{E(Tget(T, 'search', 'Search'))}</div>", unsafe_allow_html=True)

        q1, q2 = st.columns([3, 1], gap="medium")
        with q1:
            st.text_input(
                Tget(T, "query", "Query"),
                placeholder=Tget(T, "query_ph", "crowd running"),
                key="search_query",
            )
        with q2:
            st.number_input(
                Tget(T, "topk", "Top-K"),
                min_value=1,
                max_value=50,
                step=1,
                key="search_topk",
            )

        search_query = str(st.session_state.get("search_query", "") or "")
        search_topk = int(st.session_state.get("search_topk", 10) or 10)

        f1, f2 = st.columns([1, 1], gap="medium")
        with f1:
            video_options = ["(all)"] + sorted(list(runs_map.keys()))
            if st.session_state.get("search_video_filter") not in video_options:
                st.session_state["search_video_filter"] = "(all)"
            st.selectbox(
                Tget(T, "video_filter", "Video"),
                options=video_options,
                index=video_options.index(st.session_state["search_video_filter"]),
                key="search_video_filter",
            )

        with f2:
            run_options = ["(all)"]
            current_video_filter = st.session_state.get("search_video_filter", "(all)")
            if current_video_filter != "(all)":
                run_options += (runs_map.get(current_video_filter, []) or [])
            else:
                all_runs = sorted({r for rs in runs_map.values() for r in (rs or [])})
                run_options += all_runs

            if st.session_state.get("search_run_filter") not in run_options:
                st.session_state["search_run_filter"] = "(all)"

            st.selectbox(
                Tget(T, "run_filter", "Run"),
                options=run_options,
                index=run_options.index(st.session_state["search_run_filter"]),
                key="search_run_filter",
            )

        search_video_filter = str(st.session_state.get("search_video_filter", "(all)"))
        search_run_filter = str(st.session_state.get("search_run_filter", "(all)"))

        thin_rule()

        st.markdown("<div class='btn-open btn-ghost'>", unsafe_allow_html=True)
        if st.button(Tget(T, "update_index", "Update search index"), width="stretch", key="update_index_search"):
            with st.spinner(Tget(T, "building_index", "Updating the search index...")):
                try:
                    resp = client.index_rebuild()
                    soft_note(
                        Tget(T, "ok", "Completed") if resp.get("ok") else Tget(T, "index_build_failed", "Failed"),
                        kind=("ok" if resp.get("ok") else "warn"),
                    )
                except Exception as e:
                    soft_note(f"{Tget(T, 'search_error', 'Search execution error')}: {e}", kind="warn")
        st.markdown("</div>", unsafe_allow_html=True)

    if not search_query.strip():
        soft_note(Tget(T, "type_query", "Please enter a search query."), kind="info")
        return

    try:
        hits = client.search(
            query=search_query.strip(),
            top_k=int(search_topk),
            video_id=None if search_video_filter == "(all)" else search_video_filter,
            run_id=None if search_run_filter == "(all)" else search_run_filter,
            dedupe=True,
        )
    except Exception as e:
        soft_note(f"{Tget(T, 'search_error', 'Search execution error')}: {e}", kind="warn")
        return

    anchor("sec_results")
    anchor("sec_player")

    left, right = st.columns([2, 3], gap="large", vertical_alignment="top")
    with left:
        with st.container(border=True):
            st.markdown(f"<div class='section-title'>{E(Tget(T, 'results', 'Results'))}</div>", unsafe_allow_html=True)
            if not hits:
                soft_note(Tget(T, "no_results", "No results available."), kind="info")
            else:
                for idx, h in enumerate(hits):
                    with st.container(border=True):
                        b1, b2 = st.columns([1.1, 6], gap="small", vertical_alignment="center")
                        with b1:
                            st.markdown("<div class='btn-open btn-ghost btn-small'>", unsafe_allow_html=True)
                            if st.button(f"{mmss(h['start_sec'])}–{mmss(h['end_sec'])}", key=f"hit_{idx}", width="stretch"):
                                st.session_state["selected_hit"] = h
                                request_scroll_to(u, "sec_player")
                                st.rerun()
                            st.markdown("</div>", unsafe_allow_html=True)
                        with b2:
                            st.write(h["description"])
                            st.caption(
                                f"{h['video_id']} · {h['run_id']} · score={h['score']:.3f} "
                                f"(bm25={h.get('sparse_score',0.0):.3f}, dense={h.get('dense_score',0.0):.3f})"
                            )

    with right:
        with st.container(border=True):
            st.markdown(f"<div class='section-title'>{E(Tget(T, 'player', 'Player'))}</div>", unsafe_allow_html=True)
            hit = st.session_state.get("selected_hit")
            if not hit:
                soft_note(Tget(T, "pick_result", "Select a result to open the player."), kind="info")
                return

            vid = str(hit["video_id"])
            start = float(hit.get("start_sec", 0.0) or 0.0)
            end = float(hit.get("end_sec", 0.0) or 0.0)
            desc = str(hit.get("description", "") or "")

            st.write(f"**{vid}** · {hit.get('run_id','')}")
            st.write(f"[{mmss(start)}–{mmss(end)}] {desc}")

            p = vid_to_path.get(vid)
            if not p or not p.exists():
                soft_note(f"{Tget(T, 'raw_missing', 'Source video not found')}: {vid}", kind="warn")
            else:
                maybe_playback_warning(p, T)
                render_video_player(p, int(start))


def footer(T: dict) -> None:
    st.markdown("<div class='thin-rule'></div>", unsafe_allow_html=True)
    txt = Tget(T, "footer", "")
    if txt.strip():
        st.caption(txt)
