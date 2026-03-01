# app/ui.py
"""
UI building blocks for SmartCampus V2T Streamlit app.

Contains:
- HTML/CSS helpers
- i18n helpers (ui_text.json)
- Header + page head rendering
- Home/Analytics/Storage tabs (backend-driven)
"""

from __future__ import annotations

import base64
import csv
import html
import io
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

ICON_OPEN = "👁"
ICON_QUEUE = "➕"
ICON_RUN = "▶"
ICON_DELETE = "🗑"
ICON_TOP = "⏫"
ICON_UP = "▲"
ICON_DOWN = "▼"
ICON_SKIP = "⏭"
ICON_REFRESH = "⟳"

ICON_PAUSE = "⏸"
ICON_RESUME = "⏵"
ICON_SAVE = "💾"
ICON_CLEAR = "🧹"
ICON_LOAD = "⤵"
ICON_CONVERT = "🎞"
ICON_JUMP = "⏩"
ICON_CANCEL = "✖"

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


def fmt_dt(ts: Optional[float]) -> str:
    if not ts:
        return ""
    try:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(float(ts)))
    except Exception:
        return ""


def fmt_bytes(num: Optional[float]) -> str:
    if num is None:
        return ""
    try:
        n = float(num)
    except Exception:
        return ""
    if n <= 0:
        return "0 B"
    units = ["B", "KB", "MB", "GB", "TB"]
    idx = 0
    while n >= 1024.0 and idx < len(units) - 1:
        n /= 1024.0
        idx += 1
    if idx == 0:
        return f"{int(n)} {units[idx]}"
    return f"{n:.2f} {units[idx]}"


@st.cache_data(show_spinner=False)
def get_video_meta(video_path_str: str, mtime: float) -> Dict[str, Any]:
    _ = mtime
    p = Path(video_path_str)
    meta: Dict[str, Any] = {}
    if not p.exists():
        return meta

    try:
        stt = p.stat()
        meta["size_bytes"] = stt.st_size
        meta["updated_at"] = stt.st_mtime
    except Exception:
        pass

    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        return meta

    try:
        w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if w and h:
            meta["width"] = int(w)
            meta["height"] = int(h)
        if fps:
            meta["fps"] = float(fps)
        if frames:
            meta["frames"] = int(frames)
        if fps and frames and fps > 0:
            meta["duration_sec"] = float(frames) / float(fps)
    finally:
        cap.release()

    return meta


def annotations_to_csv(annotations: List[Dict[str, Any]]) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["video_id", "start_sec", "end_sec", "description"])
    for a in annotations or []:
        writer.writerow(
            [
                a.get("video_id", ""),
                a.get("start_sec", ""),
                a.get("end_sec", ""),
                a.get("description", ""),
            ]
        )
    return buf.getvalue()


def build_timeline_bins(annotations: List[Dict[str, Any]], max_bins: int = 40) -> List[Tuple[float, float, int]]:
    if not annotations:
        return []
    try:
        max_end = max(float(a.get("end_sec", 0.0) or 0.0) for a in annotations)
    except Exception:
        return []
    if max_end <= 0:
        return []
    bin_sec = max(1.0, max_end / float(max_bins))
    bins: List[Tuple[float, float, int]] = []
    i = 0
    while True:
        start = i * bin_sec
        end = min(max_end, (i + 1) * bin_sec)
        if start >= max_end:
            break
        count = 0
        for a in annotations:
            a_s = float(a.get("start_sec", 0.0) or 0.0)
            a_e = float(a.get("end_sec", 0.0) or 0.0)
            if a_s < end and a_e > start:
                count += 1
        bins.append((start, end, count))
        i += 1
    return bins


def soft_note(text: str, kind: str = "info") -> None:
    cls = {"info": "soft-note", "warn": "soft-warn", "ok": "soft-ok"}.get(kind, "soft-note")
    st.markdown(f"<div class='{cls}'>{E(text)}</div>", unsafe_allow_html=True)


def btn_label(icon: str, text: str) -> str:
    t = "" if text is None else str(text).strip()
    if not t:
        return str(icon)
    return f"{icon} {t}"


def thin_rule() -> None:
    st.markdown("<div class='thin-rule'></div>", unsafe_allow_html=True)


def render_progress_bar(
    value: Optional[float],
    *,
    label: str = "",
    sublabel: str = "",
    animate: bool = False,
    container: Optional[Any] = None,
) -> None:
    pct: Optional[int] = None
    if value is not None:
        try:
            pct = int(max(0.0, min(1.0, float(value))) * 100.0)
        except Exception:
            pct = None
    if pct is None:
        pct = int(time.time() * 7) % 100
        animate = True

    cls = "progress-bar progress-anim" if animate else "progress-bar"
    sub_html = f"<div class='progress-sub'>{E(sublabel)}</div>" if sublabel else ""
    target = container if container is not None else st
    target.markdown(
        f"""
        <div class="progress-wrap">
            <div class="progress-head">
                <div class="progress-label">{E(label)}</div>
                <div class="progress-pct">{pct}%</div>
            </div>
            <div class="progress-track">
                <div class="{cls}" style="width:{pct}%;"></div>
            </div>
            {sub_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_assistant_panel(T: Dict[str, Any]) -> None:
    anchor("sec_player")
    with st.container(border=True):
        st.markdown(f"<div class='section-title'>{E(Tget(T, 'assistant_title', 'RAG Assistant'))}</div>", unsafe_allow_html=True)
        st.caption(Tget(T, "assistant_placeholder", "Placeholder. Models will be connected later."))
        st.markdown(f"<div class='mini-title'>{E(Tget(T, 'assistant_prompt', 'Ask about...'))}</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='chip-row'>"
            f"<span class='chip'>{E(Tget(T, 'assistant_q1', 'Group results by topic'))}</span>"
            f"<span class='chip'>{E(Tget(T, 'assistant_q2', 'Find similar segments'))}</span>"
            f"<span class='chip'>{E(Tget(T, 'assistant_q3', 'Short summary for the query'))}</span>"
            "</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div class='assistant-chat'>"
            "<div class='assistant-msg assistant-bot'>"
            f"{E(Tget(T, 'assistant_placeholder', 'Placeholder. Models will be connected later.'))}"
            "</div>"
            "<div class='assistant-input'>"
            f"{E(Tget(T, 'assistant_prompt', 'Ask about...'))}"
            "</div>"
            "</div>",
            unsafe_allow_html=True,
        )


def _render_queue_panel(client: BackendClient, T: Dict[str, Any]) -> None:
    with st.container(border=True):
        st.markdown(f"<div class='section-title'>{E(Tget(T, 'queue_short', 'Queue'))}</div>", unsafe_allow_html=True)
        try:
            q = client.queue_list()
        except Exception as e:
            soft_note(f"{Tget(T, 'run_err_prefix', 'Error')}: {e}", kind="warn")
            q = {}
        queued = q.get("queued") or []
        if not queued:
            soft_note(Tget(T, "queue_empty", "Queue is empty."), kind="info")
            return
        for idx, job in enumerate(queued[:5]):
            jid = str(job.get("job_id") or "")
            vid = str(job.get("video_id") or "")
            pct = job.get("progress")
            with st.container(border=True):
                st.markdown(f"<div class='queue-item-title'>{E(vid)}</div>", unsafe_allow_html=True)
                st.caption(E(jid))
                render_progress_bar(pct, label=Tget(T, "status", "Status"))
        st.markdown("<div class='btn-open btn-ghost'>", unsafe_allow_html=True)
        if st.button(Tget(T, "show_more", "Show more"), key="queue_more_global", width="stretch"):
            st.query_params["tab"] = "storage"
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)


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
        if not isinstance(tabs, list) or len(tabs) != 3:
            raise ValueError(f"ui_text.json: '{lang}.tabs' must be a list of 3 labels")
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
            menu_lab = Tget(T0, "menu", "Menu")
            st.markdown(
                f"<div class='section-nav'><span class='section-nav-title'>{E(menu_lab)}</span>{''.join(parts)}</div>",
                unsafe_allow_html=True,
            )
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


def home_tab_legacy(client: BackendClient, cfg, ui_text: Dict[str, Dict[str, Any]]) -> None:
    u = UIState()
    u.bind_defaults()
    _ensure_defaults_for_widgets()

    langs = cfg.ui.langs or ["ru", "kz", "en"]
    Ttmp = get_T(ui_text, st.session_state["ui_lang"])
    section_links = [
        (Tget(Ttmp, "preview", "Preview"), "#sec_preview"),
        (Tget(Ttmp, "videos_panel_title", "Videos"), "#sec_videos"),
        (Tget(Ttmp, "queue", "Queue"), "#sec_queue"),
        (Tget(Ttmp, "processing_in_home", "Processing"), "#sec_processing"),
        (Tget(Ttmp, "video_analytics", "Video analytics"), "#sec_analytics"),
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
    active_lang = (st.session_state.get("ui_lang") or cfg.ui.default_lang or "en").strip().lower()

    ids = sorted([v["video_id"] for v in videos])
    if ids and (st.session_state.get("selected_video_id") not in ids):
        st.session_state["selected_video_id"] = ids[0]

    vid_to_path = {v["video_id"]: Path(v["path"]) for v in videos}

    def _fetch_outputs(video_id: str, lang: str) -> Optional[Dict[str, Any]]:
        try:
            return client.get_video_outputs(video_id, lang)
        except Exception:
            return None

    def _enqueue_job(video_id: str, *, start_now: bool) -> None:
        force_ow = bool(st.session_state.get("force_overwrite_outputs", False))
        extra = {
            "device": str(st.session_state.get("device_proc", "cuda")),
            "force_overwrite": bool(force_ow),
        }
        try:
            job = client.create_job(video_id, extra=extra)
        except Exception as e:
            soft_note(f"{Tget(T, 'run_err_prefix', 'Error')}: {e}", kind="warn")
            return

        if start_now:
            st.session_state["active_job_id"] = job.get("job_id")
            st.session_state["active_job_video_id"] = video_id
            request_scroll_to(u, "sec_processing")
        else:
            soft_note(Tget(T, "job_queued", "Job added to queue"), kind="ok")
            request_scroll_to(u, "sec_queue")
        st.rerun()

    selected_video_id = st.session_state.get("selected_video_id")
    available_outputs: List[str] = []
    if selected_video_id:
        v_sel = next((x for x in videos if x["video_id"] == selected_video_id), None)
        if isinstance(v_sel, dict):
            available_outputs = [str(x).strip().lower() for x in (v_sel.get("languages") or []) if str(x).strip()]
            available_outputs = sorted(set(available_outputs))

    if available_outputs:
        if st.session_state.get("outputs_lang_sel") not in available_outputs:
            st.session_state["outputs_lang_sel"] = active_lang if active_lang in available_outputs else available_outputs[0]
    else:
        if st.session_state.get("outputs_lang_sel") is None:
            st.session_state["outputs_lang_sel"] = active_lang

    requested_outputs_lang = (st.session_state.get("outputs_lang_sel") or active_lang).strip().lower()
    outputs_lang = requested_outputs_lang
    selected_outputs = None
    if selected_video_id:
        selected_outputs = _fetch_outputs(selected_video_id, outputs_lang)
        if selected_outputs is None:
            fallback_candidates = []
            if active_lang:
                fallback_candidates.append(active_lang)
            fallback_candidates.append("en")
            fallback_candidates.extend(available_outputs)
            for cand in fallback_candidates:
                cand = (cand or "").strip().lower()
                if not cand or cand == outputs_lang:
                    continue
                selected_outputs = _fetch_outputs(selected_video_id, cand)
                if selected_outputs is not None:
                    outputs_lang = cand
                    break

    top_left, top_right = st.columns([2.2, 1.4], gap="large", vertical_alignment="top")
    with top_left:
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
                    meta = get_video_meta(str(p), _mtime(p))

                    vcol, mcol = st.columns([1.8, 1.1], gap="large", vertical_alignment="top")
                    with vcol:
                        render_video_player(p, int(st.session_state.get("preview_seek_sec") or 0))
                    with mcol:
                        meta_items = [
                            (Tget(T, "meta_format", "Format"), p.suffix.upper().lstrip(".") or "-"),
                            (Tget(T, "meta_size", "Size"), fmt_bytes(meta.get("size_bytes")) or "-"),
                            (
                                Tget(T, "meta_duration", "Duration"),
                                hms(float(meta.get("duration_sec") or 0.0)) if meta.get("duration_sec") else "-",
                            ),
                            (
                                Tget(T, "meta_resolution", "Resolution"),
                                f"{int(meta.get('width'))}x{int(meta.get('height'))}"
                                if meta.get("width") and meta.get("height")
                                else "-",
                            ),
                            (
                                Tget(T, "meta_fps", "FPS"),
                                f"{float(meta.get('fps')):.2f}" if meta.get("fps") else "-",
                            ),
                            (
                                Tget(T, "meta_updated", "Updated"),
                                fmt_dt(float(meta.get("updated_at"))) if meta.get("updated_at") else "-",
                            ),
                        ]
                        items_html = "".join(
                            [
                                f"<div class='meta-item'><div class='meta-label'>{E(k)}</div><div class='meta-value'>{E(vv)}</div></div>"
                                for k, vv in meta_items
                            ]
                        )
                        st.markdown(f"<div class='meta-grid'>{items_html}</div>", unsafe_allow_html=True)

                        maybe_playback_warning(p, T)

                        if p.suffix.lower() in {".avi", ".mkv"}:
                            st.markdown("<div class='btn-open btn-ghost'>", unsafe_allow_html=True)
                            if st.button(
                                btn_label(ICON_CONVERT, Tget(T, "convert", "Convert to MP4")),
                                key="convert_preview",
                                width="stretch",
                            ):
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
                            st.caption(Tget(T, "convert_hint", "Recommended format: MP4 (H.264/AAC)."))

                    markers = []
                    if selected_outputs and selected_outputs.get("annotations"):
                        anns = selected_outputs.get("annotations") or []
                        for a in anns[:6]:
                            start2 = float(a.get("start_sec", 0.0) or 0.0)
                            end2 = float(a.get("end_sec", 0.0) or 0.0)
                            desc = str(a.get("description", "") or "")
                            markers.append((start2, end2, desc))

                    st.markdown(
                        f"<div class='mini-title'>{E(Tget(T, 'time_markers', 'Time markers'))}</div>",
                        unsafe_allow_html=True,
                    )
                    if not markers:
                        st.caption(Tget(T, "no_markers", "No time markers available."))
                    else:
                        for i, (start2, end2, desc) in enumerate(markers):
                            b1, b2 = st.columns([1.2, 6], gap="small", vertical_alignment="center")
                            with b1:
                                st.markdown("<div class='btn-open btn-ghost btn-small'>", unsafe_allow_html=True)
                                if st.button(
                                    btn_label(ICON_JUMP, f"{mmss(start2)}-{mmss(end2)}"),
                                    key=f"marker_{vid}_{outputs_lang}_{i}",
                                    width="stretch",
                                ):
                                    st.session_state["preview_seek_sec"] = int(max(0.0, start2))
                                    request_scroll_to(u, "sec_preview")
                                    st.rerun()
                                st.markdown("</div>", unsafe_allow_html=True)
                            with b2:
                                st.caption(desc)

    with top_right:
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

            st.markdown(f"<div class='mini-title'>{E(Tget(T, 'quick_upload', 'Quick upload'))}</div>", unsafe_allow_html=True)
            uploaded = st.file_uploader(
                Tget(T, "drop_here", "Drop a video file here"),
                type=["mp4", "mov", "mkv", "avi"],
                accept_multiple_files=False,
                key="home_uploader",
            )
            hint = Tget(T, "quick_upload_hint", "")
            if hint.strip():
                st.caption(hint)

            if uploaded is not None:
                st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
                st.markdown("<div class='btn-open btn-ghost'>", unsafe_allow_html=True)
                if st.button(btn_label(ICON_SAVE, Tget(T, "save_raw", "Save")), width="stretch", key="upload_save"):
                    try:
                        client.upload_video(uploaded.name, uploaded.getbuffer().tobytes())
                        soft_note(Tget(T, "saved", "Saved"), kind="ok")
                        st.rerun()
                    except Exception as e:
                        soft_note(f"{Tget(T, 'run_err_prefix', 'Error')}: {e}", kind="warn")
                st.markdown("</div>", unsafe_allow_html=True)

            thin_rule()

            st.markdown(f"<div class='mini-title'>{E(Tget(T, 'latest_videos', 'Latest videos'))}</div>", unsafe_allow_html=True)
            video_items = []
            for v in videos:
                path = Path(v["path"])
                video_items.append((v["video_id"], path, _mtime(path)))
            video_items.sort(key=lambda x: x[2], reverse=True)

            if not video_items:
                soft_note(Tget(T, "no_videos", "No videos available. Please upload a file above."), kind="info")
            else:
                filtered = [item for item in video_items if home_search in item[0].lower()] if home_search else video_items
                if not filtered:
                    soft_note(Tget(T, "no_matches", "No matches found"), kind="info")
                else:
                    for idx, (vid, path, _) in enumerate(filtered[:5]):
                        thumb = get_thumbnail(path, Path(cfg.paths.thumbs_dir)) if path else None
                        meta = get_video_meta(str(path), _mtime(path)) if path else {}
                        list_uid = f"list_{idx}_{vid}"

                        with st.container(border=True):
                            c1, c2 = st.columns([1.1, 2.4], gap="small", vertical_alignment="center")
                            with c1:
                                if thumb:
                                    b64 = base64.b64encode(thumb).decode("utf-8")
                                    st.markdown(
                                        f"<img class='thumb-mini' src='data:image/jpeg;base64,{b64}' />",
                                        unsafe_allow_html=True,
                                    )
                                else:
                                    st.markdown(
                                        f"<div class='thumb-empty thumb-mini-empty'>{E(Tget(T, 'no_thumbnail', 'No thumbnail'))}</div>",
                                        unsafe_allow_html=True,
                                    )
                            with c2:
                                st.markdown(f"<div class='video-list-title'>{E(vid)}</div>", unsafe_allow_html=True)
                                st.markdown(
                                    f"<div class='video-list-sub'>{E(path.name if path else '')}</div>",
                                    unsafe_allow_html=True,
                                )
                                chips = []
                                if meta.get("duration_sec"):
                                    chips.append(hms(float(meta.get("duration_sec") or 0.0)))
                                if meta.get("updated_at"):
                                    chips.append(fmt_dt(float(meta.get("updated_at"))))
                                langs_available = v.get("languages") or []
                                if langs_available:
                                    chips.append(", ".join([str(x).upper() for x in langs_available]))
                                if chips:
                                    chips_html = "".join([f"<span class='chip'>{E(c)}</span>" for c in chips])
                                    st.markdown(f"<div class='chip-row'>{chips_html}</div>", unsafe_allow_html=True)

                                if st.session_state.get("confirm_delete_video_id") == vid:
                                    st.markdown(
                                        f"<div class='delete-confirm-text'>{E(Tget(T, 'delete_confirm', 'Delete this video?'))}</div>",
                                        unsafe_allow_html=True,
                                    )
                                    d1, d2 = st.columns([1, 1], gap="small")
                                    with d1:
                                        st.markdown("<div class='btn-open btn-small'>", unsafe_allow_html=True)
                                        if st.button(
                                            btn_label(ICON_CANCEL, Tget(T, "delete_no", "Cancel")),
                                            key=f"del_cancel_{list_uid}",
                                            width="stretch",
                                        ):
                                            st.session_state["confirm_delete_video_id"] = None
                                            st.rerun()
                                        st.markdown("</div>", unsafe_allow_html=True)
                                    with d2:
                                        st.markdown("<div class='btn-del btn-small'>", unsafe_allow_html=True)
                                        if st.button(
                                            btn_label(ICON_DELETE, Tget(T, "delete_yes", "Delete")),
                                            key=f"del_ok_{list_uid}",
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
                                else:
                                    b1, b2, b3, b4 = st.columns([1, 1, 1, 1], gap="small")
                                    with b1:
                                        st.markdown("<div class='btn-open btn-ghost btn-small icon-only'>", unsafe_allow_html=True)
                                        if st.button(
                                            ICON_OPEN,
                                            key=f"open_{list_uid}",
                                            width="stretch",
                                            help=Tget(T, "open", "Open"),
                                        ):
                                            st.session_state["selected_video_id"] = vid
                                            st.session_state["preview_seek_sec"] = 0
                                            request_scroll_to(u, "sec_preview")
                                            st.rerun()
                                        st.markdown("</div>", unsafe_allow_html=True)
                                    with b2:
                                        st.markdown("<div class='btn-open btn-ghost btn-small icon-only'>", unsafe_allow_html=True)
                                        if st.button(
                                            ICON_QUEUE,
                                            key=f"queue_{list_uid}",
                                            width="stretch",
                                            help=Tget(T, "add_to_queue", "Add to queue"),
                                        ):
                                            _enqueue_job(vid, start_now=False)
                                        st.markdown("</div>", unsafe_allow_html=True)
                                    with b3:
                                        st.markdown("<div class='btn-run btn-small icon-only'>", unsafe_allow_html=True)
                                        if st.button(
                                            ICON_RUN,
                                            key=f"run_{list_uid}",
                                            width="stretch",
                                            help=Tget(T, "run_inference", "Run inference"),
                                        ):
                                            st.session_state["selected_video_id"] = vid
                                            _enqueue_job(vid, start_now=True)
                                        st.markdown("</div>", unsafe_allow_html=True)
                                    with b4:
                                        st.markdown("<div class='btn-del btn-small icon-only'>", unsafe_allow_html=True)
                                        if st.button(
                                            ICON_DELETE,
                                            key=f"del_{list_uid}",
                                            width="stretch",
                                            help=Tget(T, "delete_video", "Delete"),
                                        ):
                                            st.session_state["confirm_delete_video_id"] = vid
                                            st.rerun()
                                        st.markdown("</div>", unsafe_allow_html=True)

    anchor("sec_queue")
    with st.container(border=True):
        st.markdown(f"<div class='section-title'>{E(Tget(T, 'queue', 'Queue'))}</div>", unsafe_allow_html=True)

        try:
            q = client.queue_list()
        except Exception as e:
            soft_note(f"{Tget(T, 'run_err_prefix', 'Error')}: {e}", kind="warn")
            q = {}

        status = q.get("status") or {}
        paused = bool(status.get("paused", False))
        updated_at = fmt_dt(status.get("updated_at"))
        queued = q.get("queued") or []

        c_run, c_pause, c_refresh, c_status = st.columns([0.9, 0.9, 0.9, 3], gap="small")
        with c_run:
            sel_vid = st.session_state.get("selected_video_id")
            st.markdown("<div class='btn-run icon-only'>", unsafe_allow_html=True)
            if st.button(
                ICON_RUN,
                width="stretch",
                key="queue_run_infer",
                help=Tget(T, "run_inference", "Run inference"),
                disabled=not bool(sel_vid),
            ):
                _enqueue_job(str(sel_vid), start_now=True)
            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown(
                f"<div class='icon-label'>{E(Tget(T, 'run_inference', 'Run inference'))}</div>",
                unsafe_allow_html=True,
            )

        with c_pause:
            icon = ICON_RESUME if paused else ICON_PAUSE
            help_text = Tget(T, "resume_queue", "Resume queue") if paused else Tget(T, "pause_queue", "Pause queue")
            st.markdown("<div class='btn-open btn-ghost icon-only'>", unsafe_allow_html=True)
            if st.button(icon, width="stretch", key="queue_pause_toggle", help=help_text):
                if paused:
                    client.queue_resume()
                else:
                    client.queue_pause()
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='icon-label'>{E(help_text)}</div>", unsafe_allow_html=True)

        with c_refresh:
            st.markdown("<div class='btn-open btn-ghost icon-only'>", unsafe_allow_html=True)
            if st.button(ICON_REFRESH, width="stretch", key="queue_refresh", help=Tget(T, "refresh", "Refresh")):
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='icon-label'>{E(Tget(T, 'refresh', 'Refresh'))}</div>", unsafe_allow_html=True)

        with c_status:
            status_state = Tget(T, "status_paused", "Paused") if paused else Tget(T, "status_running", "Running")
            status_class = "paused" if paused else "running"
            meta_bits = []
            if updated_at:
                meta_bits.append(f"{Tget(T, 'meta_updated', 'Updated')}: {updated_at}")
            meta_bits.append(f"{Tget(T, 'queued', 'Queued')}: {len(queued)}")
            meta_html = (
                f"<span class='status-meta'>{E(' • '.join(meta_bits))}</span>" if meta_bits else ""
            )
            st.markdown(
                f"""
                <div class="status-row">
                    <span class="status-label">{E(Tget(T, 'status', 'Status'))}:</span>
                    <span class="status-pill {status_class}">{E(status_state)}</span>
                    {meta_html}
                </div>
                """,
                unsafe_allow_html=True,
            )

        running = q.get("running")
        if running:
            job_id = str(running.get("job_id") or "")
            video_id = str(running.get("video_id") or "")
            job_type = str(running.get("job_type") or "")
            lang = str(running.get("language") or "")
            stage = str(running.get("stage") or "") or Tget(T, "working", "Working...")
            msg = str(running.get("message") or "")
            progress_val = running.get("progress")
            meta_bits = [b for b in [video_id, job_type, lang.upper() if lang else "", f"id {job_id}" if job_id else ""] if b]
            meta_line = " • ".join(meta_bits)
            sublabel = f"{msg} | {meta_line}" if msg and meta_line else (msg or meta_line)

            st.markdown(
                f"""
                <div class="queue-running">
                    <div class="queue-running-title">{E(Tget(T, 'job', 'Job'))}</div>
                    <div class="queue-running-meta">{E(meta_line or video_id)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            render_progress_bar(progress_val, label=stage, sublabel=sublabel, animate=progress_val is None)

        if not queued:
            soft_note(Tget(T, "queue_empty", "Queue is empty."), kind="info")
        else:
            st.markdown(
                f"<div class='queue-count'>{E(Tget(T, 'queued', 'Queued'))}: {len(queued)}</div>",
                unsafe_allow_html=True,
            )
            for i, item in enumerate(queued):
                job_id = str(item.get("job_id", ""))
                video_id = str(item.get("video_id") or "")
                job_type = str(item.get("job_type") or "")
                lang = str(item.get("language") or "")
                state = str(item.get("state") or "")
                created = fmt_dt(item.get("created_at"))

                with st.container(border=True):
                    st.markdown(
                        f"<div class='queue-item-head'><div class='queue-item-title'>{E(video_id)}</div>"
                        f"<div class='queue-item-id'>{E(job_id)}</div></div>",
                        unsafe_allow_html=True,
                    )
                    chips = []
                    for val in [job_type, (lang.upper() if lang else ""), state, created]:
                        if val:
                            chips.append(f"<span class='queue-chip'>{E(val)}</span>")
                    if chips:
                        st.markdown(f"<div class='queue-chip-row'>{''.join(chips)}</div>", unsafe_allow_html=True)

                    actions = st.columns([1, 1, 1, 1, 1], gap="small")
                    with actions[0]:
                        st.markdown("<div class='btn-open btn-ghost btn-small icon-only'>", unsafe_allow_html=True)
                        if st.button(
                            ICON_TOP,
                            key=f"q_top_{job_id}",
                            help=Tget(T, "move_top", "Move to top"),
                        ):
                            client.queue_move(job_id, "top", steps=1)
                            st.rerun()
                        st.markdown("</div>", unsafe_allow_html=True)
                        st.markdown(
                            f"<div class='icon-label'>{E(Tget(T, 'move_top', 'Top'))}</div>",
                            unsafe_allow_html=True,
                        )
                    with actions[1]:
                        st.markdown("<div class='btn-open btn-ghost btn-small icon-only'>", unsafe_allow_html=True)
                        if st.button(
                            ICON_UP,
                            key=f"q_up_{job_id}",
                            help=Tget(T, "move_up", "Move up"),
                        ):
                            client.queue_move(job_id, "up", steps=1)
                            st.rerun()
                        st.markdown("</div>", unsafe_allow_html=True)
                        st.markdown(
                            f"<div class='icon-label'>{E(Tget(T, 'move_up', 'Up'))}</div>",
                            unsafe_allow_html=True,
                        )
                    with actions[2]:
                        st.markdown("<div class='btn-open btn-ghost btn-small icon-only'>", unsafe_allow_html=True)
                        if st.button(
                            ICON_DOWN,
                            key=f"q_down_{job_id}",
                            help=Tget(T, "move_down", "Move down"),
                        ):
                            client.queue_move(job_id, "down", steps=1)
                            st.rerun()
                        st.markdown("</div>", unsafe_allow_html=True)
                        st.markdown(
                            f"<div class='icon-label'>{E(Tget(T, 'move_down', 'Down'))}</div>",
                            unsafe_allow_html=True,
                        )
                    with actions[3]:
                        st.markdown("<div class='btn-open btn-ghost btn-small icon-only'>", unsafe_allow_html=True)
                        if st.button(
                            ICON_SKIP,
                            key=f"q_skip_{job_id}",
                            help=Tget(T, "skip_queue", "Skip"),
                        ):
                            client.queue_move(job_id, "bottom", steps=1)
                            st.rerun()
                        st.markdown("</div>", unsafe_allow_html=True)
                        st.markdown(
                            f"<div class='icon-label'>{E(Tget(T, 'skip_queue', 'Skip'))}</div>",
                            unsafe_allow_html=True,
                        )
                    with actions[4]:
                        st.markdown("<div class='btn-del btn-small icon-only'>", unsafe_allow_html=True)
                        if st.button(
                            ICON_DELETE,
                            key=f"q_rm_{job_id}",
                            help=Tget(T, "remove_from_queue", "Remove from queue"),
                        ):
                            client.queue_remove(job_id)
                            st.rerun()
                        st.markdown("</div>", unsafe_allow_html=True)
                        st.markdown(
                            f"<div class='icon-label'>{E(Tget(T, 'remove_from_queue', 'Remove'))}</div>",
                            unsafe_allow_html=True,
                        )

    anchor("sec_processing")
    with st.container(border=True):
        st.markdown(
            f"<div class='section-title'>{E(Tget(T, 'processing_in_home', 'Processing'))}</div>",
            unsafe_allow_html=True,
        )

        if not st.session_state.get("selected_video_id"):
            soft_note(Tget(T, "pick_video", "Select a video first."), kind="info")
        else:
            c1, c2 = st.columns([1.4, 1], gap="medium")
            with c1:
                st.markdown(
                    f"<div class='field-label'>{E(Tget(T, 'model_lang', 'Model output language'))}</div>",
                    unsafe_allow_html=True,
                )
                st.write(Tget(T, "model_lang_fixed", "English (EN)"))
                targets = ", ".join([str(x).upper() for x in (cfg.translation.target_langs or [])])
                if targets:
                    st.caption(f"{Tget(T, 'translation_targets', 'Translations')}: {targets}")
            with c2:
                st.markdown(f"<div class='field-label'>{E(Tget(T, 'device', 'Device'))}</div>", unsafe_allow_html=True)
                st.selectbox(
                    Tget(T, "device", "Device"),
                    options=["cuda", "cpu"],
                    index=0 if st.session_state.get("device_proc", st.session_state.get("device", "cuda")) == "cuda" else 1,
                    key="device_proc",
                    label_visibility="collapsed",
                )
            st.checkbox(
                Tget(T, "force_overwrite_outputs", "Force overwrite existing outputs"),
                key="force_overwrite_outputs",
            )

            st.markdown("<div class='btn-run btn-main icon-only icon-bright'>", unsafe_allow_html=True)
            run_clicked = st.button(ICON_RUN, key="run_main", help=Tget(T, "run_pipeline", "Run"), width="stretch")
            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown(
                f"<div class='icon-label'>{E(Tget(T, 'run_pipeline', 'Run'))}</div>",
                unsafe_allow_html=True,
            )

            should_run = bool(run_clicked) or bool(st.session_state.pop("_run_request", False))
            if should_run:
                force_ow = bool(st.session_state.get("force_overwrite_outputs", False))
                extra = {
                    "device": str(st.session_state.get("device_proc", "cuda")),
                    "force_overwrite": bool(force_ow),
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
                st.markdown(
                    f"<div class='queue-running'><div class='queue-running-title'>{E(Tget(T, 'job', 'Job'))}</div>"
                    f"<div class='queue-running-meta'>{E('id ' + str(job_id))}</div></div>",
                    unsafe_allow_html=True,
                )
                progress_slot = st.empty()
                render_progress_bar(
                    None,
                    label=Tget(T, "working", "Working..."),
                    sublabel=f"id {job_id}",
                    animate=True,
                    container=progress_slot,
                )

                st.markdown("<div class='btn-open btn-ghost'>", unsafe_allow_html=True)
                cancel = st.button(
                    btn_label(ICON_CANCEL, Tget(T, "cancel_job", "Cancel job")),
                    width="stretch",
                    key="cancel_job",
                )
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
                stage_label = stage or Tget(T, "working", "Working...")
                sublabel = msg or f"id {job_id}"
                render_progress_bar(
                    min(max(progress, 0.0), 1.0),
                    label=stage_label,
                    sublabel=sublabel,
                    container=progress_slot,
                )

                if state == "done":
                    st.session_state["active_job_id"] = None
                    soft_note(Tget(T, "processing_finished", "Processing finished."), kind="ok")
                    st.rerun()
                elif state == "failed":
                    st.session_state["active_job_id"] = None
                    soft_note(f"{Tget(T, 'failed', 'Failed')}: {job.get('error') or job.get('message')}", kind="warn")
                elif state == "canceled":
                    st.session_state["active_job_id"] = None
                    soft_note(Tget(T, "canceled", "Canceled"), kind="warn")
    anchor("sec_analytics")
    with st.container(border=True):
        st.markdown(
            f"<div class='section-title'>{E(Tget(T, 'video_analytics', 'Video analytics'))}</div>",
            unsafe_allow_html=True,
        )

        vid = st.session_state.get("selected_video_id")
        out = selected_outputs
        if not vid:
            soft_note(Tget(T, "pick_video", "Select a video first."), kind="info")
            return
        if available_outputs:
            lab = Tget(T, "output_language", "Output language")
            st.selectbox(
                lab,
                options=available_outputs,
                index=available_outputs.index(st.session_state.get("outputs_lang_sel", available_outputs[0])),
                key="outputs_lang_sel",
            )
        if not out:
            soft_note(Tget(T, "no_outputs", "No outputs available for this language yet."), kind="info")
            return

        if outputs_lang != requested_outputs_lang:
            soft_note(
                Tget(T, "language_fallback", "Showing available outputs while translations are processing."),
                kind="info",
            )

        manifest = out.get("manifest") or {}
        anns = out.get("annotations") or []
        met: Dict[str, Any] = out.get("metrics") or {}
        lang_meta = (manifest.get("languages") or {}).get(outputs_lang, {})

        a_left, a_right = st.columns([1.3, 1], gap="large", vertical_alignment="top")
        with a_left:
            with st.container(border=True):
                st.markdown(
                    f"<div class='inner-title'>{E(Tget(T, 'global_summary_box', 'Video summary'))}</div>",
                    unsafe_allow_html=True,
                )
                if out.get("global_summary"):
                    st.write(out["global_summary"])
                else:
                    st.caption(Tget(T, "no_global_summary", "No summary available."))
        with a_right:
            with st.container(border=True):
                st.markdown(
                    f"<div class='inner-title'>{E(Tget(T, 'output_meta', 'Output details'))}</div>",
                    unsafe_allow_html=True,
                )
                meta_items = [
                    (Tget(T, "meta_language", "Language"), outputs_lang.upper()),
                    (Tget(T, "meta_source_language", "Source"), lang_meta.get("source_lang") or met.get("language") or "-"),
                    (
                        Tget(T, "meta_updated", "Updated"),
                        fmt_dt(float(lang_meta.get("updated_at"))) if lang_meta.get("updated_at") else "-",
                    ),
                    (Tget(T, "meta_device", "Device"), met.get("device") or "-"),
                ]
                items_html = "".join(
                    [
                        f"<div class='meta-item'><div class='meta-label'>{E(k)}</div><div class='meta-value'>{E(vv)}</div></div>"
                        for k, vv in meta_items
                    ]
                )
                st.markdown(f"<div class='meta-grid'>{items_html}</div>", unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown(
                f"<div class='inner-title'>{E(Tget(T, 'metrics_block', 'Metrics'))}</div>",
                unsafe_allow_html=True,
            )
            stats = [
                (Tget(T, "metric_clips", "Clips"), met.get("num_clips")),
                (Tget(T, "metric_frames", "Frames"), met.get("num_frames")),
                (
                    Tget(T, "metric_video_duration", "Video duration"),
                    hms(float(met.get("video_duration_sec", 0.0) or 0.0)),
                ),
                (
                    Tget(T, "metric_avg_clip", "Avg clip"),
                    f"{float(met.get('avg_clip_duration_sec', 0.0) or 0.0):.2f}s",
                ),
            ]
            s_cols = st.columns(4, gap="small")
            for col, (lab, val) in zip(s_cols, stats):
                col.metric(lab, "-" if val is None else str(val))

            t_cols = st.columns(4, gap="small")
            t_cols[0].metric(Tget(T, "metrics_preprocess", "Preprocess"), hms(float(met.get("preprocess_time_sec", 0.0) or 0.0)))
            t_cols[1].metric(Tget(T, "metrics_model", "Inference"), hms(float(met.get("model_time_sec", 0.0) or 0.0)))
            t_cols[2].metric(Tget(T, "metrics_postprocess", "Postprocess"), hms(float(met.get("postprocess_time_sec", 0.0) or 0.0)))
            t_cols[3].metric(Tget(T, "metrics_total", "Total"), hms(float(met.get("total_time_sec", 0.0) or 0.0)))

        with st.container(border=True):
            st.markdown(f"<div class='inner-title'>{E(Tget(T, 'segments_timeline', 'Timeline'))}</div>", unsafe_allow_html=True)
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
                        if st.button(
                            btn_label(ICON_JUMP, f"{mmss(start2)}-{mmss(end2)}"),
                            key=f"seg_seek_{vid}_{outputs_lang}_{i}",
                            width="stretch",
                        ):
                            st.session_state["preview_seek_sec"] = int(max(0.0, start2))
                            request_scroll_to(u, "sec_preview")
                            st.rerun()
                        st.markdown("</div>", unsafe_allow_html=True)
                    with b2:
                        st.write(desc)

        if anns:
            bins = build_timeline_bins(anns, max_bins=40)
            if bins:
                with st.container(border=True):
                    st.markdown(
                        f"<div class='inner-title'>{E(Tget(T, 'timeline_overview', 'Timeline overview'))}</div>",
                        unsafe_allow_html=True,
                    )
                    st.bar_chart([b[2] for b in bins])
                    labels = [f"{mmss(b[0])}-{mmss(b[1])} ({b[2]})" for b in bins]
                    sel = st.selectbox(
                        Tget(T, "timeline_bin", "Jump to interval"),
                        options=list(range(len(labels))),
                        format_func=lambda i: labels[i],
                        key="timeline_bin_sel",
                    )
                    st.markdown("<div class='btn-open btn-ghost'>", unsafe_allow_html=True)
                    if st.button(
                        btn_label(ICON_JUMP, Tget(T, "jump", "Jump")),
                        width="stretch",
                        key="timeline_jump",
                    ):
                        st.session_state["preview_seek_sec"] = int(max(0.0, bins[int(sel)][0]))
                        request_scroll_to(u, "sec_preview")
                        st.rerun()
                    st.markdown("</div>", unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown(f"<div class='inner-title'>{E(Tget(T, 'export', 'Export'))}</div>", unsafe_allow_html=True)
            export_payload = {
                "video_id": vid,
                "language": outputs_lang,
                "manifest": out.get("manifest"),
                "annotations": out.get("annotations") or [],
                "metrics": out.get("metrics") or {},
            }
            json_bytes = json.dumps(export_payload, ensure_ascii=False, indent=2).encode("utf-8")
            csv_text = annotations_to_csv(out.get("annotations") or [])
            met_bytes = json.dumps(out.get("metrics") or {}, ensure_ascii=False, indent=2).encode("utf-8")

            d1, d2, d3 = st.columns([1, 1, 1], gap="small")
            with d1:
                st.download_button(
                    label=Tget(T, "export_json", "Export JSON"),
                    data=json_bytes,
                    file_name=f"{vid}_{outputs_lang}.json",
                    mime="application/json",
                )
            with d2:
                st.download_button(
                    label=Tget(T, "export_csv", "Export CSV"),
                    data=csv_text,
                    file_name=f"{vid}_{outputs_lang}.csv",
                    mime="text/csv",
                )
            with d3:
                st.download_button(
                    label=Tget(T, "export_metrics", "Export metrics"),
                    data=met_bytes,
                    file_name=f"{vid}_{outputs_lang}_metrics.json",
                    mime="application/json",
                )


def home_tab(client: BackendClient, cfg, ui_text: Dict[str, Dict[str, Any]]) -> None:
    u = UIState()
    u.bind_defaults()
    _ensure_defaults_for_widgets()

    langs = cfg.ui.langs or ["ru", "kz", "en"]
    Ttmp = get_T(ui_text, st.session_state["ui_lang"])
    section_links = [
        (Tget(Ttmp, "summary", "Summary"), "#sec_summary"),
        (Tget(Ttmp, "home_guide_title", "Getting started"), "#sec_guide"),
        (Tget(Ttmp, "home_drag_title", "Quick start"), "#sec_upload"),
        (Tget(Ttmp, "recent", "Recent"), "#sec_recent"),
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
    ids = sorted([v["video_id"] for v in videos])
    if ids and (st.session_state.get("selected_video_id") not in ids):
        st.session_state["selected_video_id"] = ids[0]

    left_panel, main_col, right_panel = st.columns([1.05, 4.9, 1.25], gap="large", vertical_alignment="top")
    with left_panel:
        _render_assistant_panel(T)
    with right_panel:
        _render_assistant_panel(T)

    with main_col:
        st.markdown(
            f"<div class='home-hero'>"
            f"<div class='home-hero-title'>{E(Tget(T, 'home_hero_title', 'SmartCampus V2T'))}</div>"
            f"<div class='home-hero-sub'>{E(Tget(T, 'home_hero_subtitle', 'Video-to-text system'))}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        anchor("sec_summary")
        with st.container(border=True):
            st.markdown(f"<div class='section-title'>{E(Tget(T, 'summary', 'Summary'))}</div>", unsafe_allow_html=True)
            try:
                q = client.queue_list()
            except Exception:
                q = {}
            queued = q.get("queued") or []

            today = time.strftime("%Y-%m-%d", time.localtime())
            today_count = 0
            total_duration = 0.0
            for v in videos:
                p = Path(v["path"])
                if not p.exists():
                    continue
                m = fmt_dt(_mtime(p))
                if m.startswith(today):
                    today_count += 1
                meta = get_video_meta(str(p), _mtime(p))
                if meta.get("duration_sec"):
                    total_duration += float(meta.get("duration_sec") or 0.0)

            s1, s2, s3, s4 = st.columns(4, gap="small")
            s1.metric(Tget(T, "summary_today", "Today"), str(today_count))
            s2.metric(Tget(T, "summary_queue", "In queue"), str(len(queued)))
            s3.metric(Tget(T, "summary_errors", "Errors"), "0")
            s4.metric(Tget(T, "summary_processed", "Processed"), hms(total_duration) if total_duration > 0 else "0:00")

        anchor("sec_guide")
        with st.container(border=True):
            st.markdown(f"<div class='section-title'>{E(Tget(T, 'home_guide_title', 'Getting started'))}</div>", unsafe_allow_html=True)
            st.markdown(
                "<div class='home-guide'>"
                f"<div>{E(Tget(T, 'home_step1', '1. Drop a video file'))}</div>"
                f"<div>{E(Tget(T, 'home_step2', '2. Go to Analytics'))}</div>"
                f"<div>{E(Tget(T, 'home_step3', '3. Run processing'))}</div>"
                f"<div>{E(Tget(T, 'home_step4', '4. Search events and segments'))}</div>"
                "</div>",
                unsafe_allow_html=True,
            )

        anchor("sec_upload")
        with st.container(border=True):
            st.markdown(f"<div class='section-title'>{E(Tget(T, 'home_drag_title', 'Quick start'))}</div>", unsafe_allow_html=True)
            st.caption(Tget(T, "home_drag_hint", "Drop a file here — Analytics will open after upload"))
            uploaded = st.file_uploader(
                Tget(T, "drop_here", "Drop a video file here"),
                type=["mp4", "mov", "mkv", "avi"],
                accept_multiple_files=False,
                key="home_uploader",
            )
            if uploaded is not None:
                st.markdown("<div class='btn-open btn-ghost'>", unsafe_allow_html=True)
                if st.button(btn_label(ICON_SAVE, Tget(T, "save_raw", "Save")), width="stretch", key="upload_save"):
                    try:
                        client.upload_video(uploaded.name, uploaded.getbuffer().tobytes())
                        soft_note(Tget(T, "saved", "Saved"), kind="ok")
                        st.query_params["tab"] = "analytics"
                        st.rerun()
                    except Exception as e:
                        soft_note(f"{Tget(T, 'run_err_prefix', 'Error')}: {e}", kind="warn")
                st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("<div class='btn-run btn-main'>", unsafe_allow_html=True)
            if st.button(Tget(T, "home_to_analytics", "Go to Analytics"), key="go_analytics", width="stretch"):
                st.query_params["tab"] = "analytics"
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

        anchor("sec_recent")
        with st.container(border=True):
            st.markdown(f"<div class='section-title'>{E(Tget(T, 'recent', 'Recent'))}</div>", unsafe_allow_html=True)
            video_items = []
            for v in videos:
                path = Path(v["path"])
                video_items.append((v["video_id"], path, _mtime(path)))
            video_items.sort(key=lambda x: x[2], reverse=True)

            if not video_items:
                soft_note(Tget(T, "no_videos", "No videos available. Please upload a file above."), kind="info")
            else:
                for idx, (vid, path, _) in enumerate(video_items[:10]):
                    thumb = get_thumbnail(path, Path(cfg.paths.thumbs_dir)) if path else None
                    meta = get_video_meta(str(path), _mtime(path)) if path else {}
                    list_uid = f"recent_{idx}_{vid}"
                    with st.container(border=True):
                        c1, c2 = st.columns([1.1, 3.4], gap="medium", vertical_alignment="center")
                        with c1:
                            if thumb:
                                b64 = base64.b64encode(thumb).decode("utf-8")
                                st.markdown(
                                    f"<img class='thumb-mini' src='data:image/jpeg;base64,{b64}' />",
                                    unsafe_allow_html=True,
                                )
                            else:
                                st.markdown(
                                    f"<div class='thumb-empty thumb-mini-empty'>{E(Tget(T, 'no_thumbnail', 'No thumbnail'))}</div>",
                                    unsafe_allow_html=True,
                                )
                        with c2:
                            st.markdown(f"<div class='recent-title'>{E(vid)}</div>", unsafe_allow_html=True)
                            date_line = fmt_dt(float(meta.get("updated_at"))) if meta.get("updated_at") else "-"
                            duration_line = hms(float(meta.get("duration_sec") or 0.0)) if meta.get("duration_sec") else "-"
                            st.markdown(
                                f"<div class='recent-sub'>{E(date_line)} • {E(duration_line)}</div>",
                                unsafe_allow_html=True,
                            )
                            st.caption(Tget(T, "assistant_placeholder", "Description placeholder."))
                            chips = ["плотность: средняя", "динамика: статично", "обычное"]
                            chips_html = "".join([f"<span class='chip'>{E(c)}</span>" for c in chips])
                            st.markdown(f"<div class='chip-row'>{chips_html}</div>", unsafe_allow_html=True)

                            b1, b2 = st.columns([1, 1], gap="small")
                            with b1:
                                st.markdown("<div class='btn-open btn-ghost btn-small'>", unsafe_allow_html=True)
                                if st.button(btn_label(ICON_OPEN, Tget(T, "open", "Open")), key=f"open_{list_uid}", width="stretch"):
                                    st.session_state["selected_video_id"] = vid
                                    st.session_state["preview_seek_sec"] = 0
                                    st.query_params["tab"] = "analytics"
                                    st.rerun()
                                st.markdown("</div>", unsafe_allow_html=True)
                            with b2:
                                st.markdown("<div class='btn-open btn-ghost btn-small'>", unsafe_allow_html=True)
                                if st.button(btn_label(ICON_QUEUE, Tget(T, "add_to_queue", "Add to queue")), key=f"queue_{list_uid}", width="stretch"):
                                    try:
                                        client.create_job(vid, extra={})
                                        soft_note(Tget(T, "job_queued", "Job added to queue"), kind="ok")
                                    except Exception as e:
                                        soft_note(f"{Tget(T, 'run_err_prefix', 'Error')}: {e}", kind="warn")
                                st.markdown("</div>", unsafe_allow_html=True)


def search_tab_legacy(client: BackendClient, cfg, ui_text: Dict[str, Dict[str, Any]]) -> None:
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
    active_lang = (st.session_state.get("ui_lang") or cfg.ui.default_lang or "en").strip().lower()

    videos = client.list_videos()
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

        def _parse_float(s: Any) -> Optional[float]:
            s = ("" if s is None else str(s)).strip().replace(",", ".")
            if not s:
                return None
            try:
                return float(s)
            except Exception:
                return None

        video_options = ["(all)"] + sorted([v["video_id"] for v in videos])
        if st.session_state.get("search_video_filter") not in video_options:
            st.session_state["search_video_filter"] = "(all)"
        st.selectbox(
            Tget(T, "video_filter", "Video"),
            options=video_options,
            index=video_options.index(st.session_state["search_video_filter"]),
            key="search_video_filter",
        )

        st.checkbox(
            Tget(T, "time_filter", "Filter by time range"),
            key="search_time_filter",
        )
        if st.session_state.get("search_time_filter"):
            tc1, tc2 = st.columns([1, 1], gap="medium")
            with tc1:
                st.text_input(
                    Tget(T, "time_start", "Start time (sec)"),
                    key="search_time_start",
                    placeholder="0",
                )
            with tc2:
                st.text_input(
                    Tget(T, "time_end", "End time (sec)"),
                    key="search_time_end",
                    placeholder="",
                )

        st.checkbox(
            Tget(T, "duration_filter", "Filter by segment duration"),
            key="search_duration_filter",
        )
        if st.session_state.get("search_duration_filter"):
            dc1, dc2 = st.columns([1, 1], gap="medium")
            with dc1:
                st.text_input(
                    Tget(T, "duration_min", "Min duration (sec)"),
                    key="search_dur_min",
                    placeholder="",
                )
            with dc2:
                st.text_input(
                    Tget(T, "duration_max", "Max duration (sec)"),
                    key="search_dur_max",
                    placeholder="",
                )

        with st.container(border=True):
            st.markdown(f"<div class='inner-title'>{E(Tget(T, 'saved_queries', 'Saved queries'))}</div>", unsafe_allow_html=True)
            st.text_input(
                Tget(T, "saved_query_name", "Name"),
                key="save_query_name",
                placeholder=Tget(T, "saved_query_name_ph", "e.g. Crowd running"),
            )
            save_cols = st.columns([1, 1], gap="small")
            with save_cols[0]:
                st.markdown("<div class='btn-open btn-ghost'>", unsafe_allow_html=True)
                if st.button(btn_label(ICON_SAVE, Tget(T, "save_query", "Save query")), key="save_query_btn"):
                    name = (st.session_state.get("save_query_name") or "").strip() or (search_query.strip() or "Query")
                    item = {
                        "name": name,
                        "query": search_query.strip(),
                        "top_k": int(search_topk),
                        "video_filter": st.session_state.get("search_video_filter", "(all)"),
                        "time_filter": bool(st.session_state.get("search_time_filter")),
                        "time_start": st.session_state.get("search_time_start"),
                        "time_end": st.session_state.get("search_time_end"),
                        "duration_filter": bool(st.session_state.get("search_duration_filter")),
                        "dur_min": st.session_state.get("search_dur_min"),
                        "dur_max": st.session_state.get("search_dur_max"),
                    }
                    existing = st.session_state.get("saved_queries") or []
                    existing = [x for x in existing if x.get("name") != name]
                    existing.append(item)
                    st.session_state["saved_queries"] = existing
                    soft_note(Tget(T, "saved", "Saved"), kind="ok")
                st.markdown("</div>", unsafe_allow_html=True)
            with save_cols[1]:
                st.markdown("<div class='btn-open btn-ghost'>", unsafe_allow_html=True)
                if st.button(btn_label(ICON_CLEAR, Tget(T, "clear_query", "Clear")), key="clear_query_btn"):
                    st.session_state["search_query"] = ""
                    st.session_state["search_time_start"] = ""
                    st.session_state["search_time_end"] = ""
                    st.session_state["search_dur_min"] = ""
                    st.session_state["search_dur_max"] = ""
                    st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)

            saved = st.session_state.get("saved_queries") or []
            if saved:
                options = [s.get("name", "Query") for s in saved]
                if st.session_state.get("saved_query_sel") not in options:
                    st.session_state["saved_query_sel"] = options[-1]
                lc1, lc2 = st.columns([1, 1], gap="small")
                with lc1:
                    st.selectbox(
                        Tget(T, "saved_query_pick", "Load saved query"),
                        options=options,
                        index=options.index(st.session_state["saved_query_sel"]),
                        key="saved_query_sel",
                    )
                with lc2:
                    st.markdown("<div class='btn-open btn-ghost'>", unsafe_allow_html=True)
                    if st.button(btn_label(ICON_LOAD, Tget(T, "load_query", "Load")), key="load_query_btn"):
                        sel_name = st.session_state.get("saved_query_sel")
                        item = next((x for x in saved if x.get("name") == sel_name), None)
                        if item:
                            st.session_state["search_query"] = item.get("query", "")
                            st.session_state["search_topk"] = int(item.get("top_k", 10) or 10)
                            st.session_state["search_video_filter"] = item.get("video_filter", "(all)")
                            st.session_state["search_time_filter"] = bool(item.get("time_filter"))
                            st.session_state["search_time_start"] = item.get("time_start", "")
                            st.session_state["search_time_end"] = item.get("time_end", "")
                            st.session_state["search_duration_filter"] = bool(item.get("duration_filter"))
                            st.session_state["search_dur_min"] = item.get("dur_min", "")
                            st.session_state["search_dur_max"] = item.get("dur_max", "")
                            st.rerun()
                    st.markdown("</div>", unsafe_allow_html=True)

        search_video_filter = str(st.session_state.get("search_video_filter", "(all)"))
        time_start = _parse_float(st.session_state.get("search_time_start")) if st.session_state.get("search_time_filter") else None
        time_end = _parse_float(st.session_state.get("search_time_end")) if st.session_state.get("search_time_filter") else None
        dur_min = _parse_float(st.session_state.get("search_dur_min")) if st.session_state.get("search_duration_filter") else None
        dur_max = _parse_float(st.session_state.get("search_dur_max")) if st.session_state.get("search_duration_filter") else None

        hint = Tget(T, "index_hint", "")
        if hint.strip():
            st.caption(hint)

        thin_rule()

        st.markdown("<div class='btn-open btn-ghost'>", unsafe_allow_html=True)
        if st.button(
            btn_label(ICON_REFRESH, Tget(T, "update_index", "Update search index")),
            width="stretch",
            key="update_index_search",
        ):
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
            language=active_lang,
            dedupe=True,
            start_sec=time_start,
            end_sec=time_end,
            min_duration_sec=dur_min,
            max_duration_sec=dur_max,
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
                            if st.button(
                                btn_label(ICON_OPEN, f"{mmss(h['start_sec'])}-{mmss(h['end_sec'])}"),
                                key=f"hit_{idx}",
                                width="stretch",
                            ):
                                st.session_state["selected_hit"] = h
                                request_scroll_to(u, "sec_player")
                                st.rerun()
                            st.markdown("</div>", unsafe_allow_html=True)
                        with b2:
                            st.write(h["description"])
                            st.caption(
                                f"{h['video_id']} | {str(h.get('language') or '').upper()} | score={h['score']:.3f} "
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

            st.write(f"**{vid}** | {str(hit.get('language') or '').upper()}")
            st.write(f"[{mmss(start)}-{mmss(end)}] {desc}")

            p = vid_to_path.get(vid)
            if not p or not p.exists():
                soft_note(f"{Tget(T, 'raw_missing', 'Source video not found')}: {vid}", kind="warn")
            else:
                maybe_playback_warning(p, T)
                render_video_player(p, int(start))


def search_tab(client: BackendClient, cfg, ui_text: Dict[str, Dict[str, Any]]) -> None:
    u = UIState()
    u.bind_defaults()
    _ensure_defaults_for_widgets()

    langs = cfg.ui.langs or ["ru", "kz", "en"]
    Ttmp = get_T(ui_text, st.session_state["ui_lang"])
    section_links = [
        (Tget(Ttmp, "player", "Player"), "#sec_player"),
        (Tget(Ttmp, "analytics_summary", "Summary"), "#sec_summary"),
        (Tget(Ttmp, "analytics_segments", "Segments"), "#sec_segments"),
    ]
    T = render_page_head(
        Tget(Ttmp, "search_desc_title", "Analytics"),
        ui_text,
        section_links=section_links,
        langs=langs,
        default_lang=cfg.ui.default_lang,
    )
    active_lang = (st.session_state.get("ui_lang") or cfg.ui.default_lang or "en").strip().lower()

    videos = client.list_videos()
    vid_to_path = {v["video_id"]: Path(v["path"]) for v in videos}

    def _fetch_outputs(video_id: str, lang: str) -> Optional[Dict[str, Any]]:
        try:
            return client.get_video_outputs(video_id, lang)
        except Exception:
            return None

    top_left, top_right = st.columns([2.9, 2.3], gap="large", vertical_alignment="top")
    with top_left:
        anchor("sec_player")
        with st.container(border=True):
            st.markdown(f"<div class='section-title'>{E(Tget(T, 'player', 'Player'))}</div>", unsafe_allow_html=True)
            video_ids = sorted([v["video_id"] for v in videos])
            if video_ids and (st.session_state.get("selected_video_id") not in video_ids):
                st.session_state["selected_video_id"] = video_ids[0]
            st.selectbox(
                Tget(T, "selected_video", "Selected video"),
                options=video_ids or ["-"],
                index=video_ids.index(st.session_state.get("selected_video_id")) if video_ids and st.session_state.get("selected_video_id") in video_ids else 0,
                key="selected_video_id",
            )
            selected_video_id = st.session_state.get("selected_video_id")
            if not selected_video_id:
                soft_note(Tget(T, "pick_video", "Select a video first."), kind="info")
            else:
                p = vid_to_path.get(selected_video_id)
                if not p or not p.exists():
                    soft_note(f"{Tget(T, 'raw_missing', 'Source video not found')}: {selected_video_id}", kind="warn")
                else:
                    maybe_playback_warning(p, T)
                    render_video_player(p, int(st.session_state.get("preview_seek_sec") or 0))

    with top_right:
        anchor("sec_summary")
        with st.container(border=True):
            st.markdown(f"<div class='section-title'>{E(Tget(T, 'analytics_summary', 'Summary'))}</div>", unsafe_allow_html=True)
            outputs = _fetch_outputs(str(st.session_state.get("selected_video_id") or ""), active_lang) or {}
            met = outputs.get("metrics") or {}
            summary = met.get("global_summary")
            if summary:
                st.write(summary)
            else:
                st.caption(Tget(T, "assistant_placeholder", "Description placeholder."))
            thin_rule()
            anchor("sec_segments")
            st.markdown(
                f"<div class='mini-title'>{E(Tget(T, 'analytics_segments', 'Segments and clips'))}</div>",
                unsafe_allow_html=True,
            )
            anns = outputs.get("annotations") or []
            if not anns:
                soft_note(Tget(T, "no_annotations", "No annotations available."), kind="info")
            else:
                for i, a in enumerate(anns[:12]):
                    start2 = float(a.get("start_sec", 0.0) or 0.0)
                    end2 = float(a.get("end_sec", 0.0) or 0.0)
                    desc = str(a.get("description", "") or "")
                    b1, b2 = st.columns([1.5, 6], gap="small", vertical_alignment="center")
                    with b1:
                        st.markdown("<div class='btn-open btn-ghost btn-small'>", unsafe_allow_html=True)
                        if st.button(btn_label(ICON_JUMP, f"{mmss(start2)}-{mmss(end2)}"), key=f"seg_seek_{i}", width="stretch"):
                            st.session_state["preview_seek_sec"] = int(max(0.0, start2))
                            st.rerun()
                        st.markdown("</div>", unsafe_allow_html=True)
                    with b2:
                        st.caption(desc or "-")

            thin_rule()
            chips_metrics = []
            if met:
                total = hms(float(met.get("total_time_sec", 0.0) or 0.0)) if met.get("total_time_sec") else "-"
                device = str(met.get("device") or outputs.get("device") or "-")
                clips = met.get("num_clips", "-")
                chips_metrics = [f"total: {total}", f"device: {device}", f"clips: {clips}"]
            st.markdown(
                f"<div class='chip-line-label'>{E(Tget(T, 'metrics', 'Метрики'))}:</div>",
                unsafe_allow_html=True,
            )
            if chips_metrics:
                chips_html = "".join([f"<span class='chip'>{E(c)}</span>" for c in chips_metrics])
                st.markdown(f"<div class='chip-row'>{chips_html}</div>", unsafe_allow_html=True)
            else:
                st.caption("-")

            chips_tags = ["плотность: средняя", "динамика: статично", "обычное", "локация: авто"]
            st.markdown(
                f"<div class='chip-line-label'>{E(Tget(T, 'tags', 'Теги'))}:</div>",
                unsafe_allow_html=True,
            )
            chips_tags_html = "".join([f"<span class='chip'>{E(c)}</span>" for c in chips_tags])
            st.markdown(f"<div class='chip-row'>{chips_tags_html}</div>", unsafe_allow_html=True)


def gallery_tab(client: BackendClient, cfg, ui_text: Dict[str, Dict[str, Any]]) -> None:
    u = UIState()
    u.bind_defaults()
    _ensure_defaults_for_widgets()

    langs = cfg.ui.langs or ["ru", "kz", "en"]
    Ttmp = get_T(ui_text, st.session_state["ui_lang"])
    section_links = [
        (Tget(Ttmp, "gallery_title", "Gallery"), "#sec_gallery"),
    ]
    T = render_page_head(
        Tget(Ttmp, "gallery_title", "Storage"),
        ui_text,
        section_links=section_links,
        langs=langs,
        default_lang=cfg.ui.default_lang,
    )

    scroll_if_requested(u)

    videos = client.list_videos()
    ids = sorted([v["video_id"] for v in videos])
    if ids and (st.session_state.get("selected_video_id") not in ids):
        st.session_state["selected_video_id"] = ids[0]

    def _hub_enqueue(video_id: str, *, start_now: bool) -> None:
        try:
            extra = {
                "device": str(st.session_state.get("device_proc", "cuda")),
                "force_overwrite": bool(st.session_state.get("force_overwrite_outputs", False)),
            }
            job = client.create_job(str(video_id), extra=extra)
            if start_now and job.get("job_id"):
                client.queue_move(str(job.get("job_id")), "top", steps=1)
            st.session_state["active_job_id"] = job.get("job_id")
            st.session_state["active_job_video_id"] = str(video_id)
            st.session_state["selected_video_id"] = str(video_id)
            soft_note(Tget(T, "job_queued", "Job added to queue"), kind="ok")
            if start_now:
                st.rerun()
        except Exception as e:
            soft_note(f"{Tget(T, 'run_err_prefix', 'Error')}: {e}", kind="warn")

    main_col, right_panel = st.columns([5.2, 1.8], gap="large", vertical_alignment="top")

    with main_col:
        anchor("sec_gallery")

        with st.container(border=True):
            st.markdown(f"<div class='section-title'>{E(Tget(T, 'storage_manage_title', 'Video management'))}</div>", unsafe_allow_html=True)
            c1, c2, c3, c4 = st.columns([2.4, 1.5, 1.2, 1.2], gap="medium", vertical_alignment="bottom")
            with c1:
                st.text_input(Tget(T, "search_video_label", "Search by video name"), key="gallery_search")
                st.markdown(f"<div class='field-label hub-upload-label'>{E(Tget(T, 'storage_upload_title', 'Upload'))}</div>", unsafe_allow_html=True)
                uploaded = st.file_uploader(
                    Tget(T, "drop_here", "Drop a video file here"),
                    type=["mp4", "mov", "mkv", "avi"],
                    accept_multiple_files=False,
                    key="storage_uploader",
                    label_visibility="collapsed",
                )
                if uploaded is not None:
                    st.markdown("<div class='btn-open btn-ghost btn-small'>", unsafe_allow_html=True)
                    if st.button(btn_label(ICON_SAVE, Tget(T, "save_raw", "Save")), width="stretch", key="storage_upload_save"):
                        try:
                            client.upload_video(uploaded.name, uploaded.getbuffer().tobytes())
                            soft_note(Tget(T, "storage_upload_done", "Upload complete"), kind="ok")
                            st.rerun()
                        except Exception as e:
                            soft_note(f"{Tget(T, 'run_err_prefix', 'Error')}: {e}", kind="warn")
                    st.markdown("</div>", unsafe_allow_html=True)
            with c2:
                st.markdown(f"<div class='field-label'>{E(Tget(T, 'filters', 'Filters'))}</div>", unsafe_allow_html=True)
                st.text_input(Tget(T, "filters", "Filters"), key="gallery_filter_tags", placeholder="теги / дата / длительность", label_visibility="collapsed")
            with c3:
                st.markdown(f"<div class='field-label'>{E(Tget(T, 'gallery_view', 'View'))}</div>", unsafe_allow_html=True)
                st.selectbox(
                    Tget(T, "gallery_view", "View"),
                    options=["list", "grid", "carousel"],
                    format_func=lambda x: Tget(T, "gallery_list", "List") if x == "list" else (Tget(T, "gallery_grid", "Grid") if x == "grid" else Tget(T, "gallery_carousel", "Carousel")),
                    key="gallery_view",
                    label_visibility="collapsed",
                )
            with c4:
                st.markdown(f"<div class='field-label'>{E(Tget(T, 'hub_search_type', 'Search'))}</div>", unsafe_allow_html=True)
                st.selectbox(
                    Tget(T, "search_type", "Search type"),
                    options=["title", "segments"],
                    format_func=lambda x: Tget(T, "search_by_title", "By title") if x == "title" else Tget(T, "search_by_segments", "By segments"),
                    key="hub_search_type",
                    label_visibility="collapsed",
                )

            hint = Tget(T, "filters_hint", "")
            if hint.strip():
                st.caption(hint)


        view = st.session_state.get("gallery_view") or "list"
        query = (st.session_state.get("gallery_search") or "").strip().lower()
        search_type = st.session_state.get("hub_search_type") or "title"

        items = []
        if search_type == "segments" and query:
            try:
                hits = client.search(
                    query=query,
                    top_k=20,
                    video_id=None,
                    language=(st.session_state.get("ui_lang") or "ru"),
                    dedupe=True,
                )
            except Exception as e:
                soft_note(f"{Tget(T, 'search_error', 'Search execution error')}: {e}", kind="warn")
                hits = []
            if hits:
                for idx, h in enumerate(hits):
                    vid = str(h.get("video_id") or "")
                    path = Path(next((v.get("path") for v in videos if v.get("video_id") == vid), "") or "")
                    items.append((vid, path, _mtime(path), h))
        else:
            for v in videos:
                vid = str(v.get("video_id") or "")
                if query and query not in vid.lower():
                    continue
                path = Path(v.get("path") or "")
                items.append((vid, path, _mtime(path), None))
        items.sort(key=lambda x: x[2], reverse=True)

        if not items:
            soft_note(Tget(T, "no_videos", "No videos available. Please upload a file above."), kind="info")

        if view != "list":
            soft_note(Tget(T, "assistant_placeholder", "Placeholder. Models will be connected later."), kind="info")

        if view == "list" and items:
            for idx, (vid, path, _, hit) in enumerate(items):
                    thumb = get_thumbnail(path, Path(cfg.paths.thumbs_dir)) if path else None
                    meta = get_video_meta(str(path), _mtime(path)) if path else {}
                    list_uid = f"gallery_{idx}_{vid}"

                    with st.container(border=True):
                        c1, c2 = st.columns([1.1, 3.8], gap="medium", vertical_alignment="center")
                        with c1:
                            if thumb:
                                b64 = base64.b64encode(thumb).decode("utf-8")
                                st.markdown(
                                    f"<img class='thumb-mini' src='data:image/jpeg;base64,{b64}' />",
                                    unsafe_allow_html=True,
                                )
                            else:
                                st.markdown(
                                    f"<div class='thumb-empty thumb-mini-empty'>{E(Tget(T, 'no_thumbnail', 'No thumbnail'))}</div>",
                                    unsafe_allow_html=True,
                                )
                        with c2:
                            st.markdown(f"<div class='video-list-title'>{E(vid)}</div>", unsafe_allow_html=True)
                            date_line = fmt_dt(float(meta.get("updated_at"))) if meta.get("updated_at") else "-"
                            duration_line = hms(float(meta.get("duration_sec") or 0.0)) if meta.get("duration_sec") else "-"
                            st.markdown(
                                f"<div class='video-list-sub'>{E(date_line)} • {E(duration_line)}</div>",
                                unsafe_allow_html=True,
                            )
                            if hit:
                                st.caption(f"{E(hit.get('description') or '')} • {mmss(hit.get('start_sec', 0))}-{mmss(hit.get('end_sec', 0))}")
                            else:
                                st.caption(Tget(T, "assistant_placeholder", "Description placeholder."))

                            b1, b2, b3 = st.columns([1, 1, 1], gap="small")
                            with b1:
                                st.markdown("<div class='btn-open btn-ghost btn-small icon-only'>", unsafe_allow_html=True)
                                if st.button(
                                    ICON_OPEN,
                                    key=f"g_open_{list_uid}",
                                    width="stretch",
                                    help=Tget(T, "open", "Open"),
                                ):
                                    st.session_state["selected_video_id"] = vid
                                    if hit:
                                        st.session_state["preview_seek_sec"] = int(float(hit.get("start_sec") or 0.0))
                                    else:
                                        st.session_state["preview_seek_sec"] = 0
                                    st.query_params["tab"] = "analytics"
                                    st.rerun()
                                st.markdown("</div>", unsafe_allow_html=True)
                            with b2:
                                st.markdown("<div class='btn-open btn-ghost btn-small icon-only'>", unsafe_allow_html=True)
                                if st.button(
                                    ICON_QUEUE,
                                    key=f"g_queue_{list_uid}",
                                    width="stretch",
                                    help=Tget(T, "add_to_queue", "Add to queue"),
                                ):
                                    _hub_enqueue(vid, start_now=False)
                                st.markdown("</div>", unsafe_allow_html=True)
                            with b3:
                                st.markdown("<div class='btn-del btn-small icon-only'>", unsafe_allow_html=True)
                                if st.button(
                                    ICON_DELETE,
                                    key=f"g_del_{list_uid}",
                                    width="stretch",
                                    help=Tget(T, "delete_video", "Delete"),
                                ):
                                    st.session_state["confirm_delete_video_id"] = vid
                                    st.rerun()
                                st.markdown("</div>", unsafe_allow_html=True)

                            chips = ["плотность: средняя", "динамика: статично", "обычное", "локация: авто"]
                            chips_html = "".join([f"<span class='chip'>{E(c)}</span>" for c in chips])
                            st.markdown(f"<div class='chip-row'>{chips_html}</div>", unsafe_allow_html=True)

                    if st.session_state.get("confirm_delete_video_id") == vid:
                        st.markdown(
                            f"<div class='delete-confirm-text'>{E(Tget(T, 'delete_confirm', 'Delete this video?'))}</div>",
                            unsafe_allow_html=True,
                        )
                        d1, d2 = st.columns([1, 1], gap="small")
                        with d1:
                            st.markdown("<div class='btn-open btn-small'>", unsafe_allow_html=True)
                            if st.button(
                                btn_label(ICON_CANCEL, Tget(T, "delete_no", "Cancel")),
                                key=f"g_del_cancel_{list_uid}",
                                width="stretch",
                            ):
                                st.session_state["confirm_delete_video_id"] = None
                                st.rerun()
                            st.markdown("</div>", unsafe_allow_html=True)
                        with d2:
                            st.markdown("<div class='btn-del btn-small'>", unsafe_allow_html=True)
                            if st.button(
                                btn_label(ICON_DELETE, Tget(T, "delete_yes", "Delete")),
                                key=f"g_del_ok_{list_uid}",
                                width="stretch",
                            ):
                                try:
                                    client.delete_video(vid)
                                    st.session_state["confirm_delete_video_id"] = None
                                    soft_note(Tget(T, "deleted", "Deleted"), kind="ok")
                                    st.rerun()
                                except Exception as e:
                                    soft_note(f"{Tget(T, 'run_err_prefix', 'Error')}: {e}", kind="warn")
                            st.markdown("</div>", unsafe_allow_html=True)

    with right_panel:
        st.markdown("<div class='hub-sidebar'>", unsafe_allow_html=True)

        video_ids = sorted([v["video_id"] for v in videos])
        if video_ids and (st.session_state.get("selected_video_id") not in video_ids):
            st.session_state["selected_video_id"] = video_ids[0]

        try:
            q = client.queue_list()
        except Exception as e:
            soft_note(f"{Tget(T, 'run_err_prefix', 'Error')}: {e}", kind="warn")
            q = {}
        status = q.get("status") or {}
        paused = bool(status.get("paused", False))
        updated_at = fmt_dt(status.get("updated_at"))
        queued = q.get("queued") or []
        running = q.get("running")

        with st.container(border=True):
            st.markdown(
                f"<div class='section-title'>{E(Tget(T, 'processing_in_home', 'Inference control'))}</div>",
                unsafe_allow_html=True,
            )
            st.selectbox(
                Tget(T, "selected_video", "Selected video"),
                options=video_ids or ["-"],
                index=video_ids.index(st.session_state.get("selected_video_id")) if video_ids and st.session_state.get("selected_video_id") in video_ids else 0,
                key="selected_video_id",
            )
            st.selectbox(
                Tget(T, "device", "Device"),
                options=["cuda", "cpu"],
                index=0 if st.session_state.get("device_proc", st.session_state.get("device", "cuda")) == "cuda" else 1,
                key="device_proc",
            )
            st.checkbox(
                Tget(T, "force_overwrite_outputs", "Force overwrite existing outputs"),
                key="force_overwrite_outputs",
            )
            run_col, add_col = st.columns(2, gap="small")
            with run_col:
                st.markdown("<div class='btn-run btn-main'>", unsafe_allow_html=True)
                if st.button(
                    btn_label(ICON_RUN, Tget(T, "run_inference", "Run now")),
                    width="stretch",
                    key="hub_run_now",
                    disabled=not bool(video_ids),
                ):
                    _hub_enqueue(str(st.session_state.get("selected_video_id")), start_now=True)
                st.markdown("</div>", unsafe_allow_html=True)
            with add_col:
                st.markdown("<div class='btn-open btn-ghost btn-main'>", unsafe_allow_html=True)
                if st.button(
                    btn_label(ICON_QUEUE, Tget(T, "add_to_queue", "Add to queue")),
                    width="stretch",
                    key="hub_add_queue",
                    disabled=not bool(video_ids),
                ):
                    _hub_enqueue(str(st.session_state.get("selected_video_id")), start_now=False)
                st.markdown("</div>", unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown(
                f"<div class='section-title'>{E(Tget(T, 'queue', 'Queue'))}</div>",
                unsafe_allow_html=True,
            )
            pause_col, refresh_col = st.columns(2, gap="small")
            with pause_col:
                icon = ICON_RESUME if paused else ICON_PAUSE
                text = Tget(T, "resume_queue", "Resume queue") if paused else Tget(T, "pause_queue", "Pause queue")
                st.markdown("<div class='btn-open btn-ghost'>", unsafe_allow_html=True)
                if st.button(btn_label(icon, text), width="stretch", key="hub_pause_resume"):
                    try:
                        if paused:
                            client.queue_resume()
                        else:
                            client.queue_pause()
                        st.rerun()
                    except Exception as e:
                        soft_note(f"{Tget(T, 'run_err_prefix', 'Error')}: {e}", kind="warn")
                st.markdown("</div>", unsafe_allow_html=True)
            with refresh_col:
                st.markdown("<div class='btn-open btn-ghost'>", unsafe_allow_html=True)
                if st.button(btn_label(ICON_REFRESH, Tget(T, "refresh", "Refresh")), width="stretch", key="hub_queue_refresh"):
                    st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)

            status_state = Tget(T, "status_paused", "Paused") if paused else Tget(T, "status_running", "Running")
            status_class = "paused" if paused else "running"
            status_meta = f"{Tget(T, 'queued', 'Queued')}: {len(queued)}"
            if updated_at:
                status_meta = f"{status_meta} • {Tget(T, 'meta_updated', 'Updated')}: {updated_at}"
            st.markdown(
                f"""
                <div class="status-row status-row-start">
                    <span class="status-label">{E(Tget(T, 'status', 'Status'))}:</span>
                    <span class="status-pill {status_class}">{E(status_state)}</span>
                </div>
                <div class="status-meta status-meta-block">{E(status_meta)}</div>
                """,
                unsafe_allow_html=True,
            )

            if running:
                job_id = str(running.get("job_id") or "")
                video_id = str(running.get("video_id") or "")
                stage = str(running.get("stage") or "") or Tget(T, "working", "Working...")
                progress_val = running.get("progress")
                msg = str(running.get("message") or "")
                st.markdown(
                    f"<div class='queue-running'><div class='queue-running-title'>{E(video_id or Tget(T, 'job', 'Job'))}</div>"
                    f"<div class='queue-running-meta'>{E(job_id)}</div></div>",
                    unsafe_allow_html=True,
                )
                render_progress_bar(progress_val, label=stage, sublabel=msg, animate=progress_val is None)

            if not queued:
                soft_note(Tget(T, "queue_empty", "Queue is empty."), kind="info")
            else:
                st.markdown("<div class='hub-queue-list'>", unsafe_allow_html=True)
                for item in queued[:8]:
                    job_id = str(item.get("job_id") or "")
                    video_id = str(item.get("video_id") or "")
                    state = str(item.get("state") or "")
                    created = fmt_dt(item.get("created_at"))
                    with st.container(border=True):
                        st.markdown(
                            f"<div class='queue-item-head'><div class='queue-item-title'>{E(video_id)}</div>"
                            f"<div class='queue-item-id'>{E(job_id)}</div></div>",
                            unsafe_allow_html=True,
                        )
                        chips = []
                        if state:
                            chips.append(f"<span class='queue-chip'>{E(state)}</span>")
                        if created:
                            chips.append(f"<span class='queue-chip'>{E(created)}</span>")
                        if chips:
                            st.markdown(f"<div class='queue-chip-row'>{''.join(chips)}</div>", unsafe_allow_html=True)
                        q_actions = st.columns(4, gap="small")
                        with q_actions[0]:
                            st.markdown("<div class='btn-open btn-ghost btn-small icon-only'>", unsafe_allow_html=True)
                            if st.button(ICON_TOP, key=f"hub_q_top_{job_id}", help=Tget(T, "move_top", "Top")):
                                try:
                                    client.queue_move(job_id, "top", steps=1)
                                    st.rerun()
                                except Exception as e:
                                    soft_note(f"{Tget(T, 'run_err_prefix', 'Error')}: {e}", kind="warn")
                            st.markdown("</div>", unsafe_allow_html=True)
                        with q_actions[1]:
                            st.markdown("<div class='btn-open btn-ghost btn-small icon-only'>", unsafe_allow_html=True)
                            if st.button(ICON_UP, key=f"hub_q_up_{job_id}", help=Tget(T, "move_up", "Up")):
                                try:
                                    client.queue_move(job_id, "up", steps=1)
                                    st.rerun()
                                except Exception as e:
                                    soft_note(f"{Tget(T, 'run_err_prefix', 'Error')}: {e}", kind="warn")
                            st.markdown("</div>", unsafe_allow_html=True)
                        with q_actions[2]:
                            st.markdown("<div class='btn-open btn-ghost btn-small icon-only'>", unsafe_allow_html=True)
                            if st.button(ICON_DOWN, key=f"hub_q_down_{job_id}", help=Tget(T, "move_down", "Down")):
                                try:
                                    client.queue_move(job_id, "down", steps=1)
                                    st.rerun()
                                except Exception as e:
                                    soft_note(f"{Tget(T, 'run_err_prefix', 'Error')}: {e}", kind="warn")
                            st.markdown("</div>", unsafe_allow_html=True)
                        with q_actions[3]:
                            st.markdown("<div class='btn-del btn-small icon-only'>", unsafe_allow_html=True)
                            if st.button(ICON_DELETE, key=f"hub_q_rm_{job_id}", help=Tget(T, "remove_from_queue", "Remove")):
                                try:
                                    client.queue_remove(job_id)
                                    st.rerun()
                                except Exception as e:
                                    soft_note(f"{Tget(T, 'run_err_prefix', 'Error')}: {e}", kind="warn")
                            st.markdown("</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
                if len(queued) > 8:
                    st.caption(f"{Tget(T, 'show_more', 'Show more')}: +{len(queued) - 8}")

        st.markdown("</div>", unsafe_allow_html=True)


def assistant_tab(client: BackendClient, cfg, ui_text: Dict[str, Dict[str, Any]]) -> None:
    u = UIState()
    u.bind_defaults()
    _ensure_defaults_for_widgets()

    langs = cfg.ui.langs or ["ru", "kz", "en"]
    Ttmp = get_T(ui_text, st.session_state["ui_lang"])
    T = render_page_head(
        Tget(Ttmp, "assistant_page_title", "Assistant"),
        ui_text,
        section_links=None,
        langs=langs,
        default_lang=cfg.ui.default_lang,
    )

    scroll_if_requested(u)
    _render_assistant_panel(T)


def footer(T: dict) -> None:
    st.markdown("<div class='thin-rule'></div>", unsafe_allow_html=True)
    left = Tget(T, "footer_left", "Alen Issayev's Diploma Project") or "Alen Issayev's Diploma Project"
    right = Tget(T, "footer_right", "SmartCampus V2T | User-Friendly Interface") or "SmartCampus V2T | User-Friendly Interface"
    if left.strip() or right.strip():
        st.markdown(
            f"<div class='app-footer'><div>{E(left)}</div><div>{E(right)}</div></div>",
            unsafe_allow_html=True,
        )
