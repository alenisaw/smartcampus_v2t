# app/streamlit_app.py
"""
SmartCampus V2T — Streamlit UI application.

Tabs:
- Home: video carousel + preview + processing + run outputs
- Search by description: search over annotations + player

Notes:
- No sidebar (all controls are in-page)
- Streamlit UI is styled by app/assets/styles.css
- Timeline intervals are clickable: click a segment to seek the video player to that time
"""

from __future__ import annotations

import base64
import copy
import html
import json
import shutil
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import streamlit as st
import streamlit.components.v1 as components

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
ANN_DIR = DATA_DIR / "annotations"
MET_DIR = DATA_DIR / "metrics"
INDEX_DIR = DATA_DIR / "indexes"
THUMB_DIR = DATA_DIR / "thumbs"

APP_DIR = PROJECT_ROOT / "app"
ASSETS_DIR = APP_DIR / "assets"
LOGO_PATH = ASSETS_DIR / "logo.png"
STYLES_PATH = ASSETS_DIR / "styles.css"
UI_TEXT_PATH = ASSETS_DIR / "ui_text.json"
CFG_PATH = PROJECT_ROOT / "config" / "pipeline.yaml"

RAW_DIR.mkdir(parents=True, exist_ok=True)
ANN_DIR.mkdir(parents=True, exist_ok=True)
MET_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)
THUMB_DIR.mkdir(parents=True, exist_ok=True)

from src.search import build_or_update_index, QueryEngine
from src.utils.config import load_pipeline_config
from src.preprocessing.video_io import preprocess_video
from src.pipeline.video_to_text import VideoToTextPipeline
from src.core.types import VideoMeta, FrameInfo, Annotation, RunMetrics

LANGS = ["ru", "kz", "en"]
TAB_IDS = ["home", "search"]


def E(s: Any) -> str:
    return html.escape("" if s is None else str(s), quote=True)


def _get_qp(name: str, default: str) -> str:
    try:
        v = st.query_params.get(name, default)
        return v if isinstance(v, str) else (v[0] if v else default)
    except Exception:
        qp = st.experimental_get_query_params()
        return qp.get(name, [default])[0]


def get_T(ui_text: Dict[str, Dict[str, Any]], lang: str) -> Dict[str, Any]:
    lang = (lang or "ru").strip().lower()
    return ui_text.get(lang) or ui_text["ru"]


def Tget(T: Dict[str, Any], key: str, fallback: str) -> str:
    v = T.get(key)
    return fallback if v is None else str(v)


def _mtime(p: Path) -> float:
    try:
        return float(p.stat().st_mtime)
    except Exception:
        return 0.0


@st.cache_data(show_spinner=False)
def load_ui_text(path_str: str, mtime: float) -> Dict[str, Dict[str, Any]]:
    data = json.loads(Path(path_str).read_text(encoding="utf-8"))
    for lang in LANGS:
        if lang not in data:
            raise ValueError(f"ui_text.json missing language: {lang}")
        tabs = data[lang].get("tabs")
        if not isinstance(tabs, list) or len(tabs) != 2:
            raise ValueError(f"ui_text.json: '{lang}.tabs' must be a list of 2 labels")
    return data


def load_and_apply_css() -> None:
    css = ""
    if STYLES_PATH.exists():
        css = STYLES_PATH.read_text(encoding="utf-8")
    if css.strip():
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


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


def dataclass_to_dict(obj) -> Dict[str, Any]:
    if is_dataclass(obj):
        return asdict(obj)
    if hasattr(obj, "__dict__"):
        return dict(obj.__dict__)
    raise TypeError("Object is not a dataclass or simple class instance")


def soft_note(text: str, kind: str = "info") -> None:
    cls = {"info": "soft-note", "warn": "soft-warn", "ok": "soft-ok"}.get(kind, "soft-note")
    st.markdown(f"<div class='{cls}'>{E(text)}</div>", unsafe_allow_html=True)


def thin_rule() -> None:
    st.markdown("<div class='thin-rule'></div>", unsafe_allow_html=True)


def _img_to_data_uri(p: Path) -> Optional[str]:
    if not p.exists():
        return None
    try:
        b = p.read_bytes()
        return "data:image/png;base64," + base64.b64encode(b).decode("utf-8")
    except Exception:
        return None


def render_header(T: Dict[str, Any], labels: List[str], ids: List[str], current_tab: str) -> None:
    links = []
    for lab, tid in zip(labels, ids):
        cls = "nav-link active" if tid == current_tab else "nav-link"
        links.append(f"<a class='{cls}' href='?tab={E(tid)}' target='_self'>{E(lab)}</a>")

    logo = _img_to_data_uri(LOGO_PATH)
    logo_html = ""
    if logo:
        logo_html = f"<div class='hero-logo'><img src='{logo}' alt='logo' /></div>"

    st.markdown(
        f"""
        <div class="hero">
            {logo_html}
            <div class="hero-title">{E(Tget(T, "app_title", "SmartCampus V2T"))}</div>
            <div class="hero-sub">{E(Tget(T, "app_subtitle", ""))}</div>
            <div class="hero-flow">{E(Tget(T, "app_flow", ""))}</div>
            <div class="nav-wrap">
                <div class="nav-row">
                    {''.join(links)}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_page_head(
    title: str,
    ui_text: Dict[str, Dict[str, Any]],
    section_links: Optional[List[Tuple[str, str]]] = None,
) -> Dict[str, Any]:
    if "ui_lang" not in st.session_state:
        st.session_state.ui_lang = "ru"
    if st.session_state.ui_lang not in LANGS:
        st.session_state.ui_lang = "ru"

    T0 = get_T(ui_text, st.session_state.ui_lang)

    left, right = st.columns([6, 2], gap="large", vertical_alignment="bottom")
    with left:
        st.markdown(
            f"<div class='page-head'>"
            f"<div class='page-title-text'>{E(title)}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
        if section_links:
            parts = []
            for lab, anchor in section_links:
                parts.append(f"<a class='sec-link' href='{E(anchor)}' target='_self'>{E(lab)}</a>")
            st.markdown(f"<div class='section-nav'>{''.join(parts)}</div>", unsafe_allow_html=True)
        st.markdown("<div class='page-rule'></div>", unsafe_allow_html=True)

    with right:
        lab = Tget(T0, "ui_lang", "UI language")
        st.markdown("<div class='ui-lang-row'>", unsafe_allow_html=True)
        st.markdown(f"<div class='ui-lang-label'>{E(lab)}</div>", unsafe_allow_html=True)
        st.selectbox(
            lab,
            options=LANGS,
            index=LANGS.index(st.session_state.ui_lang),
            key="ui_lang",
            label_visibility="collapsed",
        )
        st.markdown("</div>", unsafe_allow_html=True)

    return get_T(ui_text, st.session_state.ui_lang)


@st.cache_data(show_spinner=False, ttl=2)
def list_raw_videos() -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not RAW_DIR.exists():
        return out
    for p in sorted(RAW_DIR.iterdir()):
        if p.is_file() and p.suffix.lower() in {".mp4", ".mov", ".mkv", ".avi"}:
            out[p.stem] = str(p)
    return out


def list_runs_for_video(video_id: str) -> List[str]:
    base = ANN_DIR / video_id
    if not base.exists():
        return []
    runs: List[str] = []
    for p in sorted(base.iterdir()):
        if p.is_dir() and p.name.startswith("run_") and (p / "annotations.json").exists():
            runs.append(p.name)
    return runs


@st.cache_data(show_spinner=False, ttl=2)
def list_all_runs() -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    if not ANN_DIR.exists():
        return out
    for vid_dir in sorted(ANN_DIR.iterdir()):
        if not vid_dir.is_dir():
            continue
        runs = list_runs_for_video(vid_dir.name)
        if runs:
            out[vid_dir.name] = runs
    return out


def allocate_run_id(video_id: str) -> str:
    base = ANN_DIR / video_id
    base.mkdir(parents=True, exist_ok=True)
    existing = [p for p in base.iterdir() if p.is_dir() and p.name.startswith("run_")]
    nums: List[int] = []
    for p in existing:
        suf = p.name.replace("run_", "")
        if suf.isdigit():
            nums.append(int(suf))
    next_id = (max(nums) + 1) if nums else 1
    return f"run_{next_id:03d}"


def annotation_to_dict(a: Any, fallback_index: int, video_id: Optional[str] = None) -> Dict[str, Any]:
    if isinstance(a, dict):
        d = dict(a)
    elif is_dataclass(a):
        d = asdict(a)
    elif hasattr(a, "__dict__"):
        d = dict(a.__dict__)
    else:
        d = {}

    vid = d.get("video_id") or getattr(a, "video_id", None) or video_id

    start = d.get("start_sec")
    if start is None:
        start = getattr(a, "start_sec", None)

    end = d.get("end_sec")
    if end is None:
        end = getattr(a, "end_sec", None)

    desc = d.get("description")
    if desc is None:
        desc = getattr(a, "description", "")

    extra = d.get("extra")
    if extra is None:
        extra = getattr(a, "extra", None)
    if extra is None:
        extra = {}

    clip_index = (
        d.get("clip_index")
        or d.get("clip_id")
        or d.get("index")
        or getattr(a, "clip_index", None)
        or getattr(a, "clip_id", None)
        or fallback_index
    )

    try:
        clip_index_out = int(clip_index)
    except Exception:
        clip_index_out = clip_index

    return {
        "video_id": vid,
        "clip_index": clip_index_out,
        "start_sec": float(start or 0.0),
        "end_sec": float(end or 0.0),
        "description": str(desc),
        "extra": extra,
    }


def save_run_outputs(
    video_id: str,
    run_id: str,
    annotations: List[Annotation],
    metrics: RunMetrics,
    language: str,
) -> Tuple[Path, Path]:
    language = (language or "").strip().lower() or "unknown"

    ann_run_dir = ANN_DIR / video_id / run_id
    met_run_dir = MET_DIR / video_id / run_id
    ann_run_dir.mkdir(parents=True, exist_ok=True)
    met_run_dir.mkdir(parents=True, exist_ok=True)

    ann_dicts = [annotation_to_dict(a, i, video_id=video_id) for i, a in enumerate(annotations)]

    metrics_dict = dataclass_to_dict(metrics)
    if not isinstance(metrics_dict, dict):
        metrics_dict = {}

    extra = metrics_dict.get("extra")
    if not isinstance(extra, dict):
        extra = {}
    extra["language"] = extra.get("language") or language
    metrics_dict["extra"] = extra

    (ann_run_dir / "annotations.json").write_text(
        json.dumps(ann_dicts, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (met_run_dir / "metrics.json").write_text(
        json.dumps(metrics_dict, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    run_meta = {"video_id": video_id, "language": language}
    (ann_run_dir / "run_meta.json").write_text(
        json.dumps(run_meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return ann_run_dir, met_run_dir


@st.cache_data(show_spinner=False)
def _read_run_outputs_cached(
    ann_path_str: str,
    met_path_str: str,
    meta_path_str: str,
    ann_mtime: float,
    met_mtime: float,
    meta_mtime: float,
    video_id: str,
    run_id: str,
) -> Dict[str, Any]:
    ann_path = Path(ann_path_str)
    met_path = Path(met_path_str)
    meta_path = Path(meta_path_str)

    out: Dict[str, Any] = {
        "video_id": video_id,
        "run_id": run_id,
        "annotations": [],
        "metrics": None,
        "global_summary": None,
        "language": None,
    }

    if meta_path.exists():
        try:
            out["language"] = json.loads(meta_path.read_text(encoding="utf-8")).get("language")
        except Exception:
            out["language"] = None

    if ann_path.exists():
        try:
            out["annotations"] = json.loads(ann_path.read_text(encoding="utf-8"))
        except Exception:
            out["annotations"] = []

    if met_path.exists():
        try:
            metrics = json.loads(met_path.read_text(encoding="utf-8"))
            out["metrics"] = metrics
            out["global_summary"] = (metrics.get("extra") or {}).get("global_summary")
            if out["language"] is None:
                out["language"] = (metrics.get("extra") or {}).get("language")
        except Exception:
            out["metrics"] = None

    return out


def read_run_outputs(video_id: str, run_id: str) -> Dict[str, Any]:
    ann_path = ANN_DIR / video_id / run_id / "annotations.json"
    met_path = MET_DIR / video_id / run_id / "metrics.json"
    meta_path = ANN_DIR / video_id / run_id / "run_meta.json"

    return _read_run_outputs_cached(
        ann_path_str=str(ann_path),
        met_path_str=str(met_path),
        meta_path_str=str(meta_path),
        ann_mtime=_mtime(ann_path),
        met_mtime=_mtime(met_path),
        meta_mtime=_mtime(meta_path),
        video_id=video_id,
        run_id=run_id,
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
        return out if out.exists() else None
    except Exception:
        return None


def maybe_playback_warning(path: Path, T: dict) -> None:
    ext = path.suffix.lower()
    if ext in {".avi", ".mkv"}:
        soft_note(Tget(T, "playback_warn", "Playback may be unavailable due to the container or codec."), kind="warn")
        st.caption(Tget(T, "convert_hint", "Recommended format: MP4 (H.264/AAC)."))


@st.cache_data(show_spinner=False)
def make_thumbnail_bytes(video_path_str: str, mtime: float, max_w: int = 520) -> Optional[bytes]:
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
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    ok2, buf = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    if not ok2:
        return None
    return buf.tobytes()


def _thumb_file_for_video(video_path: Path) -> Path:
    return THUMB_DIR / f"{video_path.stem}.jpg"


def get_thumbnail(video_path: Path, max_w: int = 520) -> Optional[bytes]:
    tp = _thumb_file_for_video(video_path)
    v_m = _mtime(video_path)
    t_m = _mtime(tp)
    if tp.exists() and t_m >= v_m and tp.stat().st_size > 0:
        try:
            return tp.read_bytes()
        except Exception:
            pass
    b = make_thumbnail_bytes(str(video_path), v_m, max_w=max_w)
    if b:
        try:
            tp.write_bytes(b)
        except Exception:
            pass
    return b


def build_clips_from_video_meta(
    video_meta: VideoMeta,
    window_sec: float,
    stride_sec: float,
    min_clip_frames: int,
    max_clip_frames: int,
) -> Tuple[List[List[str]], List[Tuple[float, float]]]:
    if not video_meta.frames:
        return [], []

    frames: List[FrameInfo] = sorted(video_meta.frames, key=lambda f: f.timestamp_sec)
    ts = [float(f.timestamp_sec) for f in frames]
    paths = [str(f.path) for f in frames]
    duration = float(video_meta.duration_sec)

    clips: List[List[str]] = []
    clip_timestamps: List[Tuple[float, float]] = []

    if stride_sec <= 0 or window_sec <= 0:
        return clips, clip_timestamps

    n = len(frames)
    l = 0
    r = 0
    t = 0.0

    while t < duration + 1e-6:
        t_end = min(t + window_sec, duration)

        while l < n and ts[l] < t:
            l += 1
        if r < l:
            r = l
        while r < n and ts[r] <= t_end:
            r += 1

        count = r - l
        if count >= min_clip_frames:
            win_paths = paths[l:r]
            if len(win_paths) > max_clip_frames:
                step = len(win_paths) / float(max_clip_frames)
                idxs = [min(len(win_paths) - 1, int(i * step)) for i in range(max_clip_frames)]
                win_paths = [win_paths[i] for i in idxs]
            last_ts = ts[r - 1] if r - 1 >= l else t_end
            clips.append(win_paths)
            clip_timestamps.append((float(t), float(last_ts)))

        t += stride_sec

    return clips, clip_timestamps


@st.cache_resource(show_spinner=False)
def _get_pipeline_cached(device: str, language: str, cfg_path_str: str, cfg_mtime: float):
    cfg0 = load_pipeline_config(Path(cfg_path_str))
    cfg = copy.deepcopy(cfg0)
    cfg.model.device = device
    cfg.model.language = language
    return cfg, VideoToTextPipeline(cfg)


def run_pipeline_on_video(video_path: Path, device: str, language: str) -> Tuple[List[Annotation], RunMetrics]:
    cfg, pipeline = _get_pipeline_cached(
        device=device,
        language=language,
        cfg_path_str=str(CFG_PATH),
        cfg_mtime=_mtime(CFG_PATH),
    )

    video_meta: VideoMeta = preprocess_video(video_path, cfg)
    duration_sec = float(video_meta.duration_sec)
    preprocess_time_sec = float((video_meta.extra or {}).get("preprocess_time_sec", 0.0))

    clips, clip_ts = build_clips_from_video_meta(
        video_meta=video_meta,
        window_sec=cfg.clips.window_sec,
        stride_sec=cfg.clips.stride_sec,
        min_clip_frames=cfg.clips.min_clip_frames,
        max_clip_frames=cfg.clips.max_clip_frames,
    )

    annotations, metrics = pipeline.run(
        video_id=video_meta.video_id,
        video_duration_sec=duration_sec,
        clips=clips,
        clip_timestamps=clip_ts,
        preprocess_time_sec=preprocess_time_sec,
    )

    lang_norm = (language or "").strip().lower() or "unknown"
    try:
        extra = getattr(metrics, "extra", None) or {}
        if not isinstance(extra, dict):
            extra = {}
        extra["language"] = extra.get("language") or lang_norm
        setattr(metrics, "extra", extra)
    except Exception:
        pass

    return annotations, metrics


def _index_version() -> float:
    p1 = INDEX_DIR / "manifest.json"
    p2 = INDEX_DIR / "meta.json"
    return max(_mtime(p1), _mtime(p2), 0.0)


def ensure_index(T: dict) -> bool:
    before = _index_version()
    try:
        build_or_update_index(ann_root=ANN_DIR, index_dir=INDEX_DIR, model_name="intfloat/multilingual-e5-base")
        after = _index_version()
        st.session_state.index_version = after
        if after != before and "qe" in st.session_state:
            del st.session_state["qe"]
        return True
    except RuntimeError as e:
        if "sentence-transformers" in str(e):
            soft_note(Tget(T, "e5_missing", "Missing sentence-transformers"), kind="warn")
        else:
            soft_note(f"{Tget(T, 'index_build_failed', 'Index build failed')}: {e}", kind="warn")
        return False
    except Exception as e:
        soft_note(f"{Tget(T, 'index_build_failed', 'Index build failed')}: {e}", kind="warn")
        return False


def get_engine() -> Optional[QueryEngine]:
    try:
        cur_ver = _index_version()
        if "index_version" not in st.session_state:
            st.session_state.index_version = cur_ver
        if st.session_state.index_version != cur_ver and "qe" in st.session_state:
            del st.session_state["qe"]
            st.session_state.index_version = cur_ver
        if "qe" not in st.session_state:
            st.session_state.qe = QueryEngine(index_dir=INDEX_DIR)
        return st.session_state.qe
    except Exception:
        return None


def _quick_upload_block(T: dict, key_prefix: str) -> None:
    st.markdown(f"<div class='mini-title'>{E(Tget(T, 'quick_upload', 'Quick upload'))}</div>", unsafe_allow_html=True)

    if "uploader_nonce" not in st.session_state:
        st.session_state.uploader_nonce = 0

    uploaded = st.file_uploader(
        Tget(T, "drop_here", "Drop a video file here"),
        type=["mp4", "mov", "mkv", "avi"],
        accept_multiple_files=False,
        key=f"{key_prefix}_uploader_{st.session_state.uploader_nonce}",
    )

    if uploaded is None:
        st.caption(Tget(T, "quick_upload_hint", "Drag & drop or browse a file. Then click “Save”."))
        return

    target_path = RAW_DIR / uploaded.name
    st.caption(f"{Tget(T, 'target', 'Target')}: {target_path}")

    c1, c2 = st.columns([3, 2], gap="large", vertical_alignment="top")
    with c1:
        st.markdown(f"<div class='field-label'>{E(Tget(T, 'model_lang', 'Model output language'))}</div>", unsafe_allow_html=True)
        st.selectbox(
            Tget(T, "model_lang", "Model output language"),
            LANGS,
            index=LANGS.index(st.session_state.pipeline_lang),
            key="pipeline_lang",
            label_visibility="collapsed",
        )
    with c2:
        st.markdown(f"<div class='field-label'>{E(Tget(T, 'device', 'Device'))}</div>", unsafe_allow_html=True)
        st.selectbox(
            Tget(T, "device", "Device"),
            ["cuda", "cpu"],
            index=["cuda", "cpu"].index(st.session_state.device),
            key="device",
            label_visibility="collapsed",
        )

    st.markdown("<div class='btn-open btn-ghost'>", unsafe_allow_html=True)
    save_clicked = st.button(Tget(T, "save_raw", "Save"), width="stretch", key=f"{key_prefix}_save_raw")
    st.markdown("</div>", unsafe_allow_html=True)

    if save_clicked:
        try:
            target_path.write_bytes(uploaded.getbuffer())
            soft_note(Tget(T, "saved", "Saved"), kind="ok")
            st.session_state.uploader_nonce += 1
            st.rerun()
        except Exception as e:
            soft_note(f"{Tget(T, 'run_err_prefix', 'Error')}: {e}", kind="warn")


def _safe_delete_video(video_id: str, raw_videos: Dict[str, Path]) -> None:
    p = raw_videos.get(video_id)
    if p is None:
        for ext in [".mp4", ".mov", ".mkv", ".avi"]:
            cand = RAW_DIR / f"{video_id}{ext}"
            if cand.exists():
                p = cand
                break

    try:
        if p is not None and p.exists():
            p.unlink()
    except Exception:
        pass

    try:
        tp = THUMB_DIR / f"{video_id}.jpg"
        if tp.exists():
            tp.unlink()
    except Exception:
        pass

    try:
        shutil.rmtree(ANN_DIR / video_id, ignore_errors=True)
    except Exception:
        pass
    try:
        shutil.rmtree(MET_DIR / video_id, ignore_errors=True)
    except Exception:
        pass


def _pill_row(items: List[str]) -> None:
    html_pills = "".join([f"<span class='pill'>{E(x)}</span>" for x in items if x and str(x).strip()])
    if html_pills:
        st.markdown(f"<div class='pill-row'>{html_pills}</div>", unsafe_allow_html=True)


def _video_card_actions(video_id: str, T: dict) -> None:
    c1, c2, c3 = st.columns([1, 1, 1], gap="small", vertical_alignment="center")

    with c1:
        st.markdown("<div class='btn-open icon-only btn-small'>", unsafe_allow_html=True)
        open_clicked = st.button("⤢", key=f"open_{video_id}", help=Tget(T, "open", "Open"), width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown("<div class='btn-run icon-only btn-small'>", unsafe_allow_html=True)
        run_clicked = st.button("▶", key=f"run_{video_id}", help=Tget(T, "run_pipeline", "Run"), width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)

    with c3:
        st.markdown("<div class='btn-del icon-only btn-small'>", unsafe_allow_html=True)
        del_clicked = st.button("🗑", key=f"del_{video_id}", help=Tget(T, "delete_video", "Delete"), width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)

    if open_clicked:
        st.session_state.selected_video_id = video_id
        st.rerun()

    if run_clicked:
        st.session_state.selected_video_id = video_id
        st.session_state.run_request_video_id = video_id
        st.rerun()

    if del_clicked:
        st.session_state.confirm_delete_video_id = video_id
        st.rerun()


def _render_delete_confirm(video_id: str, T: dict, raw_videos: Dict[str, Path]) -> None:
    st.markdown("<div class='delete-confirm'>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='delete-confirm-text'>{E(Tget(T, 'delete_confirm', 'Delete this video?'))}</div>",
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns([1, 1], gap="small")
    with c1:
        st.markdown("<div class='btn-open icon-only btn-small'>", unsafe_allow_html=True)
        cancel = st.button("←", key=f"confirm_del_cancel_{video_id}", help=Tget(T, "delete_no", "Cancel"), width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown("<div class='btn-del icon-only btn-small'>", unsafe_allow_html=True)
        ok = st.button("🗑", key=f"confirm_del_ok_{video_id}", help=Tget(T, "delete_yes", "Delete"), width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    if ok:
        _safe_delete_video(video_id, raw_videos)
        st.session_state.confirm_delete_video_id = None
        if st.session_state.get("selected_video_id") == video_id:
            st.session_state.selected_video_id = None
        soft_note(Tget(T, "deleted", "Deleted"), kind="ok")
        st.rerun()

    if cancel:
        st.session_state.confirm_delete_video_id = None
        st.rerun()


def _anchor(anchor_id: str) -> None:
    st.markdown(f"<div id='{E(anchor_id)}' class='anchor'></div>", unsafe_allow_html=True)


def _safe_dom_id(s: str) -> str:
    # only keep characters that are safe for DOM ids
    return "".join(ch for ch in (s or "") if ch.isalnum() or ch in {"_", "-"})


def _request_scroll_to(anchor_id: str) -> None:
    st.session_state.scroll_to_anchor = _safe_dom_id(anchor_id)


def _scroll_if_requested() -> None:
    anchor_id = st.session_state.pop("scroll_to_anchor", None)
    if not anchor_id:
        return

    anchor_id = _safe_dom_id(str(anchor_id))
    if not anchor_id:
        return

    # Smooth scroll to the element (retry a few times until it exists in DOM)
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
            if (tries < 25) {{
              setTimeout(go, 120);
            }}
          }}
          setTimeout(go, 60);
        }})();
        </script>
        """,
        height=0,
        width=0,
    )


def _seek_preview_to(video_id: str, start_sec: float) -> None:
    st.session_state.selected_video_id = video_id
    st.session_state.preview_seek_sec = int(max(0.0, float(start_sec)))
    # ключевое: после rerun переносим к превью
    _request_scroll_to("sec_preview")


def _render_video_player(video_path: Path, start_sec: int) -> None:
    try:
        st.video(str(video_path), start_time=int(max(0, start_sec)))
    except TypeError:
        st.video(str(video_path))
        if start_sec > 0:
            st.caption(f"{int(start_sec)} sec")


def home_section(raw_videos: Dict[str, Path], runs_map: Dict[str, List[str]], ui_text: Dict[str, Dict[str, Any]]) -> None:
    if "pipeline_lang" not in st.session_state:
        st.session_state.pipeline_lang = "ru"
    if "device" not in st.session_state:
        st.session_state.device = "cuda"
    if "carousel_page" not in st.session_state:
        st.session_state.carousel_page = 0
    if "confirm_delete_video_id" not in st.session_state:
        st.session_state.confirm_delete_video_id = None
    if "run_request_video_id" not in st.session_state:
        st.session_state.run_request_video_id = None
    if "preview_seek_sec" not in st.session_state:
        st.session_state.preview_seek_sec = 0

    Ttmp = get_T(ui_text, st.session_state.ui_lang)
    section_links = [
        (Tget(Ttmp, "videos_panel_title", "Videos"), "#sec_videos"),
        (Tget(Ttmp, "preview", "Preview"), "#sec_preview"),
        (Tget(Ttmp, "processing_in_home", "Processing"), "#sec_processing"),
        (Tget(Ttmp, "choose_run", "Select run"), "#sec_runs"),
    ]
    T = render_page_head(Tget(Ttmp, "home_title", "Home"), ui_text, section_links=section_links)

    # если прошлым кликом попросили прокрутку — выполняем
    _scroll_if_requested()

    all_ids = sorted(raw_videos.keys())
    has_videos = len(all_ids) > 0

    if has_videos:
        if "selected_video_id" not in st.session_state or st.session_state.selected_video_id not in all_ids:
            st.session_state.selected_video_id = all_ids[0]

    _anchor("sec_videos")
    with st.container(border=True):
        st.markdown(f"<div class='section-title'>{E(Tget(T, 'videos_panel_title', 'Videos'))}</div>", unsafe_allow_html=True)

        q = st.text_input(
            Tget(T, "search_video_label", "Search by video name"),
            value=st.session_state.get("home_search", ""),
            key="home_search",
        ).strip().lower()

        _quick_upload_block(T, key_prefix="home")
        thin_rule()

        if has_videos:
            filtered = [vid for vid in all_ids if q in vid.lower()] if q else all_ids

            if not filtered:
                soft_note(Tget(T, "no_matches", "No matches found"), kind="info")
            else:
                cols_per_row = 3
                page_size = 9
                total_pages = max(1, (len(filtered) + page_size - 1) // page_size)
                st.session_state.carousel_page = min(max(0, int(st.session_state.carousel_page)), total_pages - 1)

                nav_l, nav_mid, nav_r = st.columns([1, 6, 1], gap="small", vertical_alignment="center")
                with nav_l:
                    st.markdown("<div class='btn-open btn-ghost icon-only btn-small'>", unsafe_allow_html=True)
                    prev = st.button("◀", key="car_prev", help=Tget(T, "prev", "Prev"), width="stretch", disabled=(total_pages <= 1))
                    st.markdown("</div>", unsafe_allow_html=True)

                with nav_mid:
                    st.markdown(
                        f"<div class='carousel-meta'>{E(Tget(T, 'page', 'Page'))}: {st.session_state.carousel_page + 1}/{total_pages}</div>",
                        unsafe_allow_html=True,
                    )

                with nav_r:
                    st.markdown("<div class='btn-open btn-ghost icon-only btn-small'>", unsafe_allow_html=True)
                    nxt = st.button("▶", key="car_next", help=Tget(T, "next", "Next"), width="stretch", disabled=(total_pages <= 1))
                    st.markdown("</div>", unsafe_allow_html=True)

                if prev:
                    st.session_state.carousel_page = (st.session_state.carousel_page - 1) % total_pages
                    st.rerun()
                if nxt:
                    st.session_state.carousel_page = (st.session_state.carousel_page + 1) % total_pages
                    st.rerun()

                start = st.session_state.carousel_page * page_size
                slice_ids = filtered[start : start + page_size]

                for r0 in range(0, len(slice_ids), cols_per_row):
                    row_ids = slice_ids[r0 : r0 + cols_per_row]
                    row_cols = st.columns(cols_per_row, gap="small")

                    for i in range(cols_per_row):
                        if i >= len(row_ids):
                            row_cols[i].empty()
                            continue

                        vid = row_ids[i]
                        path = raw_videos[vid]
                        run_count = len(runs_map.get(vid, []))
                        thumb = get_thumbnail(path)

                        with row_cols[i]:
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
                                    f"<div class='video-card-sub'>{E(path.name)} · {run_count} {E(Tget(T, 'runs_label', 'runs'))}</div>",
                                    unsafe_allow_html=True,
                                )

                                if st.session_state.confirm_delete_video_id == vid:
                                    _render_delete_confirm(vid, T, raw_videos)
                                else:
                                    _video_card_actions(vid, T)
        else:
            soft_note(Tget(T, "no_videos", "No videos available. Please upload a file above."), kind="info")

    _anchor("sec_preview")
    with st.container(border=True):
        st.markdown(f"<div class='section-title'>{E(Tget(T, 'preview', 'Preview'))}</div>", unsafe_allow_html=True)

        if not has_videos:
            soft_note(Tget(T, "no_videos", "No videos available. Please upload a file above."), kind="info")
            return

        selected = st.session_state.selected_video_id
        path = raw_videos[selected]

        st.markdown(
            f"<div class='selected-caption'>{E(Tget(T, 'selected_video', 'Selected video'))}: <b>{E(selected)}</b></div>",
            unsafe_allow_html=True,
        )

        start_seek = int(st.session_state.get("preview_seek_sec", 0) or 0)


        maybe_playback_warning(path, T)
        _render_video_player(path, start_seek)

        if path.suffix.lower() in {".avi", ".mkv"}:
            c1, c2 = st.columns([1.2, 2.2], gap="medium")
            with c1:
                st.markdown("<div class='btn-open btn-ghost'>", unsafe_allow_html=True)
                if st.button(Tget(T, "convert", "Convert to MP4"), key=f"convert_preview_{selected}", width="stretch"):
                    if not ffmpeg_available():
                        soft_note(Tget(T, "ffmpeg_missing", "ffmpeg not found"), kind="warn")
                    else:
                        with st.spinner(Tget(T, "converting", "Conversion in progress...")):
                            out_mp4 = convert_to_mp4(path)
                            if out_mp4 is None:
                                soft_note(Tget(T, "conversion_failed", "Conversion failed"), kind="warn")
                            else:
                                soft_note(f"{Tget(T, 'conversion_done', 'Done')}: {out_mp4.name}", kind="ok")
                                st.session_state.selected_video_id = out_mp4.stem
                                st.session_state.preview_seek_sec = 0
                                st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)
            with c2:
                st.caption(Tget(T, "convert_hint", "Recommended format: MP4 (H.264/AAC)."))

    _anchor("sec_processing")
    with st.container(border=True):
        st.markdown(f"<div class='section-title'>{E(Tget(T, 'processing_in_home', 'Processing'))}</div>", unsafe_allow_html=True)

        if not has_videos:
            soft_note(Tget(T, "no_videos", "No videos available. Please upload a file above."), kind="info")
            return

        c1, c2 = st.columns([1, 1], gap="medium")
        with c1:
            st.markdown(f"<div class='field-label'>{E(Tget(T, 'model_lang', 'Model output language'))}</div>", unsafe_allow_html=True)
            st.selectbox(
                Tget(T, "model_lang", "Model output language"),
                LANGS,
                index=LANGS.index(st.session_state.pipeline_lang),
                key="pipeline_lang",
                label_visibility="collapsed",
            )
        with c2:
            st.markdown(f"<div class='field-label'>{E(Tget(T, 'device', 'Device'))}</div>", unsafe_allow_html=True)
            st.selectbox(
                Tget(T, "device", "Device"),
                ["cuda", "cpu"],
                index=["cuda", "cpu"].index(st.session_state.device),
                key="device",
                label_visibility="collapsed",
            )

        st.markdown("<div class='btn-run btn-main icon-only icon-bright'>", unsafe_allow_html=True)
        run_clicked = st.button(
            "▶",
            key=f"run_main_{st.session_state.selected_video_id}",
            help=Tget(T, "run_pipeline", "Run"),
            width="stretch",
        )
        st.markdown("</div>", unsafe_allow_html=True)

        should_run = bool(run_clicked) or (st.session_state.run_request_video_id == st.session_state.selected_video_id)
        if should_run:
            st.session_state.run_request_video_id = None

            run_id = allocate_run_id(st.session_state.selected_video_id)
            msg = st.empty()
            progress = st.progress(0, text=Tget(T, "run_stage_init", "Loading..."))

            try:
                msg.info(Tget(T, "run_stage_init", "Loading..."))
                progress.progress(6, text=Tget(T, "run_stage_init", "Loading..."))

                msg.info(Tget(T, "run_stage_preprocess", "Preprocessing..."))
                progress.progress(22, text=Tget(T, "run_stage_preprocess", "Preprocessing..."))

                msg.info(Tget(T, "run_stage_infer", "Inference..."))
                progress.progress(52, text=Tget(T, "run_stage_infer", "Inference..."))

                annotations, metrics = run_pipeline_on_video(
                    video_path=raw_videos[st.session_state.selected_video_id],
                    device=st.session_state.device,
                    language=st.session_state.pipeline_lang,
                )

                msg.info(Tget(T, "run_stage_save", "Saving..."))
                progress.progress(78, text=Tget(T, "run_stage_save", "Saving..."))

                save_run_outputs(
                    video_id=st.session_state.selected_video_id,
                    run_id=run_id,
                    annotations=annotations,
                    metrics=metrics,
                    language=st.session_state.pipeline_lang,
                )

                msg.info(Tget(T, "run_stage_index", "Updating the search index..."))
                progress.progress(92, text=Tget(T, "run_stage_index", "Updating the search index..."))
                ok = ensure_index(T)

                progress.progress(100, text=Tget(T, "run_done", "Done ✅"))
                msg.success(Tget(T, "run_done", "Done ✅") if ok else Tget(T, "run_done_no_index", "Done, but index not updated"))

                st.session_state[f"selected_run_{st.session_state.selected_video_id}"] = run_id
                try:
                    st.toast(f"{Tget(T,'run_done','Done ✅')} · {run_id}")
                except Exception:
                    pass

                st.rerun()

            except Exception as e:
                msg.error(f"{Tget(T, 'run_err_prefix', 'Error')}: {e}")
                progress.empty()
                soft_note(str(e), kind="warn")

    _anchor("sec_runs")
    with st.container(border=True):
        st.markdown(f"<div class='section-title'>{E(Tget(T, 'choose_run', 'Select run'))}</div>", unsafe_allow_html=True)

        if not has_videos:
            soft_note(Tget(T, "no_videos", "No videos available. Please upload a file above."), kind="info")
            return

        selected = st.session_state.selected_video_id
        runs = runs_map.get(selected, [])
        if not runs:
            soft_note(Tget(T, "no_runs_home", "No runs available yet. Please start processing above."), kind="info")
            return

        key_run_state = f"selected_run_{selected}"
        if key_run_state not in st.session_state or st.session_state[key_run_state] not in runs:
            st.session_state[key_run_state] = runs[-1]

        sel_run = st.selectbox(
            Tget(T, "run_filter", "Run"),
            runs,
            index=runs.index(st.session_state[key_run_state]),
            key=f"run_select_{selected}",
        )
        st.session_state[key_run_state] = sel_run

        out = read_run_outputs(selected, sel_run)
        run_lang = (out.get("language") or "unknown").strip() or "unknown"

        m = out.get("metrics") or {}
        preprocess_sec = float(m.get("preprocess_time_sec", 0.0) or 0.0)
        model_sec = float(m.get("model_time_sec", 0.0) or 0.0)
        total_sec = float(m.get("total_time_sec", 0.0) or 0.0)

        _pill_row(
            [
                f"{Tget(T,'run_lang','Run language')}: {run_lang.upper()}",
                f"{Tget(T,'metrics_preprocess','Preprocess')}: {hms(preprocess_sec)}",
                f"{Tget(T,'metrics_model','Inference')}: {hms(model_sec)}",
                f"{Tget(T,'metrics_total','Total')}: {hms(total_sec)}",
            ]
        )

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
                    start = float(a.get("start_sec", 0.0) or 0.0)
                    end = float(a.get("end_sec", 0.0) or 0.0)
                    desc = str(a.get("description", "") or "")
                    b1, b2 = st.columns([1.1, 6], gap="small", vertical_alignment="center")
                    with b1:
                        st.markdown("<div class='btn-open btn-ghost btn-small'>", unsafe_allow_html=True)
                        if st.button(f"{mmss(start)}–{mmss(end)}", key=f"seg_seek_{selected}_{sel_run}_{i}", width="stretch"):
                            # ставим время + запрашиваем скролл к превью
                            _seek_preview_to(selected, start)
                            st.rerun()
                        st.markdown("</div>", unsafe_allow_html=True)
                    with b2:
                        st.write(desc)


def search_section(raw_videos: Dict[str, Path], runs_map: Dict[str, List[str]], ui_text: Dict[str, Dict[str, Any]]) -> None:
    Ttmp = get_T(ui_text, st.session_state.ui_lang)
    section_links = [
        (Tget(Ttmp, "search", "Search"), "#sec_search"),
        (Tget(Ttmp, "results", "Results"), "#sec_results"),
        (Tget(Ttmp, "player", "Player"), "#sec_player"),
    ]
    T = render_page_head(Tget(Ttmp, "search_desc_title", "Search by event"), ui_text, section_links=section_links)

    qe = get_engine()
    if qe is None:
        soft_note(Tget(T, "index_missing", "Search index not found."), kind="warn")
        return

    _anchor("sec_search")
    with st.container(border=True):
        st.markdown(f"<div class='section-title'>{E(Tget(T, 'search', 'Search'))}</div>", unsafe_allow_html=True)

        q1, q2 = st.columns([3, 1], gap="medium")
        with q1:
            query = st.text_input(
                Tget(T, "query", "Query"),
                placeholder=Tget(T, "query_ph", "crowd running"),
                key="search_query",
            )
        with q2:
            top_k = st.number_input(
                Tget(T, "topk", "Top-K"),
                min_value=1,
                max_value=50,
                value=10,
                step=1,
                key="search_topk",
            )

        f1, f2 = st.columns([1, 1], gap="medium")
        with f1:
            video_options = ["(all)"] + sorted(runs_map.keys())
            sel_video = st.selectbox(Tget(T, "video_filter", "Video"), video_options, index=0, key="search_video_filter")
        with f2:
            run_options = ["(all)"]
            if sel_video != "(all)":
                run_options += runs_map.get(sel_video, [])
            else:
                all_runs = sorted({r for rs in runs_map.values() for r in rs})
                run_options += all_runs
            sel_run = st.selectbox(Tget(T, "run_filter", "Run"), run_options, index=0, key="search_run_filter")

        thin_rule()

        st.markdown("<div class='btn-open btn-ghost'>", unsafe_allow_html=True)
        upd = st.button(Tget(T, "update_index", "Update search index"), width="stretch", key="update_index_search")
        st.markdown("</div>", unsafe_allow_html=True)

        if upd:
            with st.spinner(Tget(T, "building_index", "Updating the search index...")):
                ok = ensure_index(T)
            soft_note(Tget(T, "ok", "Completed") if ok else Tget(T, "index_build_failed", "Failed to update the search index"), kind=("ok" if ok else "warn"))

    if not query or not query.strip():
        soft_note(Tget(T, "type_query", "Please enter a search query."), kind="info")
        return

    try:
        hits = qe.search(
            query=query.strip(),
            top_k=int(top_k),
            video_id=None if sel_video == "(all)" else sel_video,
            run_id=None if sel_run == "(all)" else sel_run,
            dedupe=True,
        )
    except Exception as e:
        soft_note(f"{Tget(T, 'search_error', 'Search execution error')}: {e}", kind="warn")
        return

    _anchor("sec_results")
    _anchor("sec_player")
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
                            if st.button(f"{mmss(h.start_sec)}–{mmss(h.end_sec)}", key=f"hit_seek_{idx}", width="stretch"):
                                st.session_state.selected_hit = {
                                    "video_id": h.video_id,
                                    "run_id": h.run_id,
                                    "start_sec": float(h.start_sec),
                                    "end_sec": float(h.end_sec),
                                    "description": h.description,
                                }
                            st.markdown("</div>", unsafe_allow_html=True)
                        with b2:
                            st.write(h.description)
                            st.caption(
                                f"{h.video_id} · {h.run_id} · score={h.score:.3f} "
                                f"(bm25={h.sparse_score:.3f}, dense={h.dense_score:.3f})"
                            )

    with right:
        with st.container(border=True):
            st.markdown(f"<div class='section-title'>{E(Tget(T, 'player', 'Player'))}</div>", unsafe_allow_html=True)
            hit = st.session_state.get("selected_hit")
            if not hit:
                soft_note(Tget(T, "pick_result", "Select a result and click “Open”."), kind="info")
            else:
                vid = str(hit["video_id"])
                run_id = str(hit.get("run_id", ""))
                start = float(hit.get("start_sec", 0.0) or 0.0)
                end = float(hit.get("end_sec", 0.0) or 0.0)
                desc = str(hit.get("description", "") or "")

                st.write(f"**{vid}** · {run_id}")
                st.write(f"[{mmss(start)}–{mmss(end)}] {desc}")

                video_path = raw_videos.get(vid)
                if not video_path or not video_path.exists():
                    soft_note(f"{Tget(T, 'raw_missing', 'Source video not found')}: '{vid}' → {RAW_DIR}", kind="warn")
                else:
                    maybe_playback_warning(video_path, T)
                    _render_video_player(video_path, int(start))


def footer(T: dict) -> None:
    st.markdown("<div class='thin-rule'></div>", unsafe_allow_html=True)
    st.caption(Tget(T, "footer", ""))


def main() -> None:
    st.set_page_config(page_title="SmartCampus V2T", layout="wide")
    load_and_apply_css()

    try:
        ui_text = load_ui_text(str(UI_TEXT_PATH), _mtime(UI_TEXT_PATH))
    except Exception as e:
        st.error(str(e))
        st.stop()

    if "selected_hit" not in st.session_state:
        st.session_state.selected_hit = None
    if "ui_lang" not in st.session_state:
        st.session_state.ui_lang = "ru"
    if "preview_seek_sec" not in st.session_state:
        st.session_state.preview_seek_sec = 0

    T = get_T(ui_text, st.session_state.ui_lang)

    labels = T.get("tabs") or ["Home", "Search"]
    current_tab = _get_qp("tab", TAB_IDS[0])
    if current_tab not in TAB_IDS:
        current_tab = TAB_IDS[0]

    render_header(T, labels, TAB_IDS, current_tab)

    raw_videos_str = list_raw_videos()
    raw_videos: Dict[str, Path] = {k: Path(v) for k, v in raw_videos_str.items()}
    runs_map = list_all_runs()

    if current_tab == "home":
        home_section(raw_videos, runs_map, ui_text)
    elif current_tab == "search":
        search_section(raw_videos, runs_map, ui_text)

    footer(get_T(ui_text, st.session_state.ui_lang))


if __name__ == "__main__":
    main()
