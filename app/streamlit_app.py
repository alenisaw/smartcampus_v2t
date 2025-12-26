# app/streamlit_app.py
"""
SmartCampus V2T — Streamlit UI application.

Tabs:
- Home: video list + preview + run selection + quick upload (drag&drop)
- Processing: run pipeline
- Search by description: search over annotations

UI:
- Header with centered title + navigation via ?tab=
- UI language switch (ru/kz/en) in sidebar
- Sidebar: runtime + index tools
- UI text: app/assets/ui_text.json
- CSS: app/assets/styles.css (mtime-based cache busting)
"""

from __future__ import annotations

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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
ANN_DIR = DATA_DIR / "annotations"
MET_DIR = DATA_DIR / "metrics"
INDEX_DIR = DATA_DIR / "indexes"

APP_DIR = PROJECT_ROOT / "app"
ASSETS_DIR = APP_DIR / "assets"
LOGO_PATH = ASSETS_DIR / "logo.png"
STYLES_PATH = ASSETS_DIR / "styles.css"
UI_TEXT_PATH = ASSETS_DIR / "ui_text.json"

RAW_DIR.mkdir(parents=True, exist_ok=True)
ANN_DIR.mkdir(parents=True, exist_ok=True)
MET_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)

from src.search import build_or_update_index, QueryEngine
from src.utils.config import load_pipeline_config
from src.preprocessing.video_io import preprocess_video
from src.pipeline.video_to_text import VideoToTextPipeline
from src.core.types import VideoMeta, FrameInfo, Annotation, RunMetrics

LANGS = ["ru", "kz", "en"]

# 3 tabs (as you wanted)
TAB_IDS = ["home", "process", "search"]


def E(s: Any) -> str:
    return html.escape("" if s is None else str(s), quote=True)


def _mtime(p: Path) -> float:
    try:
        return p.stat().st_mtime
    except Exception:
        return 0.0


@st.cache_data(show_spinner=False)
def load_ui_text(path_str: str, mtime: float) -> Dict[str, Dict[str, Any]]:
    data = json.loads(Path(path_str).read_text(encoding="utf-8"))
    for lang in LANGS:
        if lang not in data:
            raise ValueError(f"ui_text.json missing language: {lang}")
        tabs = data[lang].get("tabs")
        if not isinstance(tabs, list) or len(tabs) != 3:
            raise ValueError(f"ui_text.json: '{lang}.tabs' must be a list of 3 labels")
    return data


@st.cache_data(show_spinner=False)
def load_css(path_str: str, mtime: float) -> str:
    p = Path(path_str)
    if not p.exists():
        return ""
    return p.read_text(encoding="utf-8")


def inject_css() -> None:
    css = load_css(str(STYLES_PATH), _mtime(STYLES_PATH))
    if not css.strip():
        css = """
        .block-container { padding-top: 2.2rem; padding-bottom: 2.5rem; }
        section[data-testid="stSidebar"] { min-width: 320px; max-width: 320px; }
        section[data-testid="stSidebar"] > div { overflow: auto; }
        """
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


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


def render_page_head(title: str, right_html: Optional[str] = None) -> None:
    right = right_html or ""
    st.markdown(
        f"""
        <div class="page-head">
          <div class="page-head-row">
            <h1 class="page-title">{E(title)}</h1>
            <div class="page-head-right">{right}</div>
          </div>
          <div class="page-rule"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_header(T: Dict[str, Any], labels: List[str], ids: List[str], current_tab: str) -> None:
    links = []
    for lab, tid in zip(labels, ids):
        cls = "nav-link active" if tid == current_tab else "nav-link"
        links.append(f"<a class='{cls}' href='?tab={E(tid)}' target='_self'>{E(lab)}</a>")

    st.markdown(
        f"""
        <div class="hero">
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


def list_raw_videos() -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for p in sorted(RAW_DIR.iterdir()):
        if p.is_file() and p.suffix.lower() in {".mp4", ".mov", ".mkv", ".avi"}:
            out[p.stem] = p
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
    """
    Guarantees language persistence:
    - data/annotations/<video_id>/<run_id>/run_meta.json  -> language
    - data/metrics/<video_id>/<run_id>/metrics.json      -> extra.language
    """
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


def read_run_outputs(video_id: str, run_id: str) -> Dict[str, Any]:
    ann_path = ANN_DIR / video_id / run_id / "annotations.json"
    met_path = MET_DIR / video_id / run_id / "metrics.json"
    meta_path = ANN_DIR / video_id / run_id / "run_meta.json"

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
        "ffmpeg", "-y", "-i", str(src),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k",
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
        soft_note(Tget(T, "playback_warn", "Playback may fail due to container/codec."), kind="warn")
        st.caption(Tget(T, "convert_hint", "Recommended: MP4 (H.264/AAC)."))


@st.cache_data(show_spinner=False)
def make_thumbnail_bytes(video_path_str: str, max_w: int = 640) -> Optional[bytes]:
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
    ok2, buf = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok2:
        return None
    return buf.tobytes()


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
    duration = float(video_meta.duration_sec)

    clips: List[List[str]] = []
    clip_timestamps: List[Tuple[float, float]] = []
    t = 0.0
    while t < duration:
        t_end = min(t + window_sec, duration)
        window_frames = [f for f in frames if t <= f.timestamp_sec <= t_end]
        if len(window_frames) >= min_clip_frames:
            paths = [str(f.path) for f in window_frames]
            if len(paths) > max_clip_frames:
                step = len(paths) / max_clip_frames
                indices = [int(i * step) for i in range(max_clip_frames)]
                paths = [paths[i] for i in indices]
            last_ts = float(window_frames[-1].timestamp_sec)
            clips.append(paths)
            clip_timestamps.append((float(t), last_ts))
        t += stride_sec
        if stride_sec <= 0:
            break
    return clips, clip_timestamps


def run_pipeline_on_video(video_path: Path, device: str, language: str) -> Tuple[List[Annotation], RunMetrics]:
    cfg = load_pipeline_config(PROJECT_ROOT / "config" / "pipeline.yaml")
    cfg.model.device = device
    cfg.model.language = language

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

    pipeline = VideoToTextPipeline(cfg)
    annotations, metrics = pipeline.run(
        video_id=video_meta.video_id,
        video_duration_sec=duration_sec,
        clips=clips,
        clip_timestamps=clip_ts,
        preprocess_time_sec=preprocess_time_sec,
    )

    # Ensure language stored in metrics object (if field exists)
    lang_norm = (language or "").strip().lower() or "unknown"
    try:
        setattr(metrics, "language", getattr(metrics, "language", None) or lang_norm)
    except Exception:
        pass
    try:
        extra = getattr(metrics, "extra", None) or {}
        if not isinstance(extra, dict):
            extra = {}
        extra["language"] = extra.get("language") or lang_norm
        setattr(metrics, "extra", extra)
    except Exception:
        pass

    return annotations, metrics


def ensure_index(T: dict) -> bool:
    try:
        build_or_update_index(ann_root=ANN_DIR, index_dir=INDEX_DIR, model_name="intfloat/multilingual-e5-base")
        if "qe" in st.session_state:
            del st.session_state["qe"]
        soft_note(Tget(T, "ok", "OK"), kind="ok")
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
        if "qe" not in st.session_state:
            st.session_state.qe = QueryEngine(index_dir=INDEX_DIR)
        return st.session_state.qe
    except Exception:
        return None


def sidebar_runtime_and_index(ui_text: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    if "ui_lang" not in st.session_state:
        st.session_state.ui_lang = "ru"
    if "pipeline_lang" not in st.session_state:
        st.session_state.pipeline_lang = "ru"
    if "device" not in st.session_state:
        st.session_state.device = "cuda"

    if LOGO_PATH.exists():
        st.sidebar.image(str(LOGO_PATH), use_container_width=True)

    st.sidebar.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    T0 = get_T(ui_text, st.session_state.ui_lang)
    st.sidebar.selectbox(
        Tget(T0, "ui_lang", "UI language"),
        options=LANGS,
        index=LANGS.index(st.session_state.ui_lang),
        key="ui_lang",
    )

    T = get_T(ui_text, st.session_state.ui_lang)

    st.sidebar.divider()

    st.sidebar.subheader(Tget(T, "runtime", "Runtime"))
    st.sidebar.selectbox(
        Tget(T, "model_lang", "Model output language"),
        LANGS,
        index=LANGS.index(st.session_state.pipeline_lang),
        key="pipeline_lang",
    )
    st.sidebar.selectbox(
        Tget(T, "device", "Device"),
        ["cuda", "cpu"],
        index=["cuda", "cpu"].index(st.session_state.device),
        key="device",
    )

    st.sidebar.divider()

    st.sidebar.subheader(Tget(T, "index", "Index"))
    st.sidebar.caption(Tget(T, "index_hint", "Index is required for search."))
    if st.sidebar.button(Tget(T, "update_index", "Update index"), use_container_width=True, key="update_index_sb"):
        with st.spinner(Tget(T, "building_index", "Building index...")):
            ensure_index(T)

    st.sidebar.divider()
    with st.sidebar.expander(Tget(T, "debug", "Debug")):
        st.caption(f"raw={RAW_DIR}")
        st.caption(f"annotations={ANN_DIR}")
        st.caption(f"metrics={MET_DIR}")
        st.caption(f"indexes={INDEX_DIR}")

    return T


def _quick_upload_block(T: dict, compact: bool = False) -> None:
    # compact=True -> smaller title, better for embedding inside other panels
    if compact:
        st.markdown(f"<div class='mini-title'>{E(Tget(T, 'quick_upload', 'Quick upload'))}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"### {Tget(T, 'quick_upload', 'Quick upload')}")

    uploaded = st.file_uploader(
        Tget(T, "drop_here", "Drop a video file here"),
        type=["mp4", "mov", "mkv", "avi"],
        accept_multiple_files=False,
        key="home_uploader",
    )

    if uploaded is None:
        st.caption(Tget(T, "quick_upload_hint", "Drag & drop or browse → Save."))
        return

    target_path = RAW_DIR / uploaded.name
    st.caption(f"{Tget(T, 'target', 'Target')}: {target_path}")

    c1, c2, c3 = st.columns([1, 1, 2], gap="medium")
    with c1:
        if st.button(Tget(T, "save_raw", "Save"), use_container_width=True, key="home_save_raw_btn"):
            target_path.write_bytes(uploaded.getbuffer())
            soft_note(f"{Tget(T, 'saved', 'Saved')}: {target_path.name}", kind="ok")
            st.session_state.selected_video_id = target_path.stem
            st.rerun()

    with c2:
        if target_path.suffix.lower() in {".avi", ".mkv"}:
            if st.button(Tget(T, "convert", "Convert to MP4"), use_container_width=True, key="home_convert_btn"):
                if not ffmpeg_available():
                    soft_note(Tget(T, "ffmpeg_missing", "ffmpeg not found"), kind="warn")
                else:
                    tmp = RAW_DIR / uploaded.name
                    if not tmp.exists():
                        tmp.write_bytes(uploaded.getbuffer())
                    with st.spinner(Tget(T, "converting", "Converting...")):
                        out_mp4 = convert_to_mp4(tmp)
                        if out_mp4 is None:
                            soft_note(Tget(T, "conversion_failed", "Conversion failed"), kind="warn")
                        else:
                            soft_note(f"{Tget(T, 'conversion_done', 'Conversion done')}: {out_mp4.name}", kind="ok")
                            st.session_state.selected_video_id = out_mp4.stem
                            st.rerun()

    with c3:
        st.caption(Tget(T, "convert_hint", "Recommended: MP4 (H.264/AAC)."))
        if target_path.suffix.lower() in {".avi", ".mkv"}:
            soft_note(Tget(T, "playback_warn", "Playback may fail due to container/codec."), kind="warn")


def home_section(raw_videos: Dict[str, Path], T: dict) -> None:
    render_page_head(Tget(T, "home_title", "Home"))

    all_ids = sorted(raw_videos.keys())
    has_videos = len(all_ids) > 0

    # init selection if possible
    if has_videos:
        if "selected_video_id" not in st.session_state or st.session_state.selected_video_id not in all_ids:
            st.session_state.selected_video_id = all_ids[0]

    left, right = st.columns([3, 2], gap="large", vertical_alignment="top")

    # RIGHT: single panel "Видео" that contains search + upload + list
    with right:
        st.markdown(f"## {Tget(T, 'videos_panel_title', 'Videos')}")
        with st.container(border=True):
            q = st.text_input(
                Tget(T, "search_video_label", "Search video by name"),
                value=st.session_state.get("home_search", ""),
                key="home_search",
            ).strip().lower()

            _quick_upload_block(T, compact=True)
            st.divider()

            if not has_videos:
                soft_note(Tget(T, "no_videos", "No videos yet. Upload one above."), kind="info")
            else:
                filtered = [vid for vid in all_ids if q in vid.lower()] if q else all_ids
                if not filtered:
                    soft_note(Tget(T, "no_matches", "No matches"), kind="info")
                else:
                    with st.container(height=520):
                        for vid in filtered:
                            path = raw_videos[vid]
                            run_count = len(list_runs_for_video(vid))
                            thumb = make_thumbnail_bytes(str(path))

                            with st.container(border=True):
                                c1, c2 = st.columns([1, 3], vertical_alignment="center", gap="medium")
                                with c1:
                                    if thumb:
                                        st.image(thumb, use_container_width=True)
                                    else:
                                        st.caption(Tget(T, "no_thumbnail", "No thumbnail"))
                                with c2:
                                    st.markdown(f"**{vid}**")
                                    runs_word = Tget(T, "runs_label", "runs")
                                    st.caption(f"{path.name} · {run_count} {runs_word}")
                                    if st.button(Tget(T, "open", "Open"), key=f"open_right_{vid}", use_container_width=True):
                                        st.session_state.selected_video_id = vid
                                        st.rerun()

    # LEFT: Preview first, then selected video title (swapped)
    with left:
        if not has_videos:
            st.markdown(f"## {Tget(T, 'preview', 'Preview')}")
            soft_note(Tget(T, "no_videos", "No videos yet. Upload one above."), kind="info")
            return

        selected = st.session_state.selected_video_id
        path = raw_videos[selected]

        st.markdown(f"## {Tget(T, 'preview', 'Preview')}")
        st.markdown(
            f"<div class='selected-caption'>"
            f"{E(Tget(T, 'selected_video', 'Selected video'))}: <b>{E(selected)}</b>"
            f"</div>",
            unsafe_allow_html=True,
        )

        with st.container(border=True):
            maybe_playback_warning(path, T)
            st.video(str(path))

        # AVI/MKV convert button below preview (kept)
        ext = path.suffix.lower()
        if ext in {".avi", ".mkv"}:
            c1, c2 = st.columns([1, 1], gap="medium")
            with c1:
                if st.button(Tget(T, "convert", "Convert to MP4"), key=f"convert_preview_{selected}", use_container_width=True):
                    if not ffmpeg_available():
                        soft_note(Tget(T, "ffmpeg_missing", "ffmpeg not found"), kind="warn")
                    else:
                        with st.spinner(Tget(T, "converting", "Converting...")):
                            out_mp4 = convert_to_mp4(path)
                            if out_mp4 is None:
                                soft_note(Tget(T, "conversion_failed", "Conversion failed"), kind="warn")
                            else:
                                soft_note(f"{Tget(T, 'conversion_done', 'Conversion done')}: {out_mp4.name}", kind="ok")
                                st.session_state.selected_video_id = out_mp4.stem
                                st.rerun()
            with c2:
                soft_note(Tget(T, "convert_hint", "Recommended: MP4 (H.264/AAC)."), kind="info")

        runs = list_runs_for_video(selected)
        if not runs:
            soft_note(Tget(T, "no_runs", "No runs for this video yet. Go to Processing."), kind="info")
            return

        st.markdown(f"### {Tget(T, 'choose_run', 'Select run')}")
        key_run = f"selected_run_{selected}"
        if key_run not in st.session_state or st.session_state[key_run] not in runs:
            st.session_state[key_run] = runs[-1]

        sel_run = st.selectbox(
            Tget(T, "run_filter", "Run"),
            runs,
            index=runs.index(st.session_state[key_run]),
            key=f"run_select_{selected}",
        )
        st.session_state[key_run] = sel_run

        out = read_run_outputs(selected, sel_run)
        run_lang = (out.get("language") or "unknown").strip() or "unknown"
        st.markdown(
            f"<span class='pill'>{E(Tget(T, 'run_lang', 'Run language'))}: {E(run_lang.upper())}</span>",
            unsafe_allow_html=True,
        )

        with st.container(border=True):
            st.markdown(f"### {Tget(T, 'global_summary_box', 'Video summary')}")
            if out.get("global_summary"):
                st.write(out["global_summary"])
            else:
                st.caption(Tget(T, "no_global_summary", "No global summary."))

        with st.container(border=True):
            st.markdown(f"### {Tget(T, 'segments_timeline', 'Timeline')}")
            anns = out.get("annotations") or []
            if not anns:
                soft_note(Tget(T, "no_annotations", "No annotations."), kind="info")
            else:
                for a in anns:
                    st.write(f"[{mmss(a['start_sec'])} - {mmss(a['end_sec'])}] {a['description']}")


def processing_section(raw_videos: Dict[str, Path], T: dict) -> None:
    render_page_head(Tget(T, "processing_title", "Processing"))

    if not raw_videos:
        soft_note(Tget(T, "no_videos", "No videos yet. Upload one on Home."), kind="info")
        return

    ids = sorted(raw_videos.keys())
    if "selected_video_id" not in st.session_state or st.session_state.selected_video_id not in ids:
        st.session_state.selected_video_id = ids[0]

    with st.container(border=True):
        c1, c2 = st.columns([4, 2], vertical_alignment="center", gap="large")

        with c1:
            vid = st.selectbox(
                Tget(T, "video_filter", "Video"),
                ids,
                index=ids.index(st.session_state.selected_video_id),
                key="process_video_select",
            )
            st.session_state.selected_video_id = vid

        with c2:
            st.markdown(
                f"""
                <div class="kpi-row right">
                  <div class="kpi">{E(Tget(T, 'model_lang', 'Model lang'))}: <b>{E(st.session_state.pipeline_lang.upper())}</b></div>
                  <div class="kpi">{E(Tget(T, 'device', 'Device'))}: <b>{E(st.session_state.device.upper())}</b></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        run_clicked = st.button(
            Tget(T, "run_pipeline", "RUN"),
            type="primary",
            use_container_width=True,
            key="process_run_btn",
        )

        if run_clicked:
            video_path = raw_videos[vid]
            run_id = allocate_run_id(video_path.stem)

            msg = st.empty()
            progress = st.progress(0, text=Tget(T, "run_stage_init", "Loading..."))

            try:
                msg.info(Tget(T, "run_stage_init", "Loading..."))
                progress.progress(5, text=Tget(T, "run_stage_init", "Loading..."))

                msg.info(Tget(T, "run_stage_preprocess", "Preprocessing..."))
                progress.progress(20, text=Tget(T, "run_stage_preprocess", "Preprocessing..."))

                msg.info(Tget(T, "run_stage_infer", "Inference..."))
                progress.progress(45, text=Tget(T, "run_stage_infer", "Inference..."))

                annotations, metrics = run_pipeline_on_video(
                    video_path=video_path,
                    device=st.session_state.device,
                    language=st.session_state.pipeline_lang,
                )

                msg.info(Tget(T, "run_stage_save", "Saving..."))
                progress.progress(75, text=Tget(T, "run_stage_save", "Saving..."))

                save_run_outputs(
                    video_id=video_path.stem,
                    run_id=run_id,
                    annotations=annotations,
                    metrics=metrics,
                    language=st.session_state.pipeline_lang,
                )
                st.session_state["last_run"] = {"video_id": video_path.stem, "run_id": run_id}

                msg.info(Tget(T, "run_stage_index", "Updating index..."))
                progress.progress(90, text=Tget(T, "run_stage_index", "Updating index..."))
                ok = ensure_index(T)

                if ok:
                    msg.success(Tget(T, "run_done", "Done"))
                    progress.progress(100, text=Tget(T, "run_done", "Done"))
                else:
                    msg.warning(Tget(T, "run_done_no_index", "Done, but index not updated"))
                    progress.progress(100, text=Tget(T, "run_done_no_index", "Done, but index not updated"))

                st.rerun()

            except Exception as e:
                msg.error(f"{Tget(T, 'run_err_prefix', 'Error')}: {e}")
                progress.empty()
                soft_note(str(e), kind="warn")

    vid = st.session_state.selected_video_id
    path = raw_videos[vid]
    with st.container(border=True):
        st.markdown(f"### {Tget(T, 'preview', 'Preview')}")
        maybe_playback_warning(path, T)
        st.video(str(path))

    last = st.session_state.get("last_run")
    if not last or last.get("video_id") != vid:
        return

    out = read_run_outputs(vid, last["run_id"])
    anns = out.get("annotations") or []
    lang = (out.get("language") or "").strip() or "unknown"

    approx_dur = 0.0
    if anns:
        approx_dur = max(float(a.get("end_sec", 0.0)) for a in anns)

    m = out.get("metrics") or {}
    preprocess_sec = float(m.get("preprocess_time_sec", 0.0) or 0.0)
    model_sec = float(m.get("model_time_sec", 0.0) or 0.0)
    total_sec = float(m.get("total_time_sec", 0.0) or 0.0)

    st.write("")
    with st.container(border=True):
        st.markdown(f"### {Tget(T, 'last_run', 'Last run')} · {out['run_id']}")

        st.markdown(
            f"""
            <div class="kpi-row">
              <div class="kpi">{E(Tget(T, 'duration', 'Duration'))}: <b>{E(hms(approx_dur) if approx_dur > 0 else Tget(T,'unknown','—'))}</b></div>
              <div class="kpi">{E(Tget(T, 'segments_count', 'Segments'))}: <b>{E(len(anns))}</b></div>
              <div class="kpi">{E(Tget(T, 'run_language', 'Run language'))}: <b>{E(lang.upper())}</b></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.container(border=True):
            st.markdown(f"### {Tget(T, 'global_summary_box', 'Video summary')}")
            if out.get("global_summary"):
                st.write(out["global_summary"])
            else:
                st.caption(Tget(T, "no_global_summary", "No global summary."))

        with st.container(border=True):
            st.markdown(f"### {Tget(T, 'segments_timeline', 'Timeline')}")
            if not anns:
                soft_note(Tget(T, "no_annotations", "No annotations."), kind="info")
            else:
                for a in anns:
                    st.write(f"[{mmss(a['start_sec'])} - {mmss(a['end_sec'])}] {a['description']}")

        with st.expander(Tget(T, "metrics", "Metrics")):
            st.write(f"{Tget(T, 'metrics_preprocess', 'Preprocess')}: {hms(preprocess_sec)}")
            st.write(f"{Tget(T, 'metrics_model', 'Model')}: {hms(model_sec)}")
            st.write(f"{Tget(T, 'metrics_total', 'Total')}: {hms(total_sec)}")


def search_section(raw_videos: Dict[str, Path], runs_map: Dict[str, List[str]], T: dict) -> None:
    render_page_head(Tget(T, "search_desc_title", "Search by description"))

    qe = get_engine()
    if qe is None:
        soft_note(Tget(T, "index_missing", "Index not found."), kind="warn")
        return

    with st.container(border=True):
        q1, q2 = st.columns([3, 1], gap="medium")
        with q1:
            query = st.text_input(
                Tget(T, "query", "Query"),
                placeholder=Tget(T, "query_ph", "толпа бежит / адамдар жүгіріп жатыр / crowd running"),
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

    if not query or not query.strip():
        soft_note(Tget(T, "type_query", "Type a query."), kind="info")
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
        soft_note(f"{Tget(T, 'search_error', 'Search error')}: {e}", kind="warn")
        return

    left, right = st.columns([2, 3], gap="large", vertical_alignment="top")

    with left:
        st.markdown(f"### {Tget(T, 'results', 'Results')}")
        if not hits:
            soft_note(Tget(T, "no_results", "No results."), kind="info")
        else:
            for idx, h in enumerate(hits):
                with st.container(border=True):
                    st.write(f"[{mmss(h.start_sec)} - {mmss(h.end_sec)}] {h.description}")
                    st.caption(
                        f"{h.video_id} · {h.run_id} · score={h.score:.3f} "
                        f"(bm25={h.sparse_score:.3f}, dense={h.dense_score:.3f})"
                    )
                    if st.button(Tget(T, "open", "Open"), key=f"open_hit_{idx}", use_container_width=True):
                        st.session_state.selected_hit = {
                            "video_id": h.video_id,
                            "run_id": h.run_id,
                            "start_sec": float(h.start_sec),
                            "end_sec": float(h.end_sec),
                            "description": h.description,
                        }

    with right:
        st.markdown(f"### {Tget(T, 'player', 'Player')}")
        hit = st.session_state.get("selected_hit")
        if not hit:
            soft_note(Tget(T, "pick_result", "Pick a result and click Open."), kind="info")
            return

        vid = hit["video_id"]
        start = float(hit["start_sec"])
        desc = hit["description"]
        st.write(f"**{vid}** · {hit['run_id']}")
        st.write(f"[{mmss(start)}] {desc}")

        video_path = raw_videos.get(vid)
        if not video_path or not video_path.exists():
            soft_note(f"{Tget(T, 'raw_missing', 'Raw video not found')}: '{vid}' → {RAW_DIR}", kind="warn")
        else:
            try:
                st.video(str(video_path), start_time=int(start))
            except TypeError:
                st.video(str(video_path))
                st.caption(f"{Tget(T, 'seek_hint', 'Seek manually to ~ ')}{int(start)} sec ({mmss(start)}).")


def footer(T: dict) -> None:
    st.markdown("---")
    st.caption(Tget(T, "footer", ""))


def main() -> None:
    st.set_page_config(page_title="SmartCampus V2T", layout="wide")
    inject_css()

    try:
        ui_text = load_ui_text(str(UI_TEXT_PATH), _mtime(UI_TEXT_PATH))
    except Exception as e:
        st.error(str(e))
        st.stop()

    if "selected_hit" not in st.session_state:
        st.session_state.selected_hit = None

    T = sidebar_runtime_and_index(ui_text)

    labels = T.get("tabs") or ["Home", "Processing", "Search"]
    current_tab = _get_qp("tab", TAB_IDS[0])
    if current_tab not in TAB_IDS:
        current_tab = TAB_IDS[0]

    render_header(T, labels, TAB_IDS, current_tab)

    raw_videos = list_raw_videos()
    runs_map = list_all_runs()

    if current_tab == "home":
        home_section(raw_videos, T)
    elif current_tab == "process":
        processing_section(raw_videos, T)
    elif current_tab == "search":
        search_section(raw_videos, runs_map, T)

    footer(T)


if __name__ == "__main__":
    main()
