# app/streamlit_app.py

"""
SmartCampus V2T — Streamlit UI application.

Tabs:
- Home: video carousel + preview + processing + run outputs
- Search by description: search over annotations + player

Notes:
- No sidebar (all controls are in-page)
- Streamlit UI is styled by CSS from cfg.ui.styles_path
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
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import streamlit as st
import streamlit.components.v1 as components

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CFG_PATH = PROJECT_ROOT / "config" / "pipeline.yaml"

from src.search import build_or_update_index, QueryEngine
from src.utils.config_loader import load_pipeline_config
from src.preprocessing.video_io import preprocess_video
from src.pipeline.video_to_text import VideoToTextPipeline
from src.core.types import VideoMeta, FrameInfo, Annotation, RunMetrics

TAB_IDS = ["home", "search"]


def E(s: Any) -> str:
    return html.escape("" if s is None else str(s), quote=True)


def _mtime(p: Path) -> float:
    try:
        return float(p.stat().st_mtime)
    except Exception:
        return 0.0


@st.cache_resource(show_spinner=False)
def get_cfg_cached(cfg_path_str: str, cfg_mtime: float):
    _ = cfg_mtime
    return load_pipeline_config(Path(cfg_path_str))


def get_cfg():
    return get_cfg_cached(str(CFG_PATH), _mtime(CFG_PATH))


def _cache_bucket(ttl_sec: int) -> int:
    ttl = max(1, int(ttl_sec or 1))
    return int(time.time()) // ttl


def _get_qp(name: str, default: str) -> str:
    try:
        v = st.query_params.get(name, default)
        return v if isinstance(v, str) else (v[0] if v else default)
    except Exception:
        qp = st.experimental_get_query_params()
        return qp.get(name, [default])[0]


def _ensure_i18n_state() -> None:
    if "_i18n_missing_set" not in st.session_state:
        st.session_state._i18n_missing_set = set()
    if "_i18n_missing_total" not in st.session_state:
        st.session_state._i18n_missing_total = 0
    if "_i18n_missing_by_lang" not in st.session_state:
        st.session_state._i18n_missing_by_lang = {}


def get_T(ui_text: Dict[str, Dict[str, Any]], lang: str) -> Dict[str, Any]:
    lang = (lang or "ru").strip().lower()
    return ui_text.get(lang) or ui_text.get("ru") or {}


def Tget(T: Dict[str, Any], key: str, fallback: str) -> str:
    v = T.get(key)
    if v is None:
        _ensure_i18n_state()
        lang = st.session_state.get("ui_lang", "ru")
        miss = f"{lang}:{key}"
        if miss not in st.session_state._i18n_missing_set:
            st.session_state._i18n_missing_set.add(miss)
            st.session_state._i18n_missing_total += 1
            st.session_state._i18n_missing_by_lang[lang] = st.session_state._i18n_missing_by_lang.get(lang, 0) + 1
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


def load_and_apply_css(styles_path: Path) -> None:
    css = ""
    if styles_path.exists():
        css = styles_path.read_text(encoding="utf-8")
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


def render_header(T: Dict[str, Any], labels: List[str], ids: List[str], current_tab: str, logo_path: Path) -> None:
    links = []
    for lab, tid in zip(labels, ids):
        cls = "nav-link active" if tid == current_tab else "nav-link"
        links.append(f"<a class='{cls}' href='?tab={E(tid)}' target='_self'>{E(lab)}</a>")

    logo = _img_to_data_uri(logo_path)
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
    langs: Optional[List[str]] = None,
    default_lang: str = "ru",
) -> Dict[str, Any]:
    langs = langs or ["ru", "kz", "en"]

    if "ui_lang" not in st.session_state:
        st.session_state.ui_lang = default_lang
    if st.session_state.ui_lang not in langs:
        st.session_state.ui_lang = default_lang

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
            options=langs,
            index=langs.index(st.session_state.ui_lang),
            key="ui_lang",
            label_visibility="collapsed",
        )
        st.markdown("</div>", unsafe_allow_html=True)

    return get_T(ui_text, st.session_state.ui_lang)


@st.cache_data(show_spinner=False)
def list_raw_videos(raw_dir_str: str, bucket: int) -> Dict[str, str]:
    _ = bucket
    out: Dict[str, str] = {}
    raw_dir = Path(raw_dir_str)
    if not raw_dir.exists():
        return out
    for p in sorted(raw_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in {".mp4", ".mov", ".mkv", ".avi"}:
            out[p.stem] = str(p)
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
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    ok2, buf = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    if not ok2:
        return None
    return buf.tobytes()


def _thumb_file_for_video(video_path: Path) -> Path:
    cfg = get_cfg()
    return cfg.paths.thumbs_dir / f"{video_path.stem}.jpg"


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


def _runs_dir(cfg) -> Path:
    p = getattr(cfg.paths, "runs_dir", None)
    if p is not None:
        return Path(p)
    ann = getattr(cfg.paths, "annotations_dir", None)
    if ann is not None:
        return Path(ann)
    return Path(cfg.paths.data_dir) / "runs"

def to_jsonable(x):
    from pathlib import Path
    from dataclasses import is_dataclass, asdict

    if isinstance(x, Path):
        return str(x)
    if is_dataclass(x):
        return to_jsonable(asdict(x))
    if isinstance(x, dict):
        return {k: to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    return x

def _sha1_hex(payload: Any) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    import hashlib

    return hashlib.sha1(raw).hexdigest()


def _best_effort_model_fingerprint(model_name_or_path: str) -> str:
    p = Path(model_name_or_path)
    if not p.exists():
        return f"str:{model_name_or_path}"
    try:
        if p.is_file():
            stt = p.stat()
            return f"file:{p.name}:{stt.st_size}:{stt.st_mtime_ns}"
        if p.is_dir():
            cand = None
            for name in ["config.json", "generation_config.json", "model.safetensors", "pytorch_model.bin"]:
                if (p / name).exists():
                    cand = p / name
                    break
            if cand is None:
                stt = p.stat()
                return f"dir:{p.name}:{stt.st_mtime_ns}"
            stt = cand.stat()
            return f"dirfile:{p.name}/{cand.name}:{stt.st_size}:{stt.st_mtime_ns}"
    except Exception:
        return f"str:{model_name_or_path}"
    return f"str:{model_name_or_path}"


def compute_inference_fingerprint(cfg) -> str:
    m = cfg.model
    c = cfg.clips
    prompt_payload = {
        "language": getattr(m, "language", None),
        "model_name_or_path": getattr(m, "model_name_or_path", None) or getattr(m, "model_name", None),
        "dtype": getattr(m, "dtype", None),
        "device": getattr(m, "device", None),
        "batch_size": getattr(m, "batch_size", None),
        "max_new_tokens": getattr(m, "max_new_tokens", None),
        "temperature": getattr(m, "temperature", None),
        "top_p": getattr(m, "top_p", None),
        "top_k": getattr(m, "top_k", None),
        "repetition_penalty": getattr(m, "repetition_penalty", None),
        "do_sample": getattr(m, "do_sample", None),
        "window_sec": getattr(c, "window_sec", None),
        "stride_sec": getattr(c, "stride_sec", None),
        "min_clip_frames": getattr(c, "min_clip_frames", None),
        "max_clip_frames": getattr(c, "max_clip_frames", None),
    }
    model_path = str(prompt_payload["model_name_or_path"] or "")
    prompt_payload["model_fingerprint"] = _best_effort_model_fingerprint(model_path)
    return _sha1_hex(prompt_payload)[:24]


def find_cached_run(runs_root: Path, video_id: str, inference_fp: str) -> Optional[str]:
    base = runs_root / video_id
    if not base.exists():
        return None
    candidates = sorted([p for p in base.iterdir() if p.is_dir() and p.name.startswith("run_")], reverse=True)
    for rd in candidates:
        mf = rd / "run_manifest.json"
        if not mf.exists():
            continue
        try:
            obj = json.loads(mf.read_text(encoding="utf-8"))
            if str(obj.get("inference_fingerprint", "")) == str(inference_fp):
                return rd.name
        except Exception:
            continue
    return None


def list_runs_for_video(runs_root: Path, video_id: str) -> List[str]:
    base = runs_root / video_id
    if not base.exists():
        return []
    runs: List[str] = []
    for p in sorted(base.iterdir()):
        if p.is_dir() and p.name.startswith("run_") and (p / "run_manifest.json").exists():
            runs.append(p.name)
    return runs


@st.cache_data(show_spinner=False)
def list_all_runs(runs_root_str: str, bucket: int) -> Dict[str, List[str]]:
    _ = bucket
    out: Dict[str, List[str]] = {}
    runs_root = Path(runs_root_str)
    if not runs_root.exists():
        return out
    for vid_dir in sorted(runs_root.iterdir()):
        if not vid_dir.is_dir():
            continue
        runs = list_runs_for_video(runs_root, vid_dir.name)
        if runs:
            out[vid_dir.name] = runs
    return out


def allocate_run_id(runs_root: Path, video_id: str) -> str:
    base = runs_root / video_id
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


def _safe_clear_dir(p: Path) -> None:
    if not p.exists():
        return
    for child in p.iterdir():
        try:
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            else:
                child.unlink(missing_ok=True)
        except Exception:
            pass


def save_run_outputs(
    runs_root: Path,
    cfg,
    video_id: str,
    run_id: str,
    annotations: List[Annotation],
    metrics: RunMetrics,
    language: str,
    device: str,
    inference_fp: str,
    prepared_meta: Optional[VideoMeta],
    force_overwrite: bool = False,
) -> Path:
    language = (language or "").strip().lower() or "unknown"
    device = (device or "").strip().lower() or "unknown"

    run_dir = runs_root / video_id / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    if force_overwrite:
        _safe_clear_dir(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)

    ann_dicts = [annotation_to_dict(a, i, video_id=video_id) for i, a in enumerate(annotations)]

    metrics_dict = dataclass_to_dict(metrics)
    if not isinstance(metrics_dict, dict):
        metrics_dict = {}

    extra = metrics_dict.get("extra")
    if not isinstance(extra, dict):
        extra = {}
    extra["language"] = extra.get("language") or language
    extra["device"] = extra.get("device") or device
    extra["inference_fingerprint"] = extra.get("inference_fingerprint") or inference_fp
    metrics_dict["extra"] = extra

    cfg_dict = to_jsonable(cfg)

    video_fp = None
    video_cfg_fp = None
    prepared_dir = None
    if prepared_meta is not None:
        try:
            cache = (prepared_meta.extra or {}).get("cache") or {}
            video_fp = cache.get("video_fingerprint")
            video_cfg_fp = cache.get("video_cfg_fingerprint")
            prepared_dir = str(prepared_meta.prepared_dir) if prepared_meta.prepared_dir else None
        except Exception:
            pass

    manifest = {
        "video_id": video_id,
        "run_id": run_id,
        "created_at_unix": int(time.time()),
        "language": language,
        "device": device,
        "inference_fingerprint": inference_fp,
        "prepared": {
            "prepared_dir": prepared_dir,
            "video_fingerprint": video_fp,
            "video_cfg_fingerprint": video_cfg_fp,
        },
        "status": "ok",
    }

    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (run_dir / "config.json").write_text(json.dumps(cfg_dict, indent=2, ensure_ascii=False), encoding="utf-8")
    (run_dir / "annotations.json").write_text(json.dumps(ann_dicts, indent=2, ensure_ascii=False), encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps(metrics_dict, indent=2, ensure_ascii=False), encoding="utf-8")

    return run_dir


@st.cache_data(show_spinner=False)
def _read_run_outputs_cached(
    run_dir_str: str,
    manifest_mtime: float,
    ann_mtime: float,
    met_mtime: float,
) -> Dict[str, Any]:
    _ = (manifest_mtime, ann_mtime, met_mtime)

    run_dir = Path(run_dir_str)

    out: Dict[str, Any] = {
        "video_id": run_dir.parent.name if run_dir.parent else None,
        "run_id": run_dir.name,
        "annotations": [],
        "metrics": None,
        "global_summary": None,
        "language": None,
        "device": None,
        "manifest": None,
    }

    mf = run_dir / "run_manifest.json"
    if mf.exists():
        try:
            out["manifest"] = json.loads(mf.read_text(encoding="utf-8"))
            out["language"] = (out["manifest"].get("language") or None)
            out["device"] = (out["manifest"].get("device") or None)
        except Exception:
            out["manifest"] = None

    ap = run_dir / "annotations.json"
    if ap.exists():
        try:
            out["annotations"] = json.loads(ap.read_text(encoding="utf-8"))
        except Exception:
            out["annotations"] = []

    mp = run_dir / "metrics.json"
    if mp.exists():
        try:
            metrics = json.loads(mp.read_text(encoding="utf-8"))
            out["metrics"] = metrics
            out["global_summary"] = (metrics.get("extra") or {}).get("global_summary")
            if out["language"] is None:
                out["language"] = (metrics.get("extra") or {}).get("language")
            if out["device"] is None:
                out["device"] = (metrics.get("extra") or {}).get("device")
        except Exception:
            out["metrics"] = None

    return out


def read_run_outputs(runs_root: Path, video_id: str, run_id: str) -> Dict[str, Any]:
    run_dir = runs_root / video_id / run_id
    mf = run_dir / "run_manifest.json"
    ap = run_dir / "annotations.json"
    mp = run_dir / "metrics.json"
    return _read_run_outputs_cached(
        run_dir_str=str(run_dir),
        manifest_mtime=_mtime(mf),
        ann_mtime=_mtime(ap),
        met_mtime=_mtime(mp),
    )


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
    _ = cfg_mtime
    cfg0 = load_pipeline_config(Path(cfg_path_str))
    cfg = copy.deepcopy(cfg0)
    cfg.model.device = device
    cfg.model.language = language
    return cfg, VideoToTextPipeline(cfg)


def run_pipeline_on_video(
    video_path: Path,
    device: str,
    language: str,
    force_overwrite: bool = False,
    overwrite_run_id: Optional[str] = None,
) -> Tuple[str, List[Annotation], RunMetrics, bool]:
    cfg, pipeline = _get_pipeline_cached(
        device=device,
        language=language,
        cfg_path_str=str(CFG_PATH),
        cfg_mtime=_mtime(CFG_PATH),
    )

    runs_root = _runs_dir(cfg)
    runs_root.mkdir(parents=True, exist_ok=True)

    inference_fp = compute_inference_fingerprint(cfg)

    cached_run_id = None
    if not force_overwrite:
        cached_run_id = find_cached_run(runs_root, video_id=video_path.stem, inference_fp=inference_fp)

    if cached_run_id is not None:
        out = read_run_outputs(runs_root, video_path.stem, cached_run_id)
        anns = out.get("annotations") or []
        merged = [
            Annotation(
                video_id=str(a.get("video_id", video_path.stem)),
                start_sec=float(a.get("start_sec", 0.0)),
                end_sec=float(a.get("end_sec", 0.0)),
                description=str(a.get("description", "")),
                extra=a.get("extra"),
            )
            for a in anns
        ]
        met = out.get("metrics") or {}
        metrics_obj = RunMetrics(
            video_id=str(met.get("video_id", video_path.stem)),
            video_duration_sec=float(met.get("video_duration_sec", 0.0)),
            num_frames=int(met.get("num_frames", 0)),
            num_clips=int(met.get("num_clips", 0)),
            avg_clip_duration_sec=float(met.get("avg_clip_duration_sec", 0.0)),
            preprocess_time_sec=float(met.get("preprocess_time_sec", 0.0)),
            model_time_sec=float(met.get("model_time_sec", 0.0)),
            postprocess_time_sec=float(met.get("postprocess_time_sec", 0.0)),
            total_time_sec=float(met.get("total_time_sec", 0.0)),
            extra=met.get("extra"),
        )
        return cached_run_id, merged, metrics_obj, True

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

    if overwrite_run_id and force_overwrite:
        run_id = overwrite_run_id
    else:
        run_id = allocate_run_id(runs_root, video_meta.video_id)

    save_run_outputs(
        runs_root=runs_root,
        cfg=cfg,
        video_id=video_meta.video_id,
        run_id=run_id,
        annotations=annotations,
        metrics=metrics,
        language=language,
        device=device,
        inference_fp=inference_fp,
        prepared_meta=video_meta,
        force_overwrite=bool(force_overwrite and overwrite_run_id),
    )

    return run_id, annotations, metrics, False


def _index_version(index_dir: Path) -> float:
    p1 = index_dir / "manifest.json"
    p2 = index_dir / "meta.json"
    return max(_mtime(p1), _mtime(p2), 0.0)


def ensure_index(T: dict) -> bool:
    cfg = get_cfg()
    before = _index_version(cfg.paths.indexes_dir)
    try:
        build_or_update_index(
            ann_root=_runs_dir(cfg),
            index_dir=cfg.paths.indexes_dir,
            model_name=cfg.search.embed_model_name,
        )
        after = _index_version(cfg.paths.indexes_dir)
        st.session_state.index_version = after
        if after != before and "qe" in st.session_state:
            del st.session_state["qe"]
        return True
    except Exception as e:
        soft_note(f"{Tget(T, 'index_build_failed', 'Index build failed')}: {e}", kind="warn")
        return False


def get_engine() -> Optional[QueryEngine]:
    cfg = get_cfg()
    try:
        cur_ver = _index_version(cfg.paths.indexes_dir)
        if "index_version" not in st.session_state:
            st.session_state.index_version = cur_ver
        if st.session_state.index_version != cur_ver and "qe" in st.session_state:
            del st.session_state["qe"]
            st.session_state.index_version = cur_ver
        if "qe" not in st.session_state:
            st.session_state.qe = QueryEngine(
                index_dir=cfg.paths.indexes_dir,
                w_bm25=cfg.search.w_bm25,
                w_dense=cfg.search.w_dense,
                candidate_k_sparse=cfg.search.candidate_k_sparse,
                candidate_k_dense=cfg.search.candidate_k_dense,
                fusion=cfg.search.fusion,
                rrf_k=cfg.search.rrf_k,
                dedupe_mode=cfg.search.dedupe_mode,
                dedupe_tol_sec=cfg.search.dedupe_tol_sec,
                dedupe_overlap_thr=cfg.search.dedupe_overlap_thr,
                embed_model_name=cfg.search.embed_model_name,
            )
        return st.session_state.qe
    except Exception:
        return None


def _quick_upload_block(T: dict, key_prefix: str, langs: List[str]) -> None:
    cfg = get_cfg()

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

    target_path = cfg.paths.raw_dir / uploaded.name
    st.caption(f"{Tget(T, 'target', 'Target')}: {target_path}")

    c1, c2 = st.columns([3, 2], gap="large", vertical_alignment="top")
    with c1:
        st.markdown(f"<div class='field-label'>{E(Tget(T, 'model_lang', 'Model output language'))}</div>", unsafe_allow_html=True)
        st.selectbox(
            Tget(T, "model_lang", "Model output language"),
            langs,
            index=langs.index(st.session_state.pipeline_lang),
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
    cfg = get_cfg()
    p = raw_videos.get(video_id)
    if p is None:
        for ext in [".mp4", ".mov", ".mkv", ".avi"]:
            cand = cfg.paths.raw_dir / f"{video_id}{ext}"
            if cand.exists():
                p = cand
                break

    try:
        if p is not None and p.exists():
            p.unlink()
    except Exception:
        pass

    try:
        tp = cfg.paths.thumbs_dir / f"{video_id}.jpg"
        if tp.exists():
            tp.unlink()
    except Exception:
        pass

    try:
        shutil.rmtree(_runs_dir(cfg) / video_id, ignore_errors=True)
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
    return "".join(ch for ch in (s or "") if ch.isalnum() or ch in {"_", "-"})  # keep '-' too


def _request_scroll_to(anchor_id: str) -> None:
    st.session_state.scroll_to_anchor = _safe_dom_id(anchor_id)


def _scroll_if_requested() -> None:
    anchor_id = st.session_state.pop("scroll_to_anchor", None)
    if not anchor_id:
        return

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
    _request_scroll_to("sec_preview")


def _render_video_player(video_path: Path, start_sec: int) -> None:
    try:
        st.video(str(video_path), start_time=int(max(0, start_sec)))
    except TypeError:
        st.video(str(video_path))
        if start_sec > 0:
            st.caption(f"{int(start_sec)} sec")


def home_section(raw_videos: Dict[str, Path], runs_map: Dict[str, List[str]], ui_text: Dict[str, Dict[str, Any]], langs: List[str]) -> None:
    if "pipeline_lang" not in st.session_state:
        st.session_state.pipeline_lang = langs[0] if langs else "ru"
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
    if "force_overwrite_run" not in st.session_state:
        st.session_state.force_overwrite_run = False
    if "overwrite_run_id" not in st.session_state:
        st.session_state.overwrite_run_id = None

    cfg = get_cfg()
    Ttmp = get_T(ui_text, st.session_state.ui_lang)
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

        _quick_upload_block(T, key_prefix="home", langs=langs)
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
                langs,
                index=langs.index(st.session_state.pipeline_lang),
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

        selected_vid = st.session_state.selected_video_id
        existing_runs = runs_map.get(selected_vid, [])

        c3, c4 = st.columns([2.2, 2.8], gap="medium", vertical_alignment="top")
        with c3:
            st.checkbox(
                Tget(T, "force_overwrite", "Force overwrite an existing run"),
                value=bool(st.session_state.force_overwrite_run),
                key="force_overwrite_run",
            )
        with c4:
            if st.session_state.force_overwrite_run:
                if not existing_runs:
                    st.session_state.overwrite_run_id = None
                    st.caption(Tget(T, "no_runs_to_overwrite", "No runs to overwrite for this video."))
                else:
                    default_overwrite = existing_runs[-1]
                    cur = st.session_state.overwrite_run_id or default_overwrite
                    if cur not in existing_runs:
                        cur = default_overwrite
                    st.selectbox(
                        Tget(T, "overwrite_run_id", "Overwrite run"),
                        existing_runs,
                        index=existing_runs.index(cur),
                        key="overwrite_run_id",
                    )
            else:
                st.session_state.overwrite_run_id = None

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

            msg = st.empty()
            progress = st.progress(0, text=Tget(T, "run_stage_init", "Loading..."))

            try:
                msg.info(Tget(T, "run_stage_init", "Loading..."))
                progress.progress(6, text=Tget(T, "run_stage_init", "Loading..."))

                msg.info(Tget(T, "run_stage_preprocess", "Preprocessing..."))
                progress.progress(22, text=Tget(T, "run_stage_preprocess", "Preprocessing..."))

                msg.info(Tget(T, "run_stage_infer", "Inference..."))
                progress.progress(52, text=Tget(T, "run_stage_infer", "Inference..."))

                force_overwrite = bool(st.session_state.force_overwrite_run and st.session_state.overwrite_run_id)
                run_id, annotations, metrics, cache_hit = run_pipeline_on_video(
                    video_path=raw_videos[st.session_state.selected_video_id],
                    device=st.session_state.device,
                    language=st.session_state.pipeline_lang,
                    force_overwrite=force_overwrite,
                    overwrite_run_id=st.session_state.overwrite_run_id if force_overwrite else None,
                )

                msg.info(Tget(T, "run_stage_index", "Updating the search index..."))
                progress.progress(92, text=Tget(T, "run_stage_index", "Updating the search index..."))
                ok = ensure_index(T)

                progress.progress(100, text=Tget(T, "run_done", "Done ✅"))
                if force_overwrite:
                    msg.success((Tget(T, "run_overwritten", "Overwritten ✅")) + f" · {run_id}" + (" · index" if ok else ""))
                else:
                    if cache_hit:
                        msg.success(Tget(T, "run_done", "Done ✅") + f" · cache hit · {run_id}")
                    else:
                        msg.success(Tget(T, "run_done", "Done ✅") if ok else Tget(T, "run_done_no_index", "Done, but index not updated"))

                st.session_state[f"selected_run_{st.session_state.selected_video_id}"] = run_id
                try:
                    st.toast(
                        (Tget(T, "run_done", "Done ✅") + f" · {run_id}")
                        + (" · cache" if cache_hit else "")
                        + (" · overwritten" if force_overwrite else "")
                    )
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
            st.markdown("<div style='margin-top: 10px'></div>", unsafe_allow_html=True)
            soft_note(Tget(T, "no_videos", "No videos available. Please upload a file above."), kind="info")
            return

        selected = st.session_state.selected_video_id
        runs = runs_map.get(selected, [])
        if not runs:
            st.markdown("<div style='margin-top: 10px'></div>", unsafe_allow_html=True)
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

        cfg = get_cfg()
        out = read_run_outputs(_runs_dir(cfg), selected, sel_run)
        run_lang = (out.get("language") or "unknown").strip() or "unknown"
        run_dev = (out.get("device") or "unknown").strip() or "unknown"

        m = out.get("metrics") or {}
        preprocess_sec = float(m.get("preprocess_time_sec", 0.0) or 0.0)
        model_sec = float(m.get("model_time_sec", 0.0) or 0.0)
        total_sec = float(m.get("total_time_sec", 0.0) or 0.0)

        _pill_row(
            [
                f"{Tget(T,'run_lang','Run language')}: {run_lang.upper()}",
                f"{Tget(T,'device','Device')}: {run_dev.upper()}",
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
                            _seek_preview_to(selected, start)
                            st.rerun()
                        st.markdown("</div>", unsafe_allow_html=True)
                    with b2:
                        st.write(desc)


def search_section(raw_videos: Dict[str, Path], runs_map: Dict[str, List[str]], ui_text: Dict[str, Dict[str, Any]], langs: List[str]) -> None:
    cfg = get_cfg()
    Ttmp = get_T(ui_text, st.session_state.ui_lang)
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
            soft_note(
                Tget(T, "ok", "Completed") if ok else Tget(T, "index_build_failed", "Failed to update the search index"),
                kind=("ok" if ok else "warn"),
            )

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
                soft_note(Tget(T, "pick_result", "Select a result and click “Open”."))
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
                    soft_note(f"{Tget(T, 'raw_missing', 'Source video not found')}: '{vid}' → {cfg.paths.raw_dir}", kind="warn")
                else:
                    maybe_playback_warning(video_path, T)
                    _render_video_player(video_path, int(start))


def footer(T: dict) -> None:
    st.markdown("<div class='thin-rule'></div>", unsafe_allow_html=True)
    st.caption(Tget(T, "footer", ""))


def _render_i18n_metrics() -> None:
    _ensure_i18n_state()
    total = int(st.session_state._i18n_missing_total or 0)
    if total <= 0:
        return
    with st.expander("i18n metrics", expanded=False):
        st.write({"missing_total": total, "missing_by_lang": dict(st.session_state._i18n_missing_by_lang)})
        st.code("\n".join(sorted(st.session_state._i18n_missing_set)))


def main() -> None:
    cfg = get_cfg()

    st.set_page_config(page_title="SmartCampus V2T", layout="wide")
    load_and_apply_css(cfg.ui.styles_path)

    langs = cfg.ui.langs or ["ru", "kz", "en"]
    langs_key = ",".join(langs)

    try:
        ui_text = load_ui_text(str(cfg.ui.ui_text_path), _mtime(cfg.ui.ui_text_path), langs_key)
    except Exception as e:
        st.error(str(e))
        st.stop()

    if "selected_hit" not in st.session_state:
        st.session_state.selected_hit = None
    if "ui_lang" not in st.session_state:
        st.session_state.ui_lang = cfg.ui.default_lang
    if "preview_seek_sec" not in st.session_state:
        st.session_state.preview_seek_sec = 0

    T = get_T(ui_text, st.session_state.ui_lang)

    labels = T.get("tabs") or ["Home", "Search"]
    current_tab = _get_qp("tab", TAB_IDS[0])
    if current_tab not in TAB_IDS:
        current_tab = TAB_IDS[0]

    render_header(T, labels, TAB_IDS, current_tab, logo_path=cfg.ui.logo_path)

    bucket = _cache_bucket(cfg.ui.cache_ttl_sec)
    raw_videos_str = list_raw_videos(str(cfg.paths.raw_dir), bucket)
    raw_videos: Dict[str, Path] = {k: Path(v) for k, v in raw_videos_str.items()}

    runs_root = _runs_dir(cfg)
    runs_map = list_all_runs(str(runs_root), bucket)

    if current_tab == "home":
        home_section(raw_videos, runs_map, ui_text, langs=langs)
    elif current_tab == "search":
        search_section(raw_videos, runs_map, ui_text, langs=langs)

    _render_i18n_metrics()
    footer(get_T(ui_text, st.session_state.ui_lang))


if __name__ == "__main__":
    main()
