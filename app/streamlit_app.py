# app/streamlit_app.py
"""
SmartCampus V2T — Streamlit UI application.

Features:
- Clean UI layout (hero header, cards, consistent spacing)
- UI language switch (ru / kz / en) in a top bar (not in sidebar)
- Sidebar kept minimal: runtime + index tools
- Video gallery with both preview grid + list view
- Drag & drop upload to data/raw
- Run V2T pipeline from UI (language-aware: ru/kz/en)
- Show outputs nicely: GLOBAL SUMMARY in a separate box + timestamped segments below
- Runs selector per selected video (placed under video preview)
- Hybrid index management (BM25 + E5)
- Search over annotations + player

Important note about video playback:
- st.video relies on browser codecs.
- .avi / .mkv often will NOT play in Chrome/Edge.
- Recommended raw format: .mp4 (H.264 video + AAC audio).
- UI includes an optional "Convert to MP4" via ffmpeg (if installed).

Dependencies:
- For E5 embeddings: pip install -U sentence-transformers
- Optional for conversion: ffmpeg in PATH
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import streamlit as st

# =========================
# Paths
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
ANN_DIR = DATA_DIR / "annotations"
MET_DIR = DATA_DIR / "metrics"
INDEX_DIR = DATA_DIR / "indexes"
ASSETS_DIR = PROJECT_ROOT / "app" / "assets"
LOGO_PATH = ASSETS_DIR / "logo.png"

RAW_DIR.mkdir(parents=True, exist_ok=True)
ANN_DIR.mkdir(parents=True, exist_ok=True)
MET_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# Project imports
# =========================
from src.search import build_or_update_index, QueryEngine
from src.utils.config import load_pipeline_config
from src.preprocessing.video_io import preprocess_video
from src.pipeline.video_to_text import VideoToTextPipeline
from src.core.types import VideoMeta, FrameInfo, Annotation, RunMetrics

# =========================
# UI text (FULL localization)
# =========================
UI_TEXT = {
    "ru": {
        "app_title": "SmartCampus V2T",
        "app_subtitle": "MLLM-based video description pipeline of surveillance systems",
        "app_flow": "Загрузка → Прогон → Индексация → Поиск по событиям на видео",
        "tabs": ["Галерея", "Загрузка", "Запуск", "Поиск", "Прогоны"],
        "ui_lang": "Язык интерфейса",
        "runtime": "Параметры запуска",
        "model_lang": "Язык вывода модели",
        "device": "Устройство",
        "index": "Индекс",
        "update_index": "Обновить индекс",
        "index_hint": "Индекс нужен для поиска по аннотациям (BM25 + E5).",
        "video_gallery": "Галерея видео",
        "selected_video": "Выбранное видео",
        "grid": "Сетка",
        "open": "Открыть",
        "preview": "Превью",
        "upload": "Загрузка (drag & drop)",
        "drop_here": "Перетащи видео сюда",
        "save_raw": "Сохранить в data/raw",
        "run": "Запуск",
        "run_pipeline": "RUN",
        "output": "Результаты",
        "global_summary": "GLOBAL SUMMARY",
        "segments": "Сегменты",
        "metrics": "Метрики",
        "search": "Поиск",
        "query": "Запрос (ru / kz / en)",
        "topk": "Top-K",
        "filters": "Фильтры",
        "video_filter": "Видео",
        "run_filter": "Прогон",
        "results": "Результаты",
        "player": "Плеер",
        "runs": "Прогоны",
        "convert": "Конвертировать в MP4",
        "convert_hint": "Для воспроизведения в браузере лучше MP4 (H.264/AAC).",
        "ffmpeg_missing": "ffmpeg не найден. Установи ffmpeg и добавь в PATH, либо конвертируй видео заранее в mp4.",
        "playback_warn": "Видео может не воспроизводиться в браузере из-за формата/кодека. Рекомендуется MP4 (H.264/AAC).",
        "no_videos": "Пока нет видео. Загрузите файл во вкладке «Загрузка».",
        "no_runs": "Пока нет прогонов для этого видео. Сделай прогон во вкладке «Запуск».",
        "runs_for_video": "Индексированные прогоны",
        "run_output": "Вывод прогона",
        "run_lang": "Язык прогона",
        "lang_missing_label": "Для выбранного языка прогона нет. Запусти пайплайн в нужном языке.",
        "index_missing": "Индекс не найден. Нажми «Обновить индекс» слева.",
        "e5_missing": "Для E5 нужен sentence-transformers. Установи в то же окружение: pip install -U sentence-transformers",
        "footer": "Сделано автором проекта SmartCampus V2T (AITU).",
        "debug": "Отладка",
        # new / refactor strings
        "view_mode": "Отображение",
        "view_grid": "Превью",
        "view_list": "Списком",
        "videos": "Видео",
        "video_list": "Список видео",
        "runs_for_selected": "Прогоны для выбранного видео",
        "choose_run": "Выбор прогона",
        "global_summary_box": "Сводка по видео",
        "segments_timeline": "Подробно по времени",
        "no_thumbnail": "Нет превью",
        "building_index": "Сборка индекса...",
        "converting": "Конвертация через ffmpeg...",
        "running_pipeline": "Запуск пайплайна...",
        "updating_index": "Обновление индекса...",
        "saved": "Сохранено",
        "conversion_failed": "Конвертация не удалась. Проверь ffmpeg/кодеки.",
        "type_query": "Введите запрос.",
        "no_results": "Нет результатов.",
        "pick_result": "Выбери результат и нажми Open.",
        "raw_missing": "Raw-видео не найдено в data/raw",
        "drag_drop_hint": "Перетащи файл → Сохрани → Запусти.",
        "target": "Куда сохраняем",
        "ok": "OK",
        "no_annotations": "Нет аннотаций в этом прогоне.",
        "no_global_summary": "Нет сводки.",
        "no_runs_found": "Прогоны не найдены в data/annotations/",
        "conversion_done": "Конвертация завершена",
        "index_build_failed": "Сборка индекса не удалась",
        "search_error": "Ошибка поиска",
        "seek_hint": "Перемотай вручную примерно до",
        "file_target": "Файл",
        # product-like run tab
        "status": "Статус",
        "status_idle": "Idle",
        "status_running": "Running",
        "status_saved": "Saved",
        "status_indexed": "Indexed",
        "duration": "Длительность",
        "segments_count": "Сегменты",
        "run_language": "Язык прогона",
        "last_run": "Последний прогон",
        "metrics_preprocess": "Preprocess",
        "metrics_model": "Model",
        "metrics_total": "Total",
        "unknown": "—",
        # run panel labels
        "select_video": "Выбор видео",
        "run_controls": "Параметры",
    },
    "en": {
        "app_title": "SmartCampus V2T",
        "app_subtitle": "MLLM-based video description pipeline of surveillance systems",
        "app_flow": "Upload → Run → Index → Search events in video",
        "tabs": ["Gallery", "Upload", "Run", "Search", "Runs"],
        "ui_lang": "UI language",
        "runtime": "Runtime",
        "model_lang": "Model output language",
        "device": "Device",
        "index": "Index",
        "update_index": "Update index",
        "index_hint": "Index is required for search (BM25 + E5).",
        "video_gallery": "Video Gallery",
        "selected_video": "Selected video",
        "grid": "Grid",
        "open": "Open",
        "preview": "Preview",
        "upload": "Upload (drag & drop)",
        "drop_here": "Drop a video file here",
        "save_raw": "Save to data/raw",
        "run": "Run",
        "run_pipeline": "RUN",
        "output": "Output",
        "global_summary": "GLOBAL SUMMARY",
        "segments": "Segments",
        "metrics": "Metrics",
        "search": "Search",
        "query": "Query (ru / kz / en)",
        "topk": "Top-K",
        "filters": "Filters",
        "video_filter": "Video",
        "run_filter": "Run",
        "results": "Results",
        "player": "Player",
        "runs": "Runs",
        "convert": "Convert to MP4",
        "convert_hint": "For browser playback, MP4 (H.264/AAC) is recommended.",
        "ffmpeg_missing": "ffmpeg not found. Install ffmpeg and add it to PATH, or convert videos to mp4 beforehand.",
        "playback_warn": "Playback may fail due to container/codec. Recommended: MP4 (H.264/AAC).",
        "no_videos": "No videos yet. Upload one in the Upload tab.",
        "no_runs": "No runs for this video yet. Run the pipeline in the Run tab.",
        "runs_for_video": "Indexed runs",
        "run_output": "Run output",
        "run_lang": "Run language",
        "lang_missing_label": "No runs for the selected language. Run the pipeline in that language.",
        "index_missing": "Index not found. Click “Update index” on the left.",
        "e5_missing": "E5 needs sentence-transformers. Install in the same env: pip install -U sentence-transformers",
        "footer": "Made by the SmartCampus V2T project author (AITU).",
        "debug": "Debug",
        # new / refactor strings
        "view_mode": "View",
        "view_grid": "Thumbnails",
        "view_list": "List",
        "videos": "Videos",
        "video_list": "Video list",
        "runs_for_selected": "Runs for selected video",
        "choose_run": "Select run",
        "global_summary_box": "Video summary",
        "segments_timeline": "Timeline details",
        "no_thumbnail": "No thumbnail",
        "building_index": "Building index...",
        "converting": "Converting via ffmpeg...",
        "running_pipeline": "Running pipeline...",
        "updating_index": "Updating index...",
        "saved": "Saved",
        "conversion_failed": "Conversion failed. Check ffmpeg/codecs.",
        "type_query": "Type a query.",
        "no_results": "No results.",
        "pick_result": "Pick a result and click Open.",
        "raw_missing": "Raw video not found in data/raw",
        "drag_drop_hint": "Drag & drop → Save → Run.",
        "target": "Target",
        "ok": "OK",
        "no_annotations": "No annotations in this run.",
        "no_global_summary": "No global summary.",
        "no_runs_found": "No runs found in data/annotations/",
        "conversion_done": "Conversion completed",
        "index_build_failed": "Index build failed",
        "search_error": "Search error",
        "seek_hint": "Seek manually to ~",
        "file_target": "File",
        # product-like run tab
        "status": "Status",
        "status_idle": "Idle",
        "status_running": "Running",
        "status_saved": "Saved",
        "status_indexed": "Indexed",
        "duration": "Duration",
        "segments_count": "Segments",
        "run_language": "Run language",
        "last_run": "Last run",
        "metrics_preprocess": "Preprocess",
        "metrics_model": "Model",
        "metrics_total": "Total",
        "unknown": "—",
        "select_video": "Select video",
        "run_controls": "Controls",
    },
    "kz": {
        "app_title": "SmartCampus V2T",
        "app_subtitle": "MLLM-based video description pipeline of surveillance systems",
        "app_flow": "Жүктеу → Іске қосу → Индекстеу → Бейне оқиғаларын іздеу",
        "tabs": ["Галерея", "Жүктеу", "Іске қосу", "Іздеу", "Жүгірулер"],
        "ui_lang": "Интерфейс тілі",
        "runtime": "Іске қосу параметрлері",
        "model_lang": "Модель шығару тілі",
        "device": "Құрылғы",
        "index": "Индекс",
        "update_index": "Индексті жаңарту",
        "index_hint": "Іздеу үшін индекс керек (BM25 + E5).",
        "video_gallery": "Бейне галереясы",
        "selected_video": "Таңдалған бейне",
        "grid": "Тор",
        "open": "Ашу",
        "preview": "Алдын ала көру",
        "upload": "Жүктеу (drag & drop)",
        "drop_here": "Бейнені осында тастаңыз",
        "save_raw": "data/raw ішіне сақтау",
        "run": "Іске қосу",
        "run_pipeline": "RUN",
        "output": "Нәтижелер",
        "global_summary": "GLOBAL SUMMARY",
        "segments": "Сегменттер",
        "metrics": "Метрикалар",
        "search": "Іздеу",
        "query": "Сұрау (ru / kz / en)",
        "topk": "Top-K",
        "filters": "Сүзгілер",
        "video_filter": "Бейне",
        "run_filter": "Жүгіру",
        "results": "Нәтижелер",
        "player": "Ойнатқыш",
        "runs": "Жүгірулер",
        "convert": "MP4-ке түрлендіру",
        "convert_hint": "Браузерде ойнау үшін MP4 (H.264/AAC) ұсынылады.",
        "ffmpeg_missing": "ffmpeg табылмады. ffmpeg орнатып PATH-қа қос, немесе видеоны алдын ала mp4-ке айналдыр.",
        "playback_warn": "Формат/кодекке байланысты ойнамауы мүмкін. Ұсыныс: MP4 (H.264/AAC).",
        "no_videos": "Әзірге бейне жоқ. «Жүктеу» бөлімінде файл жүктеңіз.",
        "no_runs": "Бұл бейне үшін жүгірулер жоқ. «Іске қосу» бөлімінде пайплайнды жүргізіңіз.",
        "runs_for_video": "Индекстелген жүгірулер",
        "run_output": "Жүгіру нәтижесі",
        "run_lang": "Жүгіру тілі",
        "lang_missing_label": "Таңдалған тіл үшін жүгіру жоқ. Сол тілде пайплайнды іске қос.",
        "index_missing": "Индекс табылмады. Сол жақтан «Индексті жаңарту» батырмасын бас.",
        "e5_missing": "E5 үшін sentence-transformers керек. Бір ортада орнат: pip install -U sentence-transformers",
        "footer": "SmartCampus V2T жоба авторы жасады (AITU).",
        "debug": "Debug",
        # new / refactor strings
        "view_mode": "Көрсету",
        "view_grid": "Превью",
        "view_list": "Тізім",
        "videos": "Бейнелер",
        "video_list": "Бейне тізімі",
        "runs_for_selected": "Таңдалған бейне үшін жүгірулер",
        "choose_run": "Жүгіруді таңдау",
        "global_summary_box": "Бейне бойынша қысқаша қорытынды",
        "segments_timeline": "Уақыт бойынша толық",
        "no_thumbnail": "Превью жоқ",
        "building_index": "Индекс құрастыру...",
        "converting": "ffmpeg арқылы түрлендіру...",
        "running_pipeline": "Пайплайн іске қосылуда...",
        "updating_index": "Индексті жаңарту...",
        "saved": "Сақталды",
        "conversion_failed": "Түрлендіру сәтсіз. ffmpeg/кодектерді тексер.",
        "type_query": "Сұрау енгізіңіз.",
        "no_results": "Нәтиже жоқ.",
        "pick_result": "Нәтижені таңдап Open бас.",
        "raw_missing": "Raw бейне data/raw ішінде табылмады",
        "drag_drop_hint": "Drag & drop → Сақтау → Іске қосу.",
        "target": "Сақтау орны",
        "ok": "OK",
        "no_annotations": "Бұл жүгіруде аннотация жоқ.",
        "no_global_summary": "Қысқаша қорытынды жоқ.",
        "no_runs_found": "data/annotations/ ішінде жүгірулер табылмады",
        "conversion_done": "Түрлендіру аяқталды",
        "index_build_failed": "Индекс құрастыру сәтсіз",
        "search_error": "Іздеу қатесі",
        "seek_hint": "Қолмен шамамен мына уақытқа жылжыт:",
        "file_target": "Файл",
        # product-like run tab
        "status": "Мәртебе",
        "status_idle": "Idle",
        "status_running": "Running",
        "status_saved": "Saved",
        "status_indexed": "Indexed",
        "duration": "Ұзақтығы",
        "segments_count": "Сегменттер",
        "run_language": "Жүгіру тілі",
        "last_run": "Соңғы жүгіру",
        "metrics_preprocess": "Preprocess",
        "metrics_model": "Model",
        "metrics_total": "Total",
        "unknown": "—",
        "select_video": "Бейнені таңдау",
        "run_controls": "Басқару",
    },
}

# =========================
# Styling (no overlays / no phantom bars)
# =========================
def inject_css() -> None:
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.2rem; padding-bottom: 2.5rem; }

        /* Move UI language selector slightly down without creating extra "phantom bars" */
        .topbar-spacer { height: 34px; }
        .ui-lang-wrap { margin-top: 10px; }

        .hero {
            margin-top: 0.6rem;
            padding: 16px 18px;
            border: 1px solid rgba(255,255,255,0.10);
            border-radius: 16px;
            background: rgba(255,255,255,0.03);
        }
        .muted { opacity: 0.80; }

        .pill {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 999px;
            border: 1px solid rgba(255,255,255,0.12);
            background: rgba(255,255,255,0.04);
            font-size: 12px;
            opacity: 0.9;
        }

        /* Compact "product-like" pills (KPIs) */
        .kpi-row { display:flex; gap:10px; flex-wrap:wrap; align-items:center; }
        .kpi {
          display:inline-flex; gap:8px; align-items:center;
          padding: 6px 10px;
          border-radius:999px;
          border:1px solid rgba(255,255,255,0.12);
          background: rgba(255,255,255,0.04);
          font-size:12px;
          opacity:0.95;
          line-height: 1;
          white-space: nowrap;
        }

        /* Status row (never overlays: pure flow layout) */
        .status-row { display:flex; gap:10px; flex-wrap:wrap; align-items:center; margin-top: 10px; }
        .status-pill {
          display:inline-flex; align-items:center; gap:8px;
          padding: 6px 10px;
          border-radius:999px;
          border:1px solid rgba(255,255,255,0.12);
          background: rgba(255,255,255,0.04);
          font-size:12px; opacity:0.95;
          line-height: 1;
          white-space: nowrap;
        }
        .dot { width:8px; height:8px; border-radius:999px; background: rgba(255,255,255,0.35); }
        .dot.on { background: rgba(120,255,180,0.85); }
        .dot.run { background: rgba(90,170,255,0.90); }

        /* Reduce default extra spacing under selectbox / inputs inside containers */
        div[data-testid="stVerticalBlock"] > div:has(> div[data-testid="stSelectbox"]) { margin-top: 0.1rem; }
        div[data-testid="stVerticalBlock"] > div:has(> div[data-testid="stTextInput"]) { margin-top: 0.1rem; }

        /* Headings tighter */
        h1, h2, h3 { letter-spacing: -0.01em; }
        </style>
        """,
        unsafe_allow_html=True,
    )

# =========================
# Helpers
# =========================
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

def save_run_outputs(
    video_id: str,
    run_id: str,
    annotations: List[Annotation],
    metrics: RunMetrics,
) -> Tuple[Path, Path]:
    ann_run_dir = (ANN_DIR / video_id / run_id)
    met_run_dir = (MET_DIR / video_id / run_id)
    ann_run_dir.mkdir(parents=True, exist_ok=True)
    met_run_dir.mkdir(parents=True, exist_ok=True)

    ann_dicts = [
        {
            "video_id": a.video_id,
            "clip_index": a.clip_index,
            "start_sec": a.start_sec,
            "end_sec": a.end_sec,
            "description": a.description,
            "extra": a.extra,
        }
        for a in annotations
    ]
    metrics_dict = dataclass_to_dict(metrics)

    (ann_run_dir / "annotations.json").write_text(
        json.dumps(ann_dicts, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (met_run_dir / "metrics.json").write_text(
        json.dumps(metrics_dict, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    run_meta = {"video_id": video_id, "language": (metrics.extra or {}).get("language")}
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
        st.warning(T["playback_warn"])
        st.caption(T["convert_hint"])

@st.cache_data(show_spinner=False)
def make_thumbnail_bytes(video_path_str: str, max_w: int = 640) -> Optional[bytes]:
    """Reads a frame using OpenCV and returns JPEG bytes. Cached by path string, so it stays fast."""
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
    ok2, buf = cv2.imencode(
        ".jpg",
        cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
        [int(cv2.IMWRITE_JPEG_QUALITY), 85],
    )
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

    extra = metrics.extra or {}
    extra["language"] = language
    metrics.extra = extra
    return annotations, metrics

def ensure_index(T: dict) -> None:
    try:
        build_or_update_index(
            ann_root=ANN_DIR,
            index_dir=INDEX_DIR,
            model_name="intfloat/multilingual-e5-base",
        )
        if "qe" in st.session_state:
            del st.session_state["qe"]
        st.success(T["ok"])
    except RuntimeError as e:
        if "sentence-transformers" in str(e):
            st.error(T["e5_missing"])
        else:
            st.error(f"{T['index_build_failed']}: {e}")
    except Exception as e:
        st.error(f"{T['index_build_failed']}: {e}")

def get_engine() -> Optional[QueryEngine]:
    try:
        if "qe" not in st.session_state:
            st.session_state.qe = QueryEngine(index_dir=INDEX_DIR)
        return st.session_state.qe
    except Exception:
        return None

# =========================
# UI blocks
# =========================
def hero(T: dict) -> None:
    st.markdown(
        f"""
        <div class="hero">
            <div style="font-size: 34px; font-weight: 800; line-height: 1.1;">{T['app_title']}</div>
            <div class="muted" style="margin-top: 6px;">{T['app_subtitle']}</div>
            <div class="muted" style="margin-top: 6px; font-size: 14px;">{T['app_flow']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")

def sidebar_runtime_and_index(T: dict) -> None:
    if "pipeline_lang" not in st.session_state:
        st.session_state.pipeline_lang = "ru"
    if "device" not in st.session_state:
        st.session_state.device = "cuda"

    if LOGO_PATH.exists():
        st.sidebar.image(str(LOGO_PATH), use_container_width=True)

    st.sidebar.subheader(T["runtime"])
    st.session_state.pipeline_lang = st.sidebar.selectbox(
        T["model_lang"],
        ["ru", "kz", "en"],
        index=["ru", "kz", "en"].index(st.session_state.pipeline_lang),
        key="pipeline_lang_sb",
    )
    st.session_state.device = st.sidebar.selectbox(
        T["device"],
        ["cuda", "cpu"],
        index=["cuda", "cpu"].index(st.session_state.device),
        key="device_sb",
    )

    st.sidebar.divider()
    st.sidebar.subheader(T["index"])
    st.sidebar.caption(T["index_hint"])

    if st.sidebar.button(T["update_index"], use_container_width=True, key="update_index_sb"):
        with st.spinner(T["building_index"]):
            ensure_index(T)

    st.sidebar.divider()
    with st.sidebar.expander(T["debug"]):
        st.caption(f"raw={RAW_DIR}")
        st.caption(f"annotations={ANN_DIR}")
        st.caption(f"metrics={MET_DIR}")
        st.caption(f"indexes={INDEX_DIR}")

def top_language_bar() -> dict:
    if "ui_lang" not in st.session_state:
        st.session_state.ui_lang = "ru"

    T = UI_TEXT[st.session_state.ui_lang]
    bar_l, bar_r = st.columns([5, 1], vertical_alignment="center")

    with bar_l:
        st.markdown("<div class='topbar-spacer'></div>", unsafe_allow_html=True)

    with bar_r:
        st.markdown("<div class='ui-lang-wrap'>", unsafe_allow_html=True)
        st.session_state.ui_lang = st.selectbox(
            T["ui_lang"],
            options=["ru", "kz", "en"],
            index=["ru", "kz", "en"].index(st.session_state.ui_lang),
            key="ui_lang_top",
            label_visibility="collapsed",
        )
        st.markdown("</div>", unsafe_allow_html=True)

    return UI_TEXT[st.session_state.ui_lang]

def gallery_section(raw_videos: Dict[str, Path], T: dict) -> None:
    st.subheader(T["video_gallery"])

    if not raw_videos:
        st.info(T["no_videos"])
        return

    ids = sorted(raw_videos.keys())

    if "selected_video_id" not in st.session_state:
        st.session_state.selected_video_id = ids[0]
    if st.session_state.selected_video_id not in ids:
        st.session_state.selected_video_id = ids[0]

    mode = st.radio(
        T["view_mode"],
        options=[T["view_grid"], T["view_list"]],
        horizontal=True,
        key="gallery_view_mode",
    )

    if mode == T["view_list"]:
        st.markdown(f"### {T['video_list']}")
        for vid in ids:
            path = raw_videos[vid]
            run_count = len(list_runs_for_video(vid))
            with st.container(border=True):
                c1, c2 = st.columns([4, 1], vertical_alignment="center")
                with c1:
                    st.markdown(f"**{vid}**")
                    st.caption(f"{path.name} · {run_count} runs")
                with c2:
                    if st.button(T["open"], key=f"open_list_{vid}", use_container_width=True):
                        st.session_state.selected_video_id = vid
                        st.rerun()
        st.markdown("---")

    else:
        st.markdown(f"### {T['grid']}")
        cols = st.columns(3, gap="large")
        for i, vid in enumerate(ids):
            path = raw_videos[vid]
            with cols[i % 3]:
                with st.container(border=True):
                    st.markdown(f"**{vid}**")
                    thumb = make_thumbnail_bytes(str(path))
                    if thumb:
                        st.image(thumb, use_container_width=True)
                    else:
                        st.caption(T["no_thumbnail"])
                    run_count = len(list_runs_for_video(vid))
                    st.markdown(f"<span class='pill'>{run_count} runs</span>", unsafe_allow_html=True)
                    if st.button(T["open"], key=f"open_grid_{vid}", use_container_width=True):
                        st.session_state.selected_video_id = vid
                        st.rerun()

        st.write("")
        st.markdown("---")

    selected = st.session_state.selected_video_id
    path = raw_videos[selected]

    st.markdown(f"## {T['selected_video']}: **{selected}**")

    # Preview FIRST
    with st.container(border=True):
        st.markdown(f"### {T['preview']}")
        maybe_playback_warning(path, T)
        st.video(str(path))

    # Convert controls under preview
    ext = path.suffix.lower()
    if ext in {".avi", ".mkv"}:
        c1, c2 = st.columns([1, 1], gap="medium")
        with c1:
            if st.button(T["convert"], key=f"convert_preview_{selected}", use_container_width=True):
                if not ffmpeg_available():
                    st.error(T["ffmpeg_missing"])
                else:
                    with st.spinner(T["converting"]):
                        out_mp4 = convert_to_mp4(path)
                        if out_mp4 is None:
                            st.error(T["conversion_failed"])
                        else:
                            st.success(f"{T['conversion_done']}: {out_mp4.name}")
                            st.rerun()
        with c2:
            st.info(T["convert_hint"])

    # Runs selector UNDER preview
    runs = list_runs_for_video(selected)
    if not runs:
        st.info(T["no_runs"])
        return

    st.markdown(f"### {T['choose_run']}")
    key_run = f"selected_run_{selected}"
    if key_run not in st.session_state:
        st.session_state[key_run] = runs[-1]
    if st.session_state[key_run] not in runs:
        st.session_state[key_run] = runs[-1]

    sel_run = st.selectbox(
        T["run_filter"],
        runs,
        index=runs.index(st.session_state[key_run]),
        key=f"run_select_{selected}",
    )
    st.session_state[key_run] = sel_run

    out = read_run_outputs(selected, sel_run)
    run_lang = out.get("language") or "unknown"

    # language hint
    desired_lang = st.session_state.pipeline_lang
    langs_present: List[str] = []
    for r in runs:
        meta = read_run_outputs(selected, r)
        if meta.get("language"):
            langs_present.append(meta["language"])
    if desired_lang and (desired_lang not in langs_present) and langs_present:
        st.warning(T["lang_missing_label"])

    st.markdown(f"<span class='pill'>{T['run_lang']}: {run_lang}</span>", unsafe_allow_html=True)

    # Global summary separate box
    with st.container(border=True):
        st.markdown(f"### {T['global_summary_box']}")
        if out.get("global_summary"):
            st.write(out["global_summary"])
        else:
            st.caption(T["no_global_summary"])

    # Timeline details below
    with st.container(border=True):
        st.markdown(f"### {T['segments_timeline']}")
        anns = out.get("annotations") or []
        if not anns:
            st.info(T["no_annotations"])
        else:
            for a in anns:
                st.write(f"[{mmss(a['start_sec'])} - {mmss(a['end_sec'])}] {a['description']}")

        with st.expander(T["metrics"]):
            m = out.get("metrics") or {}
            st.write(f"{T['metrics_preprocess']}: {hms(float(m.get('preprocess_time_sec', 0.0) or 0.0))}")
            st.write(f"{T['metrics_model']}: {hms(float(m.get('model_time_sec', 0.0) or 0.0))}")
            st.write(f"{T['metrics_total']}: {hms(float(m.get('total_time_sec', 0.0) or 0.0))}")

def upload_section(T: dict) -> None:
    st.subheader(T["upload"])
    uploaded = st.file_uploader(
        T["drop_here"],
        type=["mp4", "mov", "mkv", "avi"],
        accept_multiple_files=False,
        key="uploader_main",
    )

    if uploaded is None:
        st.info(T["drag_drop_hint"])
        return

    target_path = RAW_DIR / uploaded.name

    with st.container(border=True):
        st.caption(f"{T['target']}: {target_path}")
        c1, c2 = st.columns([1, 2], gap="medium")
        with c1:
            if st.button(T["save_raw"], use_container_width=True, key="save_raw_btn"):
                target_path.write_bytes(uploaded.getbuffer())
                st.success(f"{T['saved']}: {target_path.name}")
                st.rerun()
        with c2:
            st.caption(T["convert_hint"])
            if target_path.suffix.lower() in {".avi", ".mkv"}:
                st.warning(T["playback_warn"])

def _status_pills_html(T: dict, rs: dict) -> str:
    def pill(label: str, on: bool = False, running: bool = False) -> str:
        dot_class = "dot"
        if running:
            dot_class += " run"
        elif on:
            dot_class += " on"
        return f"<div class='status-pill'><span class='{dot_class}'></span><b>{label}</b></div>"

    # Running = blue while rs["running"]
    # Saved / Indexed = green when true
    return (
        "<div class='status-row'>"
        + pill(T["status_running"], running=bool(rs.get("running")))
        + pill(T["status_saved"], on=bool(rs.get("saved")))
        + pill(T["status_indexed"], on=bool(rs.get("indexed")))
        + "</div>"
    )

def run_section(raw_videos: Dict[str, Path], T: dict) -> None:
    """
    Product-like run page:
    - No overlay/phantom bars: only Streamlit containers + flow HTML (no absolute positioning)
    - Controls on top, preview below, logging below preview
    - Status pills always visible and correctly updated
    """
    st.subheader(T["run"])
    if not raw_videos:
        st.info(T["no_videos"])
        return

    ids = sorted(raw_videos.keys())
    if "selected_video_id" not in st.session_state or st.session_state.selected_video_id not in ids:
        st.session_state.selected_video_id = ids[0]

    # status state
    if "run_status" not in st.session_state:
        st.session_state.run_status = {"running": False, "saved": False, "indexed": False}

    # --- TOP CONTROLS PANEL (NO custom outer div -> avoids overlay) ---
    with st.container(border=True):
        # Row 1: video select + kpi pills
        c1, c2 = st.columns([4, 2], vertical_alignment="center", gap="large")

        with c1:
            vid = st.selectbox(
                T["video_filter"],
                ids,
                index=ids.index(st.session_state.selected_video_id),
                key="run_video_select",
            )
            st.session_state.selected_video_id = vid

        with c2:
            st.markdown(
                f"""
                <div class="kpi-row" style="justify-content:flex-end;">
                  <div class="kpi">{T['model_lang']}: <b>{st.session_state.pipeline_lang.upper()}</b></div>
                  <div class="kpi">{T['device']}: <b>{st.session_state.device.upper()}</b></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Row 2: RUN button
        run_clicked = st.button(T["run_pipeline"], type="primary", use_container_width=True, key="run_pipeline_btn")

        # If clicked -> run synchronously, update status flags, save last_run
        if run_clicked:
            st.session_state.run_status = {"running": True, "saved": False, "indexed": False}

            video_path = raw_videos[vid]
            run_id = allocate_run_id(video_path.stem)

            with st.spinner(T["running_pipeline"]):
                annotations, metrics = run_pipeline_on_video(
                    video_path=video_path,
                    device=st.session_state.device,
                    language=st.session_state.pipeline_lang,
                )
                save_run_outputs(
                    video_id=video_path.stem,
                    run_id=run_id,
                    annotations=annotations,
                    metrics=metrics,
                )

            st.session_state["last_run"] = {"video_id": video_path.stem, "run_id": run_id}
            st.session_state.run_status["running"] = False
            st.session_state.run_status["saved"] = True

            with st.spinner(T["updating_index"]):
                ensure_index(T)
            st.session_state.run_status["indexed"] = True
            st.rerun()

        # Row 3: status pills (always after potential run)
        st.markdown(_status_pills_html(T, st.session_state.run_status), unsafe_allow_html=True)

    # --- PREVIEW PANEL ---
    vid = st.session_state.selected_video_id
    path = raw_videos[vid]
    with st.container(border=True):
        st.markdown(f"### {T['preview']}")
        maybe_playback_warning(path, T)
        st.video(str(path))

    # --- LAST RUN LOGGING PANEL (if exists and matches current video) ---
    last = st.session_state.get("last_run")
    if not last or last.get("video_id") != vid:
        return

    out = read_run_outputs(vid, last["run_id"])
    anns = out.get("annotations") or []
    lang = (out.get("language") or "").strip() or "unknown"

    # duration: prefer approx from segments, fallback to 0
    approx_dur = 0.0
    if anns:
        approx_dur = max(float(a.get("end_sec", 0.0)) for a in anns)

    m = out.get("metrics") or {}
    preprocess_sec = float(m.get("preprocess_time_sec", 0.0) or 0.0)
    model_sec = float(m.get("model_time_sec", 0.0) or 0.0)
    total_sec = float(m.get("total_time_sec", 0.0) or 0.0)

    st.write("")
    with st.container(border=True):
        st.markdown(f"### {T['last_run']} · {out['run_id']}")

        st.markdown(
            f"""
            <div class="kpi-row">
              <div class="kpi">{T['duration']}: <b>{hms(approx_dur) if approx_dur > 0 else T['unknown']}</b></div>
              <div class="kpi">{T['segments_count']}: <b>{len(anns)}</b></div>
              <div class="kpi">{T['run_language']}: <b>{lang.upper()}</b></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Summary
        with st.container(border=True):
            st.markdown(f"### {T['global_summary_box']}")
            if out.get("global_summary"):
                st.write(out["global_summary"])
            else:
                st.caption(T["no_global_summary"])

        # Timeline
        with st.container(border=True):
            st.markdown(f"### {T['segments_timeline']}")
            if not anns:
                st.info(T["no_annotations"])
            else:
                for a in anns:
                    st.write(f"[{mmss(a['start_sec'])} - {mmss(a['end_sec'])}] {a['description']}")

        # Metrics in mm:ss (no sec)
        with st.expander(T["metrics"]):
            st.write(f"{T['metrics_preprocess']}: {hms(preprocess_sec)}")
            st.write(f"{T['metrics_model']}: {hms(model_sec)}")
            st.write(f"{T['metrics_total']}: {hms(total_sec)}")

def search_section(raw_videos: Dict[str, Path], runs_map: Dict[str, List[str]], T: dict) -> None:
    st.subheader(T["search"])
    qe = get_engine()
    if qe is None:
        st.warning(T["index_missing"])
        return

    with st.container(border=True):
        q1, q2 = st.columns([3, 1], gap="medium")
        with q1:
            query = st.text_input(
                T["query"],
                placeholder="толпа бежит / адамдар жүгіріп жатыр / crowd running",
                key="search_query",
            )
        with q2:
            top_k = st.number_input(T["topk"], min_value=1, max_value=50, value=10, step=1, key="search_topk")

        f1, f2 = st.columns([1, 1], gap="medium")
        with f1:
            video_options = ["(all)"] + sorted(runs_map.keys())
            sel_video = st.selectbox(T["video_filter"], video_options, index=0, key="search_video_filter")
        with f2:
            run_options = ["(all)"]
            if sel_video != "(all)":
                run_options += runs_map.get(sel_video, [])
            else:
                all_runs = sorted({r for rs in runs_map.values() for r in rs})
                run_options += all_runs
            sel_run = st.selectbox(T["run_filter"], run_options, index=0, key="search_run_filter")

    if not query or not query.strip():
        st.info(T["type_query"])
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
        st.error(f"{T['search_error']}: {e}")
        return

    left, right = st.columns([2, 3], gap="large", vertical_alignment="top")

    with left:
        st.markdown(f"### {T['results']}")
        if not hits:
            st.info(T["no_results"])
        else:
            for idx, h in enumerate(hits):
                with st.container(border=True):
                    st.write(f"[{mmss(h.start_sec)} - {mmss(h.end_sec)}] {h.description}")
                    st.caption(
                        f"{h.video_id} · {h.run_id} · score={h.score:.3f} "
                        f"(bm25={h.sparse_score:.3f}, dense={h.dense_score:.3f})"
                    )
                    if st.button(T["open"], key=f"open_hit_{idx}", use_container_width=True):
                        st.session_state.selected_hit = {
                            "video_id": h.video_id,
                            "run_id": h.run_id,
                            "start_sec": float(h.start_sec),
                            "end_sec": float(h.end_sec),
                            "description": h.description,
                        }

    with right:
        st.markdown(f"### {T['player']}")
        hit = st.session_state.get("selected_hit")
        if not hit:
            st.info(T["pick_result"])
            return

        vid = hit["video_id"]
        start = float(hit["start_sec"])
        desc = hit["description"]
        st.write(f"**{vid}** · {hit['run_id']}")
        st.write(f"[{mmss(start)}] {desc}")

        video_path = raw_videos.get(vid)
        if not video_path or not video_path.exists():
            st.warning(f"{T['raw_missing']}: '{vid}' → {RAW_DIR}")
        else:
            try:
                st.video(str(video_path), start_time=int(start))
            except TypeError:
                st.video(str(video_path))
                st.info(f"{T['seek_hint']}{int(start)} sec ({mmss(start)}).")

def runs_section(runs_map: Dict[str, List[str]], T: dict) -> None:
    st.subheader(T["runs"])
    if not runs_map:
        st.info(T["no_runs_found"])
        return

    for vid, runs in sorted(runs_map.items()):
        with st.expander(f"{vid} — {len(runs)}"):
            st.write(", ".join(runs))

def footer(T: dict) -> None:
    st.markdown("---")
    st.caption(T["footer"])

# =========================
# Main
# =========================
def main() -> None:
    st.set_page_config(page_title="SmartCampus V2T", layout="wide")
    inject_css()

    if "selected_hit" not in st.session_state:
        st.session_state.selected_hit = None

    # Top bar for UI language
    T = top_language_bar()

    # Sidebar minimal: runtime + index
    sidebar_runtime_and_index(T)

    # Hero
    hero(T)

    raw_videos = list_raw_videos()
    runs_map = list_all_runs()

    tabs = st.tabs(T["tabs"])
    with tabs[0]:
        gallery_section(raw_videos, T)
    with tabs[1]:
        upload_section(T)
    with tabs[2]:
        run_section(raw_videos, T)
    with tabs[3]:
        search_section(raw_videos, runs_map, T)
    with tabs[4]:
        runs_section(runs_map, T)

    footer(T)

if __name__ == "__main__":
    main()
