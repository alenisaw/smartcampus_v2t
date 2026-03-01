# backend/deps.py
"""
Backend dependencies for SmartCampus V2T.

Purpose:
- Load configs/pipeline.yaml and return PipelineConfig + raw dict.
- Resolve backend filesystem paths (jobs/queue/locks/index_state/queue_state).
- Provide safe JSON read/write helpers, timestamps, host id.
- Provide helpers to list videos and outputs using the per-video layout.
"""
from __future__ import annotations

import json
import os
import socket
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, List

import yaml

from src.utils.config_loader import load_pipeline_config


def now_ts() -> float:
    return float(time.time())


def host_id() -> str:
    try:
        hn = socket.gethostname()
    except Exception:
        hn = "host"
    return f"{hn}:{os.getpid()}"


def read_json(path: Path, default: Any = None) -> Any:
    try:
        if not path.exists():
            return default
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


@dataclass(frozen=True)
class BackendPaths:
    project_root: Path
    config_path: Path

    videos_dir: Path
    indexes_dir: Path

    jobs_dir: Path
    queue_dir: Path
    locks_dir: Path

    index_state_path: Path
    queue_state_path: Path


def load_cfg_and_raw(project_root: Optional[Path] = None, cfg_path: Optional[Path] = None):
    root = project_root or Path(__file__).resolve().parents[1]
    cpath = cfg_path or (root / "configs" / "pipeline.yaml")
    raw = yaml.safe_load(cpath.read_text(encoding="utf-8"))

    cfg = load_pipeline_config(cpath)
    return cfg, raw


def _resolve_path(root: Path, value: Optional[str], default_rel: str) -> Path:
    if value is None or str(value).strip() == "":
        return (root / default_rel).resolve()
    p = Path(str(value))
    return p if p.is_absolute() else (root / p).resolve()


def get_backend_paths(cfg, raw: Dict[str, Any], project_root: Optional[Path] = None) -> BackendPaths:
    root = project_root or Path(__file__).resolve().parents[1]
    cfg_path = root / "configs" / "pipeline.yaml"

    videos_dir = Path(cfg.paths.videos_dir)
    indexes_dir = Path(cfg.paths.indexes_dir)

    jobs_raw = raw.get("jobs") or {}
    queue_raw = raw.get("queue") or {}
    locks_raw = raw.get("locks") or {}

    jobs_dir = _resolve_path(root, jobs_raw.get("dir"), "data/jobs")
    queue_dir = _resolve_path(root, queue_raw.get("dir"), "data/queue")
    locks_dir = _resolve_path(root, locks_raw.get("dir"), "data/locks")

    index_state_path = indexes_dir / "index_state.json"
    queue_state_path = (root / "data" / "queue_state.json").resolve()

    for p in [videos_dir, indexes_dir, jobs_dir, queue_dir, locks_dir]:
        p.mkdir(parents=True, exist_ok=True)

    if not queue_state_path.exists():
        atomic_write_json(queue_state_path, {"paused": False, "updated_at": now_ts()})

    return BackendPaths(
        project_root=root,
        config_path=cfg_path,
        videos_dir=videos_dir,
        indexes_dir=indexes_dir,
        jobs_dir=jobs_dir,
        queue_dir=queue_dir,
        locks_dir=locks_dir,
        index_state_path=index_state_path,
        queue_state_path=queue_state_path,
    )


def new_job_id(prefix: str = "job") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def list_videos(videos_dir: Path) -> List[Dict[str, Any]]:
    from src.utils.video_store import list_videos as _list_videos

    return _list_videos(Path(videos_dir))


def read_video_outputs(videos_dir: Path, video_id: str, lang: str) -> Dict[str, Any]:
    from src.utils.video_store import (
        outputs_manifest_path,
        metrics_path,
        read_metrics,
        read_segments,
        read_summary,
        segments_path,
        summary_path,
    )

    out: Dict[str, Any] = {
        "video_id": video_id,
        "language": lang,
        "manifest": None,
        "annotations": [],
        "metrics": None,
        "global_summary": None,
    }

    mp = outputs_manifest_path(Path(videos_dir), video_id)
    out["manifest"] = read_json(mp, default=None)

    seg = segments_path(Path(videos_dir), video_id, lang)
    out["annotations"] = read_segments(seg)

    sp = summary_path(Path(videos_dir), video_id, lang)
    summ = read_summary(sp)
    if isinstance(summ, dict):
        out["global_summary"] = summ.get("summary")

    met = read_metrics(metrics_path(Path(videos_dir), video_id))
    if isinstance(met, dict):
        out["metrics"] = met

    return out
