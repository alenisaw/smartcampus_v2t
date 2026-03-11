# backend/deps.py
"""
Backend dependency helpers for SmartCampus V2T.

Purpose:
- Load effective config and resolve backend filesystem paths.
- Provide shared JSON, timestamp, and output-loading helpers for backend modules.
"""

from __future__ import annotations

import json
import os
import socket
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils.config_loader import load_pipeline_bundle


def now_ts() -> float:
    return float(time.time())


def host_id() -> str:
    try:
        hostname = socket.gethostname()
    except Exception:
        hostname = "host"
    return f"{hostname}:{os.getpid()}"


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


def load_cfg_and_raw(
    project_root: Optional[Path] = None,
    cfg_path: Optional[Path] = None,
    *,
    profile: Optional[str] = None,
    variant: Optional[str] = None,
):
    """Load the merged config and its raw form for backend and worker entrypoints."""

    root = project_root or Path(__file__).resolve().parents[1]
    cpath = cfg_path or (root / "configs" / "profiles" / "main.yaml")
    cfg, raw = load_pipeline_bundle(cpath, profile=profile, variant=variant)
    return cfg, raw


def _resolve_path(root: Path, value: Optional[str], default_rel: str) -> Path:
    if value is None or str(value).strip() == "":
        return (root / default_rel).resolve()
    path = Path(str(value))
    return path if path.is_absolute() else (root / path).resolve()


def get_backend_paths(cfg, raw: Dict[str, Any], project_root: Optional[Path] = None) -> BackendPaths:
    root = project_root or Path(__file__).resolve().parents[1]
    runtime_context = raw.get("runtime_context") or {}
    cfg_path = Path(str(runtime_context.get("config_path") or (root / "configs" / "profiles" / "main.yaml"))).resolve()

    videos_dir = Path(cfg.paths.videos_dir)
    indexes_dir = Path(cfg.paths.indexes_dir)

    jobs_dir = Path(getattr(cfg.jobs, "dir", _resolve_path(root, None, "data/jobs")))
    queue_dir = Path(getattr(cfg.queue, "dir", _resolve_path(root, None, "data/queue")))
    locks_dir = Path(getattr(cfg.locks, "dir", _resolve_path(root, None, "data/locks")))

    index_state_path = indexes_dir / "index_state.json"
    queue_state_path = (root / "data" / "queue_state.json").resolve()

    for path in [videos_dir, indexes_dir, jobs_dir, queue_dir, locks_dir]:
        path.mkdir(parents=True, exist_ok=True)

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


def read_video_outputs(videos_dir: Path, video_id: str, lang: str, variant: Optional[str] = None) -> Dict[str, Any]:
    from src.utils.video_store import (
        batch_manifest_path,
        metrics_path,
        outputs_manifest_path,
        read_metrics,
        read_segments,
        read_summary,
        run_manifest_path,
        segments_path,
        summary_path,
    )

    out: Dict[str, Any] = {
        "video_id": video_id,
        "language": lang,
        "variant": variant,
        "manifest": None,
        "run_manifest": None,
        "batch_manifest": None,
        "annotations": [],
        "metrics": None,
        "global_summary": None,
    }

    out["manifest"] = read_json(outputs_manifest_path(Path(videos_dir), video_id, variant=variant), default=None)
    out["run_manifest"] = read_json(run_manifest_path(Path(videos_dir), video_id, variant=variant), default=None)
    out["batch_manifest"] = read_json(batch_manifest_path(Path(videos_dir), video_id), default=None)

    seg_path = segments_path(Path(videos_dir), video_id, lang, variant=variant)
    out["annotations"] = read_segments(seg_path)

    summary_obj = read_summary(summary_path(Path(videos_dir), video_id, lang, variant=variant))
    if isinstance(summary_obj, dict):
        out["global_summary"] = summary_obj.get("global_summary", summary_obj.get("summary"))

    metrics_obj = read_metrics(metrics_path(Path(videos_dir), video_id, variant=variant))
    if isinstance(metrics_obj, dict):
        out["metrics"] = metrics_obj

    return out
