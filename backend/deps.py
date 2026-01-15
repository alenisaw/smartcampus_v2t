# backend/deps.py
"""
Backend dependencies for SmartCampus V2T.

Purpose:
- Load configs/pipeline.yaml and return PipelineConfig + raw dict.
- Resolve backend filesystem paths (jobs/queue/locks/index_state/queue_state).
- Provide safe JSON read/write helpers, timestamps, host id.
- Provide helpers to list videos and runs using the monolith-compatible layout.
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

    raw_dir: Path
    prepared_dir: Path
    runs_dir: Path
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


def get_backend_paths(cfg, raw: Dict[str, Any], project_root: Optional[Path] = None) -> BackendPaths:
    root = project_root or Path(__file__).resolve().parents[1]
    cfg_path = root / "configs" / "pipeline.yaml"

    raw_dir = Path(cfg.paths.raw_dir)
    prepared_dir = Path(cfg.paths.prepared_dir)
    runs_dir = Path(cfg.paths.runs_dir)
    indexes_dir = Path(cfg.paths.indexes_dir)

    jobs_root = root / "data"
    jobs_dir = jobs_root / "jobs"
    queue_dir = jobs_root / "queue"
    locks_dir = jobs_root / "locks"

    index_state_path = indexes_dir / "index_state.json"
    queue_state_path = jobs_root / "queue_state.json"

    for p in [raw_dir, prepared_dir, runs_dir, indexes_dir, jobs_dir, queue_dir, locks_dir]:
        p.mkdir(parents=True, exist_ok=True)

    if not queue_state_path.exists():
        atomic_write_json(queue_state_path, {"paused": False, "updated_at": now_ts()})

    return BackendPaths(
        project_root=root,
        config_path=cfg_path,
        raw_dir=raw_dir,
        prepared_dir=prepared_dir,
        runs_dir=runs_dir,
        indexes_dir=indexes_dir,
        jobs_dir=jobs_dir,
        queue_dir=queue_dir,
        locks_dir=locks_dir,
        index_state_path=index_state_path,
        queue_state_path=queue_state_path,
    )


def new_job_id(prefix: str = "job") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def list_raw_videos(raw_dir: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not raw_dir.exists():
        return out
    for p in sorted(raw_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in {".mp4", ".mov", ".mkv", ".avi"}:
            try:
                st = p.stat()
                out.append(
                    {
                        "video_id": p.stem,
                        "filename": p.name,
                        "path": str(p),
                        "size_bytes": int(st.st_size),
                        "mtime": float(st.st_mtime),
                    }
                )
            except Exception:
                out.append({"video_id": p.stem, "filename": p.name, "path": str(p)})
    return out


def list_runs_for_video(runs_dir: Path, video_id: str) -> List[str]:
    base = runs_dir / video_id
    if not base.exists():
        return []
    runs: List[str] = []
    for p in sorted(base.iterdir()):
        if p.is_dir() and p.name.startswith("run_") and (p / "run_manifest.json").exists():
            runs.append(p.name)
    return runs


def list_all_runs(runs_dir: Path) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    if not runs_dir.exists():
        return out
    for vid_dir in sorted(runs_dir.iterdir()):
        if not vid_dir.is_dir():
            continue
        runs = list_runs_for_video(runs_dir, vid_dir.name)
        if runs:
            out[vid_dir.name] = runs
    return out


def read_run_outputs(runs_dir: Path, video_id: str, run_id: str) -> Dict[str, Any]:
    run_dir = runs_dir / video_id / run_id
    out: Dict[str, Any] = {
        "video_id": video_id,
        "run_id": run_id,
        "manifest": None,
        "annotations": [],
        "metrics": None,
        "global_summary": None,
        "language": None,
        "device": None,
    }

    mf = run_dir / "run_manifest.json"
    ap = run_dir / "annotations.json"
    mp = run_dir / "metrics.json"
    if mf.exists():
        out["manifest"] = read_json(mf, default=None)
        if isinstance(out["manifest"], dict):
            out["language"] = out["manifest"].get("language")
            out["device"] = out["manifest"].get("device")

    if ap.exists():
        out["annotations"] = read_json(ap, default=[]) or []

    if mp.exists():
        met = read_json(mp, default=None)
        out["metrics"] = met
        if isinstance(met, dict):
            extra = met.get("extra") or {}
            out["global_summary"] = extra.get("global_summary")
            out["language"] = out["language"] or extra.get("language")
            out["device"] = out["device"] or extra.get("device")

    return out
