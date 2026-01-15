# backend/worker.py
"""
Background worker for SmartCampus V2T.

Purpose:
- Consume filesystem queue (data/queue/*.q).
- Maintain per-job lock + lease (Windows-safe).
- Run preprocess -> clips -> inference -> write run outputs.
- Optionally update search index after successful run.
- Support queue pause/resume via data/queue_state.json.
"""
from __future__ import annotations

import os
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Optional, List

from backend.deps import (
    atomic_write_json,
    get_backend_paths,
    host_id,
    load_cfg_and_raw,
    now_ts,
    read_json,
)

from src.preprocessing.video_io import preprocess_video
from src.pipeline.video_to_text import VideoToTextPipeline, build_clips_from_video_meta
from src.search import build_or_update_index


def _list_queue_items(queue_dir: Path) -> List[Path]:
    return sorted(queue_dir.glob("p*__*__*.q"))


def _parse_job_id_from_queue_item(p: Path) -> Optional[str]:
    try:
        parts = p.name.split("__")
        if len(parts) < 3:
            return None
        job_part = parts[2]
        if job_part.endswith(".q"):
            job_part = job_part[:-2]
        return job_part
    except Exception:
        return None


def _job_path(paths, job_id: str) -> Path:
    return paths.jobs_dir / f"{job_id}.json"


def _read_job(paths, job_id: str) -> Dict[str, Any]:
    p = _job_path(paths, job_id)
    job = read_json(p, default=None)
    if not isinstance(job, dict):
        raise RuntimeError(f"Job file missing or corrupted: {p}")
    return job


def _write_job(paths, job: Dict[str, Any]) -> None:
    atomic_write_json(_job_path(paths, job["job_id"]), job)


def _job_state(job: Dict[str, Any]) -> str:
    return str(job.get("state", "queued"))


def _is_terminal(job: Dict[str, Any]) -> bool:
    return _job_state(job) in {"done", "failed", "canceled"}


def _lock_path(paths, job_id: str) -> Path:
    return paths.locks_dir / f"{job_id}.lock"


def _try_create_lock(paths, job_id: str, owner: str) -> bool:
    lp = _lock_path(paths, job_id)
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    try:
        fd = os.open(str(lp), flags)
        try:
            os.write(fd, f"{owner}\n{now_ts()}\n".encode("utf-8", errors="ignore"))
        finally:
            os.close(fd)
        return True
    except FileExistsError:
        return False
    except Exception:
        return False


def _remove_lock_if_exists(paths, job_id: str) -> None:
    try:
        _lock_path(paths, job_id).unlink(missing_ok=True)
    except Exception:
        pass


def _lease_expired(job: Dict[str, Any]) -> bool:
    lease = job.get("lease") or {}
    exp = lease.get("expires_at")
    try:
        return (exp is None) or (float(exp) <= now_ts())
    except Exception:
        return True


def _try_lease(paths, job_id: str, owner: str, lease_sec: float) -> bool:
    job = _read_job(paths, job_id)
    if _is_terminal(job):
        return False

    lease = job.get("lease") or {}
    cur_owner = lease.get("owner")
    if cur_owner and cur_owner != owner and not _lease_expired(job):
        return False

    job["lease"] = {"owner": owner, "expires_at": now_ts() + float(lease_sec)}
    job["state"] = "leased"
    job["stage"] = "leased"
    job["updated_at"] = now_ts()
    _write_job(paths, job)
    return True


def _check_cancel(paths, job_id: str) -> bool:
    job = _read_job(paths, job_id)
    return bool(job.get("cancel_requested", False))


def _set_state(
    paths,
    job_id: str,
    state: str,
    *,
    stage: Optional[str] = None,
    progress: Optional[float] = None,
    message: Optional[str] = None,
    run_id: Optional[str] = None,
    error: Optional[str] = None,
) -> None:
    job = _read_job(paths, job_id)
    job["state"] = state
    if stage is not None:
        job["stage"] = stage
    if progress is not None:
        job["progress"] = float(progress)
    if message is not None:
        job["message"] = message
    if run_id is not None:
        job["run_id"] = run_id
    job["updated_at"] = now_ts()
    if state == "running" and job.get("started_at") is None:
        job["started_at"] = now_ts()
    if state in {"done", "failed", "canceled"}:
        job["finished_at"] = now_ts()
    if error is not None:
        job["error"] = {"type": "WorkerError", "message": error}
    _write_job(paths, job)


def _allocate_run_id(runs_root: Path, video_id: str) -> str:
    vdir = runs_root / video_id
    vdir.mkdir(parents=True, exist_ok=True)
    existing = sorted([p.name for p in vdir.glob("run_*") if p.is_dir()])
    nums: List[int] = []
    for name in existing:
        try:
            nums.append(int(name.split("_")[1]))
        except Exception:
            pass
    nxt = (max(nums) + 1) if nums else 1
    return f"run_{nxt:03d}"


def _run_dir(runs_root: Path, video_id: str, run_id: str) -> Path:
    return runs_root / video_id / run_id


def _pick_video_path_by_stem(root: Path, stem: str) -> Optional[Path]:
    if not root.exists():
        return None
    cands = list(root.glob(f"{stem}.*"))
    if not cands:
        return None

    pref = [".mp4", ".mov", ".m4v", ".mkv", ".avi"]

    def key(p: Path):
        ext = p.suffix.lower()
        try:
            return (pref.index(ext), p.name.lower())
        except ValueError:
            return (len(pref), p.name.lower())

    cands.sort(key=key)
    return cands[0]


def _write_run_outputs(
    run_dir: Path,
    *,
    manifest: Dict[str, Any],
    config: Dict[str, Any],
    annotations: Any,
    metrics: Any,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(run_dir / "run_manifest.json", manifest)
    atomic_write_json(run_dir / "config.json", config)
    atomic_write_json(run_dir / "annotations.json", annotations)
    atomic_write_json(run_dir / "metrics.json", metrics)


def _queue_paused(paths) -> bool:
    stt = read_json(paths.queue_state_path, default={"paused": False}) or {}
    return bool(stt.get("paused", False))


def worker_main() -> None:
    cfg, raw = load_cfg_and_raw()
    paths = get_backend_paths(cfg, raw)

    paths.jobs_dir.mkdir(parents=True, exist_ok=True)
    paths.queue_dir.mkdir(parents=True, exist_ok=True)
    paths.locks_dir.mkdir(parents=True, exist_ok=True)

    wcfg = raw.get("worker") or {}
    poll_interval = float(wcfg.get("poll_interval_sec", 1))
    lease_sec = float(wcfg.get("lease_sec", 30))
    auto_index = bool((raw.get("index") or {}).get("auto_update_on_done", True))

    owner = host_id()
    pipeline = VideoToTextPipeline(cfg)

    print(f"[worker] owner={owner} jobs={paths.jobs_dir} queue={paths.queue_dir} locks={paths.locks_dir}")
    active_job_id: Optional[str] = None

    while True:
        try:
            if _queue_paused(paths):
                time.sleep(max(0.4, poll_interval))
                continue

            items = _list_queue_items(paths.queue_dir)
            if not items:
                time.sleep(poll_interval)
                continue

            qitem = items[0]
            job_id = _parse_job_id_from_queue_item(qitem)
            if not job_id:
                try:
                    qitem.unlink(missing_ok=True)
                except Exception:
                    pass
                continue

            if not _try_create_lock(paths, job_id, owner):
                try:
                    job = _read_job(paths, job_id)
                    if _lease_expired(job):
                        _remove_lock_if_exists(paths, job_id)
                except Exception:
                    pass
                time.sleep(0.1)
                continue

            try:
                qitem.unlink(missing_ok=True)
            except Exception:
                pass

            if not _try_lease(paths, job_id, owner, lease_sec):
                _remove_lock_if_exists(paths, job_id)
                continue

            active_job_id = job_id
            _set_state(paths, job_id, "running", stage="starting", progress=0.01, message="Started")

            job = _read_job(paths, job_id)
            extra = job.get("extra") or {}
            index_only = bool(extra.get("index_only", False))
            language = extra.get("language") or cfg.model.language
            device = extra.get("device") or cfg.model.device
            force_overwrite = bool(extra.get("force_overwrite", False))
            overwrite_run_id = extra.get("overwrite_run_id")

            if _check_cancel(paths, job_id):
                _set_state(paths, job_id, "canceled", stage="canceled", progress=0.0, message="Canceled before start")
                _remove_lock_if_exists(paths, job_id)
                active_job_id = None
                continue

            if index_only:
                _set_state(paths, job_id, "indexing", stage="indexing", progress=0.2, message="Index rebuild (index_only)")
                build_or_update_index(
                    runs_root=Path(cfg.paths.runs_dir),
                    index_dir=Path(cfg.paths.indexes_dir),
                    model_name=cfg.search.embed_model_name,
                )
                stt = {"built_at": now_ts(), "updated_at": now_ts(), "version": now_ts(), "last_error": None}
                atomic_write_json(paths.index_state_path, stt)
                _set_state(paths, job_id, "done", stage="done", progress=1.0, message="Index rebuilt")
                _remove_lock_if_exists(paths, job_id)
                active_job_id = None
                continue

            video_id = job.get("video_id")
            video_path = _pick_video_path_by_stem(Path(cfg.paths.raw_dir), str(video_id))
            if video_path is None:
                video_path = _pick_video_path_by_stem(Path(cfg.paths.prepared_dir), str(video_id))
            if video_path is None:
                raise RuntimeError(f"Video not found for job video_id={video_id}")

            cfg.model.device = str(device)
            cfg.model.language = str(language)

            _set_state(paths, job_id, "running", stage="preprocess", progress=0.08, message="Preprocessing")
            video_meta = preprocess_video(video_path, cfg)

            if _check_cancel(paths, job_id):
                _set_state(paths, job_id, "canceled", stage="canceled", progress=0.0, message="Canceled after preprocess")
                _remove_lock_if_exists(paths, job_id)
                active_job_id = None
                continue

            clips, clip_ts = build_clips_from_video_meta(
                video_meta=video_meta,
                window_sec=cfg.clips.window_sec,
                stride_sec=cfg.clips.stride_sec,
                min_clip_frames=cfg.clips.min_clip_frames,
                max_clip_frames=cfg.clips.max_clip_frames,
            )

            _set_state(paths, job_id, "running", stage="inference", progress=0.20, message="Inference")

            annotations, metrics = pipeline.run(
                video_id=video_meta.video_id,
                video_duration_sec=float(video_meta.duration_sec),
                clips=clips,
                clip_timestamps=clip_ts,
                preprocess_time_sec=float((video_meta.extra or {}).get("preprocess_time_sec", 0.0)),
            )

            if _check_cancel(paths, job_id):
                _set_state(paths, job_id, "canceled", stage="canceled", progress=0.0, message="Canceled after inference")
                _remove_lock_if_exists(paths, job_id)
                active_job_id = None
                continue

            runs_root = Path(cfg.paths.runs_dir)
            if overwrite_run_id and force_overwrite:
                run_id = str(overwrite_run_id)
            else:
                run_id = _allocate_run_id(runs_root, video_meta.video_id)

            _set_state(paths, job_id, "running", stage="saving", progress=0.85, message="Saving outputs", run_id=run_id)
            out_dir = _run_dir(runs_root, video_meta.video_id, run_id)

            manifest = {
                "video_id": video_meta.video_id,
                "run_id": run_id,
                "created_at": now_ts(),
                "status": "done",
                "language": str(language),
                "device": str(device),
            }

            config_snapshot = {
                "config_path": str(paths.config_path),
                "effective": {
                    "model": {
                        "device": str(device),
                        "language": str(language),
                        "dtype": str(cfg.model.dtype),
                        "batch_size": int(cfg.model.batch_size),
                        "max_new_tokens": int(cfg.model.max_new_tokens),
                    },
                    "search": {
                        "embed_model_name": str(cfg.search.embed_model_name),
                        "w_bm25": float(cfg.search.w_bm25),
                        "w_dense": float(cfg.search.w_dense),
                    },
                },
            }

            ann_payload: List[Dict[str, Any]] = []
            for a in annotations or []:
                if isinstance(a, dict):
                    ann_payload.append(a)
                else:
                    ann_payload.append(
                        {
                            "video_id": getattr(a, "video_id", video_meta.video_id),
                            "start_sec": float(getattr(a, "start_sec", 0.0)),
                            "end_sec": float(getattr(a, "end_sec", 0.0)),
                            "description": str(getattr(a, "description", "")),
                            "extra": getattr(a, "extra", None),
                        }
                    )

            met_payload = (
                metrics
                if isinstance(metrics, dict)
                else getattr(metrics, "model_dump", lambda: None)() or getattr(metrics, "__dict__", {})
            )
            _write_run_outputs(out_dir, manifest=manifest, config=config_snapshot, annotations=ann_payload, metrics=met_payload)

            if auto_index:
                _set_state(paths, job_id, "indexing", stage="indexing", progress=0.92, message="Index update")
                build_or_update_index(
                    runs_root=Path(cfg.paths.runs_dir),
                    index_dir=Path(cfg.paths.indexes_dir),
                    model_name=cfg.search.embed_model_name,
                )
                stt = {"built_at": now_ts(), "updated_at": now_ts(), "version": now_ts(), "last_error": None}
                atomic_write_json(paths.index_state_path, stt)

            _set_state(paths, job_id, "done", stage="done", progress=1.0, message="Done", run_id=run_id)

            _remove_lock_if_exists(paths, job_id)
            active_job_id = None

        except Exception as e:
            err = f"{e}\n{traceback.format_exc()}"
            if active_job_id:
                try:
                    _set_state(
                        paths,
                        active_job_id,
                        "failed",
                        stage="failed",
                        progress=0.0,
                        message="Failed",
                        error=str(e),
                    )
                except Exception:
                    pass
                try:
                    _remove_lock_if_exists(paths, active_job_id)
                except Exception:
                    pass
                active_job_id = None

            try:
                stt = read_json(paths.index_state_path, default={}) or {}
                stt["updated_at"] = now_ts()
                stt["last_error"] = str(e)
                atomic_write_json(paths.index_state_path, stt)
            except Exception:
                pass

            print(f"[worker] ERROR:\n{err}")
            time.sleep(0.5)


if __name__ == "__main__":
    worker_main()
