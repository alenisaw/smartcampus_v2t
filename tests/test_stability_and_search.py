from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

import backend.api as api
from backend.deps import atomic_write_json
from backend.jobs.store import build_job_record, write_job
from backend.retrieval_runtime import append_unique_hits
from backend.schemas import SearchHit
from backend.jobs.process_runtime import _build_process_context
from src.utils.video_store import (
    clip_observations_path,
    metrics_path,
    outputs_manifest_path,
    run_manifest_path,
    segments_path,
    summary_path,
    validate_process_outputs,
    validate_translation_outputs,
    write_metrics,
    write_run_manifest,
    write_segments,
    write_summary,
)


def _paths(root: Path) -> SimpleNamespace:
    return SimpleNamespace(
        videos_dir=root / "videos",
        jobs_dir=root / "jobs",
        queue_dir=root / "queue",
        locks_dir=root / "locks",
        index_state_path=root / "indexes" / "index_state.json",
        queue_state_path=root / "queue_state.json",
    )


def _prepare_dirs(paths: SimpleNamespace) -> None:
    for path in [paths.videos_dir, paths.jobs_dir, paths.queue_dir, paths.locks_dir, paths.index_state_path.parent]:
        path.mkdir(parents=True, exist_ok=True)
    atomic_write_json(paths.queue_state_path, {"paused": False, "updated_at": 0.0})


def _video_root(videos_dir: Path, video_id: str) -> Path:
    raw_dir = videos_dir / video_id / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / f"{video_id}.mp4").write_bytes(b"video")
    return raw_dir.parent


def test_validate_process_outputs_detects_missing_clip_observations(tmp_path: Path) -> None:
    videos_dir = tmp_path / "videos"
    video_id = "v1"
    _video_root(videos_dir, video_id)
    write_segments(segments_path(videos_dir, video_id, "en"), [{"segment_id": "seg_1", "start_sec": 0.0, "end_sec": 1.0, "extra": {"merged_from": [0]}}])
    write_summary(summary_path(videos_dir, video_id, "en"), {"global_summary": "ok"}, "en")
    write_metrics(metrics_path(videos_dir, video_id), {"video_duration_sec": 10.0})
    write_run_manifest(videos_dir, video_id, {"profile": "main", "language": "en", "status": "ready"})
    atomic_write_json(outputs_manifest_path(videos_dir, video_id), {"languages": {"en": {"status": "ready"}}})

    health = validate_process_outputs(videos_dir, video_id, "en")

    assert health["complete"] is False
    assert health["classification"] == "manifest_ready_but_incomplete"
    assert "clip_observations" in health["missing_artifacts"]


def test_validate_translation_outputs_detects_missing_summary(tmp_path: Path) -> None:
    videos_dir = tmp_path / "videos"
    video_id = "v1"
    _video_root(videos_dir, video_id)
    write_segments(segments_path(videos_dir, video_id, "ru"), [{"segment_id": "seg_1", "start_sec": 0.0, "end_sec": 1.0}])
    atomic_write_json(outputs_manifest_path(videos_dir, video_id), {"languages": {"ru": {"status": "ready"}}})

    health = validate_translation_outputs(videos_dir, video_id, "ru")

    assert health["complete"] is False
    assert health["classification"] == "manifest_ready_but_incomplete"
    assert "summary" in health["missing_artifacts"]


def test_append_unique_hits_dedupes_across_languages() -> None:
    hits = [
        SimpleNamespace(video_id="v1", variant=None, segment_id="seg_1", start_sec=0.0, end_sec=1.0, language="en", description="a", score=1.0, sparse_score=1.0, dense_score=1.0, tags=[], objects=[], anomaly_flag=False),
        SimpleNamespace(video_id="v1", variant=None, segment_id="seg_1", start_sec=0.0, end_sec=1.0, language="ru", description="b", score=0.9, sparse_score=0.9, dense_score=0.9, tags=[], objects=[], anomaly_flag=False),
    ]
    out: list[SearchHit] = []
    seen: set = set()

    append_unique_hits(out, seen, hits, limit=10)

    assert len(out) == 1
    assert out[0].segment_id == "seg_1"


def test_jobs_cancel_and_delete_video_lifecycle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _paths(tmp_path)
    _prepare_dirs(paths)
    _video_root(paths.videos_dir, "v1")
    context = SimpleNamespace(paths=paths, cfg=SimpleNamespace())
    monkeypatch.setattr(api, "get_api_context", lambda: context)

    queued_job = build_job_record(
        job_id="job_queued",
        video_id="v1",
        job_type="process",
        profile="main",
        variant=None,
        language="en",
        source_language=None,
    )
    write_job(paths, queued_job)
    (paths.queue_dir / "p010__t__job_queued.q").write_text("job_queued", encoding="utf-8")

    canceled = api.jobs_cancel("job_queued")
    assert canceled.state == "canceled"

    queued_for_delete = build_job_record(
        job_id="job_delete",
        video_id="v1",
        job_type="process",
        profile="main",
        variant=None,
        language="en",
        source_language=None,
    )
    write_job(paths, queued_for_delete)
    (paths.queue_dir / "p010__t__job_delete.q").write_text("job_delete", encoding="utf-8")

    assert api.videos_delete("v1") == {"ok": True}
    assert not (paths.videos_dir / "v1").exists()
    assert not (paths.jobs_dir / "job_delete.json").exists()


def test_videos_delete_rejects_running_job(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _paths(tmp_path)
    _prepare_dirs(paths)
    _video_root(paths.videos_dir, "v2")
    context = SimpleNamespace(paths=paths, cfg=SimpleNamespace())
    monkeypatch.setattr(api, "get_api_context", lambda: context)

    running = build_job_record(
        job_id="job_running",
        video_id="v2",
        job_type="process",
        profile="main",
        variant=None,
        language="en",
        source_language=None,
    )
    running["state"] = "running"
    running["stage"] = "inference"
    write_job(paths, running)

    with pytest.raises(HTTPException) as exc:
        api.videos_delete("v2")
    assert exc.value.status_code == 409


def test_build_process_context_does_not_mutate_cfg(tmp_path: Path) -> None:
    videos_dir = tmp_path / "videos"
    _video_root(videos_dir, "v3")
    cfg = SimpleNamespace(
        paths=SimpleNamespace(videos_dir=str(videos_dir)),
        translation=SimpleNamespace(source_lang="en"),
        model=SimpleNamespace(language="en", device="cuda"),
        active_variant=None,
    )
    context = _build_process_context(
        cfg=cfg,
        paths=SimpleNamespace(),
        cfg_fp="fp",
        job_id="job",
        job={"video_id": "v3", "extra": {}},
        device="cpu",
        force_overwrite=False,
        auto_index=False,
        webhook_cfg={},
        services=SimpleNamespace(),
    )

    assert context.device == "cpu"
    assert cfg.model.device == "cuda"
    assert cfg.model.language == "en"
