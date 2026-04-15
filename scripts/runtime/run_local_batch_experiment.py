# scripts/runtime/run_local_batch_experiment.py
"""
Purpose:
- Import loose videos from `data/videos` into the structured library layout.
- Enqueue process jobs for videos that do not yet have ready outputs.
- Wait for the local worker to finish base + translation outputs.
- Export a diploma-friendly metrics bundle and a subset zip for the processed batch.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.deps import atomic_write_json, get_backend_paths, load_cfg_and_raw, now_ts, read_json
from backend.http.common import normalize_uploaded_video
from backend.jobs.control import create_job
from scripts.collect_metrics import export_metrics_bundle
from src.utils.video_store import (
    VIDEO_EXTS,
    ensure_video_dirs,
    metrics_path,
    outputs_manifest_path,
    pick_video_file,
    run_manifest_path,
    summary_path,
    segments_path,
    clip_observations_path,
    video_manifest_path,
)


def _read_json(path: Path, default: Any = None) -> Any:
    return read_json(path, default=default)


def _is_loose_video(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in VIDEO_EXTS


def _structured_video_ids(videos_dir: Path) -> set[str]:
    out: set[str] = set()
    for item in sorted(videos_dir.iterdir()):
        if not item.is_dir():
            continue
        if pick_video_file(item / "raw") is not None:
            out.add(item.name)
    return out


def _import_loose_videos(videos_dir: Path) -> List[str]:
    imported: List[str] = []
    existing_ids = _structured_video_ids(videos_dir)
    for path in sorted(videos_dir.iterdir()):
        if not _is_loose_video(path):
            continue
        video_id = path.stem
        if video_id in existing_ids:
            continue

        dirs = ensure_video_dirs(videos_dir, video_id)
        target = dirs["raw"] / path.name
        shutil.copy2(path, target)
        target = normalize_uploaded_video(target)
        st = target.stat()
        atomic_write_json(
            video_manifest_path(videos_dir, video_id),
            {
                "video_id": video_id,
                "filename": target.name,
                "path": str(target),
                "size_bytes": int(st.st_size),
                "mtime": float(st.st_mtime),
                "uploaded_at": now_ts(),
            },
        )
        imported.append(video_id)
        existing_ids.add(video_id)
    return imported


def _language_ready(videos_dir: Path, video_id: str, lang: str) -> bool:
    manifest = _read_json(outputs_manifest_path(videos_dir, video_id), default=None)
    if not isinstance(manifest, dict):
        return False
    languages = manifest.get("languages")
    if not isinstance(languages, dict):
        return False
    payload = languages.get(lang)
    if not isinstance(payload, dict):
        return False
    return str(payload.get("status") or "").strip().lower() == "ready"


def _video_fully_ready(videos_dir: Path, video_id: str) -> bool:
    if not metrics_path(videos_dir, video_id).exists():
        return False
    return all(_language_ready(videos_dir, video_id, lang) for lang in ("en", "ru", "kz"))


def _discover_unprocessed(videos_dir: Path) -> List[str]:
    targets: List[str] = []
    for video_id in sorted(_structured_video_ids(videos_dir)):
        if not _video_fully_ready(videos_dir, video_id):
            targets.append(video_id)
    return targets


def _enqueue_targets(paths: Any, cfg: Any, video_ids: Iterable[str], profile: str, variant: Optional[str], videos_dir: Path) -> List[Dict[str, Any]]:
    jobs: List[Dict[str, Any]] = []
    for video_id in video_ids:
        en_ready = _language_ready(videos_dir, video_id, "en")
        if not en_ready or not metrics_path(videos_dir, video_id).exists():
            job = create_job(
                paths,
                video_id=video_id,
                job_type="process",
                profile=profile,
                variant=variant,
                language=str(cfg.model.language),
                extra={"job_type": "process", "profile": profile, "variant": variant},
            )
            jobs.append(job)
            continue

        for tgt_lang in ("ru", "kz"):
            if _language_ready(videos_dir, video_id, tgt_lang):
                continue
            job = create_job(
                paths,
                video_id=video_id,
                job_type="translate",
                profile=profile,
                variant=variant,
                language=tgt_lang,
                source_language="en",
                extra={
                    "job_type": "translate",
                    "profile": profile,
                    "variant": variant,
                    "language": tgt_lang,
                    "source_language": "en",
                },
                priority="020",
            )
            jobs.append(job)
    return jobs


def _pending_videos(videos_dir: Path, targets: Iterable[str]) -> List[str]:
    return [video_id for video_id in sorted(set(targets)) if not _video_fully_ready(videos_dir, video_id)]


def _wait_for_targets(videos_dir: Path, targets: List[str], poll_sec: float, timeout_sec: float) -> List[str]:
    deadline = time.time() + float(timeout_sec)
    while True:
        pending = _pending_videos(videos_dir, targets)
        if not pending:
            return []
        if time.time() >= deadline:
            return pending
        print(f"Pending videos: {', '.join(pending)}", flush=True)
        time.sleep(float(poll_sec))


def _copy_if_exists(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_file():
        shutil.copy2(src, dst)


def _copy_tree_files(src_dir: Path, dst_dir: Path) -> None:
    if not src_dir.exists():
        return
    for file_path in src_dir.rglob("*"):
        if not file_path.is_file():
            continue
        rel = file_path.relative_to(src_dir)
        target = dst_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file_path, target)


def _build_subset_metrics_csv(videos_dir: Path, targets: List[str], csv_path: Path) -> None:
    fields = [
        "video_id",
        "profile",
        "variant",
        "language",
        "video_duration_sec",
        "num_frames",
        "num_clips",
        "preprocess_time_sec",
        "model_time_sec",
        "postprocess_time_sec",
        "total_time_sec",
        "translation_total_time_sec",
        "index_total_time_sec",
        "batch_count",
        "batch_padding_ratio",
        "model_frames_per_sec",
        "model_clips_per_sec",
        "throughput_frames_per_sec",
        "throughput_clips_per_sec",
        "video_seconds_per_compute_second",
    ]
    rows: List[Dict[str, Any]] = []
    for video_id in targets:
        metrics = _read_json(metrics_path(videos_dir, video_id), default={}) or {}
        run_manifest = _read_json(run_manifest_path(videos_dir, video_id), default={}) or {}
        extra = metrics.get("extra") if isinstance(metrics.get("extra"), dict) else {}
        batching = extra.get("batching") if isinstance(extra.get("batching"), dict) else {}
        translations = metrics.get("translations") if isinstance(metrics.get("translations"), dict) else {}
        indexing = metrics.get("indexing") if isinstance(metrics.get("indexing"), dict) else {}
        translation_total = sum(float((payload or {}).get("time_sec") or 0.0) for payload in translations.values() if isinstance(payload, dict))
        index_total = sum(float((payload or {}).get("time_sec") or 0.0) for payload in indexing.values() if isinstance(payload, dict))
        total_time = float(metrics.get("total_time_sec") or 0.0)
        num_frames = float(metrics.get("num_frames") or 0.0)
        num_clips = float(metrics.get("num_clips") or 0.0)
        video_duration = float(metrics.get("video_duration_sec") or 0.0)
        rows.append(
            {
                "video_id": video_id,
                "profile": str(run_manifest.get("profile") or ""),
                "variant": str(run_manifest.get("variant") or ""),
                "language": str(metrics.get("language") or ""),
                "video_duration_sec": metrics.get("video_duration_sec"),
                "num_frames": metrics.get("num_frames"),
                "num_clips": metrics.get("num_clips"),
                "preprocess_time_sec": metrics.get("preprocess_time_sec"),
                "model_time_sec": metrics.get("model_time_sec"),
                "postprocess_time_sec": metrics.get("postprocess_time_sec"),
                "total_time_sec": metrics.get("total_time_sec"),
                "translation_total_time_sec": translation_total,
                "index_total_time_sec": index_total,
                "batch_count": batching.get("actual_batches"),
                "batch_padding_ratio": batching.get("actual_padding_ratio"),
                "model_frames_per_sec": batching.get("model_frames_per_sec"),
                "model_clips_per_sec": batching.get("model_clips_per_sec"),
                "throughput_frames_per_sec": (num_frames / total_time) if total_time else None,
                "throughput_clips_per_sec": (num_clips / total_time) if total_time else None,
                "video_seconds_per_compute_second": (video_duration / total_time) if total_time else None,
            }
        )

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})


def _zip_dir(src_dir: Path, zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_path in sorted(src_dir.rglob("*")):
            if not file_path.is_file():
                continue
            archive.write(file_path, arcname=file_path.relative_to(src_dir))


def _package_batch(videos_dir: Path, out_dir: Path, targets: List[str], jobs: List[Dict[str, Any]], profile: str, variant: Optional[str]) -> Dict[str, Path]:
    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    batch_dir = out_dir / f"diploma_batch_{stamp}"
    outputs_root = batch_dir / "video_outputs"
    batch_dir.mkdir(parents=True, exist_ok=True)

    (batch_dir / "targets.txt").write_text("\n".join(targets) + "\n", encoding="utf-8")
    (batch_dir / "jobs.json").write_text(json.dumps(jobs, ensure_ascii=False, indent=2), encoding="utf-8")
    (batch_dir / "notes.txt").write_text(
        "Subset batch package for local diploma experiment.\n"
        f"profile={profile}\nvariant={variant or ''}\n"
        "Includes per-video manifests, metrics, summaries, segments, clip observations, and metrics exports.\n",
        encoding="utf-8",
    )

    manifest_payload = {
        "created_at": now_ts(),
        "profile": profile,
        "variant": variant,
        "targets": targets,
        "target_count": len(targets),
    }
    (batch_dir / "batch_manifest.json").write_text(json.dumps(manifest_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    for video_id in targets:
        dst = outputs_root / video_id
        _copy_if_exists(video_manifest_path(videos_dir, video_id), dst / "manifest.json")
        _copy_if_exists(outputs_manifest_path(videos_dir, video_id), dst / "outputs" / "manifest.json")
        _copy_if_exists(run_manifest_path(videos_dir, video_id), dst / "outputs" / "run_manifest.json")
        _copy_if_exists(metrics_path(videos_dir, video_id), dst / "outputs" / "metrics.json")
        _copy_if_exists(clip_observations_path(videos_dir, video_id), dst / "outputs" / "clip_observations.json")
        for lang in ("en", "ru", "kz"):
            _copy_if_exists(summary_path(videos_dir, video_id, lang), dst / "outputs" / "summaries" / f"{lang}.json")
            zst_path = segments_path(videos_dir, video_id, lang)
            _copy_if_exists(zst_path, dst / "outputs" / "segments" / zst_path.name)

    metrics_result = export_metrics_bundle(
        videos_dir=videos_dir,
        out_dir=batch_dir,
        profile_filter=profile,
        variant_filter=variant or "",
    )
    _build_subset_metrics_csv(videos_dir, targets, batch_dir / "batch_metrics.csv")

    zip_path = out_dir / f"{batch_dir.name}.zip"
    _zip_dir(batch_dir, zip_path)
    return {
        "batch_dir": batch_dir,
        "zip_path": zip_path,
        "metrics_zip_path": Path(metrics_result["zip_path"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos-dir", type=str, default="data/videos")
    parser.add_argument("--out-dir", type=str, default="data/research")
    parser.add_argument("--profile", type=str, default="main")
    parser.add_argument("--variant", type=str, default="")
    parser.add_argument("--poll-sec", type=float, default=15.0)
    parser.add_argument("--timeout-sec", type=float, default=21600.0)
    args = parser.parse_args()

    cfg, raw = load_cfg_and_raw(profile=args.profile, variant=(args.variant or None))
    paths = get_backend_paths(cfg, raw)
    videos_dir = Path(args.videos_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    profile = str(args.profile or cfg.active_profile).strip().lower() or cfg.active_profile
    variant = str(args.variant or "").strip().lower() or None

    imported = _import_loose_videos(videos_dir)
    targets = _discover_unprocessed(videos_dir)
    jobs = _enqueue_targets(paths, cfg, targets, profile, variant, videos_dir)

    print(f"Imported loose videos: {len(imported)}", flush=True)
    print(f"Targets enqueued: {len(targets)}", flush=True)
    if not targets:
        print("No unprocessed targets found.", flush=True)
        package = _package_batch(videos_dir, out_dir, [], jobs, profile, variant)
        print(f"Batch dir: {package['batch_dir']}", flush=True)
        print(f"Batch zip: {package['zip_path']}", flush=True)
        return

    pending = _wait_for_targets(videos_dir, targets, args.poll_sec, args.timeout_sec)
    if pending:
        raise SystemExit(f"Timed out waiting for: {', '.join(pending)}")

    package = _package_batch(videos_dir, out_dir, targets, jobs, profile, variant)
    print(f"Batch dir: {package['batch_dir']}", flush=True)
    print(f"Batch zip: {package['zip_path']}", flush=True)
    print(f"Metrics zip: {package['metrics_zip_path']}", flush=True)


if __name__ == "__main__":
    main()
