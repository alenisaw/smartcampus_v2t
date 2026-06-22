# scripts/experiments/run_v08_pipeline.py
import argparse
import csv
import json
import hashlib
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.deps import get_backend_paths, load_cfg_and_raw, now_ts, read_json, atomic_write_json
from backend.jobs.control import create_job, read_job, set_state
from backend.http.common import normalize_uploaded_video
from src.utils.video_store import (
    VIDEO_EXTS,
    ensure_video_dirs,
    metrics_path,
    outputs_manifest_path,
    pick_video_file,
    run_manifest_path,
    video_manifest_path,
)
from scripts.collect_metrics import export_metrics_bundle

TERMINAL_FAILURE_STATES = {"failed", "error", "canceled", "cancelled"}

def import_loose_videos(videos_dir: Path) -> List[str]:
    imported: List[str] = []
    # Discover structured video ids
    structured_ids = set()
    if videos_dir.exists():
        for item in sorted(videos_dir.iterdir()):
            if item.is_dir() and pick_video_file(item / "raw") is not None:
                structured_ids.add(item.name)
                
        # Import loose files
        for path in sorted(videos_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in VIDEO_EXTS:
                video_id = path.stem
                if video_id in structured_ids:
                    continue
                dirs = ensure_video_dirs(videos_dir, video_id)
                target = dirs["raw"] / path.name
                try:
                    os.link(path, target)
                except OSError:
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
                structured_ids.add(video_id)
    return imported

def import_manifest_videos(videos_dir: Path, rows: List[Dict[str, str]]) -> List[str]:
    """Import manifest prepared paths into the canonical structured library."""
    imported: List[str] = []
    for row in rows:
        video_id = str(row.get("video_id") or "").strip()
        source = Path(str(row.get("prepared_video_path") or "").strip()).expanduser()
        if not video_id:
            raise ValueError("Manifest row is missing video_id")
        if pick_video_file(videos_dir / video_id / "raw") is not None:
            continue
        if not source.is_absolute():
            source = (PROJECT_ROOT / source).resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Prepared video missing for {video_id}: {source}")
        dirs = ensure_video_dirs(videos_dir, video_id)
        target = dirs["raw"] / source.name
        try:
            os.link(source, target)
        except OSError:
            shutil.copy2(source, target)
        target = normalize_uploaded_video(target)
        st = target.stat()
        atomic_write_json(video_manifest_path(videos_dir, video_id), {
            "video_id": video_id, "filename": target.name, "path": str(target),
            "size_bytes": int(st.st_size), "mtime": float(st.st_mtime),
            "uploaded_at": now_ts(), "source_path": str(source),
        })
        imported.append(video_id)
    return imported

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def get_structured_video_ids(videos_dir: Path) -> List[str]:
    out = []
    if videos_dir.exists():
        for item in sorted(videos_dir.iterdir()):
            if item.is_dir() and pick_video_file(item / "raw") is not None:
                out.append(item.name)
    return out

def is_language_ready(videos_dir: Path, video_id: str, lang: str, variant: Optional[str]) -> bool:
    manifest = read_json(outputs_manifest_path(videos_dir, video_id, variant=variant), default=None)
    if not isinstance(manifest, dict):
        return False
    languages = manifest.get("languages")
    if not isinstance(languages, dict):
        return False
    payload = languages.get(lang)
    if not isinstance(payload, dict):
        return False
    return str(payload.get("status") or "").strip().lower() == "ready"

def is_video_fully_ready(videos_dir: Path, video_id: str, variant: Optional[str]) -> bool:
    if not metrics_path(videos_dir, video_id, variant=variant).exists():
        return False
    # English must be ready. Multilingual is optional but let's check en.
    # Note: v0.8 plan requires English for metrics. We evaluate English.
    return is_language_ready(videos_dir, video_id, "en", variant=variant)

def write_progress_report(report_path: Path, status_dict: Dict[str, Any]):
    report_path.parent.mkdir(parents=True, exist_ok=True)
    md = f"""# Pipeline Progress Report
**Time**: {time.strftime("%Y-%m-%d %H:%M:%S")}
**Completed videos**: {status_dict['completed_count']}/{status_dict['total_count']}
**Failed videos**: {status_dict['failed_count']}
**Current video**: {status_dict['current_video']}
**Average runtime/video**: {status_dict['avg_runtime_sec']} sec
**GPU memory**: {status_dict['gpu_mem']}
**Disk used**: {status_dict['disk_used']}
**Estimated remaining**: {status_dict['est_remaining_sec']} sec
**Issues**: {status_dict['issues']}
"""
    report_path.write_text(md, encoding="utf-8")

def main() -> int:
    parser = argparse.ArgumentParser(description="Run v0.8 Pipeline Reproducible Experiment")
    parser.add_argument("--manifest", type=str, default="data/manifests/v08_combined.csv")
    parser.add_argument("--profile", type=str, default="configs/generated/v08_auto_server.yaml")
    parser.add_argument("--variants", type=str, default="base,no_merge")
    parser.add_argument("--out", type=str, default="data/research/v08")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--dataset-filter", type=str, default=None)
    parser.add_argument("--start-from", type=str, default=None)
    parser.add_argument("--stop-after", type=str, default=None)
    parser.add_argument("--poll-sec", type=float, default=5.0)
    parser.add_argument("--timeout-sec", type=float, default=3600.0)
    args = parser.parse_args()
    
    # Load profile config to get paths
    profile_arg = str(args.profile).strip()
    profile_path = Path(profile_arg).expanduser()
    if not profile_path.is_absolute():
        profile_path = (PROJECT_ROOT / profile_path).resolve()
    if profile_path.is_file():
        cfg, raw_cfg = load_cfg_and_raw(cfg_path=profile_path)
    else:
        cfg, raw_cfg = load_cfg_and_raw(profile=profile_arg)
    profile_name = str(cfg.active_profile)
    profile_selector = str(cfg.config_path)
    paths = get_backend_paths(cfg, raw_cfg)
    videos_dir = Path(cfg.paths.videos_dir).resolve()
    out_dir = Path(args.out).resolve()
    
    # Read input manifest to get list of target videos
    if not os.path.exists(args.manifest):
        print(f"Error: Manifest not found: {args.manifest}")
        return 1
        
    manifest_videos = []
    with open(args.manifest, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            manifest_videos.append(row)
            
    # Apply filters
    if args.dataset_filter:
        manifest_videos = [v for v in manifest_videos if v["dataset_id"] == args.dataset_filter]
        
    # Start/stop limits
    if args.start_from:
        started = False
        filtered = []
        for v in manifest_videos:
            if v["video_id"] == args.start_from:
                started = True
            if started:
                filtered.append(v)
        manifest_videos = filtered
        
    if args.stop_after:
        filtered = []
        for v in manifest_videos:
            filtered.append(v)
            if v["video_id"] == args.stop_after:
                break
        manifest_videos = filtered
        
    if args.max_videos:
        manifest_videos = manifest_videos[:args.max_videos]
        
    target_video_ids = [v["video_id"] for v in manifest_videos]
    if not target_video_ids:
        print("Error: no videos matched the manifest and requested filters.", file=sys.stderr)
        return 1
    print(f"Target videos to process: {len(target_video_ids)} ({', '.join(target_video_ids)})")
    
    # Import loose videos from data/videos into structured raw layout
    import_loose_videos(videos_dir)
    try:
        import_manifest_videos(videos_dir, manifest_videos)
    except (OSError, ValueError) as exc:
        print(f"Error importing manifest videos: {exc}", file=sys.stderr)
        return 1
    
    variants_list = [v.strip() for v in args.variants.split(",") if v.strip()]
    if not variants_list:
        print("Error: at least one variant is required.", file=sys.stderr)
        return 1
    failed_reports = []
    
    # Build list of jobs to monitor
    for variant in variants_list:
        print(f"\n--- Starting variant: {variant} ---")
        
        # Check completed vs pending
        pending_ids = []
        for vid in target_video_ids:
            if args.resume and not args.force and is_video_fully_ready(videos_dir, vid, variant=variant):
                print(f"Video {vid} is already ready for variant {variant}, skipping.")
                continue
            pending_ids.append(vid)
            
        if not pending_ids:
            print(f"All videos ready for variant {variant}.")
            export_metrics_bundle(videos_dir=videos_dir, out_dir=out_dir, profile_filter=profile_name, variant_filter=variant)
            continue
            
        # Submit jobs
        print(f"Queueing jobs for: {', '.join(pending_ids)}")
        jobs_map = {}
        for vid in pending_ids:
            # Enqueue process job
            job = create_job(
                paths,
                video_id=vid,
                job_type="process",
                profile=profile_selector,
                variant=variant,
                language=str(cfg.model.language),
                extra={"job_type": "process", "profile": profile_name, "variant": variant, "force_overwrite": args.force},
            )
            jobs_map[vid] = job["job_id"]
            print(f"Created process job {job['job_id']} for {vid}")
            
        # Spawn local worker in background
        env = os.environ.copy()
        env["SMARTCAMPUS_PROFILE"] = profile_selector
        env["SMARTCAMPUS_VARIANT"] = variant
        env["SMARTCAMPUS_WORKER_ROLE"] = "all"
        
        print("Launching background worker...")
        out_dir.mkdir(parents=True, exist_ok=True)
        log_file = open(out_dir / f"worker_{variant}.log", "w", encoding="utf-8")
        worker_proc = subprocess.Popen(
            [sys.executable, "-m", "backend.worker"],
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT
        )
        
        # Wait and monitor
        t_start = time.time()
        timeout = t_start + args.timeout_sec
        completed = set()
        failed = set()
        
        report_path = out_dir / "progress_report.md"
        failed_csv_path = out_dir / "failed_videos_report.csv"
        
        try:
            while len(completed) + len(failed) < len(pending_ids):
                if time.time() > timeout:
                    print("Execution timed out!")
                    for vid in pending_ids:
                        if vid not in completed and vid not in failed:
                            failed.add(vid)
                            failed_reports.append([vid, variant, "experiment timeout"])
                    break
                    
                time.sleep(args.poll_sec)
                
                # Check target statuses
                current_video = "None"
                for vid in pending_ids:
                    if vid in completed or vid in failed:
                        continue
                        
                    job_id = jobs_map[vid]
                    try:
                        job_record = read_job(paths, job_id)
                        state = job_record.get("state", "queued").lower()
                        current_video = f"{vid} ({state})"
                        
                        # Check outputs manifest for readiness
                        if is_video_fully_ready(videos_dir, vid, variant=variant):
                            completed.add(vid)
                            print(f"Video {vid} completed successfully.")
                        elif state in TERMINAL_FAILURE_STATES:
                            failed.add(vid)
                            err_msg = job_record.get("message", "unknown error")
                            print(f"Video {vid} failed: {err_msg}")
                            failed_reports.append([vid, variant, err_msg])
                    except (OSError, json.JSONDecodeError, ValueError) as exc:
                        print(f"Warning: could not read job {job_id} for {vid}: {exc}", file=sys.stderr)
                        
                # Update progress report
                done_count = len(completed)
                err_count = len(failed)
                elapsed = time.time() - t_start
                avg_time = round(elapsed / max(1, done_count), 2)
                est_rem = round(avg_time * (len(pending_ids) - done_count - err_count), 2)
                
                # Get GPU info if possible
                gpu_mem = "N/A"
                try:
                    smi = subprocess.check_output("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits", shell=True).decode().strip()
                    gpu_mem = f"{smi} MB used"
                except Exception:
                    pass
                    
                total, used, free = shutil.disk_usage(os.getcwd())
                disk_used = f"{round(used / (1024**3), 2)} GB"
                
                write_progress_report(report_path, {
                    "completed_count": done_count,
                    "total_count": len(pending_ids),
                    "failed_count": err_count,
                    "current_video": current_video,
                    "avg_runtime_sec": avg_time,
                    "gpu_mem": gpu_mem,
                    "disk_used": disk_used,
                    "est_remaining_sec": est_rem,
                    "issues": "None" if err_count == 0 else f"{err_count} videos failed"
                })
                
        finally:
            print("Terminating background worker...")
            worker_proc.terminate()
            try:
                worker_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                worker_proc.kill()
            try:
                log_file.close()
            except Exception:
                pass
                
        # Export metrics for this variant
        export_metrics_bundle(videos_dir=videos_dir, out_dir=out_dir, profile_filter=profile_name, variant_filter=variant)
        
    # Write failed videos report
    if failed_reports:
        failed_csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(failed_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["video_id", "variant", "error_message"])
            writer.writerows(failed_reports)
            
    # Save experiment manifest
    manifest_path = Path(args.manifest).resolve()
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
        git_dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=PROJECT_ROOT, text=True).strip())
    except (OSError, subprocess.SubprocessError):
        git_commit, git_dirty = "unknown", None
    exp_manifest = {
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "profile": profile_name,
        "config_path": str(cfg.config_path),
        "config_fingerprint": str(cfg.config_fingerprint),
        "variants": variants_list,
        "target_video_ids": target_video_ids,
        "target_video_count": len(target_video_ids),
        "failures": [{"video_id": r[0], "variant": r[1], "error": r[2]} for r in failed_reports],
        "status": "failed" if failed_reports else "completed",
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "run_id": f"run_{time.strftime('%Y%m%d_%H%M%S')}",
        "timestamp": now_ts()
    }
    atomic_write_json(out_dir / "experiment_manifest.json", exp_manifest)
    print("Experiment pipeline run completed." if not failed_reports else "Experiment pipeline run completed with failures.")
    return 1 if failed_reports else 0

if __name__ == "__main__":
    raise SystemExit(main())
