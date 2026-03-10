"""
Run a local experiment matrix over the existing SmartCampus filesystem queue.

Purpose:
- Submit process jobs for one or more videos across profile/variant combinations.
- Wait for terminal artifacts using the current worker and manifest model.
- Persist per-run snapshots under data/research/experiments so repeated runs do not get lost.
- Refresh the research metrics bundle after each completed unit.

Expected usage:
- Start API/worker stack separately.
- Run this script from the repository root.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.deps import get_backend_paths, load_cfg_and_raw, now_ts, read_json
from backend.job_control import create_job, read_job
from scripts.collect_metrics import export_metrics_bundle
from src.utils.video_store import (
    batch_manifest_path,
    list_videos,
    metrics_path,
    outputs_manifest_path,
    run_manifest_path,
)


TERMINAL_JOB_STATES = {"done", "failed", "canceled"}
SUCCESS_OUTPUT_STATUS = {"ready"}
FAILED_OUTPUT_STATUS = {"failed", "canceled"}


def _parse_csv(value: str) -> List[str]:
    """Parse a comma-separated list into normalized tokens."""

    items: List[str] = []
    for part in str(value or "").split(","):
        token = str(part).strip()
        if token:
            items.append(token)
    return items


def _resolve_path(arg: str) -> Path:
    """Resolve a path against the current working directory."""

    path = Path(arg)
    if path.is_absolute():
        return path
    return (Path.cwd() / path).resolve()


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    """Write JSON with stable UTF-8 formatting."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    """Read a JSON object or return None."""

    obj = read_json(path, default=None)
    return obj if isinstance(obj, dict) else None


def _slug(text: str) -> str:
    """Convert free text into a filesystem-safe token."""

    cleaned = "".join(ch if ch.isalnum() or ch in "-._" else "_" for ch in str(text or "").strip())
    cleaned = cleaned.strip("._-")
    return cleaned or "experiment"


def _default_label() -> str:
    """Build a default experiment label."""

    return time.strftime("exp_%Y%m%d_%H%M%S", time.localtime())


def _selected_video_ids(paths: Any, requested: Sequence[str], use_all: bool) -> List[str]:
    """Resolve the selected video ids from args or the local video store."""

    if use_all:
        return [str(item.get("video_id") or "") for item in list_videos(Path(paths.videos_dir)) if str(item.get("video_id") or "")]
    video_ids = [str(item).strip() for item in requested if str(item).strip()]
    if not video_ids:
        raise RuntimeError("No video ids provided. Use --video-ids or --all-videos.")
    return video_ids


def _expected_languages(cfg: Any) -> List[str]:
    """Return the expected output languages for one process run."""

    base_lang = str(getattr(cfg.translation, "source_lang", None) or getattr(cfg.model, "language", "en")).strip().lower() or "en"
    langs = [base_lang]
    for lang in getattr(cfg.translation, "target_langs", []) or []:
        token = str(lang).strip().lower()
        if token and token not in langs:
            langs.append(token)
    return langs


def _poll_job(paths: Any, job_id: str, *, timeout_sec: float, poll_sec: float) -> Dict[str, Any]:
    """Wait until a job reaches a terminal state."""

    started_at = time.time()
    while True:
        job = read_job(paths, job_id)
        state = str(job.get("state") or "").strip().lower()
        if state in TERMINAL_JOB_STATES:
            return job
        if time.time() - started_at > timeout_sec:
            raise TimeoutError(f"Timed out waiting for job {job_id} to finish")
        time.sleep(max(0.2, poll_sec))


def _manifest_lang_statuses(manifest: Dict[str, Any], expected_langs: Iterable[str], *, min_updated_at: float) -> Tuple[bool, List[str]]:
    """Check whether all expected language entries are terminal after a given timestamp."""

    languages = manifest.get("languages")
    if not isinstance(languages, dict):
        return False, ["manifest_missing_languages"]

    failures: List[str] = []
    pending = False
    for lang in expected_langs:
        entry = languages.get(str(lang))
        if not isinstance(entry, dict):
            pending = True
            continue
        status = str(entry.get("status") or "").strip().lower()
        updated_at = entry.get("updated_at")
        try:
            updated_ts = float(updated_at)
        except Exception:
            updated_ts = 0.0

        if updated_ts < float(min_updated_at):
            pending = True
            continue
        if status in SUCCESS_OUTPUT_STATUS:
            continue
        if status in FAILED_OUTPUT_STATUS:
            failures.append(f"{lang}:{status}")
            continue
        pending = True

    return (not pending), failures


def _wait_for_outputs(
    *,
    videos_dir: Path,
    video_id: str,
    variant: Optional[str],
    expected_langs: Sequence[str],
    min_updated_at: float,
    timeout_sec: float,
    poll_sec: float,
) -> Dict[str, Any]:
    """Wait until all expected output languages become terminal for one run."""

    manifest_path = outputs_manifest_path(videos_dir, video_id, variant=variant)
    started_at = time.time()
    last_manifest: Dict[str, Any] = {}
    while True:
        manifest = _read_json(manifest_path) or {}
        last_manifest = manifest
        ready, failures = _manifest_lang_statuses(manifest, expected_langs, min_updated_at=min_updated_at)
        if ready:
            if failures:
                raise RuntimeError(f"Outputs failed for video={video_id} variant={variant or 'base'}: {', '.join(failures)}")
            return manifest
        if time.time() - started_at > timeout_sec:
            raise TimeoutError(f"Timed out waiting for outputs manifest {manifest_path}")
        time.sleep(max(0.2, poll_sec))


def _collect_variant_children(videos_dir: Path, video_id: str) -> Dict[str, Any]:
    """Load the top-level experimental batch manifest when available."""

    return _read_json(batch_manifest_path(videos_dir, video_id)) or {}


def _refresh_bundle(videos_dir: Path, out_dir: Path) -> Dict[str, Any]:
    """Refresh the shared research metrics bundle."""

    return export_metrics_bundle(videos_dir=videos_dir, out_dir=out_dir)


def _snapshot_run(
    *,
    exp_dir: Path,
    unit_key: str,
    summary: Dict[str, Any],
    video_id: str,
    variant: Optional[str],
    videos_dir: Path,
) -> Path:
    """Persist one per-run snapshot independent of mutable output paths."""

    payload = {
        "summary": summary,
        "artifacts": {
            "metrics": _read_json(metrics_path(videos_dir, video_id, variant=variant)),
            "run_manifest": _read_json(run_manifest_path(videos_dir, video_id, variant=variant)),
            "outputs_manifest": _read_json(outputs_manifest_path(videos_dir, video_id, variant=variant)),
            "batch_manifest": _collect_variant_children(videos_dir, video_id) if variant is None else None,
        },
    }
    target = exp_dir / "runs" / f"{unit_key}.json"
    _write_json(target, payload)
    return target


def _matrix_entries(
    *,
    video_ids: Sequence[str],
    profiles: Sequence[str],
    variants: Sequence[str],
    runs: int,
    experimental_fanout: bool,
) -> List[Dict[str, Any]]:
    """Build the ordered experiment matrix."""

    entries: List[Dict[str, Any]] = []
    for run_idx in range(1, max(1, int(runs)) + 1):
        for video_id in video_ids:
            for profile in profiles:
                normalized_profile = str(profile).strip().lower() or "main"
                if normalized_profile == "experimental" and experimental_fanout and not variants:
                    entries.append(
                        {
                            "video_id": str(video_id),
                            "profile": normalized_profile,
                            "variant": None,
                            "run_idx": run_idx,
                            "mode": "experimental_fanout",
                        }
                    )
                    continue
                variant_list = list(variants) if variants else [None]
                for variant in variant_list:
                    entries.append(
                        {
                            "video_id": str(video_id),
                            "profile": normalized_profile,
                            "variant": (str(variant).strip().lower() if variant else None),
                            "run_idx": run_idx,
                            "mode": "single_run",
                        }
                    )
    return entries


def run_matrix(
    *,
    label: str,
    video_ids: Sequence[str],
    profiles: Sequence[str],
    variants: Sequence[str],
    runs: int,
    poll_sec: float,
    timeout_sec: float,
    force_overwrite: bool,
    experimental_fanout: bool,
    out_dir: Path,
) -> Dict[str, Any]:
    """Submit and monitor the full experiment matrix."""

    base_cfg, raw = load_cfg_and_raw()
    paths = get_backend_paths(base_cfg, raw)
    videos_dir = Path(paths.videos_dir)
    research_out_dir = out_dir.resolve()
    exp_dir = research_out_dir / _slug(label)
    exp_dir.mkdir(parents=True, exist_ok=True)

    matrix = _matrix_entries(
        video_ids=video_ids,
        profiles=profiles,
        variants=variants,
        runs=runs,
        experimental_fanout=experimental_fanout,
    )

    state: Dict[str, Any] = {
        "label": label,
        "created_at": now_ts(),
        "updated_at": now_ts(),
        "videos": list(video_ids),
        "profiles": list(profiles),
        "variants": list(variants),
        "runs": int(runs),
        "force_overwrite": bool(force_overwrite),
        "experimental_fanout": bool(experimental_fanout),
        "results": [],
    }
    _write_json(exp_dir / "experiment.json", state)

    for index, entry in enumerate(matrix, start=1):
        profile = str(entry["profile"])
        variant = entry.get("variant")
        cfg, _raw = load_cfg_and_raw(profile=profile, variant=variant)
        expected_langs = _expected_languages(cfg)
        unit_key = (
            f"{index:03d}__run_{int(entry['run_idx']):02d}"
            f"__video_{entry['video_id']}__profile_{profile}__variant_{variant or 'base'}"
        )
        extra = {
            "force_overwrite": bool(force_overwrite),
            "experiment_id": _slug(label),
            "experiment_label": label,
            "matrix_index": index,
            "matrix_total": len(matrix),
            "matrix_mode": str(entry.get('mode') or 'single_run'),
            "run_idx": int(entry["run_idx"]),
            "requested_profile": profile,
            "requested_variant": variant,
            "requested_seed": int(getattr(cfg.runtime, "seed", 0) or 0),
            "expected_languages": expected_langs,
        }
        job = create_job(
            paths,
            video_id=str(entry["video_id"]),
            job_type="process",
            profile=profile,
            variant=variant,
            language=str(expected_langs[0] if expected_langs else "en"),
            source_language=None,
            extra=extra,
        )

        result: Dict[str, Any] = {
            "unit_key": unit_key,
            "matrix_index": index,
            "matrix_total": len(matrix),
            "video_id": str(entry["video_id"]),
            "profile": profile,
            "variant": variant,
            "run_idx": int(entry["run_idx"]),
            "mode": str(entry.get("mode") or "single_run"),
            "seed": int(getattr(cfg.runtime, "seed", 0) or 0),
            "job_id": str(job["job_id"]),
            "status": "submitted",
            "submitted_at": now_ts(),
        }

        try:
            parent_job = _poll_job(paths, str(job["job_id"]), timeout_sec=timeout_sec, poll_sec=poll_sec)
            result["job_state"] = str(parent_job.get("state") or "")
            result["job_stage"] = str(parent_job.get("stage") or "")
            result["job_finished_at"] = parent_job.get("finished_at")

            if profile == "experimental" and variant is None and experimental_fanout:
                batch_manifest = _collect_variant_children(videos_dir, str(entry["video_id"]))
                result["batch_manifest_status"] = str(batch_manifest.get("status") or "")
                child_job_ids = list((parent_job.get("extra") or {}).get("child_job_ids") or [])
                result["child_job_ids"] = child_job_ids
                for child_job_id in child_job_ids:
                    _poll_job(paths, str(child_job_id), timeout_sec=timeout_sec, poll_sec=poll_sec)

                variant_results: Dict[str, Any] = {}
                for child_variant in getattr(cfg.experiment, "variant_ids", []) or []:
                    child_variant_token = str(child_variant).strip().lower() or None
                    child_cfg, _ = load_cfg_and_raw(profile=profile, variant=child_variant_token)
                    child_langs = _expected_languages(child_cfg)
                    manifest = _wait_for_outputs(
                        videos_dir=videos_dir,
                        video_id=str(entry["video_id"]),
                        variant=child_variant_token,
                        expected_langs=child_langs,
                        min_updated_at=float(parent_job.get("created_at") or job.get("created_at") or 0.0),
                        timeout_sec=timeout_sec,
                        poll_sec=poll_sec,
                    )
                    variant_results[str(child_variant_token)] = {
                        "expected_languages": child_langs,
                        "outputs_manifest_status": str(manifest.get("status") or ""),
                        "metrics_path": str(metrics_path(videos_dir, str(entry["video_id"]), variant=child_variant_token)),
                        "run_manifest_path": str(run_manifest_path(videos_dir, str(entry["video_id"]), variant=child_variant_token)),
                    }
                result["variant_results"] = variant_results
                result["status"] = "done"
            else:
                if str(parent_job.get("state") or "").strip().lower() != "done":
                    raise RuntimeError(f"Process job {parent_job.get('job_id')} finished with state={parent_job.get('state')}")
                manifest = _wait_for_outputs(
                    videos_dir=videos_dir,
                    video_id=str(entry["video_id"]),
                    variant=variant,
                    expected_langs=expected_langs,
                    min_updated_at=float(parent_job.get("created_at") or job.get("created_at") or 0.0),
                    timeout_sec=timeout_sec,
                    poll_sec=poll_sec,
                )
                result["outputs_manifest_status"] = str(manifest.get("status") or "")
                result["metrics_path"] = str(metrics_path(videos_dir, str(entry["video_id"]), variant=variant))
                result["run_manifest_path"] = str(run_manifest_path(videos_dir, str(entry["video_id"]), variant=variant))
                result["status"] = "done"

            bundle = _refresh_bundle(videos_dir, Path(base_cfg.paths.data_dir) / "research")
            result["bundle_rows"] = int(bundle.get("rows", 0) or 0)
            result["bundle_zip_path"] = str(bundle.get("zip_path") or "")
        except Exception as exc:
            result["status"] = "failed"
            result["error"] = str(exc)

        result["snapshot_path"] = str(
            _snapshot_run(
                exp_dir=exp_dir,
                unit_key=unit_key,
                summary=result,
                video_id=str(entry["video_id"]),
                variant=variant,
                videos_dir=videos_dir,
            )
        )
        result["finished_at"] = now_ts()
        state["results"].append(result)
        state["updated_at"] = now_ts()
        _write_json(exp_dir / "experiment.json", state)

    done = sum(1 for item in state["results"] if str(item.get("status") or "") == "done")
    failed = sum(1 for item in state["results"] if str(item.get("status") or "") == "failed")
    state["summary"] = {
        "units_total": len(state["results"]),
        "units_done": done,
        "units_failed": failed,
        "completed_at": now_ts(),
    }
    state["updated_at"] = now_ts()
    _write_json(exp_dir / "experiment.json", state)
    return state


def _run_relevance_eval_if_requested(
    *,
    labels_path: Optional[Path],
    experiment_manifest: Path,
    ks_csv: str,
) -> Optional[Dict[str, Any]]:
    """Optionally run relevance evaluation on the produced experiment manifest."""

    if labels_path is None:
        return None
    if not labels_path.exists():
        raise RuntimeError(f"Relevance labels file not found: {labels_path}")

    from scripts.eval_relevance import evaluate_relevance

    ks = [int(x) for x in ks_csv.split(",") if str(x).strip().isdigit() and int(str(x).strip()) > 0]
    if not ks:
        ks = [1, 3, 5, 10]

    result = evaluate_relevance(
        labels_path=labels_path,
        out_dir=(PROJECT_ROOT / "data" / "research" / "relevance"),
        ks=ks,
        experiment_manifest=experiment_manifest,
        profiles=[],
        variants=[],
        video_ids=[],
        runs=1,
        default_top_k=20,
    )
    return result


def main() -> None:
    """CLI entrypoint for experiment matrix orchestration."""

    parser = argparse.ArgumentParser(description="Run a local SmartCampus experiment matrix through the filesystem queue.")
    parser.add_argument("--label", type=str, default="", help="Experiment label. Default: timestamp-based.")
    parser.add_argument("--video-ids", nargs="*", default=[], help="One or more video ids.")
    parser.add_argument("--all-videos", action="store_true", help="Run the matrix over all locally stored videos.")
    parser.add_argument("--profiles", type=str, default="main,experimental", help="Comma-separated profiles.")
    parser.add_argument("--variants", type=str, default="", help="Comma-separated explicit variants. Empty means base run.")
    parser.add_argument("--runs", type=int, default=1, help="How many times to repeat each matrix unit.")
    parser.add_argument("--poll-sec", type=float, default=2.0, help="Polling interval in seconds.")
    parser.add_argument("--timeout-sec", type=float, default=7200.0, help="Timeout per matrix unit in seconds.")
    parser.add_argument("--force-overwrite", action="store_true", help="Force process jobs to overwrite existing outputs.")
    parser.add_argument("--no-experimental-fanout", action="store_true", help="Disable profile-level experimental fan-out when variant is empty.")
    parser.add_argument("--out-dir", type=str, default="data/research/experiments", help="Directory for experiment manifests and snapshots.")
    parser.add_argument("--relevance-labels", type=str, default="", help="Optional labels dataset path to run relevance eval after the matrix.")
    parser.add_argument("--relevance-ks", type=str, default="1,3,5,10", help="K values for optional relevance eval, for example '1,3,5,10'.")
    parser.add_argument("--no-comparison-report", action="store_true", help="Disable automatic config comparison report generation.")
    parser.add_argument("--comparison-json", type=str, default="data/research/experiments/comparison_latest.json", help="Comparison JSON output path.")
    parser.add_argument("--comparison-csv", type=str, default="data/research/experiments/comparison_latest.csv", help="Comparison CSV output path.")
    parser.add_argument("--comparison-by-config-dir", type=str, default="data/research/experiments/by_config", help="Per-config latest metrics directory.")
    parser.add_argument("--w-quality", type=float, default=1.0, help="Weight for normalized quality in balanced score.")
    parser.add_argument("--w-latency", type=float, default=1.0, help="Weight for normalized latency in balanced score.")
    parser.add_argument("--w-cost", type=float, default=1.0, help="Weight for normalized cost in balanced score.")
    args = parser.parse_args()

    base_cfg, raw = load_cfg_and_raw()
    paths = get_backend_paths(base_cfg, raw)
    video_ids = _selected_video_ids(paths, args.video_ids, args.all_videos)
    profiles = _parse_csv(args.profiles) or ["main", "experimental"]
    variants = _parse_csv(args.variants)

    result = run_matrix(
        label=args.label.strip() or _default_label(),
        video_ids=video_ids,
        profiles=profiles,
        variants=variants,
        runs=max(1, int(args.runs)),
        poll_sec=max(0.2, float(args.poll_sec)),
        timeout_sec=max(30.0, float(args.timeout_sec)),
        force_overwrite=bool(args.force_overwrite),
        experimental_fanout=not bool(args.no_experimental_fanout),
        out_dir=_resolve_path(args.out_dir),
    )

    summary = result.get("summary") or {}
    print(f"Experiment: {result.get('label')}")
    print(f"Units total: {int(summary.get('units_total', 0) or 0)}")
    print(f"Units done: {int(summary.get('units_done', 0) or 0)}")
    print(f"Units failed: {int(summary.get('units_failed', 0) or 0)}")
    manifest_path = _resolve_path(args.out_dir) / _slug(str(result.get("label") or "")) / "experiment.json"
    print(f"Manifest: {manifest_path}")

    labels_arg = str(args.relevance_labels or "").strip()
    if labels_arg:
        eval_result = _run_relevance_eval_if_requested(
            labels_path=_resolve_path(labels_arg),
            experiment_manifest=manifest_path,
            ks_csv=str(args.relevance_ks or ""),
        )
        if isinstance(eval_result, dict):
            print(f"Relevance rows: {int(eval_result.get('rows_total', 0) or 0)}")
            print(f"Relevance targets: {int(eval_result.get('targets_total', 0) or 0)}")
            print(f"Relevance JSON: {eval_result.get('json_path')}")
            print(f"Relevance CSV: {eval_result.get('csv_path')}")

    if not bool(args.no_comparison_report):
        from scripts.build_comparison_report import build_comparison_report

        comp = build_comparison_report(
            experiment_manifest=manifest_path,
            by_config_dir=_resolve_path(args.comparison_by_config_dir),
            out_json=_resolve_path(args.comparison_json),
            out_csv=_resolve_path(args.comparison_csv),
            w_quality=float(args.w_quality),
            w_latency=float(args.w_latency),
            w_cost=float(args.w_cost),
        )
        print(f"Comparison rows: {int(comp.get('rows_total', 0) or 0)}")
        print(f"Comparison best config: {comp.get('best_config_key')}")
        print(f"Comparison JSON: {comp.get('json_path')}")
        print(f"Comparison CSV: {comp.get('csv_path')}")


if __name__ == "__main__":
    main()
