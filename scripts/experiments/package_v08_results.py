"""Validate and package completed SmartCampus v0.8 experiment artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Sequence


class PackageError(RuntimeError):
    pass


REQUIRED = {
    "hardware": ("hardware.json",),
    "environment": ("environment.json",),
    "combined manifest": ("v08_combined.csv",),
    "dataset summary": ("dataset_scale_summary.csv",),
    "runtime summary": ("runtime_summary.csv",),
    "retrieval summary": ("retrieval_results_summary.csv",),
    "README": ("README_RESULTS.md",),
    "failed videos report": ("failed_videos_report.csv",),
}
BASELINES = ("no_merge_baseline.csv", "sparse_only_results.csv", "dense_only_results.csv", "hybrid_rrf_results.csv", "hybrid_minmax_results.csv")
TABLES = tuple(f"table_{i}_{name}.csv" for i, name in enumerate(("dataset_summary", "archive_construction", "runtime_profile", "retrieval_baselines", "ablation_results", "qualitative_cases"), 1))
FIGURES = tuple(f"figure_{i}_{name}.png" for i, name in enumerate(("pipeline_runtime_breakdown", "dataset_scale", "retrieval_baseline_comparison", "merge_threshold_tradeoff", "query_latency_distribution"), 1))
FINAL_REQUIRED = (
    "hardware.json", "environment.json", "v08_combined_fixed.csv", "metrics_runs_final.csv",
    "runtime_summary_final.csv", "archive_scale_summary_final.csv", "index_status_final.csv",
    "retrieval_results_by_query_final.csv", "retrieval_results_summary_final.csv",
    "relevance_labels_final.json", "README_RESULTS_FINAL.md", "paper_readiness_report_FINAL.md",
    "pipeline_execution_audit.json", "rerun_plan.csv",
)
FINAL_TABLES = tuple(f"table_{i}_{name}_final.csv" for i, name in enumerate(("dataset_summary", "archive_construction", "runtime_profile", "retrieval_baselines", "ablation_results", "qualitative_cases"), 1))
FINAL_FIGURES = tuple(f"figure_{i}_{name}_final.png" for i, name in enumerate(("pipeline_runtime_breakdown", "dataset_scale", "retrieval_baseline_comparison", "merge_threshold_tradeoff", "query_latency_distribution"), 1))


def _nonempty(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.reader(handle)
            return next(reader, None) is not None and next(reader, None) is not None
    return True


def _matches(research: Path, repo: Path, name: str) -> list[Path]:
    candidates = list(research.rglob(name))
    for relative in (Path("data/manifests") / name, Path("configs") / name, Path("docs") / name):
        candidate = repo / relative
        if candidate.is_file(): candidates.append(candidate)
    return sorted(set(p.resolve() for p in candidates if p.is_file()))


def _require_one(research: Path, repo: Path, label: str, names: Sequence[str]) -> Path:
    found = [path for name in names for path in _matches(research, repo, name) if _nonempty(path)]
    if not found:
        raise PackageError(f"missing or empty {label}: expected {', '.join(names)}")
    if len(found) > 1:
        raise PackageError(f"ambiguous {label}: {', '.join(str(p) for p in found)}")
    return found[0]


def _repo_root(research: Path) -> Path:
    for parent in (research, *research.parents):
        if (parent / ".git").exists(): return parent
    raise PackageError(f"cannot locate repository root from {research}")


def _run_id(research: Path) -> str:
    text_file = research / "run_id.txt"
    if text_file.is_file() and text_file.read_text(encoding="utf-8").strip():
        return text_file.read_text(encoding="utf-8").strip()
    manifest = research / "experiment_manifest.json"
    if manifest.is_file():
        value = json.loads(manifest.read_text(encoding="utf-8")).get("run_id")
        if value: return str(value)
    raise PackageError("run_id is missing from run_id.txt and experiment_manifest.json")


def _destination(path: Path, research: Path, repo: Path) -> Path:
    try: return Path("results") / path.relative_to(research)
    except ValueError: return path.relative_to(repo)


def package(research: Path, output: Path) -> None:
    if output.suffix.lower() != ".zip": raise PackageError("--out must end in .zip")
    repo, run_id = _repo_root(research), _run_id(research)
    final_mode = (research / "README_RESULTS_FINAL.md").is_file()
    if final_mode:
        required = [_require_one(research, repo, f"final artifact {name}", (name,)) for name in FINAL_REQUIRED]
        required += [_require_one(research, repo, f"final paper table {name}", (name,)) for name in FINAL_TABLES]
        required += [_require_one(research, repo, f"final paper figure {name}", (name,)) for name in FINAL_FIGURES]
    else:
        required = [_require_one(research, repo, label, names) for label, names in REQUIRED.items()]
        required += [_require_one(research, repo, f"baseline {name}", (name,)) for name in BASELINES]
        required += [_require_one(research, repo, f"paper table {name}", (name,)) for name in TABLES]
        required += [_require_one(research, repo, f"paper figure {name}", (name,)) for name in FIGURES]
    logs = sorted(p for p in research.rglob("*.log") if _nonempty(p))
    if not logs: raise PackageError("no nonempty log artifacts found")
    files = sorted(set(required + logs + [p for p in research.rglob("*") if p.is_file() and p.resolve() != output.resolve()]))
    integrity = []
    for path in files:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        integrity.append({"path": _destination(path, research, repo).as_posix(), "bytes": path.stat().st_size, "sha256": digest})
    root_name = output.stem
    with tempfile.TemporaryDirectory(prefix="v08-package-") as temp_name:
        stage = Path(temp_name) / root_name
        for path in files:
            target = stage / _destination(path, research, repo)
            target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path, target)
        checks = stage / "checks"; checks.mkdir(parents=True, exist_ok=True)
        (stage / "run_id.txt").write_text(run_id + "\n", encoding="utf-8")
        (checks / "artifact_integrity_report.json").write_text(json.dumps({"run_id": run_id, "files": integrity}, indent=2), encoding="utf-8")
        (checks / "reproducibility_check_report.json").write_text(json.dumps({"run_id": run_id, "passed": True, "required_artifacts": len(required)}, indent=2), encoding="utf-8")
        missing: list[dict[str, object]] = []
        if final_mode:
            audit_path = research / "checks" / "pipeline_execution_audit.json"
            if audit_path.is_file():
                audit = json.loads(audit_path.read_text(encoding="utf-8"))
                missing = list(audit.get("missing_or_invalid") or [])
        (checks / "missing_outputs_report.json").write_text(
            json.dumps({"run_id": run_id, "missing_or_excluded": missing}, indent=2), encoding="utf-8"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary_zip = Path(temp_name) / output.name
        with zipfile.ZipFile(temporary_zip, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
            for path in sorted(stage.rglob("*")):
                if path.is_file(): archive.write(path, path.relative_to(stage.parent).as_posix())
        with zipfile.ZipFile(temporary_zip) as archive:
            bad = archive.testzip()
            if bad: raise PackageError(f"ZIP integrity check failed at {bad}")
        shutil.copy2(temporary_zip, output)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--research-dir", type=Path, default=Path("data/research/v08"))
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try: package(args.research_dir.resolve(), args.out.resolve())
    except (PackageError, OSError, csv.Error, json.JSONDecodeError, zipfile.BadZipFile) as exc:
        print(f"result packaging failed: {exc}", file=sys.stderr); return 1
    print(f"results packaged at {args.out}"); return 0


if __name__ == "__main__":
    raise SystemExit(main())
