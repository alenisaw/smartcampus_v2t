import csv
import json
import zipfile
from pathlib import Path

import pytest

from scripts.experiments.export_v08_paper_tables import ExportError, export
from scripts.experiments.package_v08_results import PackageError, package


def _csv(path: Path, rows=None):
    rows = rows or [{"run_id": "run-1", "dataset_id": "avenue", "config_fingerprint": "abc", "metric": "1.25"}]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def _export_sources(root: Path):
    for name in (
        "dataset_scale_summary.csv", "archive_scale_summary.csv", "runtime_summary.csv",
        "retrieval_results_summary.csv", "ablation_results.csv", "qualitative_cases.csv",
    ):
        _csv(root / name)


def test_export_uses_nonempty_real_source_rows(tmp_path):
    from unittest.mock import patch
    research = tmp_path / "v08"; _export_sources(research)
    out = research / "paper_tables"
    with patch("scripts.experiments.export_v08_paper_tables._render_figure") as mock_render:
        def dummy_render(path, *args, **kwargs):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"dummy png bytes")
        mock_render.side_effect = dummy_render
        export(research, out)
    assert len(list(out.glob("table_*.csv"))) == 6
    assert len(list((research / "paper_figures").glob("figure_*.png"))) == 5
    with (out / "table_1_dataset_summary.csv").open(newline="", encoding="utf-8") as handle:
        assert list(csv.DictReader(handle))[0]["metric"] == "1.25"


def test_export_rejects_missing_source(tmp_path):
    research = tmp_path / "v08"; research.mkdir()
    with pytest.raises(ExportError, match="dataset_scale_summary"):
        export(research, research / "paper_tables")


def _package_fixture(repo: Path) -> Path:
    (repo / ".git").mkdir(parents=True)
    research = repo / "data" / "research" / "v08"; research.mkdir(parents=True)
    (research / "run_id.txt").write_text("run-1\n", encoding="utf-8")
    (research / "hardware.json").write_text("{}", encoding="utf-8")
    (research / "environment.json").write_text("{}", encoding="utf-8")
    (research / "README_RESULTS.md").write_text("real run\n", encoding="utf-8")
    (research / "pipeline.log").write_text("completed\n", encoding="utf-8")
    for name in (
        "v08_combined.csv", "dataset_scale_summary.csv", "runtime_summary.csv",
        "retrieval_results_summary.csv", "failed_videos_report.csv", "no_merge_baseline.csv",
        "sparse_only_results.csv", "dense_only_results.csv", "hybrid_rrf_results.csv",
        "hybrid_minmax_results.csv",
    ):
        _csv(research / name)
    table_names = ("dataset_summary", "archive_construction", "runtime_profile", "retrieval_baselines", "ablation_results", "qualitative_cases")
    for i, name in enumerate(table_names, 1): _csv(research / "paper_tables" / f"table_{i}_{name}.csv")
    figure_names = ("pipeline_runtime_breakdown", "dataset_scale", "retrieval_baseline_comparison", "merge_threshold_tradeoff", "query_latency_distribution")
    for i, name in enumerate(figure_names, 1):
        path = research / "paper_figures" / f"figure_{i}_{name}.png"; path.parent.mkdir(exist_ok=True); path.write_bytes(b"real-png-artifact")
    return research


def test_packager_validates_and_creates_readable_zip(tmp_path):
    research = _package_fixture(tmp_path / "repo")
    output = research.parent / "smartcampus_v08_results_run-1.zip"
    package(research, output)
    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None
        names = set(archive.namelist())
        prefix = output.stem
        assert f"{prefix}/checks/artifact_integrity_report.json" in names
        assert f"{prefix}/results/paper_tables/table_1_dataset_summary.csv" in names


def test_packager_refuses_missing_artifacts(tmp_path):
    repo = tmp_path / "repo"; (repo / ".git").mkdir(parents=True)
    research = repo / "data" / "research" / "v08"; research.mkdir(parents=True)
    (research / "run_id.txt").write_text("run-1", encoding="utf-8")
    with pytest.raises(PackageError, match="hardware"):
        package(research, tmp_path / "out.zip")
