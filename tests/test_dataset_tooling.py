import csv
import subprocess
import sys
from pathlib import Path
from unittest import mock

from scripts.datasets.import_manifest_to_library import import_to_library
from scripts.datasets.merge_manifests import REQUIRED_COLUMNS, merge
from scripts.datasets.validate_manifest import validate


def _write_manifest(path: Path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REQUIRED_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _row(video_id: str, prepared_path: Path, source="source.mp4"):
    return {
        "dataset_id": "dataset",
        "video_id": video_id,
        "source_path": source,
        "prepared_video_path": str(prepared_path),
        "split": "train",
        "scene_id": "scene",
        "label_type": "none",
        "has_anomaly": "no",
        "duration_sec": "1.0",
        "fps": "25",
        "width": "640",
        "height": "480",
        "num_frames": "25",
        "notes": "",
    }


def test_prepare_scripts_require_explicit_mock(tmp_path):
    for script in ("prepare_avenue.py", "prepare_shanghaitech.py"):
        result = subprocess.run(
            [sys.executable, f"scripts/datasets/{script}", "--root", str(tmp_path / "missing"),
             "--out", str(tmp_path / "out"), "--manifest", str(tmp_path / f"{script}.csv")],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert not (tmp_path / f"{script}.csv").exists()


def test_merge_rejects_schema_mismatch_and_duplicate_ids(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("video_id\nonly\n", encoding="utf-8")
    assert not merge([str(bad)], str(tmp_path / "merged.csv"))

    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    _write_manifest(first, [_row("duplicate", tmp_path / "a.mp4")])
    _write_manifest(second, [_row("duplicate", tmp_path / "b.mp4")])
    assert not merge([str(first), str(second)], str(tmp_path / "merged.csv"))


def test_validate_rejects_mock_without_opt_in_and_sets_invalid_report(tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"placeholder")
    manifest = tmp_path / "manifest.csv"
    report = tmp_path / "report.json"
    _write_manifest(manifest, [_row("video", video, source="mock://video.mp4")])

    capture = mock.Mock()
    capture.isOpened.return_value = True
    capture.get.side_effect = [640, 480]
    with mock.patch("scripts.datasets.validate_manifest.cv2.VideoCapture", return_value=capture):
        assert not validate(str(manifest), str(report))
    assert '"valid": false' in report.read_text(encoding="utf-8")


def test_import_preflights_all_sources_before_copying(tmp_path):
    existing = tmp_path / "existing.mp4"
    existing.write_bytes(b"video")
    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest, [
        _row("existing", existing),
        _row("missing", tmp_path / "missing.mp4"),
    ])
    library = tmp_path / "library"
    assert not import_to_library(str(manifest), str(library), "copy")
    assert not library.exists()
