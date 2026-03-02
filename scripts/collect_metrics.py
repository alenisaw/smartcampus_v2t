"""
Export a research-ready metrics bundle for all stored video runs.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import zipfile
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.video_store import (
    list_output_languages,
    list_output_variants,
    metrics_path,
    run_manifest_path,
)


TRACKED_FIELDS = [
    "num_frames",
    "num_clips",
    "avg_clip_duration_sec",
    "preprocess_time_sec",
    "model_time_sec",
    "postprocess_time_sec",
    "total_time_sec",
    "dark_drop_ratio",
    "lazy_drop_ratio",
    "blur_flag_ratio",
]

CSV_FIELDS = [
    "video_id",
    "profile",
    "variant",
    "config_fingerprint",
    "languages",
    "num_frames",
    "num_clips",
    "avg_clip_duration_sec",
    "preprocess_time_sec",
    "model_time_sec",
    "postprocess_time_sec",
    "total_time_sec",
    "dark_drop_ratio",
    "lazy_drop_ratio",
    "blur_flag_ratio",
    "anonymized",
]


def _resolve_path(arg: str) -> Path:
    """Resolve relative paths against the current working directory."""

    path = Path(arg)
    if path.is_absolute():
        return path
    return (Path.cwd() / path).resolve()


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    """Read a JSON object or return None."""

    if not path.exists():
        return None
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    """Write JSON with stable UTF-8 formatting."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    """Write flat run metrics to CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in CSV_FIELDS})


def _float_or_none(value: Any) -> Optional[float]:
    """Convert scalar values to float when possible."""

    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        text = str(value).strip()
        if not text:
            return None
        return float(text)
    except Exception:
        return None


def _numeric_stats(rows: List[Dict[str, Any]], field: str) -> Dict[str, Any]:
    """Compute count, mean, and std for one numeric field."""

    values = [_float_or_none(item.get(field)) for item in rows]
    clean = [value for value in values if value is not None]
    if not clean:
        return {"count": 0, "mean": None, "std": None}
    return {
        "count": len(clean),
        "mean": mean(clean),
        "std": 0.0 if len(clean) == 1 else pstdev(clean),
    }


def _iter_targets(videos_dir: Path) -> Iterable[Tuple[str, Optional[str]]]:
    """Yield base and variant targets for every video folder."""

    if not videos_dir.exists():
        return []
    targets: List[Tuple[str, Optional[str]]] = []
    for vdir in sorted(videos_dir.iterdir()):
        if not vdir.is_dir():
            continue
        video_id = vdir.name
        targets.append((video_id, None))
        for variant in list_output_variants(videos_dir, video_id):
            targets.append((video_id, variant))
    return targets


def _build_row(videos_dir: Path, video_id: str, variant: Optional[str]) -> Optional[Dict[str, Any]]:
    """Build one flat metrics row from metrics + run manifest."""

    metrics = _read_json(metrics_path(videos_dir, video_id, variant=variant))
    run_manifest = _read_json(run_manifest_path(videos_dir, video_id, variant=variant))
    if not metrics and not run_manifest:
        return None

    metrics = metrics or {}
    run_manifest = run_manifest or {}
    extra = metrics.get("extra") if isinstance(metrics.get("extra"), dict) else {}

    return {
        "video_id": video_id,
        "variant": variant or "",
        "profile": str(run_manifest.get("profile") or ""),
        "config_fingerprint": str(run_manifest.get("config_fingerprint") or ""),
        "languages": ",".join(sorted(list_output_languages(videos_dir, video_id, variant=variant))),
        "num_frames": metrics.get("num_frames"),
        "num_clips": metrics.get("num_clips"),
        "avg_clip_duration_sec": metrics.get("avg_clip_duration_sec"),
        "preprocess_time_sec": metrics.get("preprocess_time_sec"),
        "model_time_sec": metrics.get("model_time_sec"),
        "postprocess_time_sec": metrics.get("postprocess_time_sec"),
        "total_time_sec": metrics.get("total_time_sec"),
        "dark_drop_ratio": extra.get("dark_drop_ratio"),
        "lazy_drop_ratio": extra.get("lazy_drop_ratio"),
        "blur_flag_ratio": extra.get("blur_flag_ratio"),
        "anonymized": extra.get("anonymized"),
    }


def _group_rows(rows: List[Dict[str, Any]], *keys: str) -> List[Dict[str, Any]]:
    """Group rows by one or more keys and compute tracked-field aggregates."""

    buckets: Dict[Tuple[str, ...], List[Dict[str, Any]]] = {}
    for row in rows:
        group_key = tuple(str(row.get(key) or "") for key in keys)
        buckets.setdefault(group_key, []).append(row)

    grouped: List[Dict[str, Any]] = []
    for group_key, group_rows in sorted(buckets.items()):
        item: Dict[str, Any] = {
            "runs": len(group_rows),
            "videos": sorted({str(row.get("video_id") or "") for row in group_rows if str(row.get("video_id") or "")}),
        }
        for index, key in enumerate(keys):
            item[key] = group_key[index]
        for field in TRACKED_FIELDS:
            item[field] = _numeric_stats(group_rows, field)
        grouped.append(item)
    return grouped


def _build_system_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build top-level aggregate metrics for the entire dataset."""

    summary: Dict[str, Any] = {
        "runs_total": len(rows),
        "videos_total": len({str(row.get("video_id") or "") for row in rows if str(row.get("video_id") or "")}),
        "profiles": sorted({str(row.get("profile") or "") for row in rows if str(row.get("profile") or "")}),
        "variants": sorted({str(row.get("variant") or "") for row in rows if str(row.get("variant") or "")}),
    }
    for field in TRACKED_FIELDS:
        summary[field] = _numeric_stats(rows, field)
    return summary


def _write_bundle_zip(bundle_dir: Path, zip_path: Path) -> None:
    """Pack all generated metric exports into one zip archive."""

    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_path in sorted(bundle_dir.rglob("*")):
            if not file_path.is_file():
                continue
            archive.write(file_path, arcname=file_path.relative_to(bundle_dir))


def export_metrics_bundle(
    *,
    videos_dir: Path,
    out_dir: Path,
    profile_filter: str = "",
    variant_filter: str = "",
) -> Dict[str, Any]:
    """Export a zipped metrics bundle with per-run, per-video, and system summaries."""

    videos_dir = _resolve_path(str(videos_dir))
    out_dir = _resolve_path(str(out_dir))
    profile_filter = profile_filter.strip().lower()
    variant_filter = variant_filter.strip().lower()

    rows: List[Dict[str, Any]] = []
    for video_id, variant in _iter_targets(videos_dir):
        row = _build_row(videos_dir, video_id, variant)
        if not row:
            continue
        row_profile = str(row.get("profile") or "").strip().lower()
        row_variant = str(row.get("variant") or "").strip().lower()
        if profile_filter and row_profile != profile_filter:
            continue
        if variant_filter and row_variant != variant_filter:
            continue
        rows.append(row)

    rows.sort(key=lambda item: (str(item.get("video_id") or ""), str(item.get("variant") or ""), str(item.get("profile") or "")))

    bundle_dir = out_dir / "metrics_bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    runs_csv_path = bundle_dir / "metrics_runs.csv"
    snapshot_json_path = bundle_dir / "metrics_snapshot.json"
    per_video_json_path = bundle_dir / "metrics_by_video.json"
    by_profile_json_path = bundle_dir / "metrics_by_profile_variant.json"
    system_json_path = bundle_dir / "system_metrics.json"
    manifest_json_path = bundle_dir / "bundle_manifest.json"
    zip_path = out_dir / "metrics_bundle.zip"

    per_video = _group_rows(rows, "video_id")
    by_profile_variant = _group_rows(rows, "profile", "variant")
    system_summary = _build_system_summary(rows)

    _write_csv(runs_csv_path, rows)
    _write_json(
        snapshot_json_path,
        {
            "filters": {
                "profile": profile_filter or None,
                "variant": variant_filter or None,
            },
            "rows": rows,
            "grouped_by_profile_variant": by_profile_variant,
            "grouped_by_video": per_video,
            "system": system_summary,
        },
    )
    _write_json(per_video_json_path, {"videos": per_video})
    _write_json(by_profile_json_path, {"groups": by_profile_variant})
    _write_json(system_json_path, {"system": system_summary})
    _write_json(
        manifest_json_path,
        {
            "rows_total": len(rows),
            "filters": {
                "profile": profile_filter or None,
                "variant": variant_filter or None,
            },
            "artifacts": {
                "metrics_runs_csv": runs_csv_path.name,
                "metrics_snapshot_json": snapshot_json_path.name,
                "metrics_by_video_json": per_video_json_path.name,
                "metrics_by_profile_variant_json": by_profile_json_path.name,
                "system_metrics_json": system_json_path.name,
            },
        },
    )

    _write_bundle_zip(bundle_dir, zip_path)

    return {
        "rows": len(rows),
        "bundle_dir": bundle_dir,
        "zip_path": zip_path,
        "system": system_summary,
    }


def main() -> None:
    """Export a zipped metrics bundle across all stored runs."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--videos-dir", type=str, default="data/videos")
    parser.add_argument("--out-dir", type=str, default="data/research")
    parser.add_argument("--profile", type=str, default="")
    parser.add_argument("--variant", type=str, default="")
    args = parser.parse_args()

    result = export_metrics_bundle(
        videos_dir=_resolve_path(args.videos_dir),
        out_dir=_resolve_path(args.out_dir),
        profile_filter=args.profile,
        variant_filter=args.variant,
    )

    print(f"Rows collected: {int(result['rows'])}")
    print(f"Bundle directory: {result['bundle_dir']}")
    print(f"Bundle zip: {result['zip_path']}")


if __name__ == "__main__":
    main()
