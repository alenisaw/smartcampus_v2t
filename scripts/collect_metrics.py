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
    "video_duration_sec",
    "num_frames",
    "num_clips",
    "clip_frames_min",
    "clip_frames_max",
    "clip_frames_avg",
    "avg_clip_duration_sec",
    "preprocess_time_sec",
    "model_time_sec",
    "postprocess_time_sec",
    "total_time_sec",
    "decode_time_sec",
    "source_fps",
    "processed_fps",
    "dark_drop_ratio",
    "lazy_drop_ratio",
    "blur_flag_ratio",
    "translation_total_time_sec",
    "translation_languages_count",
    "translation_segments_total",
    "translation_selected_candidates_total",
    "translation_selected_post_edited_total",
    "translation_summary_post_edited_count",
    "index_total_time_sec",
    "index_languages_count",
    "throughput_frames_per_sec",
    "throughput_clips_per_sec",
    "video_seconds_per_compute_second",
    "preprocess_share_pct",
    "model_share_pct",
    "postprocess_share_pct",
]

CSV_FIELDS = [
    "video_id",
    "language",
    "profile",
    "variant",
    "config_fingerprint",
    "languages",
    "video_duration_sec",
    "num_frames",
    "num_clips",
    "clip_frames_min",
    "clip_frames_max",
    "clip_frames_avg",
    "avg_clip_duration_sec",
    "preprocess_time_sec",
    "model_time_sec",
    "postprocess_time_sec",
    "total_time_sec",
    "decode_time_sec",
    "source_fps",
    "processed_fps",
    "dark_drop_ratio",
    "lazy_drop_ratio",
    "blur_flag_ratio",
    "anonymized",
    "translation_total_time_sec",
    "translation_languages_count",
    "translation_segments_total",
    "translation_selected_candidates_total",
    "translation_selected_post_edited_total",
    "translation_summary_post_edited_count",
    "index_total_time_sec",
    "index_languages_count",
    "throughput_frames_per_sec",
    "throughput_clips_per_sec",
    "video_seconds_per_compute_second",
    "preprocess_share_pct",
    "model_share_pct",
    "postprocess_share_pct",
]

STAGE_RUN_FIELDS = [
    "video_id",
    "language",
    "profile",
    "variant",
    "config_fingerprint",
    "stage",
    "count",
    "mean_sec",
    "std_sec",
    "min_sec",
    "max_sec",
    "median_sec",
]

STAGE_AGG_FIELDS = [
    "scope",
    "profile",
    "variant",
    "stage",
    "count",
    "mean_sec",
    "std_sec",
    "min_sec",
    "max_sec",
    "median_sec",
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


def _flatten_run_stage_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Flatten nested per-run stage timings for tabular exports."""

    out: List[Dict[str, Any]] = []
    for row in rows:
        stage_stats = row.get("stage_stats_sec")
        if not isinstance(stage_stats, dict):
            continue
        for stage_name, payload in sorted(stage_stats.items()):
            if not isinstance(payload, dict):
                continue
            out.append(
                {
                    "video_id": row.get("video_id"),
                    "language": row.get("language"),
                    "profile": row.get("profile"),
                    "variant": row.get("variant"),
                    "config_fingerprint": row.get("config_fingerprint"),
                    "stage": str(stage_name),
                    "count": payload.get("count"),
                    "mean_sec": payload.get("mean_sec"),
                    "std_sec": payload.get("std_sec"),
                    "min_sec": payload.get("min_sec"),
                    "max_sec": payload.get("max_sec"),
                    "median_sec": payload.get("median_sec"),
                }
            )
    return out


def _write_stage_run_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    """Write flattened stage timings for each run."""

    stage_rows = _flatten_run_stage_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=STAGE_RUN_FIELDS)
        writer.writeheader()
        for row in stage_rows:
            writer.writerow({key: row.get(key) for key in STAGE_RUN_FIELDS})


def _write_stage_aggregate_csv(path: Path, rows: List[Dict[str, Any]], grouped_by_profile_variant: List[Dict[str, Any]]) -> None:
    """Write system-level and profile/variant stage timing aggregates."""

    out_rows: List[Dict[str, Any]] = []

    system_stages = _aggregate_stage_stats(rows)
    for stage_name, payload in sorted(system_stages.items()):
        out_rows.append(
            {
                "scope": "system",
                "profile": "",
                "variant": "",
                "stage": str(stage_name),
                "count": payload.get("count"),
                "mean_sec": payload.get("mean_sec"),
                "std_sec": payload.get("std_sec"),
                "min_sec": payload.get("min_sec"),
                "max_sec": payload.get("max_sec"),
                "median_sec": payload.get("median_sec"),
            }
        )

    for group in grouped_by_profile_variant:
        if not isinstance(group, dict):
            continue
        stage_timings = group.get("stage_timings_sec")
        if not isinstance(stage_timings, dict):
            continue
        for stage_name, payload in sorted(stage_timings.items()):
            if not isinstance(payload, dict):
                continue
            out_rows.append(
                {
                    "scope": "profile_variant",
                    "profile": group.get("profile"),
                    "variant": group.get("variant"),
                    "stage": str(stage_name),
                    "count": payload.get("count"),
                    "mean_sec": payload.get("mean_sec"),
                    "std_sec": payload.get("std_sec"),
                    "min_sec": payload.get("min_sec"),
                    "max_sec": payload.get("max_sec"),
                    "median_sec": payload.get("median_sec"),
                }
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=STAGE_AGG_FIELDS)
        writer.writeheader()
        for row in out_rows:
            writer.writerow({key: row.get(key) for key in STAGE_AGG_FIELDS})


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


def _safe_div(num: Any, den: Any) -> Optional[float]:
    """Return a float ratio when both operands are valid and denominator is non-zero."""

    n = _float_or_none(num)
    d = _float_or_none(den)
    if n is None or d is None or d == 0:
        return None
    return float(n / d)


def _stage_payload(values: List[float]) -> Dict[str, Any]:
    """Build richer aggregate statistics for one stage across runs."""

    ordered = sorted(float(value) for value in values)
    n = len(ordered)
    if n == 0:
        return {"count": 0, "mean_sec": None, "std_sec": None, "min_sec": None, "max_sec": None, "median_sec": None}
    mid = n // 2
    median = ordered[mid] if n % 2 == 1 else (ordered[mid - 1] + ordered[mid]) / 2.0
    return {
        "count": n,
        "mean_sec": mean(ordered),
        "std_sec": 0.0 if n == 1 else pstdev(ordered),
        "min_sec": ordered[0],
        "max_sec": ordered[-1],
        "median_sec": median,
    }


def _fallback_stage_stats_from_metrics(metrics: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Synthesize minimal stage timings when older metrics files lack explicit stage stats."""

    fallback: Dict[str, Dict[str, Any]] = {}

    def add_stage(name: str, raw_value: Any) -> None:
        value = _float_or_none(raw_value)
        if value is None:
            return
        fallback[name] = {
            "count": 1,
            "mean_sec": float(value),
            "std_sec": 0.0,
            "min_sec": float(value),
            "max_sec": float(value),
            "median_sec": float(value),
        }

    add_stage("preprocess_video", metrics.get("preprocess_time_sec"))
    add_stage("run_vlm_pipeline", metrics.get("model_time_sec"))
    add_stage("postprocess_pipeline", metrics.get("postprocess_time_sec"))
    add_stage("process_total", metrics.get("total_time_sec"))

    translations = metrics.get("translations")
    if isinstance(translations, dict):
        for lang, payload in translations.items():
            if not isinstance(payload, dict):
                continue
            add_stage(f"translate_total:{lang}", payload.get("time_sec"))

    indexing = metrics.get("indexing")
    if isinstance(indexing, dict):
        for lang, payload in indexing.items():
            if not isinstance(payload, dict):
                continue
            add_stage(f"index_build:{lang}", payload.get("time_sec"))

    return fallback


def _aggregate_stage_stats(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Aggregate per-stage timing means across runs."""

    buckets: Dict[str, List[float]] = {}
    for row in rows:
        stage_stats = row.get("stage_stats_sec")
        if not isinstance(stage_stats, dict):
            continue
        for stage_name, payload in stage_stats.items():
            if not isinstance(payload, dict):
                continue
            value = _float_or_none(payload.get("mean_sec"))
            if value is None:
                value = _float_or_none(payload.get("time_sec"))
            if value is None:
                continue
            buckets.setdefault(str(stage_name), []).append(float(value))

    out: Dict[str, Dict[str, Any]] = {}
    for stage_name, values in sorted(buckets.items()):
        if not values:
            continue
        out[stage_name] = _stage_payload(values)
    return out


def _summarize_translations(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten translation metrics for research exports."""

    translations = metrics.get("translations")
    if not isinstance(translations, dict):
        return {
            "translation_total_time_sec": None,
            "translation_languages_count": 0,
            "translation_segments_total": 0,
            "translation_selected_candidates_total": 0,
            "translation_selected_post_edited_total": 0,
            "translation_summary_post_edited_count": 0,
            "translation_languages": [],
        }

    langs = sorted(str(lang) for lang in translations.keys() if str(lang).strip())
    time_values: List[float] = []
    segments_total = 0
    selected_candidates_total = 0
    selected_post_edited_total = 0
    summary_post_edited_count = 0
    for payload in translations.values():
        if not isinstance(payload, dict):
            continue
        time_value = _float_or_none(payload.get("time_sec"))
        if time_value is not None:
            time_values.append(time_value)
        segments_total += int(_float_or_none(payload.get("num_segments")) or 0)
        selected_candidates_total += int(_float_or_none(payload.get("selected_segments_candidates")) or 0)
        selected_post_edited_total += int(_float_or_none(payload.get("selected_segments_post_edited")) or 0)
        summary_post_edited_count += 1 if bool(payload.get("summary_post_edited")) else 0

    return {
        "translation_total_time_sec": float(sum(time_values)) if time_values else None,
        "translation_languages_count": len(langs),
        "translation_segments_total": segments_total,
        "translation_selected_candidates_total": selected_candidates_total,
        "translation_selected_post_edited_total": selected_post_edited_total,
        "translation_summary_post_edited_count": summary_post_edited_count,
        "translation_languages": langs,
    }


def _summarize_indexing(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten indexing metrics for research exports."""

    indexing = metrics.get("indexing")
    if not isinstance(indexing, dict):
        return {
            "index_total_time_sec": None,
            "index_languages_count": 0,
            "index_languages": [],
        }

    langs = sorted(str(lang) for lang in indexing.keys() if str(lang).strip())
    time_values: List[float] = []
    for payload in indexing.values():
        if not isinstance(payload, dict):
            continue
        time_value = _float_or_none(payload.get("time_sec"))
        if time_value is not None:
            time_values.append(time_value)
    return {
        "index_total_time_sec": float(sum(time_values)) if time_values else None,
        "index_languages_count": len(langs),
        "index_languages": langs,
    }


def _normalized_metric_language(metrics: Dict[str, Any], run_manifest: Dict[str, Any]) -> str:
    """Prefer an explicit non-placeholder language from metrics or run manifest."""

    metrics_lang = str(metrics.get("language") or "").strip().lower()
    if metrics_lang and metrics_lang not in {"unknown", "unset", "none", "null"}:
        return metrics_lang
    run_lang = str(run_manifest.get("language") or "").strip().lower()
    if run_lang and run_lang not in {"unknown", "unset", "none", "null"}:
        return run_lang
    return metrics_lang or run_lang


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
    clip_stats = extra.get("clip_stats") if isinstance(extra.get("clip_stats"), dict) else {}
    stage_stats = metrics.get("stage_stats_sec") if isinstance(metrics.get("stage_stats_sec"), dict) else {}
    if not stage_stats:
        stage_stats = _fallback_stage_stats_from_metrics(metrics)

    translation_summary = _summarize_translations(metrics)
    indexing_summary = _summarize_indexing(metrics)

    total_time_sec = _float_or_none(metrics.get("total_time_sec"))
    num_frames = _float_or_none(metrics.get("num_frames"))
    num_clips = _float_or_none(metrics.get("num_clips"))
    video_duration_sec = _float_or_none(metrics.get("video_duration_sec"))
    preprocess_time_sec = _float_or_none(metrics.get("preprocess_time_sec"))
    model_time_sec = _float_or_none(metrics.get("model_time_sec"))
    postprocess_time_sec = _float_or_none(metrics.get("postprocess_time_sec"))

    return {
        "video_id": video_id,
        "language": _normalized_metric_language(metrics, run_manifest),
        "variant": variant or "",
        "profile": str(run_manifest.get("profile") or ""),
        "config_fingerprint": str(run_manifest.get("config_fingerprint") or ""),
        "languages": ",".join(sorted(list_output_languages(videos_dir, video_id, variant=variant))),
        "video_duration_sec": metrics.get("video_duration_sec"),
        "num_frames": metrics.get("num_frames"),
        "num_clips": metrics.get("num_clips"),
        "clip_frames_min": clip_stats.get("frames_min"),
        "clip_frames_max": clip_stats.get("frames_max"),
        "clip_frames_avg": clip_stats.get("frames_avg"),
        "avg_clip_duration_sec": metrics.get("avg_clip_duration_sec"),
        "preprocess_time_sec": metrics.get("preprocess_time_sec"),
        "model_time_sec": metrics.get("model_time_sec"),
        "postprocess_time_sec": metrics.get("postprocess_time_sec"),
        "total_time_sec": metrics.get("total_time_sec"),
        "decode_time_sec": extra.get("decode_time_sec"),
        "source_fps": extra.get("source_fps"),
        "processed_fps": extra.get("processed_fps"),
        "dark_drop_ratio": extra.get("dark_drop_ratio"),
        "lazy_drop_ratio": extra.get("lazy_drop_ratio"),
        "blur_flag_ratio": extra.get("blur_flag_ratio"),
        "anonymized": extra.get("anonymized"),
        "translation_total_time_sec": translation_summary.get("translation_total_time_sec"),
        "translation_languages_count": translation_summary.get("translation_languages_count"),
        "translation_segments_total": translation_summary.get("translation_segments_total"),
        "translation_selected_candidates_total": translation_summary.get("translation_selected_candidates_total"),
        "translation_selected_post_edited_total": translation_summary.get("translation_selected_post_edited_total"),
        "translation_summary_post_edited_count": translation_summary.get("translation_summary_post_edited_count"),
        "translation_languages": translation_summary.get("translation_languages"),
        "index_total_time_sec": indexing_summary.get("index_total_time_sec"),
        "index_languages_count": indexing_summary.get("index_languages_count"),
        "index_languages": indexing_summary.get("index_languages"),
        "throughput_frames_per_sec": _safe_div(num_frames, total_time_sec),
        "throughput_clips_per_sec": _safe_div(num_clips, total_time_sec),
        "video_seconds_per_compute_second": _safe_div(video_duration_sec, total_time_sec),
        "preprocess_share_pct": None if total_time_sec in (None, 0.0) or preprocess_time_sec is None else float(preprocess_time_sec / total_time_sec * 100.0),
        "model_share_pct": None if total_time_sec in (None, 0.0) or model_time_sec is None else float(model_time_sec / total_time_sec * 100.0),
        "postprocess_share_pct": None if total_time_sec in (None, 0.0) or postprocess_time_sec is None else float(postprocess_time_sec / total_time_sec * 100.0),
        "stage_stats_sec": stage_stats,
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
        item["stage_timings_sec"] = _aggregate_stage_stats(group_rows)
        grouped.append(item)
    return grouped


def _build_system_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build top-level aggregate metrics for the entire dataset."""

    summary: Dict[str, Any] = {
        "runs_total": len(rows),
        "videos_total": len({str(row.get("video_id") or "") for row in rows if str(row.get("video_id") or "")}),
        "languages": sorted({str(row.get("language") or "") for row in rows if str(row.get("language") or "")}),
        "profiles": sorted({str(row.get("profile") or "") for row in rows if str(row.get("profile") or "")}),
        "variants": sorted({str(row.get("variant") or "") for row in rows if str(row.get("variant") or "")}),
    }
    for field in TRACKED_FIELDS:
        summary[field] = _numeric_stats(rows, field)
    summary["stage_timings_sec"] = _aggregate_stage_stats(rows)
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
    stage_runs_csv_path = bundle_dir / "metrics_stage_runs.csv"
    stage_aggregate_csv_path = bundle_dir / "metrics_stage_aggregate.csv"
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
    _write_stage_run_csv(stage_runs_csv_path, rows)
    _write_stage_aggregate_csv(stage_aggregate_csv_path, rows, by_profile_variant)
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
            "stage_timings_sec": _aggregate_stage_stats(rows),
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
                "metrics_stage_runs_csv": stage_runs_csv_path.name,
                "metrics_stage_aggregate_csv": stage_aggregate_csv_path.name,
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
