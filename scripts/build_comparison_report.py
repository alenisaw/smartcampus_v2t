"""
Build config-level comparison reports from experiment matrix manifests.

Storage policy:
- One latest file per config under data/research/experiments/by_config/.
- Same config is overwritten by newer runs.
- Different configs create separate files.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.deps import now_ts
from src.utils.video_store import read_segments


def _resolve_path(arg: str) -> Path:
    path = Path(arg)
    if path.is_absolute():
        return path
    return (Path.cwd() / path).resolve()


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        if not path.exists():
            return {}
        text = path.read_text(encoding="utf-8-sig")
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _slug(text: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(text or "").strip())
    cleaned = cleaned.strip("._-")
    return cleaned or "default"


def _float_or_none(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        text = str(value or "").strip()
        if not text:
            return None
        return float(text)
    except Exception:
        return None


def _safe_div(num: Optional[float], den: Optional[float]) -> Optional[float]:
    if num is None or den is None or den == 0:
        return None
    return float(num / den)


def _translation_totals(metrics: Dict[str, Any]) -> Tuple[float, int, int]:
    translations = metrics.get("translations")
    if not isinstance(translations, dict):
        return 0.0, 0, 0
    total_time = 0.0
    total_segments = 0
    lang_count = 0
    for payload in translations.values():
        if not isinstance(payload, dict):
            continue
        lang_count += 1
        total_time += float(_float_or_none(payload.get("time_sec")) or 0.0)
        total_segments += int(_float_or_none(payload.get("num_segments")) or 0)
    return float(total_time), int(lang_count), int(total_segments)


def _index_total(metrics: Dict[str, Any]) -> Tuple[float, int]:
    indexing = metrics.get("indexing")
    if not isinstance(indexing, dict):
        return 0.0, 0
    total_time = 0.0
    lang_count = 0
    for payload in indexing.values():
        if not isinstance(payload, dict):
            continue
        lang_count += 1
        total_time += float(_float_or_none(payload.get("time_sec")) or 0.0)
    return float(total_time), int(lang_count)


def _stage_mean(metrics: Dict[str, Any], name: str) -> Optional[float]:
    stage = metrics.get("stage_stats_sec")
    if not isinstance(stage, dict):
        return None
    payload = stage.get(name)
    if not isinstance(payload, dict):
        return None
    return _float_or_none(payload.get("mean_sec"))


def _quality_features(metrics: Dict[str, Any], run_manifest: Dict[str, Any]) -> Dict[str, Any]:
    seg_path_text = None
    output_paths = run_manifest.get("output_paths")
    if isinstance(output_paths, dict):
        seg_path_text = output_paths.get("segments")
    seg_path = Path(str(seg_path_text)) if seg_path_text else None

    segments: List[Dict[str, Any]] = []
    if seg_path is not None and seg_path.exists():
        try:
            segments = [x for x in read_segments(seg_path) if isinstance(x, dict)]
        except Exception:
            segments = []

    total = len(segments)
    if total == 0:
        return {
            "segments_total": 0,
            "structured_ratio": None,
            "enrichment_ratio": None,
            "anomaly_ratio": None,
            "summary_present": 0.0,
            "quality_proxy_raw": None,
        }

    structured = 0
    enriched = 0
    anomalies = 0
    for seg in segments:
        event_type = str(seg.get("event_type", "unclassified") or "unclassified").strip().lower()
        risk_level = str(seg.get("risk_level", "normal") or "normal").strip().lower()
        tags = seg.get("tags")
        objects = seg.get("objects")
        has_tags = isinstance(tags, list) and any(str(x or "").strip() for x in tags)
        has_objects = isinstance(objects, list) and any(str(x or "").strip() for x in objects)
        anomaly_flag = bool(seg.get("anomaly_flag", False))

        if event_type != "unclassified" or risk_level != "normal" or anomaly_flag:
            structured += 1
        if has_tags or has_objects:
            enriched += 1
        if anomaly_flag:
            anomalies += 1

    summary_present = 0.0
    if isinstance(output_paths, dict):
        summary_path = output_paths.get("summary")
        if summary_path and Path(str(summary_path)).exists():
            summary_present = 1.0

    structured_ratio = float(structured / total)
    enrichment_ratio = float(enriched / total)
    anomaly_ratio = float(anomalies / total)
    quality_proxy_raw = float(0.5 * structured_ratio + 0.3 * enrichment_ratio + 0.2 * summary_present)
    return {
        "segments_total": int(total),
        "structured_ratio": structured_ratio,
        "enrichment_ratio": enrichment_ratio,
        "anomaly_ratio": anomaly_ratio,
        "summary_present": float(summary_present),
        "quality_proxy_raw": quality_proxy_raw,
    }


def _iter_targets(experiment: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    results = experiment.get("results")
    if not isinstance(results, list):
        return []

    out: List[Dict[str, Any]] = []
    for item in results:
        if not isinstance(item, dict):
            continue
        if str(item.get("status") or "").strip().lower() != "done":
            continue

        base = {
            "unit_key": str(item.get("unit_key") or ""),
            "video_id": str(item.get("video_id") or ""),
            "profile": str(item.get("profile") or ""),
            "run_idx": int(item.get("run_idx") or 1),
            "finished_at": _float_or_none(item.get("finished_at")) or _float_or_none(item.get("job_finished_at")) or 0.0,
        }
        variant_results = item.get("variant_results")
        if isinstance(variant_results, dict) and variant_results:
            for variant, payload in variant_results.items():
                if not isinstance(payload, dict):
                    continue
                out.append(
                    {
                        **base,
                        "variant": str(variant or ""),
                        "metrics_path": str(payload.get("metrics_path") or ""),
                        "run_manifest_path": str(payload.get("run_manifest_path") or ""),
                    }
                )
        else:
            out.append(
                {
                    **base,
                    "variant": str(item.get("variant") or ""),
                    "metrics_path": str(item.get("metrics_path") or ""),
                    "run_manifest_path": str(item.get("run_manifest_path") or ""),
                }
            )
    return out


def _row_from_target(target: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    metrics_path = Path(str(target.get("metrics_path") or ""))
    run_manifest_path = Path(str(target.get("run_manifest_path") or ""))
    metrics = _read_json(metrics_path) if metrics_path.exists() else {}
    run_manifest = _read_json(run_manifest_path) if run_manifest_path.exists() else {}
    if not metrics and not run_manifest:
        return None

    profile = str(run_manifest.get("profile") or target.get("profile") or "").strip().lower() or "main"
    variant = str(run_manifest.get("variant") or target.get("variant") or "").strip().lower()
    language = str(metrics.get("language") or run_manifest.get("language") or "").strip().lower() or "en"
    video_id = str(target.get("video_id") or run_manifest.get("video_id") or "").strip()
    config_fp = str(run_manifest.get("config_fingerprint") or metrics.get("config_fingerprint") or "")

    preprocess_time = _float_or_none(metrics.get("preprocess_time_sec")) or 0.0
    model_time = _float_or_none(metrics.get("model_time_sec")) or 0.0
    postprocess_time = _float_or_none(metrics.get("postprocess_time_sec")) or 0.0
    total_time = _float_or_none(metrics.get("total_time_sec")) or (preprocess_time + model_time + postprocess_time)
    num_frames = _float_or_none(metrics.get("num_frames"))
    num_clips = _float_or_none(metrics.get("num_clips"))

    translation_time, translation_langs, translation_segments = _translation_totals(metrics)
    index_time, index_langs = _index_total(metrics)
    total_pipeline_time = float(total_time + translation_time + index_time)
    cost_proxy = float(model_time + translation_time + index_time)

    quality = _quality_features(metrics, run_manifest)

    row: Dict[str, Any] = {
        "config_key": f"{profile}__{variant or 'base'}",
        "profile": profile,
        "variant": variant,
        "language": language,
        "video_id": video_id,
        "run_idx": int(target.get("run_idx") or 1),
        "unit_key": str(target.get("unit_key") or ""),
        "config_fingerprint": config_fp,
        "finished_at": float(target.get("finished_at") or 0.0),
        "metrics_path": str(metrics_path),
        "run_manifest_path": str(run_manifest_path),
        "num_frames": int(num_frames) if num_frames is not None else None,
        "num_clips": int(num_clips) if num_clips is not None else None,
        "preprocess_time_sec": float(preprocess_time),
        "model_time_sec": float(model_time),
        "postprocess_time_sec": float(postprocess_time),
        "process_total_time_sec": float(total_time),
        "translation_total_time_sec": float(translation_time),
        "translation_languages_count": int(translation_langs),
        "translation_segments_total": int(translation_segments),
        "index_total_time_sec": float(index_time),
        "index_languages_count": int(index_langs),
        "total_pipeline_time_sec": float(total_pipeline_time),
        "cost_proxy_sec": float(cost_proxy),
        "throughput_frames_per_sec": _safe_div(num_frames, total_time),
        "throughput_clips_per_sec": _safe_div(num_clips, total_time),
        "stage_preprocess_sec": _stage_mean(metrics, "preprocess_video"),
        "stage_vlm_sec": _stage_mean(metrics, "run_vlm_pipeline"),
        "stage_structuring_sec": _stage_mean(metrics, "structuring_segments"),
        "stage_summary_sec": _stage_mean(metrics, "build_summary"),
        "stage_process_total_sec": _stage_mean(metrics, "process_total"),
        **quality,
    }
    return row


def _normalize(values: List[Optional[float]]) -> List[Optional[float]]:
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return [None for _ in values]
    lo = min(clean)
    hi = max(clean)
    if math.isclose(lo, hi):
        return [0.5 if v is not None else None for v in values]
    out: List[Optional[float]] = []
    for v in values:
        if v is None:
            out.append(None)
        else:
            out.append(float((float(v) - lo) / (hi - lo)))
    return out


def _load_existing_config_rows(by_config_dir: Path) -> Dict[str, Dict[str, Any]]:
    latest: Dict[str, Dict[str, Any]] = {}
    if not by_config_dir.exists():
        return latest
    for path in by_config_dir.glob("*.json"):
        row = _read_json(path)
        if not isinstance(row, dict):
            continue
        key = str(row.get("config_key") or "").strip()
        if not key:
            key = str(path.stem or "").strip()
            if not key:
                continue
            row["config_key"] = key
        latest[key] = row
    return latest


def _save_config_rows(rows: List[Dict[str, Any]], by_config_dir: Path) -> Dict[str, Dict[str, Any]]:
    latest = _load_existing_config_rows(by_config_dir)
    for row in rows:
        key = str(row.get("config_key") or "")
        if not key:
            continue
        prev = latest.get(key)
        if prev is None or float(row.get("finished_at") or 0.0) >= float(prev.get("finished_at") or 0.0):
            latest[key] = row

    by_config_dir.mkdir(parents=True, exist_ok=True)
    for key, row in latest.items():
        path = by_config_dir / f"{_slug(key)}.json"
        _write_json(path, row)
    return latest


def _comparison_payload(
    latest: Dict[str, Dict[str, Any]],
    *,
    w_quality: float,
    w_latency: float,
    w_cost: float,
) -> Dict[str, Any]:
    rows = list(latest.values())
    quality_vals = [(_float_or_none(r.get("quality_proxy_raw"))) for r in rows]
    latency_vals = [(_float_or_none(r.get("total_pipeline_time_sec"))) for r in rows]
    cost_vals = [(_float_or_none(r.get("cost_proxy_sec"))) for r in rows]

    q_norm = _normalize(quality_vals)
    l_norm = _normalize(latency_vals)
    c_norm = _normalize(cost_vals)

    comp_rows: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows):
        qn = q_norm[idx] if idx < len(q_norm) else None
        ln = l_norm[idx] if idx < len(l_norm) else None
        cn = c_norm[idx] if idx < len(c_norm) else None
        qn_val = float(qn) if qn is not None else 0.0
        ln_val = float(ln) if ln is not None else 0.0
        cn_val = float(cn) if cn is not None else 0.0
        balanced = float((w_quality * qn_val) - (w_latency * ln_val) - (w_cost * cn_val))
        item = dict(row)
        item["quality_norm"] = qn
        item["latency_norm"] = ln
        item["cost_norm"] = cn
        item["balanced_score"] = balanced
        comp_rows.append(item)

    comp_rows.sort(key=lambda x: float(x.get("balanced_score") or 0.0), reverse=True)
    for rank, row in enumerate(comp_rows, start=1):
        row["rank"] = rank

    return {
        "created_at": now_ts(),
        "weights": {
            "quality": float(w_quality),
            "latency": float(w_latency),
            "cost": float(w_cost),
        },
        "rows_total": len(comp_rows),
        "rows": comp_rows,
        "summary": {
            "best_config_key": str(comp_rows[0].get("config_key") or "") if comp_rows else None,
            "avg_balanced_score": float(mean([float(x.get("balanced_score") or 0.0) for x in comp_rows])) if comp_rows else None,
        },
    }


def _write_comparison_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fields = [
        "rank",
        "config_key",
        "profile",
        "variant",
        "language",
        "video_id",
        "run_idx",
        "config_fingerprint",
        "total_pipeline_time_sec",
        "cost_proxy_sec",
        "quality_proxy_raw",
        "quality_norm",
        "latency_norm",
        "cost_norm",
        "balanced_score",
        "num_frames",
        "num_clips",
        "preprocess_time_sec",
        "model_time_sec",
        "postprocess_time_sec",
        "process_total_time_sec",
        "translation_total_time_sec",
        "index_total_time_sec",
        "throughput_frames_per_sec",
        "throughput_clips_per_sec",
        "structured_ratio",
        "enrichment_ratio",
        "anomaly_ratio",
        "summary_present",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({f: row.get(f) for f in fields})


def build_comparison_report(
    *,
    experiment_manifest: Path,
    by_config_dir: Path,
    out_json: Path,
    out_csv: Path,
    w_quality: float,
    w_latency: float,
    w_cost: float,
) -> Dict[str, Any]:
    experiment = _read_json(experiment_manifest)
    targets = list(_iter_targets(experiment))
    rows: List[Dict[str, Any]] = []
    for target in targets:
        row = _row_from_target(target)
        if row is not None:
            rows.append(row)
    if not rows:
        raise RuntimeError(f"No completed config rows extracted from {experiment_manifest}")

    latest = _save_config_rows(rows, by_config_dir)
    payload = _comparison_payload(
        latest,
        w_quality=float(w_quality),
        w_latency=float(w_latency),
        w_cost=float(w_cost),
    )
    _write_json(out_json, payload)
    _write_comparison_csv(out_csv, list(payload.get("rows") or []))
    return {
        "rows_total": int(payload.get("rows_total") or 0),
        "best_config_key": str((payload.get("summary") or {}).get("best_config_key") or ""),
        "json_path": str(out_json),
        "csv_path": str(out_csv),
        "by_config_dir": str(by_config_dir),
    }


def _latest_experiment_manifest(experiments_dir: Path) -> Optional[Path]:
    if not experiments_dir.exists():
        return None
    candidates: List[Tuple[float, Path]] = []
    for path in experiments_dir.rglob("experiment.json"):
        try:
            mt = float(path.stat().st_mtime)
        except Exception:
            mt = 0.0
        candidates.append((mt, path))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build per-config latest metrics and comparison report from experiment manifest.")
    parser.add_argument("--experiment-manifest", type=str, default="", help="Path to experiment.json. If empty, latest under --experiments-dir is used.")
    parser.add_argument("--experiments-dir", type=str, default="data/research/experiments", help="Experiments root for automatic manifest discovery.")
    parser.add_argument("--by-config-dir", type=str, default="data/research/experiments/by_config", help="Directory with latest per-config rows.")
    parser.add_argument("--out-json", type=str, default="data/research/experiments/comparison_latest.json", help="Comparison JSON output path.")
    parser.add_argument("--out-csv", type=str, default="data/research/experiments/comparison_latest.csv", help="Comparison CSV output path.")
    parser.add_argument("--w-quality", type=float, default=1.0, help="Weight for normalized quality component.")
    parser.add_argument("--w-latency", type=float, default=1.0, help="Weight for normalized latency component.")
    parser.add_argument("--w-cost", type=float, default=1.0, help="Weight for normalized cost component.")
    args = parser.parse_args()

    manifest_arg = str(args.experiment_manifest or "").strip()
    if manifest_arg:
        manifest_path = _resolve_path(manifest_arg)
    else:
        manifest_path = _latest_experiment_manifest(_resolve_path(args.experiments_dir))
        if manifest_path is None:
            raise RuntimeError("No experiment manifest found. Provide --experiment-manifest or run matrix first.")

    if not manifest_path.exists():
        raise RuntimeError(f"Experiment manifest not found: {manifest_path}")

    result = build_comparison_report(
        experiment_manifest=manifest_path,
        by_config_dir=_resolve_path(args.by_config_dir),
        out_json=_resolve_path(args.out_json),
        out_csv=_resolve_path(args.out_csv),
        w_quality=float(args.w_quality),
        w_latency=float(args.w_latency),
        w_cost=float(args.w_cost),
    )
    print(f"Rows total: {int(result['rows_total'])}")
    print(f"Best config: {result['best_config_key']}")
    print(f"JSON report: {result['json_path']}")
    print(f"CSV report: {result['csv_path']}")
    print(f"Per-config dir: {result['by_config_dir']}")


if __name__ == "__main__":
    main()
