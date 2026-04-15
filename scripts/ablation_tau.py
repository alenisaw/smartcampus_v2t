"""
Threshold ablation for semantic timeline merging in SmartCampus V2T.

Purpose:
- Reuse persisted clip-level observation artifacts without rerunning video inference.
- Rebuild merged semantic timelines for multiple tau values and export intrinsic metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.deps import load_cfg_and_raw, now_ts
from scripts.common import float_or_none, resolve_path, write_json
from src.core.timeline_metrics import intrinsic_timeline_metrics, mean_ignore_none, segment_embeddings_from_members
from src.core.types import Annotation
from src.search.embed import EmbeddingCache, build_text_embedder, select_embedding_model_ref
from src.video.describe import semantic_merge_annotations


DEFAULT_TAUS = [0.85, 0.90, 0.95]


@dataclass
class TargetRun:
    """Resolved ablation target for one persisted video run."""

    video_id: str
    variant: str
    run_id: str
    video_dir: Path
    outputs_dir: Path
    clip_observations_path: Path
    metrics_path: Path
    run_manifest_path: Path


@dataclass
class EmbedderContext:
    """Embedder and cache bundle keyed by effective config."""

    model_ref: str
    embedder: Any
    cache: Optional[EmbeddingCache]


def _read_json(path: Path, *, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _slug_tau(tau: float) -> str:
    return f"{float(tau):.2f}".replace(".", "_")


def _format_float(value: Optional[float], *, digits: int = 4, dash: str = "--") -> str:
    if value is None:
        return dash
    if isinstance(value, float) and math.isnan(value):
        return dash
    return f"{float(value):.{digits}f}"


def _resolve_targets_root(input_path: Path) -> Path:
    """Resolve supported inputs to a directory containing per-video folders."""

    if not input_path.exists():
        raise RuntimeError(f"Input path not found: {input_path}")
    if input_path.is_file():
        raise RuntimeError(f"Expected a directory, got file: {input_path}")
    if (input_path / "video_outputs").is_dir():
        return (input_path / "video_outputs").resolve()
    return input_path.resolve()


def _iter_video_dirs(root: Path) -> List[Path]:
    """Return all directories that look like persisted video outputs."""

    if (root / "outputs").is_dir():
        return [root]
    out: List[Path] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if (child / "outputs").is_dir():
            out.append(child)
    return out


def _build_target(video_dir: Path) -> TargetRun:
    """Resolve the canonical persisted artifacts for one video directory."""

    outputs_dir = video_dir / "outputs"
    clip_observations_path = outputs_dir / "clip_observations.json"
    metrics_path = outputs_dir / "metrics.json"
    run_manifest_path = outputs_dir / "run_manifest.json"
    missing = [str(path) for path in [clip_observations_path, metrics_path, run_manifest_path] if not path.exists()]
    if missing:
        raise RuntimeError(f"Missing required artifacts for {video_dir.name}: {', '.join(missing)}")
    return TargetRun(
        video_id=str(video_dir.name),
        variant="",
        run_id=str(video_dir.name),
        video_dir=video_dir,
        outputs_dir=outputs_dir,
        clip_observations_path=clip_observations_path,
        metrics_path=metrics_path,
        run_manifest_path=run_manifest_path,
    )


def _load_targets(root: Path, *, video_ids: Sequence[str]) -> List[TargetRun]:
    """Load all requested target runs from the resolved root."""

    allowed = {str(item).strip() for item in video_ids if str(item).strip()}
    targets: List[TargetRun] = []
    for video_dir in _iter_video_dirs(root):
        if allowed and video_dir.name not in allowed:
            continue
        targets.append(_build_target(video_dir))
    if not targets:
        suffix = f" for video_ids={sorted(allowed)}" if allowed else ""
        raise RuntimeError(f"No eligible persisted runs found under {root}{suffix}")
    return targets


def _observations_to_annotations(rows: List[Dict[str, Any]]) -> List[Annotation]:
    """Convert persisted clip observations into annotation objects."""

    out: List[Annotation] = []
    for index, item in enumerate(rows):
        if not isinstance(item, dict):
            continue
        out.append(
            Annotation(
                video_id=str(item.get("video_id") or ""),
                start_sec=float(item.get("start_sec") or 0.0),
                end_sec=float(item.get("end_sec") or 0.0),
                description=str(item.get("description") or ""),
                extra={"merged_from": [index], "clip_id": str(item.get("clip_id") or f"clip_{index + 1:06d}")},
                anomaly_flag=bool(item.get("anomaly_flag", False)),
                anomaly_confidence=float(item.get("anomaly_confidence") or 0.0),
                anomaly_notes=list(item.get("anomaly_notes") or []),
            )
        )
    return out


def _load_run_identity(run_manifest_path: Path) -> Tuple[str, Optional[str]]:
    """Read profile and variant from the persisted run manifest."""

    payload = _read_json(run_manifest_path, default={}) or {}
    profile = str(payload.get("profile") or "main").strip().lower() or "main"
    variant = str(payload.get("variant") or "").strip().lower() or None
    return profile, variant


def _get_embedder_context(
    cache: Dict[Tuple[str, str], EmbedderContext],
    *,
    profile: str,
    variant: Optional[str],
) -> EmbedderContext:
    """Reuse one embedder per effective profile/variant config."""

    key = (str(profile), str(variant or ""))
    if key in cache:
        return cache[key]

    cfg, _raw = load_cfg_and_raw(profile=profile, variant=variant)
    model_ref = str(select_embedding_model_ref(cfg.search, models_dir=Path(cfg.paths.models_dir)))
    fallback_model_name = str(getattr(cfg.search, "embed_model_name", "") or "")
    embedder = build_text_embedder(
        model_name=model_ref,
        device=str(getattr(cfg.model, "device", "") or ""),
        query_prefix=str(getattr(cfg.search, "query_prefix", "query: ")),
        passage_prefix=str(getattr(cfg.search, "passage_prefix", "passage: ")),
        backend=str(getattr(cfg.search, "embedding_backend", "auto")),
        fallback_model_name=fallback_model_name,
    )
    embed_cache = EmbeddingCache(Path(cfg.paths.cache_dir), model_ref) if bool(getattr(cfg.search, "embed_cache", True)) else None
    context = EmbedderContext(model_ref=model_ref, embedder=embedder, cache=embed_cache)
    cache[key] = context
    return context


def _cache_rows(cache: Optional[EmbeddingCache], texts: List[str]) -> Tuple[Dict[int, np.ndarray], List[int], List[str], List[str]]:
    """Load cached vectors when available and return unresolved positions."""

    if cache is None:
        return {}, list(range(len(texts))), [], []

    import hashlib

    hashes = [hashlib.sha1(text.encode("utf-8")).hexdigest() for text in texts]
    cached_map = cache.get_many(hashes)
    loaded: Dict[int, np.ndarray] = {}
    missing_idx: List[int] = []
    missing_hashes: List[str] = []
    missing_texts: List[str] = []
    for index, hval in enumerate(hashes):
        cached = cached_map.get(hval)
        if cached is None:
            missing_idx.append(index)
            missing_hashes.append(hval)
            missing_texts.append(texts[index])
            continue
        try:
            dim, dtype, blob = cached
            dt = np.float16 if str(dtype) == "float16" else np.float32
            arr = np.frombuffer(blob, dtype=dt)
            if int(dim) and int(dim) != int(arr.shape[0]):
                raise ValueError("dimension mismatch")
            loaded[index] = np.asarray(arr, dtype=np.float32)
        except Exception:
            missing_idx.append(index)
            missing_hashes.append(hval)
            missing_texts.append(texts[index])
    return loaded, missing_idx, missing_hashes, missing_texts


def _encode_texts(context: EmbedderContext, texts: List[str]) -> List[Optional[np.ndarray]]:
    """Encode texts with cache reuse and deterministic ordering."""

    if not texts:
        return []
    loaded, missing_idx, missing_hashes, missing_texts = _cache_rows(context.cache, texts)
    out: List[Optional[np.ndarray]] = [loaded.get(index) for index in range(len(texts))]
    if missing_texts:
        arr = np.asarray(context.embedder.encode_passages(missing_texts, batch_size=16), dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        for offset, index in enumerate(missing_idx):
            out[index] = arr[offset]
        if context.cache is not None:
            cache_rows: Dict[str, Any] = {}
            for offset, hval in enumerate(missing_hashes):
                vec = arr[offset]
                cache_rows[hval] = (int(vec.shape[0]), "float16", vec.astype(np.float16).tobytes())
            context.cache.put_many(cache_rows)
    return out


def _merged_rows(merged_annotations: List[Annotation]) -> List[Dict[str, Any]]:
    """Convert merged annotations into flat row dictionaries."""

    rows: List[Dict[str, Any]] = []
    for index, ann in enumerate(merged_annotations, start=1):
        merged_from = list(((ann.extra or {}).get("merged_from") or []))
        rows.append(
            {
                "segment_index": int(index),
                "start_sec": float(ann.start_sec),
                "end_sec": float(ann.end_sec),
                "description": str(ann.description or ""),
                "merged_from": merged_from,
                "member_count": int(len(merged_from)) if merged_from else 1,
                "anomaly_flag": bool(ann.anomaly_flag),
                "anomaly_confidence": float(ann.anomaly_confidence or 0.0),
            }
        )
    return rows


def _collection_summary(rows: List[Dict[str, Any]], taus: Sequence[float]) -> List[Dict[str, Any]]:
    """Aggregate collection-level means for every tau."""

    out: List[Dict[str, Any]] = []
    for tau in taus:
        tau_rows = [row for row in rows if float(row.get("tau") or 0.0) == float(tau)]
        out.append(
            {
                "tau": float(tau),
                "runs": int(len(tau_rows)),
                "mean_final_segments": mean_ignore_none([float_or_none(row.get("final_segments")) for row in tau_rows]),
                "mean_compression_ratio": mean_ignore_none([float_or_none(row.get("compression_ratio")) for row in tau_rows]),
                "mean_dd": mean_ignore_none([float_or_none(row.get("dd")) for row in tau_rows]),
                "mean_mean_segment_duration": mean_ignore_none([float_or_none(row.get("mean_segment_duration")) for row in tau_rows]),
                "mean_tcs": mean_ignore_none([float_or_none(row.get("tcs")) for row in tau_rows]),
                "mean_srr": mean_ignore_none([float_or_none(row.get("srr")) for row in tau_rows]),
                "mean_sns": mean_ignore_none([float_or_none(row.get("sns")) for row in tau_rows]),
                "mean_sdi": mean_ignore_none([float_or_none(row.get("sdi")) for row in tau_rows]),
            }
        )
    return out


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    """Write flat CSV rows with stable field ordering."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def _write_latex_summary(path: Path, rows: List[Dict[str, Any]]) -> None:
    """Write a compact LaTeX-ready table for collection-level tau ablation."""

    lines = [
        r"\begin{tabular}{lrrrrrrrr}",
        r"\hline",
        r"$\tau$ & Final seg. & Compression & DD & Mean dur. & TCS & SRR & SNS & SDI \\",
        r"\hline",
    ]
    for row in rows:
        lines.append(
            " & ".join(
                [
                    _format_float(float_or_none(row.get("tau")), digits=2),
                    _format_float(float_or_none(row.get("mean_final_segments")), digits=2),
                    _format_float(float_or_none(row.get("mean_compression_ratio")), digits=3),
                    _format_float(float_or_none(row.get("mean_dd")), digits=3),
                    _format_float(float_or_none(row.get("mean_mean_segment_duration")), digits=3),
                    _format_float(float_or_none(row.get("mean_tcs")), digits=4),
                    _format_float(float_or_none(row.get("mean_srr")), digits=4),
                    _format_float(float_or_none(row.get("mean_sns")), digits=4),
                    _format_float(float_or_none(row.get("mean_sdi")), digits=4),
                ]
            )
            + r" \\"
        )
    lines.extend(
        [
            r"\hline",
            r"\end{tabular}",
            "",
            "% CLSA is intentionally omitted here.",
            "% Russian and Kazakh outputs in this repository are synchronized translations",
            "% of the English canonical semantic layer, not independent multilingual grounding runs.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_notes(path: Path) -> None:
    """Write a short research note for the ablation bundle."""

    path.write_text(
        "Tau ablation operates on persisted canonical English clip observations.\n"
        "The workflow does not rerun video preprocessing or clip-level VLM generation.\n"
        "Russian and Kazakh layers are synchronized translations from English in this repository,\n"
        "so autonomous cross-lingual grounding metrics such as CLSA are intentionally omitted.\n",
        encoding="utf-8",
    )


def _run_ablation_for_target(
    target: TargetRun,
    *,
    taus: Sequence[float],
    gap_tolerance_sec: float,
    embedder_cache: Dict[Tuple[str, str], EmbedderContext],
) -> List[Dict[str, Any]]:
    """Run all tau values for one persisted target and return per-tau metric rows."""

    observations_payload = _read_json(target.clip_observations_path, default=[])
    if not isinstance(observations_payload, list) or not observations_payload:
        raise RuntimeError(f"No clip observations found in {target.clip_observations_path}")

    metrics_payload = _read_json(target.metrics_path, default={}) or {}
    profile, variant = _load_run_identity(target.run_manifest_path)
    context = _get_embedder_context(embedder_cache, profile=profile, variant=variant)

    annotations = _observations_to_annotations(observations_payload)
    raw_texts = [str(item.description or "") for item in annotations]
    raw_embeddings = _encode_texts(context, raw_texts)
    duration_seconds = float_or_none(metrics_payload.get("video_duration_sec"))
    if duration_seconds is None:
        duration_seconds = max(float(item.end_sec) for item in annotations) if annotations else None

    rows: List[Dict[str, Any]] = []
    for tau in taus:
        merged = semantic_merge_annotations(
            annotations,
            embeddings=raw_embeddings,
            tau=float(tau),
            gap_tolerance=float(gap_tolerance_sec),
        )
        merged_rows = _merged_rows(merged)
        member_groups = [list(item.get("merged_from") or []) for item in merged_rows]
        merged_embeddings = segment_embeddings_from_members(raw_embeddings, member_groups)
        metrics = intrinsic_timeline_metrics(
            duration_seconds=duration_seconds,
            raw_clips=len(annotations),
            merged_segments=merged_rows,
            segment_embeddings=merged_embeddings,
        )
        rows.append(
            {
                "run_id": target.run_id,
                "video_id": target.video_id,
                "profile": profile,
                "variant": variant or "",
                "language": "en",
                "tau": float(tau),
                **metrics,
            }
        )
    return rows


def _sanity_check(summary_rows: List[Dict[str, Any]]) -> str:
    """Build a short sanity-check summary from collection-level means."""

    ordered = sorted(summary_rows, key=lambda item: float(item.get("tau") or 0.0))
    if len(ordered) < 2:
        return "Not enough tau values for a monotonicity sanity check."

    final_segments = [float_or_none(item.get("mean_final_segments")) for item in ordered]
    compression = [float_or_none(item.get("mean_compression_ratio")) for item in ordered]

    monotonic_segments = all(
        left is not None and right is not None and left <= right
        for left, right in zip(final_segments, final_segments[1:])
    )
    monotonic_compression = all(
        left is not None and right is not None and left >= right
        for left, right in zip(compression, compression[1:])
    )
    return (
        f"Sanity check: final_segments monotonic={'yes' if monotonic_segments else 'no'}, "
        f"compression_ratio monotonic={'yes' if monotonic_compression else 'no'}."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Tau ablation over persisted semantic timeline artifacts.")
    parser.add_argument("--input", type=str, required=True, help="Path to data/videos, one video dir, or a batch package root.")
    parser.add_argument("--output", type=str, required=True, help="Directory for ablation CSV and LaTeX outputs.")
    parser.add_argument("--taus", nargs="+", type=float, default=DEFAULT_TAUS, help="Tau grid, for example: 0.85 0.90 0.95")
    parser.add_argument("--video-ids", nargs="*", default=[], help="Optional subset of video ids to ablate.")
    parser.add_argument("--gap-tolerance-sec", type=float, default=1.0, help="Maximum temporal gap for adjacent merges.")
    parser.add_argument("--language", type=str, default="en", help="Canonical layer to analyze. Only 'en' is supported for tau ablation.")
    args = parser.parse_args()

    language = str(args.language or "en").strip().lower() or "en"
    if language != "en":
        raise RuntimeError(
            "Tau ablation is defined on the canonical English clip-observation layer only. "
            "Translated Russian/Kazakh outputs are synchronized layers and are not treated as independent grounding runs."
        )

    taus = [float(value) for value in args.taus]
    if not taus:
        raise RuntimeError("Provide at least one tau value.")

    input_root = _resolve_targets_root(resolve_path(args.input))
    output_dir = resolve_path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    targets = _load_targets(input_root, video_ids=args.video_ids)
    print(f"Resolved targets: {len(targets)} from {input_root}")
    print(f"Tau grid: {', '.join(f'{tau:.2f}' for tau in taus)}")

    all_rows: List[Dict[str, Any]] = []
    embedder_cache: Dict[Tuple[str, str], EmbedderContext] = {}
    started_at = time.perf_counter()
    for index, target in enumerate(targets, start=1):
        print(f"[{index}/{len(targets)}] {target.run_id}: loading persisted observations")
        rows = _run_ablation_for_target(
            target,
            taus=taus,
            gap_tolerance_sec=float(args.gap_tolerance_sec),
            embedder_cache=embedder_cache,
        )
        all_rows.extend(rows)
        short = ", ".join(f"tau={row['tau']:.2f}: final={row['final_segments']}" for row in rows)
        print(f"    {short}")

    summary_rows = _collection_summary(all_rows, taus)

    per_run_csv = output_dir / "tau_ablation_per_run.csv"
    summary_csv = output_dir / "tau_ablation_summary.csv"
    latex_path = output_dir / "tau_ablation_summary.tex"
    manifest_path = output_dir / "tau_ablation_manifest.json"
    notes_path = output_dir / "tau_ablation_notes.txt"

    _write_csv(
        per_run_csv,
        all_rows,
        [
            "run_id",
            "video_id",
            "profile",
            "variant",
            "language",
            "tau",
            "duration_seconds",
            "raw_clips",
            "final_segments",
            "compression_ratio",
            "dd",
            "mean_segment_duration",
            "tcs",
            "srr",
            "sns",
            "sdi",
        ],
    )
    _write_csv(
        summary_csv,
        summary_rows,
        [
            "tau",
            "runs",
            "mean_final_segments",
            "mean_compression_ratio",
            "mean_dd",
            "mean_mean_segment_duration",
            "mean_tcs",
            "mean_srr",
            "mean_sns",
            "mean_sdi",
        ],
    )
    _write_latex_summary(latex_path, summary_rows)
    _write_notes(notes_path)

    manifest = {
        "created_at": now_ts(),
        "input_root": str(input_root),
        "output_dir": str(output_dir),
        "taus": [float(tau) for tau in taus],
        "targets": [target.run_id for target in targets],
        "target_count": int(len(targets)),
        "language": language,
        "gap_tolerance_sec": float(args.gap_tolerance_sec),
        "elapsed_sec": float(time.perf_counter() - started_at),
        "outputs": {
            "per_run_csv": str(per_run_csv),
            "summary_csv": str(summary_csv),
            "latex_table": str(latex_path),
            "notes": str(notes_path),
        },
        "notes": {
            "canonical_layer_only": True,
            "multilingual_clsa_omitted": True,
            "reason": "Russian and Kazakh outputs are synchronized translations from the English semantic layer.",
        },
    }
    write_json(manifest_path, manifest)

    print(f"Per-run CSV: {per_run_csv}")
    print(f"Summary CSV: {summary_csv}")
    print(f"LaTeX table: {latex_path}")
    print(_sanity_check(summary_rows))


if __name__ == "__main__":
    main()
