"""Export v0.8 paper tables and figures from completed experiment artifacts.

This command deliberately performs no metric computation.  It selects and
validates existing result rows, combines like-for-like artifact files, and
renders figures from numeric columns already present in those rows.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Iterable, Sequence


TABLE_SOURCES = {
    "table_1_dataset_summary.csv": (("dataset_scale_summary.csv",),),
    "table_2_archive_construction.csv": (("archive_scale_summary.csv",),),
    "table_3_runtime_profile.csv": (("runtime_summary.csv",),),
    "table_4_retrieval_baselines.csv": (("retrieval_results_summary.csv",),),
    "table_5_ablation_results.csv": (
        ("merge_ablation.csv", "threshold_ablation.csv", "alpha_ablation.csv", "rerank_ablation.csv"),
        ("ablation_results.csv",),
    ),
    "table_6_qualitative_cases.csv": (
        ("qualitative_success_cases.csv", "qualitative_failure_cases.csv"),
        ("qualitative_cases.csv",),
    ),
}

FIGURES = {
    "figure_1_pipeline_runtime_breakdown.png": ("table_3_runtime_profile.csv", ("runtime", "latency", "seconds", "duration")),
    "figure_2_dataset_scale.png": ("table_1_dataset_summary.csv", ("videos", "duration", "clips", "segments")),
    "figure_3_retrieval_baseline_comparison.png": ("table_4_retrieval_baselines.csv", ("map", "ndcg", "recall", "precision", "mrr")),
    "figure_4_merge_threshold_tradeoff.png": ("table_5_ablation_results.csv", ("threshold", "map", "ndcg", "runtime", "segments")),
    "figure_5_query_latency_distribution.png": ("table_4_retrieval_baselines.csv", ("latency", "seconds", "ms")),
}

IDENTIFIER_ALIASES = {
    "run_id": ("run_id",),
    "dataset_id": ("dataset_id", "dataset", "dataset_name"),
    "config_fingerprint": ("config_fingerprint", "fingerprint"),
}


class ExportError(RuntimeError):
    pass


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file() or path.stat().st_size == 0:
        raise ExportError(f"required CSV is missing or empty: {path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = [{key: (value or "").strip() for key, value in row.items()} for row in reader]
    if not fields or not rows:
        raise ExportError(f"required CSV has no header or data rows: {path}")
    return fields, rows


def _find_unique(root: Path, name: str) -> Path:
    matches = sorted(p for p in root.rglob(name) if p.is_file() and "paper_tables" not in p.parts)
    if not matches:
        raise ExportError(f"required source artifact not found: {name}")
    if len(matches) > 1:
        raise ExportError(f"ambiguous source artifact {name}: {', '.join(str(p) for p in matches)}")
    return matches[0]


def _select_sources(root: Path, alternatives: Sequence[Sequence[str]]) -> list[Path]:
    errors: list[str] = []
    for group in alternatives:
        try:
            return [_find_unique(root, name) for name in group]
        except ExportError as exc:
            errors.append(str(exc))
    raise ExportError("; ".join(errors))


def _combine(paths: Sequence[Path], category: bool = False) -> tuple[list[str], list[dict[str, str]]]:
    fields: list[str] = []
    all_rows: list[dict[str, str]] = []
    for path in paths:
        source_fields, rows = _read_csv(path)
        if category and "artifact" not in source_fields:
            source_fields = ["artifact", *source_fields]
            for row in rows:
                row["artifact"] = path.stem
        for field in source_fields:
            if field not in fields:
                fields.append(field)
        all_rows.extend(rows)
    return fields, all_rows


def _validate_identifiers(tables: dict[str, tuple[list[str], list[dict[str, str]]]]) -> None:
    available = {field for fields, _ in tables.values() for field in fields}
    for canonical, aliases in IDENTIFIER_ALIASES.items():
        matching = [field for field in aliases if field in available]
        if not matching or not any(row.get(field) for fields, rows in tables.values() for row in rows for field in matching):
            raise ExportError(f"required identifier is missing or empty: {canonical}")


def _metric_fields(fields: Sequence[str], rows: Sequence[dict[str, str]]) -> list[str]:
    identifiers = {alias for aliases in IDENTIFIER_ALIASES.values() for alias in aliases}
    metrics: list[str] = []
    for field in fields:
        if field in identifiers or field in {"artifact", "method", "variant", "split", "case_type", "video_id", "segment_id"}:
            continue
        values = [row.get(field, "") for row in rows]
        numeric = 0
        for value in values:
            if not value:
                continue
            try:
                number = float(value)
                numeric += math.isfinite(number)
            except ValueError:
                pass
        if numeric:
            metrics.append(field)
            if not any(values):
                raise ExportError(f"metric column is fully empty: {field}")
    return metrics


def _write_csv(path: Path, fields: Sequence[str], rows: Sequence[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _render_figure(path: Path, fields: Sequence[str], rows: Sequence[dict[str, str]], hints: Sequence[str]) -> None:
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise ExportError("Pillow is required to render paper figures") from exc
    numeric = _metric_fields(fields, rows)
    candidates = [f for f in numeric if any(h in f.lower() for h in hints)] or numeric
    if not candidates:
        raise ExportError(f"no real numeric metric is available for {path.name}")
    metric = candidates[0]
    points = [(index, float(row[metric])) for index, row in enumerate(rows) if row.get(metric)]
    points = [(x, y) for x, y in points if math.isfinite(y)]
    if not points:
        raise ExportError(f"selected metric {metric} has no finite values for {path.name}")
    width, height, margin = 1000, 600, 70
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((margin, 20), f"{metric} (source: {path.stem})", fill="black")
    draw.line((margin, margin, margin, height - margin), fill="black", width=2)
    draw.line((margin, height - margin, width - margin, height - margin), fill="black", width=2)
    values = [y for _, y in points]
    low, high = min(values), max(values)
    span = high - low or 1.0
    bar_width = max(1, (width - 2 * margin) // max(1, len(points)))
    for position, (_, value) in enumerate(points):
        x0 = margin + position * bar_width
        if x0 >= width - margin:
            break
        x1 = min(width - margin, x0 + max(1, bar_width - 2))
        y0 = height - margin - int((value - low) / span * (height - 2 * margin - 20))
        draw.rectangle((x0, y0, x1, height - margin), fill="#2474B5")
    image.save(path, format="PNG")


def export(research_dir: Path, out: Path) -> None:
    if not research_dir.is_dir():
        raise ExportError(f"research directory does not exist: {research_dir}")
    out = out.resolve()
    figures_out = out.parent / "paper_figures"
    with tempfile.TemporaryDirectory(prefix="v08-paper-") as temp_name:
        temp = Path(temp_name)
        table_temp, figure_temp = temp / "paper_tables", temp / "paper_figures"
        table_temp.mkdir(); figure_temp.mkdir()
        tables: dict[str, tuple[list[str], list[dict[str, str]]]] = {}
        for output_name, alternatives in TABLE_SOURCES.items():
            sources = _select_sources(research_dir, alternatives)
            fields, rows = _combine(sources, category=len(sources) > 1)
            if not rows:
                raise ExportError(f"table would contain zero rows: {output_name}")
            if not _metric_fields(fields, rows):
                raise ExportError(f"table contains no populated numeric metric column: {output_name}")
            tables[output_name] = (fields, rows)
            _write_csv(table_temp / output_name, fields, rows)
        _validate_identifiers(tables)
        for figure_name, (table_name, hints) in FIGURES.items():
            fields, rows = tables[table_name]
            _render_figure(figure_temp / figure_name, fields, rows, hints)
        if out.exists(): shutil.rmtree(out)
        if figures_out.exists(): shutil.rmtree(figures_out)
        shutil.copytree(table_temp, out)
        shutil.copytree(figure_temp, figures_out)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--research-dir", type=Path, default=Path("data/research/v08"))
    parser.add_argument("--out", type=Path, default=Path("data/research/v08/paper_tables"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        export(args.research_dir.resolve(), args.out)
    except (ExportError, OSError, csv.Error, json.JSONDecodeError) as exc:
        print(f"paper export failed: {exc}", file=sys.stderr)
        return 1
    print(f"paper artifacts exported to {args.out} and {args.out.parent / 'paper_figures'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
