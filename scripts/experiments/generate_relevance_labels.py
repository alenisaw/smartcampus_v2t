"""Build and validate human-reviewed v0.8 retrieval relevance labels.

The command never infers relevance.  Template mode exposes the real processed
segment catalogue to an annotator; export mode accepts only an explicitly
reviewed annotation file and verifies every target against that catalogue.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.video_store import read_segments


class LabelError(RuntimeError):
    pass


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _segment_files(videos_dir: Path, language: str, variant: str) -> list[Path]:
    suffixes = (f"{language}.jsonl.zst", f"{language}.jsonl.gz")
    candidates = [path for suffix in suffixes for path in videos_dir.rglob(suffix)]
    selected: dict[tuple[str, str], Path] = {}
    for path in sorted(candidates):
        relative = path.relative_to(videos_dir)
        parts = relative.parts
        if len(parts) < 4 or "outputs" not in parts or parts[-2] != "segments":
            continue
        video_id = parts[0]
        try:
            output_index = parts.index("outputs")
        except ValueError:
            continue
        actual_variant = "base"
        if len(parts) > output_index + 3 and parts[output_index + 1] == "variants":
            actual_variant = parts[output_index + 2]
        if actual_variant != variant:
            continue
        key = (video_id, actual_variant)
        previous = selected.get(key)
        if previous is None or path.suffix == ".zst":
            selected[key] = path
    return list(selected.values())


def load_catalog(videos_dir: Path, language: str = "en", variant: str = "base") -> dict[str, dict[str, dict[str, Any]]]:
    if not videos_dir.is_dir():
        raise LabelError(f"videos directory does not exist: {videos_dir}")
    files = _segment_files(videos_dir, language, variant)
    if not files:
        raise LabelError(f"no processed {language!r} segment artifacts found for variant {variant!r} in {videos_dir}")
    catalog: dict[str, dict[str, dict[str, Any]]] = {}
    for path in files:
        video_id = path.relative_to(videos_dir).parts[0]
        rows = read_segments(path)
        if not rows and not path.exists():
            raise LabelError(f"processed segment artifact is empty or unreadable: {path}")
        by_id = catalog.setdefault(video_id, {})
        for index, row in enumerate(rows, 1):
            segment_id = str(row.get("segment_id") or "").strip()
            if not segment_id:
                raise LabelError(f"segment {index} has no segment_id: {path}")
            if segment_id in by_id:
                raise LabelError(f"duplicate segment_id {segment_id!r} for video {video_id!r}")
            try:
                start, end = float(row["start_sec"]), float(row["end_sec"])
            except (KeyError, TypeError, ValueError) as exc:
                raise LabelError(f"segment {video_id}/{segment_id} has invalid timestamps") from exc
            if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
                raise LabelError(f"segment {video_id}/{segment_id} has invalid interval [{start}, {end}]")
            by_id[segment_id] = {**row, "segment_id": segment_id, "start_sec": start, "end_sec": end}
    return catalog


def build_template(catalog: dict[str, dict[str, dict[str, Any]]], language: str, variant: str) -> dict[str, Any]:
    segments = []
    for video_id in sorted(catalog):
        for segment in sorted(catalog[video_id].values(), key=lambda row: (row["start_sec"], row["segment_id"])):
            entry = {
                "video_id": video_id,
                "segment_id": segment["segment_id"],
                "start_sec": segment["start_sec"],
                "end_sec": segment["end_sec"],
            }
            for field in ("description", "summary", "event_type", "tags", "objects"):
                if segment.get(field) not in (None, "", []):
                    entry[field] = segment[field]
            segments.append(entry)
    return {
        "version": "1.0",
        "reviewed": False,
        "instructions": "Add queries and relevance grades only after human review; then set reviewed=true.",
        "artifact_language": language,
        "variant": variant,
        "segment_catalog": segments,
        "queries": [],
    }


def _manifest_datasets(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    if not path.is_file():
        raise LabelError(f"manifest does not exist: {path}")
    result: dict[str, str] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            video_id, dataset_id = str(row.get("video_id") or "").strip(), str(row.get("dataset_id") or "").strip()
            if not video_id or not dataset_id:
                raise LabelError("manifest rows must contain nonempty video_id and dataset_id")
            if video_id in result and result[video_id] != dataset_id:
                raise LabelError(f"manifest assigns multiple datasets to video {video_id!r}")
            result[video_id] = dataset_id
    return result


def _interval_target(video: dict[str, dict[str, Any]], start: float, end: float, tolerance: float = 1e-6) -> bool:
    return any(start >= segment["start_sec"] - tolerance and end <= segment["end_sec"] + tolerance for segment in video.values())


def validate_annotations(
    payload: dict[str, Any], catalog: dict[str, dict[str, dict[str, Any]]], datasets: dict[str, str]
) -> list[dict[str, Any]]:
    if payload.get("reviewed") is not True:
        raise LabelError("annotation input must explicitly contain reviewed=true")
    queries = payload.get("queries")
    if not isinstance(queries, list) or not queries:
        raise LabelError("annotation input must contain a nonempty queries list")
    seen_queries: set[str] = set()
    cleaned: list[dict[str, Any]] = []
    for q_index, raw_query in enumerate(queries, 1):
        if not isinstance(raw_query, dict):
            raise LabelError(f"query {q_index} must be an object")
        query = copy.deepcopy(raw_query)
        query_id = str(query.get("query_id") or "").strip()
        text = str(query.get("query") or "").strip()
        language = str(query.get("language") or "").strip().lower()
        if not query_id or query_id in seen_queries or not text or not language:
            raise LabelError(f"query {q_index} needs unique query_id plus nonempty query and language")
        seen_queries.add(query_id)
        relevant = query.get("relevant")
        if not isinstance(relevant, list):
            raise LabelError(f"query {query_id} relevant must be a list")
        if not relevant and query.get("negative_query") is not True:
            raise LabelError(f"query {query_id} has no targets and is not explicitly marked negative_query=true")
        seen_labels: set[str] = set()
        for label in relevant:
            if not isinstance(label, dict):
                raise LabelError(f"query {query_id} contains a non-object relevance label")
            label_id = str(label.get("label_id") or "").strip()
            video_id = str(label.get("video_id") or query.get("video_id") or "").strip()
            if not label_id or label_id in seen_labels or video_id not in catalog:
                raise LabelError(f"query {query_id} label needs a unique label_id and existing video_id")
            seen_labels.add(label_id)
            try:
                grade = int(label.get("grade"))
            except (TypeError, ValueError) as exc:
                raise LabelError(f"query {query_id}/{label_id} grade must be an explicit integer 1, 2, or 3") from exc
            if grade not in (1, 2, 3) or isinstance(label.get("grade"), bool):
                raise LabelError(f"query {query_id}/{label_id} grade must be 1, 2, or 3")
            label["grade"], label["video_id"] = grade, video_id
            segment_id = str(label.get("segment_id") or "").strip()
            has_interval = label.get("start_sec") is not None or label.get("end_sec") is not None
            if not segment_id and not has_interval:
                raise LabelError(f"query {query_id}/{label_id} needs segment_id or start_sec/end_sec")
            segment = catalog[video_id].get(segment_id) if segment_id else None
            if segment_id and segment is None:
                raise LabelError(f"query {query_id}/{label_id} references missing segment {video_id}/{segment_id}")
            if has_interval:
                try:
                    start, end = float(label["start_sec"]), float(label["end_sec"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise LabelError(f"query {query_id}/{label_id} requires numeric start_sec and end_sec") from exc
                if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
                    raise LabelError(f"query {query_id}/{label_id} has an invalid interval")
                if segment and not (start >= segment["start_sec"] - 1e-6 and end <= segment["end_sec"] + 1e-6):
                    raise LabelError(f"query {query_id}/{label_id} interval is outside referenced segment")
                if not segment and not _interval_target(catalog[video_id], start, end):
                    raise LabelError(f"query {query_id}/{label_id} interval is outside all processed segments")
                label["start_sec"], label["end_sec"] = start, end
            query_dataset = str(query.get("dataset_id") or "").strip()
            if query_dataset and datasets and datasets.get(video_id) != query_dataset:
                raise LabelError(f"query {query_id}/{label_id} video does not belong to dataset_id {query_dataset!r}")
        cleaned.append(query)
    return cleaned


def _payload(queries: list[dict[str, Any]], default_language: str) -> dict[str, Any]:
    return {"version": "1.0", "defaults": {"language": default_language, "top_k": 20, "iou_threshold": 0.5}, "queries": queries}


def export_labels(
    annotations: Path, catalog: dict[str, dict[str, dict[str, Any]]], datasets: dict[str, str],
    out_en: Path, out_multilingual: Path | None,
) -> None:
    try:
        source = json.loads(annotations.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise LabelError(f"annotation input does not exist: {annotations}") from exc
    if not isinstance(source, dict):
        raise LabelError("annotation input root must be an object")
    queries = validate_annotations(source, catalog, datasets)
    english = [query for query in queries if query["language"].lower() == "en"]
    multilingual = [query for query in queries if query["language"].lower() != "en"]
    if not english:
        raise LabelError("reviewed annotations contain no English queries")
    if multilingual and out_multilingual is None:
        raise LabelError("reviewed annotations contain multilingual queries but --out-multilingual was not provided")
    if out_multilingual is not None and not multilingual:
        raise LabelError("--out-multilingual was requested but no reviewed non-English queries exist")
    _write_json(out_en, _payload(english, "en"))
    if out_multilingual is not None:
        _write_json(out_multilingual, _payload(multilingual, "multilingual"))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos-dir", type=Path, default=Path("data/videos"))
    parser.add_argument("--artifact-language", default="en")
    parser.add_argument("--variant", default="base")
    parser.add_argument("--manifest", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--build-template", type=Path, metavar="PATH")
    mode.add_argument("--annotations", type=Path, metavar="PATH")
    parser.add_argument("--out-en", type=Path, default=Path("data/relevance/v08_queries_en.json"))
    parser.add_argument("--out-multilingual", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        catalog = load_catalog(args.videos_dir.resolve(), args.artifact_language, args.variant)
        if args.build_template:
            _write_json(args.build_template, build_template(catalog, args.artifact_language, args.variant))
            print(f"review template written to {args.build_template}; no relevance grades were assigned")
        else:
            export_labels(args.annotations, catalog, _manifest_datasets(args.manifest), args.out_en, args.out_multilingual)
            print(f"validated English labels written to {args.out_en}")
    except (LabelError, OSError, json.JSONDecodeError, csv.Error, RuntimeError) as exc:
        print(f"relevance label generation failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
