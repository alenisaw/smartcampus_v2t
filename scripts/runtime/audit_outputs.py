"""
Audit and repair helper for persisted SmartCampus V2T outputs.

Purpose:
- Classify existing outputs as complete or incomplete without mutating artifacts.
- Create safe repair jobs for incomplete process or translation outputs when requested.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.deps import get_backend_paths, load_cfg_and_raw, read_json
from backend.jobs.control import create_job
from scripts.common import resolve_path
from src.utils.video_store import (
    list_output_languages,
    list_output_variants,
    outputs_manifest_path,
    read_run_manifest,
    run_manifest_path,
    validate_process_outputs,
    validate_translation_outputs,
)


def _read_manifest_languages(path: Path) -> List[str]:
    """Read the language keys from one outputs manifest."""

    payload = read_json(path, default=None)
    if not isinstance(payload, dict):
        return []
    languages = payload.get("languages")
    if not isinstance(languages, dict):
        return []
    return sorted(str(lang).strip().lower() for lang in languages.keys() if str(lang).strip())


def _iter_targets(videos_dir: Path, video_ids: Sequence[str]) -> Iterable[Tuple[str, Optional[str]]]:
    """Yield base and variant targets for all requested videos."""

    allowed = {str(item).strip() for item in video_ids if str(item).strip()}
    for child in sorted(videos_dir.iterdir()):
        if not child.is_dir():
            continue
        if allowed and child.name not in allowed:
            continue
        yield child.name, None
        for variant in list_output_variants(videos_dir, child.name):
            yield child.name, variant


def _run_identity(videos_dir: Path, video_id: str, variant: Optional[str]) -> Tuple[str, Optional[str], str]:
    """Resolve profile, variant, and canonical base language for one target."""

    payload = read_run_manifest(run_manifest_path(videos_dir, video_id, variant=variant)) or {}
    profile = str(payload.get("profile") or "main").strip().lower() or "main"
    base_lang = str(payload.get("language") or "en").strip().lower() or "en"
    normalized_variant = str(payload.get("variant") or "").strip().lower() or variant
    return profile, normalized_variant, base_lang


def _translation_languages(videos_dir: Path, video_id: str, variant: Optional[str], base_lang: str) -> List[str]:
    """Return all translated languages that should be audited."""

    langs = set(list_output_languages(videos_dir, video_id, variant=variant))
    langs.update(_read_manifest_languages(outputs_manifest_path(videos_dir, video_id, variant=variant)))
    langs = {lang for lang in langs if lang and lang != base_lang}
    return sorted(langs)


def _repair_job(
    *,
    paths: Any,
    video_id: str,
    profile: str,
    variant: Optional[str],
    language: str,
    base_lang: str,
    scope: str,
) -> Dict[str, Any]:
    """Create one safe repair job using the existing filesystem queue."""

    if scope == "process":
        return create_job(
            paths,
            video_id=video_id,
            job_type="process",
            profile=profile,
            variant=variant,
            language=base_lang,
            source_language=None,
            extra={"profile": profile, "variant": variant, "force_overwrite": True, "repair_mode": True},
            priority="010",
        )
    return create_job(
        paths,
        video_id=video_id,
        job_type="translate",
        profile=profile,
        variant=variant,
        language=language,
        source_language=base_lang,
        extra={"profile": profile, "variant": variant, "force_overwrite": True, "repair_mode": True},
        priority="020",
    )


def _audit_rows(videos_dir: Path, *, video_ids: Sequence[str], repair: bool) -> List[Dict[str, Any]]:
    """Audit all targets and optionally enqueue repair jobs."""

    cfg, raw = load_cfg_and_raw()
    paths = get_backend_paths(cfg, raw)
    rows: List[Dict[str, Any]] = []

    for video_id, variant in _iter_targets(videos_dir, video_ids):
        profile, normalized_variant, base_lang = _run_identity(videos_dir, video_id, variant)
        process_health = validate_process_outputs(videos_dir, video_id, base_lang, variant=normalized_variant)
        repair_job = None
        if repair and not bool(process_health.get("complete")):
            repair_job = _repair_job(
                paths=paths,
                video_id=video_id,
                profile=profile,
                variant=normalized_variant,
                language=base_lang,
                base_lang=base_lang,
                scope="process",
            )
        rows.append(
            {
                "video_id": video_id,
                "variant": normalized_variant or "",
                "scope": "process",
                "language": base_lang,
                "profile": profile,
                "classification": process_health.get("classification"),
                "manifest_status": process_health.get("manifest_status"),
                "complete": bool(process_health.get("complete")),
                "missing_artifacts": ",".join(process_health.get("missing_artifacts") or []),
                "corrupted_artifacts": ",".join(process_health.get("corrupted_artifacts") or []),
                "repair_job_id": str(repair_job.get("job_id") or "") if isinstance(repair_job, dict) else "",
            }
        )

        for lang in _translation_languages(videos_dir, video_id, normalized_variant, base_lang):
            translation_health = validate_translation_outputs(videos_dir, video_id, lang, variant=normalized_variant)
            repair_job = None
            if repair and not bool(translation_health.get("complete")) and bool(process_health.get("complete")):
                repair_job = _repair_job(
                    paths=paths,
                    video_id=video_id,
                    profile=profile,
                    variant=normalized_variant,
                    language=lang,
                    base_lang=base_lang,
                    scope="translation",
                )
            rows.append(
                {
                    "video_id": video_id,
                    "variant": normalized_variant or "",
                    "scope": "translation",
                    "language": lang,
                    "profile": profile,
                    "classification": translation_health.get("classification"),
                    "manifest_status": translation_health.get("manifest_status"),
                    "complete": bool(translation_health.get("complete")),
                    "missing_artifacts": ",".join(translation_health.get("missing_artifacts") or []),
                    "corrupted_artifacts": ",".join(translation_health.get("corrupted_artifacts") or []),
                    "repair_job_id": str(repair_job.get("job_id") or "") if isinstance(repair_job, dict) else "",
                }
            )
    return rows


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    """Write flat audit rows to CSV."""

    fieldnames = [
        "video_id",
        "variant",
        "scope",
        "language",
        "profile",
        "classification",
        "manifest_status",
        "complete",
        "missing_artifacts",
        "corrupted_artifacts",
        "repair_job_id",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def main() -> None:
    """Run the output audit and optional repair planner."""

    parser = argparse.ArgumentParser(description="Audit or repair persisted SmartCampus V2T outputs.")
    parser.add_argument("--videos-dir", default="data/videos", help="Directory with persisted per-video outputs.")
    parser.add_argument("--output", default="data/research/output_audit", help="Directory for JSON/CSV audit outputs.")
    parser.add_argument("--video-ids", nargs="*", default=[], help="Optional subset of video ids to inspect.")
    parser.add_argument("--repair", action="store_true", help="Enqueue safe repair jobs for incomplete outputs.")
    args = parser.parse_args()

    videos_dir = resolve_path(args.videos_dir)
    output_dir = resolve_path(args.output)
    rows = _audit_rows(videos_dir, video_ids=args.video_ids, repair=bool(args.repair))

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "output_audit.csv"
    json_path = output_dir / "output_audit.json"
    _write_csv(csv_path, rows)
    json_path.write_text(json.dumps({"rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")

    complete = sum(1 for row in rows if bool(row.get("complete")))
    print(f"audit_rows={len(rows)} complete={complete} repair={'on' if args.repair else 'off'}")
    print(csv_path)
    print(json_path)


if __name__ == "__main__":
    main()
