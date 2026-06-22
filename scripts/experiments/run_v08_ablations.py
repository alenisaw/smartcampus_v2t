"""Assemble v0.8 ablations strictly from persisted observations and retrieval results."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.deps import load_cfg_and_raw
from src.search.builder import search_config_fingerprint

ALPHAS = (0.0, 0.25, 0.50, 0.75, 1.0)


class AblationError(RuntimeError):
    pass


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        raise AblationError(f"Required artifact not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise AblationError(f"Artifact has no rows: {path}")
    return rows


def _write(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        raise AblationError(f"Refusing to write empty ablation: {path.name}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def _aggregate(rows: Iterable[Dict[str, str]], groups: Sequence[str], values: Iterable[str]) -> List[Dict[str, object]]:
    buckets: Dict[tuple, List[Dict[str, str]]] = {}
    for row in rows:
        key = tuple(str(row.get(group, "")).strip() for group in groups)
        if any(not value for value in key):
            raise AblationError(f"Missing one of {list(groups)} in retrieval results")
        buckets.setdefault(key, []).append(row)
    output = []
    for key, members in sorted(buckets.items()):
        item: Dict[str, object] = {group: value for group, value in zip(groups, key)}
        item["queries"] = len(members)
        for column in values:
            try:
                item[column] = mean(float(row[column]) for row in members)
            except (KeyError, ValueError) as exc:
                raise AblationError(f"Missing/non-numeric {column} in retrieval results") from exc
        output.append(item)
    return output


def run_ablations(*, videos_dir: Path, retrieval_by_query: Path, out_dir: Path, taus: List[float]) -> None:
    retrieval = _read_csv(retrieval_by_query)
    required_identity = {"dataset_id", "profile", "variant", "config_fingerprint"}
    if not required_identity.issubset(retrieval[0]):
        raise AblationError(f"Retrieval artifact lacks identity columns: {sorted(required_identity - set(retrieval[0]))}")
    if "alpha" not in retrieval[0]:
        raise AblationError("Retrieval artifact must contain an actual alpha column for alpha ablation")
    found = {round(float(row["alpha"]), 2) for row in retrieval}
    missing = set(ALPHAS) - found
    if missing:
        raise AblationError(f"Missing measured alpha values: {sorted(missing)}")
    if "rerank_enabled" not in retrieval[0]:
        raise AblationError("Retrieval artifact must contain rerank_enabled for rerank ablation")
    flags = {str(row["rerank_enabled"]).strip().lower() for row in retrieval}
    if not {"true", "false"}.issubset(flags):
        raise AblationError("Both measured rerank_enabled=true and false rows are required")
    identity_names = ("dataset_id", "profile", "variant", "config_fingerprint")
    identities = {tuple(row[name] for name in identity_names) for row in retrieval}
    for identity_key in identities:
        scoped = [row for row in retrieval if tuple(row[name] for name in identity_names) == identity_key]
        scoped_alphas = {round(float(row["alpha"]), 2) for row in scoped}
        if set(ALPHAS) - scoped_alphas:
            raise AblationError(f"Incomplete alpha grid for identity {identity_key}")
        reference_flags = {str(row["rerank_enabled"]).strip().lower() for row in scoped if round(float(row["alpha"]), 2) == 0.5}
        if not {"true", "false"}.issubset(reference_flags):
            raise AblationError(f"Missing rerank on/off measurements at alpha=0.50 for identity {identity_key}")

    out_dir.mkdir(parents=True, exist_ok=True)
    tau_dir = out_dir / "tau"
    command = [sys.executable, str(PROJECT_ROOT / "scripts" / "ablation_tau.py"), "--input", str(videos_dir), "--output", str(tau_dir), "--taus", *[str(v) for v in taus]]
    completed = subprocess.run(command, cwd=PROJECT_ROOT)
    if completed.returncode != 0:
        raise AblationError("Persisted-observation tau ablation failed")
    threshold_rows = _read_csv(tau_dir / "tau_ablation_per_run.csv")
    fp_cache: Dict[tuple, str] = {}
    normalized_threshold = []
    for row in threshold_rows:
        profile = str(row.get("profile") or "").strip()
        variant = str(row.get("variant") or "").strip()
        if not profile:
            raise AblationError("Tau artifact is missing profile identity")
        key = (profile, variant)
        if key not in fp_cache:
            cfg, _raw = load_cfg_and_raw(profile=profile, variant=variant or None)
            fp_cache[key] = search_config_fingerprint(cfg)
        video_id = str(row.get("video_id") or "")
        dataset_id = "avenue_full" if video_id.startswith("avenue_") else "shanghaitech_full" if video_id.startswith("shanghaitech_") else ""
        if not dataset_id:
            raise AblationError(f"Cannot determine dataset_id from persisted video_id={video_id!r}")
        normalized_threshold.append({**row, "dataset_id": dataset_id, "profile": profile, "variant": variant or "base", "config_fingerprint": fp_cache[key]})

    # Attach retrieval metrics only from an explicitly measured reference setting.
    reference_rows = [row for row in retrieval if round(float(row["alpha"]), 2) == 0.5 and str(row["rerank_enabled"]).lower() == "false"]
    reference = {tuple(row[name] for name in ("dataset_id", "profile", "variant", "config_fingerprint")): row for row in reference_rows}
    for row in normalized_threshold:
        match = reference.get(tuple(str(row[name]) for name in ("dataset_id", "profile", "variant", "config_fingerprint")))
        row["P@5"] = match.get("p@5", "") if match else ""
        row["MRR"] = match.get("mrr", "") if match else ""
        row["runtime"] = match.get("latency_ms", "") if match else ""
    _write(out_dir / "threshold_ablation.csv", normalized_threshold)

    identity = list(identity_names)
    alpha_rows = _aggregate(retrieval, [*identity, "alpha"], ("p@5", "mrr", "ndcg@5", "latency_ms"))
    archive_fields = ("raw_clips", "final_segments", "compression_ratio", "dd", "mean_segment_duration", "tcs", "srr", "sns", "sdi")
    archive_reference: Dict[tuple, Dict[str, object]] = {}
    for row in normalized_threshold:
        if abs(float(row.get("tau") or 0.0) - 0.90) > 1e-6:
            continue
        key = tuple(str(row[name]) for name in identity)
        archive_reference.setdefault(key, row)
    for result_rows in (alpha_rows,):
        for row in result_rows:
            archived = archive_reference.get(tuple(str(row[name]) for name in identity))
            if archived is None:
                raise AblationError(f"No tau=0.90 archive metrics for identity {tuple(row[name] for name in identity)}")
            for field in archive_fields:
                row[field] = archived.get(field, "")
            row["P@5"] = row.pop("p@5")
            row["MRR"] = row.pop("mrr")
            row["runtime"] = row.pop("latency_ms")
    _write(out_dir / "alpha_ablation.csv", alpha_rows)
    rerank_reference = [row for row in retrieval if round(float(row["alpha"]), 2) == 0.5]
    rerank_rows = _aggregate(rerank_reference, [*identity, "rerank_enabled"], ("p@5", "mrr", "ndcg@5", "latency_ms"))
    for row in rerank_rows:
        archived = archive_reference.get(tuple(str(row[name]) for name in identity))
        if archived is None:
            raise AblationError(f"No tau=0.90 archive metrics for identity {tuple(row[name] for name in identity)}")
        for field in archive_fields:
            row[field] = archived.get(field, "")
        row["P@5"] = row.pop("p@5")
        row["MRR"] = row.pop("mrr")
        row["runtime"] = row.pop("latency_ms")
    _write(out_dir / "rerank_ablation.csv", rerank_rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos-dir", default="data/videos")
    parser.add_argument("--retrieval-by-query", required=True, help="Measured per-query CSV containing alpha and rerank_enabled")
    parser.add_argument("--out", default="data/research/v08/ablations")
    parser.add_argument("--taus", nargs="+", type=float, default=[0.85, 0.90, 0.95])
    args = parser.parse_args()
    try:
        run_ablations(videos_dir=Path(args.videos_dir).resolve(), retrieval_by_query=Path(args.retrieval_by_query).resolve(), out_dir=Path(args.out).resolve(), taus=args.taus)
    except Exception as exc:
        print(f"Ablation run failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
