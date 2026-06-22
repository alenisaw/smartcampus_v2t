"""Build reproducible v0.8 search indexes from persisted segment artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.deps import load_cfg_and_raw
from backend.jobs.index_runtime import build_index_for_language
from src.search.builder import search_config_fingerprint


class IndexBuildError(RuntimeError):
    pass


def _tokens(value: str) -> List[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def rebuild_indexes(
    *, profile: str, variants: Iterable[str], languages: Iterable[str], out_dir: Path,
    loader: Callable[..., Any] = load_cfg_and_raw,
    builder: Callable[..., Dict[str, Any]] = build_index_for_language,
) -> List[Dict[str, Any]]:
    variants = list(variants)
    languages = list(languages)
    if not variants or not languages:
        raise IndexBuildError("At least one variant and language are required")

    rows: List[Dict[str, Any]] = []
    for variant_name in variants:
        variant = variant_name
        cfg, _raw = loader(profile=profile, variant=variant)
        cfg_fp = search_config_fingerprint(cfg)
        for language in languages:
            payload = builder(cfg=cfg, cfg_fp=cfg_fp, language=language)
            row = {
                "language": language,
                "variant": variant_name,
                "num_docs": int(payload.get("num_docs") or 0),
                "dense_valid_count": int(payload.get("dense_valid_count") or 0),
                "ann_backend": str(payload.get("ann_backend") or ""),
                "ann_index_type": str(payload.get("ann_index_type") or ""),
                "embedding_dim": int(payload.get("embed_dim") or 0),
                "config_fingerprint": cfg_fp,
                "build_time_sec": float(payload.get("time_sec") or 0.0),
                "index_dir": str(payload.get("index_dir") or ""),
            }
            if row["num_docs"] <= 0 or not row["index_dir"]:
                raise IndexBuildError(f"Empty index for language={language}, variant={variant_name}")
            rows.append(row)

    out_dir.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with (out_dir / "index_status.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    (out_dir / "index_status.json").write_text(
        json.dumps({"profile": profile, "indexes": rows}, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="v08_auto_server")
    parser.add_argument("--variants", default="base,no_merge")
    parser.add_argument("--languages", default="en")
    parser.add_argument("--out", default="data/research/v08/index_status")
    args = parser.parse_args()
    try:
        rebuild_indexes(profile=args.profile, variants=_tokens(args.variants), languages=_tokens(args.languages), out_dir=Path(args.out).resolve())
    except Exception as exc:
        print(f"Index rebuild failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
