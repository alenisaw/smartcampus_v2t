"""Download and verify every runtime model inside the repository."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict


REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS: Dict[str, Dict[str, str]] = {
    "vlm": {"repo_id": "Qwen/Qwen3-VL-2B-Instruct", "local_dir": "models/qwen3-vl-2b-instruct"},
    "llm": {"repo_id": "Qwen/Qwen3-4B-Instruct-2507", "local_dir": "models/qwen3-4b-instruct-2507"},
    "guard": {"repo_id": "Qwen/Qwen3Guard-Gen-0.6B", "local_dir": "models/qwen3guard-gen-0.6b"},
    "embedding": {"repo_id": "Qwen/Qwen3-Embedding-4B", "local_dir": "models/qwen3-embedding-4b"},
    "reranker": {"repo_id": "Qwen/Qwen3-VL-Reranker-2B", "local_dir": "models/qwen3-vl-reranker-2b"},
    "bge_m3": {"repo_id": "BAAI/bge-m3", "local_dir": "models/bge-m3"},
    "mt_en_ru": {"repo_id": "Helsinki-NLP/opus-mt-en-ru", "local_dir": "models/opus-mt-en-ru"},
    "mt_ru_en": {"repo_id": "Helsinki-NLP/opus-mt-ru-en", "local_dir": "models/opus-mt-ru-en"},
    "mt_kk_ru": {"repo_id": "deepvk/kazRush-kk-ru", "local_dir": "models/kazrush-kk-ru"},
    "mt_ru_kk": {"repo_id": "deepvk/kazRush-ru-kk", "local_dir": "models/kazrush-ru-kk"},
}


def _model_status(name: str, spec: Dict[str, str]) -> Dict[str, object]:
    path = (REPO_ROOT / spec["local_dir"]).resolve()
    if REPO_ROOT not in path.parents:
        raise RuntimeError(f"Model path escapes repository: {path}")
    files = [item for item in path.rglob("*") if item.is_file()] if path.is_dir() else []
    incomplete = [item for item in files if item.name.endswith((".incomplete", ".lock"))]
    has_config = any(item.name in {"config.json", "tokenizer_config.json", "modules.json"} for item in files)
    has_weights = any(item.suffix in {".safetensors", ".bin", ".model"} for item in files)
    return {
        "name": name,
        "repo_id": spec["repo_id"],
        "path": str(path),
        "files": len(files),
        "bytes": sum(item.stat().st_size for item in files),
        "complete": bool(files and has_config and has_weights and not incomplete),
        "incomplete_files": [str(item) for item in incomplete],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", choices=sorted(MODELS), help="Download only this model (repeatable).")
    parser.add_argument("--verify-only", action="store_true", help="Do not access the network; only validate repo-local files.")
    parser.add_argument("--report", default="data/research/model_download_report.json")
    parser.add_argument("--max-workers", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected = args.model or list(MODELS)
    failures = []
    if not args.verify_only:
        # Keep Hugging Face metadata, locks, and temporary blobs off C: too.
        os.environ.setdefault("HF_HOME", str(REPO_ROOT / "models" / ".cache" / "huggingface"))
        os.environ.setdefault("HF_HUB_CACHE", str(REPO_ROOT / "models" / ".cache" / "huggingface" / "hub"))
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise SystemExit("huggingface_hub is required to download models") from exc
        for name in selected:
            spec = MODELS[name]
            local_dir = (REPO_ROOT / spec["local_dir"]).resolve()
            local_dir.mkdir(parents=True, exist_ok=True)
            started = time.monotonic()
            print(f"[{name}] {spec['repo_id']} -> {local_dir}", flush=True)
            try:
                snapshot_download(
                    repo_id=spec["repo_id"],
                    local_dir=str(local_dir),
                    max_workers=max(1, args.max_workers),
                    ignore_patterns=["*.msgpack", "*.h5", "*.ot"],
                )
                print(f"[{name}] downloaded in {time.monotonic() - started:.1f}s", flush=True)
            except Exception as exc:  # preserve remaining downloads, but fail overall
                failures.append(f"{name}: {exc}")
                print(f"[{name}] FAILED: {exc}", file=sys.stderr, flush=True)

    statuses = [_model_status(name, MODELS[name]) for name in selected]
    failures.extend(f"{row['name']}: files are missing or incomplete" for row in statuses if not row["complete"])
    report_path = (REPO_ROOT / args.report).resolve()
    if REPO_ROOT not in report_path.parents:
        raise RuntimeError(f"Report path escapes repository: {report_path}")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps({"models": statuses, "failures": failures}, indent=2), encoding="utf-8")
    for row in statuses:
        print(f"[{row['name']}] complete={row['complete']} files={row['files']} bytes={row['bytes']}")
    if failures:
        print(f"Model verification failed ({len(failures)} issue(s)); see {report_path}", file=sys.stderr)
        return 1
    print(f"All selected models are complete and repo-local. Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
