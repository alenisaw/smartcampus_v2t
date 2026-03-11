# src/utils/video_store.py
"""
Video storage helpers for SmartCampus V2T.

Purpose:
- Manage per-video artifact paths, manifests, and output persistence helpers.
- Keep storage layout logic consistent across pipeline, backend, and UI code.
"""

from __future__ import annotations

import io
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    import zstandard as zstd
except Exception:
    zstd = None

VIDEO_EXTS = (".mp4", ".mov", ".m4v", ".mkv", ".avi")


def now_ts() -> float:
    """Return the current Unix timestamp as float."""

    return float(time.time())


def _normalize_variant(variant: Optional[str]) -> Optional[str]:
    """Normalize a variant identifier for path usage."""

    if variant is None:
        return None
    text = str(variant).strip().lower()
    return text or None


def video_dir(videos_dir: Path, video_id: str) -> Path:
    """Return the root folder for one video."""

    return Path(videos_dir) / str(video_id)


def raw_dir(videos_dir: Path, video_id: str) -> Path:
    """Return the raw video folder."""

    return video_dir(videos_dir, video_id) / "raw"


def cache_dir(videos_dir: Path, video_id: str) -> Path:
    """Return the per-video cache folder."""

    return video_dir(videos_dir, video_id) / "cache"


def outputs_root_dir(videos_dir: Path, video_id: str) -> Path:
    """Return the shared outputs root."""

    return video_dir(videos_dir, video_id) / "outputs"


def variant_root_dir(videos_dir: Path, video_id: str) -> Path:
    """Return the folder holding all variant-scoped outputs."""

    return outputs_root_dir(videos_dir, video_id) / "variants"


def outputs_dir(videos_dir: Path, video_id: str, variant: Optional[str] = None) -> Path:
    """Return the output directory for base or variant-scoped artifacts."""

    normalized = _normalize_variant(variant)
    if normalized is None:
        return outputs_root_dir(videos_dir, video_id)
    return variant_root_dir(videos_dir, video_id) / normalized


def segments_dir(videos_dir: Path, video_id: str, variant: Optional[str] = None) -> Path:
    """Return the segments output directory."""

    return outputs_dir(videos_dir, video_id, variant=variant) / "segments"


def summaries_dir(videos_dir: Path, video_id: str, variant: Optional[str] = None) -> Path:
    """Return the summaries output directory."""

    return outputs_dir(videos_dir, video_id, variant=variant) / "summaries"


def segments_path(videos_dir: Path, video_id: str, lang: str, variant: Optional[str] = None) -> Path:
    """Return the path of the segments artifact for one language."""

    suffix = ".jsonl.zst" if zstd is not None else ".jsonl.gz"
    return segments_dir(videos_dir, video_id, variant=variant) / f"{lang}{suffix}"


def summary_path(videos_dir: Path, video_id: str, lang: str, variant: Optional[str] = None) -> Path:
    """Return the path of the summary artifact for one language."""

    return summaries_dir(videos_dir, video_id, variant=variant) / f"{lang}.json"


def metrics_path(videos_dir: Path, video_id: str, variant: Optional[str] = None) -> Path:
    """Return the metrics file path."""

    return outputs_dir(videos_dir, video_id, variant=variant) / "metrics.json"


def outputs_manifest_path(videos_dir: Path, video_id: str, variant: Optional[str] = None) -> Path:
    """Return the manifest path for base or variant-scoped outputs."""

    return outputs_dir(videos_dir, video_id, variant=variant) / "manifest.json"


def run_manifest_path(videos_dir: Path, video_id: str, variant: Optional[str] = None) -> Path:
    """Return the per-run manifest path for base or variant-scoped outputs."""

    return outputs_dir(videos_dir, video_id, variant=variant) / "run_manifest.json"


def batch_manifest_path(videos_dir: Path, video_id: str) -> Path:
    """Return the top-level experimental batch manifest path."""

    return outputs_root_dir(videos_dir, video_id) / "experimental_manifest.json"


def video_manifest_path(videos_dir: Path, video_id: str) -> Path:
    """Return the per-video manifest path."""

    return video_dir(videos_dir, video_id) / "manifest.json"


def ensure_video_dirs(videos_dir: Path, video_id: str, variant: Optional[str] = None) -> Dict[str, Path]:
    """Ensure the core directory structure exists for a video and optional variant."""

    vdir = video_dir(videos_dir, video_id)
    rdir = raw_dir(videos_dir, video_id)
    cdir = cache_dir(videos_dir, video_id)
    odir = outputs_dir(videos_dir, video_id, variant=variant)
    sdir = odir / "segments"
    sumdir = odir / "summaries"

    required = [vdir, rdir, cdir, outputs_root_dir(videos_dir, video_id), sdir, sumdir]
    if variant is not None:
        required.append(variant_root_dir(videos_dir, video_id))

    for item in required:
        item.mkdir(parents=True, exist_ok=True)

    return {
        "video": vdir,
        "raw": rdir,
        "cache": cdir,
        "outputs": odir,
        "segments": sdir,
        "summaries": sumdir,
    }


def pick_video_file(raw_folder: Path) -> Optional[Path]:
    """Pick the preferred raw video file from a folder."""

    if not raw_folder.exists():
        return None
    files = [item for item in raw_folder.iterdir() if item.is_file() and item.suffix.lower() in VIDEO_EXTS]
    if not files:
        return None

    preference = list(VIDEO_EXTS)

    def key(path: Path) -> tuple:
        ext = path.suffix.lower()
        try:
            return (preference.index(ext), path.name.lower())
        except ValueError:
            return (len(preference), path.name.lower())

    files.sort(key=key)
    return files[0]


def find_video_file(videos_dir: Path, video_id: str) -> Optional[Path]:
    """Find the primary raw video file for a video id."""

    return pick_video_file(raw_dir(videos_dir, video_id))


def list_output_languages(videos_dir: Path, video_id: str, variant: Optional[str] = None) -> List[str]:
    """List languages available in base or variant-scoped outputs."""

    folder = segments_dir(videos_dir, video_id, variant=variant)
    if not folder.exists():
        return []

    langs: List[str] = []
    for item in sorted(folder.glob("*.jsonl.zst")):
        name = item.name
        langs.append(name[: -len(".jsonl.zst")])
    for item in sorted(folder.glob("*.jsonl.gz")):
        name = item.name
        langs.append(name[: -len(".jsonl.gz")])
    return langs


def list_output_variants(videos_dir: Path, video_id: str) -> List[str]:
    """List variant ids that have their own output folders."""

    root = variant_root_dir(videos_dir, video_id)
    if not root.exists():
        return []
    return [item.name for item in sorted(root.iterdir()) if item.is_dir()]


def list_videos(videos_dir: Path) -> List[Dict[str, Any]]:
    """List videos with base and variant output metadata."""

    out: List[Dict[str, Any]] = []
    root = Path(videos_dir)
    if not root.exists():
        return out

    for vdir in sorted(root.iterdir()):
        if not vdir.is_dir():
            continue

        video_id = vdir.name
        raw_file = pick_video_file(vdir / "raw")
        if raw_file is None:
            continue

        item: Dict[str, Any] = {
            "video_id": video_id,
            "filename": raw_file.name,
            "path": str(raw_file),
            "languages": list_output_languages(videos_dir, video_id),
            "variants": {},
        }

        try:
            stat = raw_file.stat()
            item["size_bytes"] = int(stat.st_size)
            item["mtime"] = float(stat.st_mtime)
        except Exception:
            pass

        for variant in list_output_variants(videos_dir, video_id):
            item["variants"][variant] = {
                "languages": list_output_languages(videos_dir, video_id, variant=variant),
            }

        out.append(item)
    return out


def _read_json(path: Path, default: Any = None) -> Any:
    """Read JSON and return a default on missing or invalid files."""

    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _write_json(path: Path, data: Any) -> None:
    """Write JSON atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _read_jsonl_zst(path: Path) -> List[Dict[str, Any]]:
    """Read `.jsonl.zst` rows."""

    if zstd is None:
        raise RuntimeError("zstandard is not installed")

    rows: List[Dict[str, Any]] = []
    with path.open("rb") as handle:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(handle) as reader:
            stream = io.TextIOWrapper(reader, encoding="utf-8")
            for line in stream:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if isinstance(obj, dict):
                    rows.append(obj)
    return rows


def _read_jsonl_gz(path: Path) -> List[Dict[str, Any]]:
    """Read `.jsonl.gz` rows."""

    import gzip

    rows: List[Dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def _write_jsonl_gz(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    """Write `.jsonl.gz` rows atomically."""

    import gzip

    tmp = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    tmp.replace(path)


def read_segments(path: Path) -> List[Dict[str, Any]]:
    """Read segments from zstd or gzip output."""

    if path.exists():
        try:
            if path.suffix == ".zst":
                if zstd is None:
                    gz_path = path.with_suffix(".jsonl.gz")
                    return _read_jsonl_gz(gz_path) if gz_path.exists() else []
                return _read_jsonl_zst(path)
            if path.suffix == ".gz":
                return _read_jsonl_gz(path)
        except Exception:
            return []

    if path.suffix == ".zst":
        fallback = path.with_suffix(".jsonl.gz")
        if fallback.exists():
            try:
                return _read_jsonl_gz(fallback)
            except Exception:
                return []

    if path.suffix == ".gz":
        fallback = path.with_suffix(".jsonl.zst")
        if fallback.exists() and zstd is not None:
            try:
                return _read_jsonl_zst(fallback)
            except Exception:
                return []
    return []


def write_segments(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    """Write segments to zstd or gzip output."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".gz":
        _write_jsonl_gz(path, rows)
        return
    if path.suffix == ".zst" and zstd is None:
        _write_jsonl_gz(path.with_suffix(".jsonl.gz"), rows)
        return

    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        cctx = zstd.ZstdCompressor(level=6)
        with cctx.stream_writer(handle) as writer:
            stream = io.TextIOWrapper(writer, encoding="utf-8", newline="\n")
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            stream.flush()
    tmp.replace(path)


def read_summary(path: Path) -> Optional[Dict[str, Any]]:
    """Read a summary JSON payload."""

    obj = _read_json(path, default=None)
    return obj if isinstance(obj, dict) else None


def write_summary(path: Path, summary: Any, lang: str, extra: Optional[Dict[str, Any]] = None) -> None:
    """Write a summary JSON payload from either plain text or schema v2 dicts."""

    if isinstance(summary, dict):
        payload = dict(summary)
        payload.setdefault("language", str(lang))
        payload.setdefault("updated_at", now_ts())
        if extra:
            merged_extra = payload.get("extra")
            if not isinstance(merged_extra, dict):
                merged_extra = {}
            merged_extra.update(extra)
            payload["extra"] = merged_extra
    else:
        payload = {
            "language": str(lang),
            "summary": str(summary),
            "updated_at": now_ts(),
            "extra": extra or {},
        }
    _write_json(path, payload)


def read_metrics(path: Path) -> Optional[Dict[str, Any]]:
    """Read metrics JSON."""

    obj = _read_json(path, default=None)
    return obj if isinstance(obj, dict) else None


def write_metrics(path: Path, metrics: Dict[str, Any]) -> None:
    """Write metrics JSON."""

    _write_json(path, metrics)


def update_metrics(path: Path, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Merge metrics payload into an existing metrics file."""

    current = read_metrics(path) or {}
    if not isinstance(current, dict):
        current = {}

    for key, value in (payload or {}).items():
        if isinstance(value, dict) and isinstance(current.get(key), dict):
            current[key].update(value)
        else:
            current[key] = value

    _write_json(path, current)
    return current


def write_batch_manifest(videos_dir: Path, video_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Write the top-level experimental batch manifest."""

    target = batch_manifest_path(videos_dir, video_id)
    current = _read_json(target, default={}) or {}
    if not isinstance(current, dict):
        current = {}
    current.update(payload or {})
    current["video_id"] = str(video_id)
    current["updated_at"] = now_ts()
    _write_json(target, current)
    return current


def write_run_manifest(videos_dir: Path, video_id: str, payload: Dict[str, Any], variant: Optional[str] = None) -> Dict[str, Any]:
    """Write a run manifest with config and output references."""

    target = run_manifest_path(videos_dir, video_id, variant=variant)
    current = _read_json(target, default={}) or {}
    if not isinstance(current, dict):
        current = {}
    current.update(payload or {})
    current["video_id"] = str(video_id)
    current["variant"] = _normalize_variant(variant)
    current["updated_at"] = now_ts()
    _write_json(target, current)
    return current


def update_outputs_manifest(
    videos_dir: Path,
    video_id: str,
    lang: str,
    *,
    variant: Optional[str] = None,
    source_lang: Optional[str] = None,
    model_name: Optional[str] = None,
    status: Optional[str] = None,
    error: Optional[str] = None,
    job_id: Optional[str] = None,
    note: Optional[str] = None,
) -> None:
    """Update base or variant-scoped outputs manifest and keep a root variant index."""

    normalized_variant = _normalize_variant(variant)
    manifest_path = outputs_manifest_path(videos_dir, video_id, variant=normalized_variant)
    manifest = _read_json(manifest_path, default={}) or {}
    if not isinstance(manifest, dict):
        manifest = {}

    languages = manifest.get("languages")
    if not isinstance(languages, dict):
        languages = {}

    entry = languages.get(lang)
    if not isinstance(entry, dict):
        entry = {}

    entry["updated_at"] = now_ts()
    if normalized_variant:
        entry["variant"] = normalized_variant
    if source_lang:
        entry["source_lang"] = str(source_lang)
    if model_name:
        entry["model_name"] = str(model_name)
    if status:
        entry["status"] = str(status)
    if error:
        entry["last_error"] = str(error)
    if job_id:
        entry["job_id"] = str(job_id)
    if note:
        entry["note"] = str(note)

    languages[str(lang)] = entry
    manifest["video_id"] = str(video_id)
    manifest["updated_at"] = now_ts()
    manifest["variant"] = normalized_variant
    manifest["languages"] = languages
    _write_json(manifest_path, manifest)

    if normalized_variant:
        root_manifest_path = outputs_manifest_path(videos_dir, video_id)
        root_manifest = _read_json(root_manifest_path, default={}) or {}
        if not isinstance(root_manifest, dict):
            root_manifest = {}
        variants = root_manifest.get("variants")
        if not isinstance(variants, dict):
            variants = {}
        variants[normalized_variant] = {
            "manifest_path": str(manifest_path),
            "updated_at": now_ts(),
            "languages": sorted(list(languages.keys())),
        }
        root_manifest["video_id"] = str(video_id)
        root_manifest["updated_at"] = now_ts()
        root_manifest["variants"] = variants
        _write_json(root_manifest_path, root_manifest)
