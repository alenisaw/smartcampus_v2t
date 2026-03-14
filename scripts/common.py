# scripts/common.py
"""
Shared script utilities for SmartCampus V2T.

Purpose:
- Keep repeated path, JSON, slug, and numeric helper functions in one place for local ops scripts.
- Reduce copy-paste across metrics, evaluation, experiment, and reporting entrypoints.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


def resolve_path(arg: str) -> Path:
    """Resolve relative paths against the current working directory."""

    path = Path(arg)
    if path.is_absolute():
        return path
    return (Path.cwd() / path).resolve()


def read_json_dict(
    path: Path,
    *,
    default: Optional[Dict[str, Any]] = None,
    encoding: str = "utf-8-sig",
) -> Optional[Dict[str, Any]]:
    """Read a JSON object and return a caller-provided default on missing or invalid files."""

    if not path.exists():
        return default
    try:
        obj = json.loads(path.read_text(encoding=encoding))
    except Exception:
        return default
    return obj if isinstance(obj, dict) else default


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    """Write JSON with stable UTF-8 formatting."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def slugify(text: str, *, default: str) -> str:
    """Convert free text into a filesystem-safe token."""

    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(text or "").strip())
    cleaned = cleaned.strip("._-")
    return cleaned or default


def float_or_none(value: Any) -> Optional[float]:
    """Convert scalar values to float when possible."""

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


def safe_div(num: Any, den: Any) -> Optional[float]:
    """Return a float ratio when both operands are valid and denominator is non-zero."""

    n = float_or_none(num)
    d = float_or_none(den)
    if n is None or d is None or d == 0:
        return None
    return float(n / d)
