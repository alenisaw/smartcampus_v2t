# backend/http/common.py
"""
Shared backend API helpers for SmartCampus V2T.

Purpose:
- Provide common config/path loading and upload helpers for FastAPI route modules.
- Keep grounded-response helper logic out of the main backend API entrypoint.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.deps import get_backend_paths, load_cfg_and_raw
from backend.retrieval_runtime import (
    generate_grounded_text as _generate_grounded_text,
    grounded_hit_payload as _grounded_hit_payload,
    guard_output_text as _guard_output_text,
    translate_output_text as _translate_output_text,
)


@dataclass(frozen=True)
class ApiContext:
    """Runtime config bundle reused by backend route handlers."""

    cfg: Any
    raw: Dict[str, Any]
    paths: Any


def get_api_context() -> ApiContext:
    """Load the active backend config and derived filesystem paths."""

    cfg, raw = load_cfg_and_raw()
    return ApiContext(cfg=cfg, raw=raw, paths=get_backend_paths(cfg, raw))


def normalize_uploaded_video(raw_path: Path) -> Path:
    """Convert uploaded videos to mp4 when ffmpeg is available."""

    if raw_path.suffix.lower() == ".mp4":
        return raw_path
    if not shutil.which("ffmpeg"):
        return raw_path

    mp4_path = raw_path.with_suffix(".mp4")
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(raw_path),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        str(mp4_path),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode == 0 and mp4_path.exists() and mp4_path.stat().st_size > 0:
            try:
                raw_path.unlink(missing_ok=True)
            except Exception:
                pass
            return mp4_path
    except Exception:
        pass
    return raw_path


def grounded_response_payload(hits: List[Any]) -> Dict[str, Any]:
    """Build shared citation and supporting-hit payloads for grounded responses."""

    return _grounded_hit_payload(hits)


def grounded_text_result(
    cfg: Any,
    *,
    task: str,
    user_input: str,
    hits: List[Any],
    fallback_text: str,
    target_lang: str,
    target_name: str,
) -> tuple[str, str, Optional[str]]:
    """Generate, translate, and guard one grounded text response."""

    text, mode, context = _generate_grounded_text(
        cfg,
        task=task,
        user_input=user_input,
        hits=hits,
        fallback_text=fallback_text,
    )
    text = _translate_output_text(cfg, text, target_lang=target_lang, target_name=target_name)
    return _guard_output_text(cfg, text), mode, context


def variant_filters(variant: Optional[str]) -> Optional[Dict[str, Any]]:
    """Return a minimal variant filter payload when a variant is specified."""

    variant_id = str(variant or "").strip().lower()
    return {"variant": variant_id} if variant_id else None
