"""
Inspect resolved runtime config and key Stage 3 knobs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.runtime import load_pipeline_config


def _payload(cfg: Any) -> Dict[str, Any]:
    return {
        "profile": str(cfg.active_profile),
        "variant": str(cfg.active_variant or ""),
        "config_path": str(cfg.config_path),
        "config_fingerprint": str(cfg.config_fingerprint),
        "backend": {
            "host": str(cfg.backend.host),
            "port": int(cfg.backend.port),
        },
        "video": {
            "target_fps": float(cfg.video.target_fps),
            "analysis_fps": float(cfg.video.analysis_fps),
            "decode_resolution": list(cfg.video.decode_resolution),
            "max_frames": cfg.video.max_frames,
        },
        "clips": {
            "window_sec": float(cfg.clips.window_sec),
            "stride_sec": float(cfg.clips.stride_sec),
            "min_clip_frames": int(cfg.clips.min_clip_frames),
            "max_clip_frames": int(cfg.clips.max_clip_frames),
            "keyframe_policy": str(cfg.clips.keyframe_policy),
        },
        "model": {
            "model_name_or_path": str(cfg.model.model_name_or_path),
            "device": str(cfg.model.device),
            "dtype": str(cfg.model.dtype),
            "batch_size": int(cfg.model.batch_size),
            "max_batch_clips": int(cfg.model.max_batch_clips),
            "max_batch_frames": int(cfg.model.max_batch_frames),
            "batch_frame_tolerance": int(cfg.model.batch_frame_tolerance),
            "max_new_tokens": int(cfg.model.max_new_tokens),
            "attn_implementation": str(cfg.model.attn_implementation),
        },
        "llm": {
            "backend": str(cfg.llm.backend),
            "model_id": str(cfg.llm.model_id),
            "vllm_base_url": str(cfg.llm.vllm_base_url),
            "vllm_served_model_name": str(cfg.llm.vllm_served_model_name),
        },
        "search": {
            "ann_backend": str(cfg.search.ann_backend),
            "ann_index_type": str(cfg.search.ann_index_type),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect resolved SmartCampus runtime config.")
    parser.add_argument("--profile", default="main", help="Runtime profile name.")
    parser.add_argument("--variant", default="", help="Optional runtime variant.")
    args = parser.parse_args()

    cfg = load_pipeline_config(profile=args.profile, variant=(args.variant or None))
    print(json.dumps(_payload(cfg), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
