# src/pipeline/video_to_text.py

"""
End-to-end video→text pipeline for SmartCampus (Qwen3-VL only).

Pipeline:
1) Takes temporal clips = lists of frame paths
2) For each clip calls Qwen3-VL backend with language-specific prompt
3) Formats timestamps as [0:00 - 0:04]
4) Returns Annotation list + RunMetrics
"""

import time
from pathlib import Path
from typing import List, Sequence, Tuple

from src.core.qwen_vl_backend import QwenVLBackend
from src.core.types import Annotation, RunMetrics
from src.pipeline.pipeline_config import PipelineConfig


def build_prompt(lang: str) -> str:
    if lang == "ru":
        return (
            "Ты — аналитик видеонаблюдения. Кратко и ясно опиши, что происходит на "
            "этом фрагменте видео. Укажи основное действие, участников, их поведение "
            "и любые необычные события. Опирайся только на визуально видимую информацию. "
            "Пиши на русском языке."
        )

    elif lang == "kz":
        return (
            "Сен — бейнебақылау аналитигісің. Бұл бейне үзіндісінде не болып жатқанын "
            "қысқа әрі анық сипатта. Негізгі әрекетті, қатысушыларды, олардың "
            "іс-әрекеттерін және ерекше жайттарды көрсет. Тек көзге көрінетін "
            "ақпаратқа сүйен. Жауапты қазақ тілінде жаз."
        )

    else:
        return (
            "You are a video surveillance analyst. Briefly and clearly describe what is "
            "happening in this video segment. Mention the main action, people involved, "
            "their behavior, and any unusual events. Describe only what is visible. "
            "Answer in English."
        )


def format_ts(sec: float) -> str:
    m = int(sec // 60)
    s = int(sec % 60)
    return f"{m}:{s:02d}"


class VideoToTextPipeline:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.backend = QwenVLBackend.from_pipeline_config(cfg)

    def run(
        self,
        video_id: str,
        video_duration_sec: float,
        clips: List[List[str]],
        clip_timestamps: List[Tuple[float, float]],
        preprocess_time_sec: float = 0.0,
    ):
        if not clips:
            metrics = RunMetrics(
                video_id=video_id,
                video_duration_sec=float(video_duration_sec),
                num_frames=0,
                num_clips=0,
                avg_clip_duration_sec=0.0,
                preprocess_time_sec=float(preprocess_time_sec),
                model_time_sec=0.0,
                postprocess_time_sec=0.0,
                total_time_sec=float(preprocess_time_sec),
                extra={"reason": "no_clips"},
            )
            return [], metrics

        if len(clips) != len(clip_timestamps):
            raise ValueError("clips and clip_timestamps must match in length")

        num_clips = len(clips)
        num_frames = sum(len(c) for c in clips)
        clip_durations = [end - start for start, end in clip_timestamps]
        avg_clip_duration = float(sum(clip_durations) / len(clip_durations))

        annotations: List[Annotation] = []
        lang = self.cfg.model.language
        prompt = build_prompt(lang)

        t_gen_start = time.perf_counter()

        for idx, (frame_paths, (start, end)) in enumerate(zip(clips, clip_timestamps)):
            paths_seq: Sequence[Path] = [Path(p) for p in frame_paths]

            description = self.backend.describe_clip(
                frame_paths=paths_seq,
                prompt=prompt,
            )

            ts_str = f"[{format_ts(start)} - {format_ts(end)}] "
            description = ts_str + description

            annotations.append(
                Annotation(
                    video_id=video_id,
                    start_sec=float(start),
                    end_sec=float(end),
                    description=description,
                    clip_index=idx,
                    extra=None,
                )
            )

        generation_time = time.perf_counter() - t_gen_start
        total_time = float(preprocess_time_sec) + float(generation_time)

        metrics = RunMetrics(
            video_id=video_id,
            video_duration_sec=float(video_duration_sec),
            num_frames=int(num_frames),
            num_clips=int(num_clips),
            avg_clip_duration_sec=float(avg_clip_duration),
            preprocess_time_sec=float(preprocess_time_sec),
            model_time_sec=float(generation_time),
            postprocess_time_sec=0.0,
            total_time_sec=float(total_time),
            extra=None,
        )

        return annotations, metrics

    def run_for_ui(self, *args, **kwargs):
        annotations, metrics = self.run(*args, **kwargs)
        rows = [
            {
                "clip_index": a.clip_index,
                "start_sec": a.start_sec,
                "end_sec": a.end_sec,
                "description": a.description,
            }
            for a in annotations
        ]
        return rows, metrics
