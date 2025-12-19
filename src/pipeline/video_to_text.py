# src/pipeline/video_to_text.py

"""
Video-to-text pipeline for SmartCampus V2T.
Handles batched clip generation, multilingual prompts and temporal smoothing.
"""

import time
import re
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

from src.core.qwen_vl_backend import QwenVLBackend
from src.core.types import Annotation, RunMetrics
from src.pipeline.pipeline_config import PipelineConfig


def build_prompt(lang: str) -> str:
    if lang == "ru":
        return (
            "Ты — аналитик городской системы видеонаблюдения. На последовательности кадров показан один фрагмент "
            "из камеры наблюдения. Твоя задача — кратко и по фактам описать, что происходит на этом фрагменте, "
            "с акцентом на людях и их поведении.\n\n"
            "Обязательно упомяни: (1) сколько людей примерно и плотность, (2) что они делают, "
            "(3) движение/направление или отсутствие движения, (4) есть ли необычное/опасное.\n\n"
            "Правила:\n"
            "- описывай только то, что однозначно видно;\n"
            "- не придумывай личности, эмоции и причины;\n"
            "- если что-то определить невозможно, напиши «не видно».\n\n"
            "Формат вывода (ОЧЕНЬ ВАЖНО):\n"
            "1) РОВНО 2–4 коротких предложения.\n"
            "2) НЕЛЬЗЯ списки и маркеры: запрещены строки, начинающиеся с '-', '•', '*', а также любые списки/нумерации.\n"
            "3) НЕЛЬЗЯ Markdown (никаких **, __, ```).\n"
            "4) Не повторяй одно и то же слово подряд. Если начинаешь повторяться — остановись и завершай ответ.\n"
            "5) Пиши сплошным текстом, без переносов строк."
        )

    elif lang == "kz":
        return (
            "Сен қалалық бейнебақылау жүйесінің аналитигісің. Кадрлар бір камерадан алынған қысқа фрагментті көрсетеді. "
            "Міндетің — кадрда не болып жатқанын қысқа әрі нақты сипаттау (адамдар және олардың әрекеті).\n\n"
            "Міндетті түрде айт: (1) адамдар саны шамамен және тығыздық, (2) негізгі әрекет, "
            "(3) қозғалыс бағыты/қозғалыс жоқ па, (4) ерекше/қауіпті жағдай бар ма.\n\n"
            "Ережелер:\n"
            "- тек анық көрінетін фактілер;\n"
            "- эмоция/ниет/себеп ойдан қоспа;\n"
            "- анықталмаса, «көрінбейді» деп жаз.\n\n"
            "Шығыс форматы (ӨТЕ МАҢЫЗДЫ):\n"
            "1) ДӘЛ 2–4 қысқа сөйлем.\n"
            "2) ТІЗІМ ҚОЛДАНБА: '-', '•', '*', нөмірлеу, маркерлер және тізім форматтары толық тыйым салынады.\n"
            "3) Markdown қолданба (**, __, ``` жоқ).\n"
            "4) Бір сөзді қатарынан қайталама. Қайталана бастасаң — тоқта да, жауапты аяқта.\n"
            "5) Бір жолмен жаз, жолға бөлме."
        )

    else:
        return (
            "You are an analyst of an urban CCTV system. The clip shows a short segment from a surveillance camera. "
            "Describe briefly and factually what is happening, focusing on people and their behavior.\n\n"
            "Mention: (1) approximate people count and density, (2) main actions, "
            "(3) motion/direction or no motion, (4) any unusual/unsafe situation.\n\n"
            "Rules: only what is clearly visible; no identities/emotions/motives; if unclear say “not visible”.\n\n"
            "Output format (VERY IMPORTANT):\n"
            "1) EXACTLY 2–4 short sentences.\n"
            "2) NO lists/bullets: do NOT start lines with '-', '•', '*', and do not use numbering.\n"
            "3) NO markdown (no **, __, ```).\n"
            "4) Do not repeat the same word consecutively. If repetition starts — stop and end the answer.\n"
            "5) Single paragraph, no line breaks."
        )


def format_ts(sec: float) -> str:
    m = int(sec // 60)
    s = int(sec % 60)
    return f"{m}:{s:02d}"


def strip_prefix(text: str) -> str:
    return re.sub(r"^\[[0-9:\s\-]+\]\s*", "", text)


def _dedupe_repeated_words(text: str, max_repeat: int = 2) -> str:
    words = text.split()
    out: List[str] = []
    run_word: Optional[str] = None
    run_len = 0
    cut = False

    for w in words:
        key = w.lower()
        if key == run_word:
            run_len += 1
            if run_len >= max_repeat:
                cut = True
                break
        else:
            run_word = key
            run_len = 0
        out.append(w)

    t = " ".join(out).strip()
    if cut and t and t[-1] not in ".!?":
        t += "."
    return t


def norm_tokens(text: str) -> List[str]:
    text = strip_prefix(text.lower())
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return [t for t in text.split() if len(t) > 2]


def text_sim(a: str, b: str) -> float:
    ta = set(norm_tokens(a))
    tb = set(norm_tokens(b))
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    return inter / float(min(len(ta), len(tb)))


def smooth_annotations(
    anns: List[Annotation],
    sim_threshold: float = 0.7,
    gap_tolerance: float = 1.0,
) -> List[Annotation]:
    """
    Merge adjacent/nearby segments if they are text-similar.
    No clip_index anywhere; time range is the anchor.
    """
    if not anns:
        return []

    anns = sorted(anns, key=lambda a: a.start_sec)
    merged: List[Annotation] = []

    # Track provenance purely as input positions (stable for this run only).
    # This is optional metadata and safe to ignore in UI/storage.
    current = Annotation(
        video_id=anns[0].video_id,
        start_sec=float(anns[0].start_sec),
        end_sec=float(anns[0].end_sec),
        description=strip_prefix(anns[0].description),
        extra={"merged_from": [0]},
    )

    for src_i, nxt in enumerate(anns[1:], start=1):
        gap = float(nxt.start_sec) - float(current.end_sec)
        sim = text_sim(current.description, nxt.description)

        if gap <= gap_tolerance and sim >= sim_threshold:
            current.end_sec = float(nxt.end_sec)
            if current.extra is None:
                current.extra = {}
            current.extra.setdefault("merged_from", []).append(src_i)
        else:
            merged.append(current)
            current = Annotation(
                video_id=nxt.video_id,
                start_sec=float(nxt.start_sec),
                end_sec=float(nxt.end_sec),
                description=strip_prefix(nxt.description),
                extra={"merged_from": [src_i]},
            )

    merged.append(current)
    return merged


def build_global_summary_prompt(
    lang: str,
    video_id: str,
    duration_sec: float,
    merged_anns: List[Annotation],
) -> str:
    header: List[str] = []

    if lang == "ru":
        header.append(
            "Ты — аналитик системы видеонаблюдения университета. Ниже приведены краткие описания фрагментов одного видео."
        )
        header.append(
            "Составь итоговое обобщённое описание всего видео, используя ТОЛЬКО факты из этих фрагментов. "
            "Не добавляй новых людей, объектов, действий или интерпретаций."
        )
        header.append("Ответ строго в следующей структуре (без угловых скобок):")
        header.append("Краткое описание: 1 предложение.")
        header.append("Основное действие в видео: 1–2 предложения.")
        header.append("Аналитические параметры сцены: строго 5 строк как ниже (каждая строка начинается с '- ').")
        header.append("- Тип сцены: 1–3 слова или 'неизвестно'")
        header.append("- Плотность людей: нет людей / низкая / средняя / высокая")
        header.append("- Тип движения: нет / слабое / стабильное / усиливающееся / хаотичное")
        header.append("- Аномалии: 1–4 слова или 'нет'")
        header.append("- Класс безопасности: норма / подозрительно / опасно")
        header.append(
            "\nВажно: В блоке 'Аналитические параметры сцены' НЕ ПИШИ ничего кроме этих 5 строк. "
            "Только эти 5 строк и ничего лишнего."
        )
        header.append(f"\nВидео: {video_id}, длительность ~{int(duration_sec)} секунд.\n")
        header.append("Фрагменты:")

    elif lang == "kz":
        header.append(
            "Сен университеттің бейнебақылау аналитигісің. Төменде бір бейненің бірнеше фрагментінің сипаттамасы берілген."
        )
        header.append(
            "Тек осы фрагменттердегі фактілерге сүйеніп, бүкіл бейнеге қысқа қорытынды жаз. Жаңа нәрсе ойдан қоспа."
        )
        header.append("Жауап құрылымы:")
        header.append("Қысқаша сипаттама: 1 сөйлем.")
        header.append("Негізгі әрекет: 1–2 сөйлем.")
        header.append("Көрініс параметрлері: төмендегі 5 жолды ҚАТАҢ форматта бер (әр жол '- ' деп басталсын).")
        header.append("- Сахна түрі: 1–3 сөз немесе 'анық емес'")
        header.append("- Адамдар тығыздығы: жоқ / төмен / орташа / жоғары")
        header.append("- Қозғалыс түрі: жоқ / әлсіз / тұрақты / күшейіп жатыр / хаотикалық")
        header.append("- Аномалиялар: 1–4 сөз немесе 'жоқ'")
        header.append("- Қауіп классы: норма / күмәнді / қауіпті")
        header.append("\nМаңызды: Осы 5 жолдан басқа ештеңе қоспа. Бұл блокта тек осы 5 жол болуы керек.")
        header.append(f"\nБейне: {video_id}, ұзақтығы ~{int(duration_sec)} секунд.\n")
        header.append("Фрагменттер:")

    else:
        header.append("You are a university CCTV analyst. Below are several segment descriptions of the same video.")
        header.append("Write a concise global summary using ONLY facts from the segments. Do not invent anything.")
        header.append("Strict structure:")
        header.append("Short summary: 1 sentence.")
        header.append("Main action: 1–2 sentences.")
        header.append("Scene analytics: output EXACTLY 5 lines below (each line must start with '- ').")
        header.append("- Scene type: 1–3 words or 'unknown'")
        header.append("- People density: none / low / medium / high")
        header.append("- Motion type: none / weak / stable / increasing / chaotic")
        header.append("- Anomalies: 1–4 words or 'none'")
        header.append("- Risk class: normal / suspicious / dangerous")
        header.append("\nImportant: In 'Scene analytics' output ONLY those 5 lines, nothing else.")
        header.append(f"\nVideo: {video_id}, duration ~{int(duration_sec)} seconds.\n")
        header.append("Segments:")

    body = [
        f"[{format_ts(ann.start_sec)} - {format_ts(ann.end_sec)}] {strip_prefix(ann.description)}"
        for ann in merged_anns
    ]

    return "\n".join(header + [""] + body)


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
        avg_clip_duration = float(sum((e - s) for s, e in clip_timestamps) / len(clip_timestamps))

        annotations: List[Annotation] = []
        clip_prompt = build_prompt(self.cfg.model.language)
        batch_size = max(1, int(getattr(self.cfg.model, "batch_size", 1)))

        t_gen_start = time.perf_counter()

        for batch_start in range(0, num_clips, batch_size):
            batch_end = min(batch_start + batch_size, num_clips)
            batch_clips = clips[batch_start:batch_end]
            batch_ts = clip_timestamps[batch_start:batch_end]

            batch_paths: List[List[Path]] = [[Path(p) for p in clip_paths] for clip_paths in batch_clips]

            max_len = max(len(seq) for seq in batch_paths)
            padded_batch: List[List[Path]] = []
            for seq in batch_paths:
                if len(seq) == max_len:
                    padded_batch.append(seq)
                else:
                    pad_img = seq[-1]
                    padded_batch.append(seq + [pad_img] * (max_len - len(seq)))

            texts = self.backend.describe_clips_batch(
                batch_frame_paths=padded_batch,
                prompt=clip_prompt,
            )

            # Only hard-stop repetition; everything else enforced by prompt.
            texts = [_dedupe_repeated_words(strip_prefix(t), max_repeat=2) for t in texts]

            for (start, end), text in zip(batch_ts, texts):
                annotations.append(
                    Annotation(
                        video_id=video_id,
                        start_sec=float(start),
                        end_sec=float(end),
                        description=text,
                        extra=None,
                    )
                )

        model_time = time.perf_counter() - t_gen_start
        total_time = float(preprocess_time_sec) + float(model_time)

        merged = smooth_annotations(annotations)

        global_summary: Optional[str] = None
        try:
            if merged:
                max_for_summary = 10
                subset = merged[:max_for_summary]
                summary_prompt = build_global_summary_prompt(
                    lang=self.cfg.model.language,
                    video_id=video_id,
                    duration_sec=video_duration_sec,
                    merged_anns=subset,
                )
                global_summary = self.backend.generate_text(summary_prompt)
        except Exception:
            global_summary = None

        extra: Dict[str, Any] = {}
        if global_summary:
            extra["global_summary"] = global_summary

        metrics = RunMetrics(
            video_id=video_id,
            video_duration_sec=float(video_duration_sec),
            num_frames=int(num_frames),
            num_clips=int(num_clips),
            avg_clip_duration_sec=float(avg_clip_duration),
            preprocess_time_sec=float(preprocess_time_sec),
            model_time_sec=float(model_time),
            postprocess_time_sec=0.0,
            total_time_sec=float(total_time),
            extra=extra or None,
        )

        return merged, metrics

    def run_for_ui(self, *args, **kwargs):
        annotations, metrics = self.run(*args, **kwargs)
        rows = [
            {
                "start_sec": a.start_sec,
                "end_sec": a.end_sec,
                "description": a.description,
            }
            for a in annotations
        ]
        return rows, metrics
