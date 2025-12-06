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
            "Нужно описать:\n"
            "- примерное количество людей и их плотность;\n"
            "- основные действия (идут, стоят, бегут, взаимодействуют, ссорятся и т.д.);\n"
            "- направление движения или отсутствие движения;\n"
            "- любые необычные или потенциально опасные ситуации, если они видны.\n\n"
            "Правила:\n"
            "- описывай только то, что однозначно видно;\n"
            "- не придумывай личности, эмоции, причины происходящего;\n"
            "- если что-то определить невозможно, напиши «не видно».\n\n"
            "Формат ответа: строго 2–4 коротких предложения. "
            "Не используй списки, маркеры и угловые скобки. "
            "Каждое предложение должно быть коротким и содержать только одну мысль."
        )

    elif lang == "kz":
        return (
            "Сен қалалық бейнебақылау жүйесінің аналитигісің. Кадрлар бір камерадан алынған фрагментті көрсетеді. "
            "Міндетің — адамдар мен олардың әрекеттеріне назар аудара отырып, не болып жатқанын қысқа әрі нақты сипаттау.\n\n"
            "Көрсету керек:\n"
            "- адамдардың шамамен саны;\n"
            "- негізгі әрекеттері;\n"
            "- қозғалыс бағыты;\n"
            "- ерекше немесе қауіпті жағдайлар болса, оларды атап өту.\n\n"
            "Ережелер:\n"
            "- тек анық көрінетін фактілерді сипатта;\n"
            "- адамдардың сезімін, кәсібін немесе ниетін ойдан қоспа;\n"
            "- анықталмайтын нәрсе болса, оны «көрінбейді» деп жаз.\n\n"
            "Формат: қатаң түрде 2–4 қысқа сөйлем. "
            "Тізімдер, маркерлер және бұрыштық жақшалар қолдануға болмайды. "
            "Әр сөйлем қысқа және бір ғана ойды қамтуы тиіс."
        )

    else:
        return (
            "You are an analyst of an urban CCTV system. The clip shows a short segment from a surveillance camera. "
            "Describe briefly and factually what is happening, focusing on people and their behavior.\n\n"
            "Include:\n"
            "- approximate number of people;\n"
            "- main actions;\n"
            "- movement pattern;\n"
            "- any unusual or unsafe situations.\n\n"
            "Rules:\n"
            "- describe only what is clearly visible;\n"
            "- do not infer identity, emotions, or motivations;\n"
            "- if something cannot be determined, explicitly say “not visible”.\n\n"
            "Output format: strictly 2–4 short sentences. "
            "No lists, no bullet points, no angle brackets. "
            "Each sentence must be brief and contain only one idea."
        )





def format_ts(sec: float) -> str:
    m = int(sec // 60)
    s = int(sec % 60)
    return f"{m}:{s:02d}"


def strip_prefix(text: str) -> str:
    return re.sub(r"^\[[0-9:\s\-]+\]\s*", "", text)


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

    if not anns:
        return []

    anns = sorted(anns, key=lambda a: a.start_sec)
    merged: List[Annotation] = []

    current = Annotation(
        video_id=anns[0].video_id,
        start_sec=anns[0].start_sec,
        end_sec=anns[0].end_sec,
        description=strip_prefix(anns[0].description),
        clip_index=0,
        extra={"merged_clip_indices": [anns[0].clip_index]},
    )

    for nxt in anns[1:]:
        gap = float(nxt.start_sec) - float(current.end_sec)
        sim = text_sim(current.description, nxt.description)

        if gap <= gap_tolerance and sim >= sim_threshold:

            current.end_sec = float(nxt.end_sec)
            current.extra["merged_clip_indices"].append(nxt.clip_index)
        else:

            merged.append(current)
            current = Annotation(
                video_id=nxt.video_id,
                start_sec=nxt.start_sec,
                end_sec=nxt.end_sec,
                description=strip_prefix(nxt.description),
                clip_index=0,
                extra={"merged_clip_indices": [nxt.clip_index]},
            )

    merged.append(current)


    for i, ann in enumerate(merged):
        ann.clip_index = i

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
            "Ты — аналитик системы видеонаблюдения университета. Ниже приведены описания фрагментов одного видео."
        )
        header.append(
            "На их основе составь обобщённое описание всего видео. Используй только факты из описаний, "
            "не выдумывай новые объекты, людей, игры или события."
        )
        header.append(
            "Ответ должен быть на русском языке и строго состоять из следующих пунктов "
            "(каждый пункт 1–2 предложения, без угловых скобок):"
        )
        header.append(
            "Краткое описание: общее содержание всего видео одним коротким предложением."
        )
        header.append(
            "Основное действие в видео: какие ключевые действия происходят в целом, что в основном делают люди "
            "или главный участник."
        )
        header.append(
            "Участники: кто участвует в видео (один человек, небольшая группа, толпа), примерное количество людей "
            "и роли, но только если роли однозначно видны по изображению."
        )
        header.append(
            "Контекст: где происходит действие (улица, коридор, холл, аудитория, транспорт и т.п.), какие объекты "
            "или элементы интерфейса заметны, есть ли понятное время суток; если контекст определить нельзя, так и напиши."
        )
        header.append(
            "Аналитические параметры сцены: оцени тип сцены (коридор, холл, улица, помещение и т.п.), "
            "плотность людей (низкая / средняя / высокая), динамику движения "
            "(движение слабое / стабильное / усиливающееся / хаотичное) и наличие аномалий "
            "(нет / есть, и очень кратко — каких именно, если они однозначно видны: бег, толкотня, резкие ускорения, "
            "агрессивные действия, оставленные предметы, пересечение запрещённой зоны, длительное стояние без движения и др.). "
            "Заверши оценкой общей ситуации: спокойная / нейтральная / напряжённая / потенциально опасная."
        )
        header.append(
            "Класс безопасности сцены: оцени уровень безопасности одним словом — строго выбери ОДНО значение "
            "из списка: норма / подозрительно / опасно. Никаких других формулировок не используй."
        )
        header.append(
            f"\nВидео: {video_id}, примерная длительность — около {int(duration_sec)} секунд.\n"
        )
        header.append("Описания фрагментов:")

    elif lang == "kz":
        header.append(
            "Сен университеттің бейнебақылау аналитигісің. Төменде бір бейненің бірнеше фрагментінің сипаттамалары берілген."
        )
        header.append(
            "Сол сипаттамаларға сүйене отырып, бүкіл бейненің жалпы мазмұнын сипатта. Жаңа объектілерді немесе адамдарды ойдан қоспа."
        )
        header.append(
            "Жауап қазақ тілінде болсын және келесі тармақтардан тұрсын "
            "(әр тармақ 1–2 сөйлем, бұрыштық жақшаларсыз):"
        )
        header.append(
            "Қысқаша сипаттама: бейненің жалпы мәнін бір қысқа сөйлеммен бер."
        )
        header.append(
            "Бейнедегі негізгі әрекет: бейнеде жалпы қандай негізгі әрекеттер болып жатыр."
        )
        header.append(
            "Қатысушылар: бейнеде кімдер бар (бір адам, шағын топ, көп адам), шамамен қанша адам және "
            "егер айқын көрінсе, қандай рөлдерде."
        )
        header.append(
            "Контекст: әрекет қай жерде болып жатыр (көше, дәліз, холл, аудитория, көлік және т.б.), "
            "қандай объектілер немесе интерфейс элементтері көрінеді, тәулік уақыты анық па; "
            "егер контекст анықталмаса, соны жаз."
        )
        header.append(
            "Көріністің аналитикалық параметрлері: көрініс типін сипатта (мысалы, дәліз, холл, көше, ғимарат іші), "
            "адамдар тығыздығын бағала (төмен / орташа / жоғары), қозғалыс динамикасын көрсет "
            "(қозғалыс әлсіз / тұрақты / күшейіп жатыр / хаотикалық) және аномалиялардың бар-жоғын жаз "
            "(жоқ / бар, егер айқын көрінсе, қысқа түрде мысалы: жүгіру, итеріс, күрт жеделдеу, агрессивті әрекеттер, "
            "тастап кеткен заттар, шектелген аймаққа өту, ұзақ уақыт қозғалмай тұру және т.б.). "
            "Соңында жалпы жағдайды бағала: тыныш / бейтарап / шиеленісті / әлеуетті қауіпті."
        )
        header.append(
            "Көріністің қауіпсіздік класы: қауіп деңгейін бір сөзбен бағала. Төмендегі нұсқалардың бірін ғана таңда: "
            "норма / күмәнді / қауіпті. Басқа сөздер қолданба."
        )
        header.append(
            f"\nБейне: {video_id}, шамамен ұзақтығы — {int(duration_sec)} секунд.\n"
        )
        header.append("Фрагмент сипаттамалары:")

    else:
        header.append(
            "You are a university CCTV analyst. Below are descriptions of several segments of the same video."
        )
        header.append(
            "Based on them, produce a global description of the whole video. Use only information from the segments; "
            "do not invent new objects, people or events."
        )
        header.append(
            "Answer in English and strictly follow this structure "
            "(1–2 sentences per item, no angle brackets):"
        )
        header.append(
            "Short summary: one short sentence with the overall content of the video."
        )
        header.append(
            "Main action in the video: what key actions are happening overall, what people or the main subject mostly do."
        )
        header.append(
            "Participants: who appears in the video (single person, small group, crowd), approximate number of people "
            "and roles, but only if roles are clearly visible."
        )
        header.append(
            "Context: where the scene takes place (street, corridor, hall, classroom, vehicle, etc.), "
            "what objects or interface elements are visible, and whether the time of day is clear; "
            "if the context cannot be determined, say so."
        )
        header.append(
            "Scene analytics: briefly describe the scene type (e.g. corridor, hall, street, indoor), "
            "people density (low / medium / high), motion dynamics "
            "(weak / stable / increasing / chaotic), and anomaly presence "
            "(none / present, and very briefly what kind: running, pushing, sudden acceleration, aggressive actions, "
            "abandoned objects, crossing restricted area, long-standing without movement, etc.). "
            "Finish with an overall situation assessment: calm / neutral / tense / potentially risky."
        )
        header.append(
            "Scene risk class: assign a single-word safety label using exactly ONE of these values: "
            "normal / suspicious / dangerous. Do not use any other wording."
        )
        header.append(
            f"\nVideo: {video_id}, approximate duration — about {int(duration_sec)} seconds.\n"
        )
        header.append("Segment descriptions:")

    body: List[str] = []
    for ann in merged_anns:
        body.append(
            f"- [{format_ts(ann.start_sec)} - {format_ts(ann.end_sec)}] {ann.description}"
        )

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
        avg_clip_duration = float(
            sum((e - s) for s, e in clip_timestamps) / len(clip_timestamps)
        )

        annotations: List[Annotation] = []
        clip_prompt = build_prompt(self.cfg.model.language)
        batch_size = max(1, int(getattr(self.cfg.model, "batch_size", 1)))

        t_gen_start = time.perf_counter()


        for batch_start in range(0, num_clips, batch_size):
            batch_end = min(batch_start + batch_size, num_clips)
            batch_clips = clips[batch_start:batch_end]
            batch_ts = clip_timestamps[batch_start:batch_end]

            batch_paths: List[List[Path]] = [
                [Path(p) for p in clip_paths] for clip_paths in batch_clips
            ]

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

            for local_idx, ((start, end), text) in enumerate(zip(batch_ts, texts)):
                global_idx = batch_start + local_idx
                annotations.append(
                    Annotation(
                        video_id=video_id,
                        start_sec=float(start),
                        end_sec=float(end),
                        description=text,
                        clip_index=global_idx,
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
                "clip_index": a.clip_index,
                "start_sec": a.start_sec,
                "end_sec": a.end_sec,
                "description": a.description,
            }
            for a in annotations
        ]
        return rows, metrics
