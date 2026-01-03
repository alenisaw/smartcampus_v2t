# src/pipeline/video_to_text.py

"""
Video-to-text pipeline for SmartCampus V2T.
Handles batched clip generation, multilingual prompts and temporal smoothing.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.core.qwen_vl_backend import QwenVLBackend
from src.core.types import Annotation, RunMetrics
from src.pipeline.pipeline_config import PipelineConfig


def build_prompt(lang: str) -> str:
    if lang == "ru":
        return (
            "Ты — CCTV аналитик. В 1–2 коротких информативных предложениях опиши, что видно на фрагменте: "
            "что делают люди/транспорт и есть ли движение (направление только если очевидно). "
            "Если явно видно необычное/опасное действие — обязательно кратко упомяни это (можно вторым предложением). "
            "Если ничего необычного не видно — не упоминай аномалии вообще. "
            "Строго по видимым фактам: без догадок, причин, эмоций, ролей, возраста, пола; без списков и переносов строк."
        )

    if lang == "kz":
        return (
            "Сен CCTV аналитигісің. 1–2 қысқа, мазмұнды сөйлеммен фрагментте не көрінетінін сипатта: "
            "адамдар/көлік не істеп жатыр және қозғалыс бар ма (бағытын тек анық болса айт). "
            "Егер анық ерекше/қауіпті әрекет көрінсе — міндетті түрде қысқа атап өт (қаласаң екінші сөйлеммен). "
            "Егер ерекше нәрсе болмаса — аномалия туралы мүлде жазба. "
            "Тек көрінетін фактілер: ойдан қоспа; себеп/эмоция/рөл/жас/жыныс жоқ; тізім мен жол ауыстырусыз."
        )

    return (
        "You are a CCTV analyst. In 1–2 short informative sentences, describe what is visible in the segment: "
        "what people/vehicles are doing and whether there is motion (direction only if obvious). "
        "If clearly visible unusual/unsafe behavior exists, you MUST mention it briefly (you may use the second sentence). "
        "If nothing unusual is visible, do not mention anomalies at all. "
        "Only visible facts: no guesses, motives, emotions, roles, age, gender; no lists or line breaks."
    )


def build_global_summary_prompt(
    lang: str,
    video_id: str,
    duration_sec: float,
    merged_anns: List["Annotation"],
) -> str:
    header: List[str] = []

    if lang == "ru":
        header.append("Ты — аналитик системы видеонаблюдения университета. Ниже приведены описания фрагментов одного видео.")
        header.append("Составь итоговую сводку видео, используя ТОЛЬКО факты из фрагментов. Ничего не выдумывай.")
        header.append("Ответ строго в следующей структуре (без угловых скобок):")
        header.append("Краткое описание: 1–2 предложения, очень кратко, без повторов и без лишних деталей.")
        header.append("Далее выведи СРАЗУ 5 строк, каждая строка начинается с '- ' и строго как ниже:")
        header.append("- Тип сцены: 1–3 слова или 'неизвестно'")
        header.append("- Плотность людей: нет людей / низкая / средняя / высокая")
        header.append("- Тип движения: нет / слабое / стабильное / усиливающееся / хаотичное")
        header.append("- Аномалии: если есть — 1–3 пункта через '; ' в формате «событие (0:01-0:10)»; если нет — 'нет'")
        header.append("- Класс безопасности: норма / подозрительно / опасно")
        header.append("Важно: не добавляй никаких других заголовков/блоков. Только эта структура.")
        header.append(f"\nВидео: {video_id}, длительность ~{int(duration_sec)} секунд.\n")
        header.append("Фрагменты:")

    elif lang == "kz":
        header.append("Сен университеттің CCTV аналитигісің. Төменде бір бейненің фрагмент сипаттамалары берілген.")
        header.append("Тек осы фрагменттердегі фактілерге сүйеніп қорытынды жаса. Жаңа нәрсе ойдан қоспа.")
        header.append("Жауап құрылымы:")
        header.append("Қысқаша сипаттама: 1–2 сөйлем, өте қысқа, қайталамасыз.")
        header.append("Содан кейін ДӘЛ 5 жол бер: әр жол '- ' деп басталсын және төмендегідей болсын:")
        header.append("- Сахна түрі: 1–3 сөз немесе 'анық емес'")
        header.append("- Адамдар тығыздығы: жоқ / төмен / орташа / жоғары")
        header.append("- Қозғалыс түрі: жоқ / әлсіз / тұрақты / күшейіп жатыр / хаотикалық")
        header.append("- Аномалиялар: бар болса — 1–3 пункт '; ' арқылы «оқиға (0:01-0:10)» форматымен; жоқ болса — 'жоқ'")
        header.append("- Қауіп классы: норма / күмәнді / қауіпті")
        header.append("Маңызды: басқа тақырып/блок қоспа. Тек осы құрылым.")
        header.append(f"\nБейне: {video_id}, ұзақтығы ~{int(duration_sec)} секунд.\n")
        header.append("Фрагменттер:")

    else:
        header.append("You are a university CCTV analyst. Below are segment descriptions of the same video.")
        header.append("Write a global summary using ONLY facts from the segments. Do not invent anything.")
        header.append("Strict structure:")
        header.append("Short summary: 1–2 sentences, very concise, no repetition.")
        header.append("Then output EXACTLY 5 lines, each starting with '- ', exactly as follows:")
        header.append("- Scene type: 1–3 words or 'unknown'")
        header.append("- People density: none / low / medium / high")
        header.append("- Motion type: none / weak / stable / increasing / chaotic")
        header.append("- Anomalies: if any — 1–3 items separated by '; ' in the form 'event (0:01-0:10)'; if none — 'none'")
        header.append("- Risk class: normal / suspicious / dangerous")
        header.append("Important: do not add any extra headings/blocks. Only this structure.")
        header.append(f"\nVideo: {video_id}, duration ~{int(duration_sec)} seconds.\n")
        header.append("Segments:")

    body = [
        f"[{format_ts(ann.start_sec)} - {format_ts(ann.end_sec)}] {strip_prefix(ann.description)}"
        for ann in merged_anns
    ]
    return "\n".join(header + [""] + body)


_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_TIMED_ITEM_RE = re.compile(r"\b(.+?)\s*\(\s*(\d+:\d{2})\s*-\s*(\d+:\d{2})\s*\)\b")
_WS_RE = re.compile(r"\s+")


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


def _collapse_spaces(s: str) -> str:
    s = s.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    return _WS_RE.sub(" ", s).strip()


def _remove_list_prefixes(s: str) -> str:
    s = re.sub(r"^\s*[-•*—]+\s*", "", s)
    s = re.sub(r"^\s*\(?\d+\)?[.)]\s*", "", s)
    return s.strip()


def _force_max_sentences(s: str, max_sentences: int = 2) -> str:
    s = s.strip()
    if not s:
        return s

    parts = _SENT_SPLIT_RE.split(s)
    parts = [p.strip() for p in parts if p.strip()]
    if not parts:
        return s

    kept = parts[: max(1, int(max_sentences))]
    out = " ".join(kept).strip()
    out = _collapse_spaces(out)
    out = re.sub(r"\s*([.!?])\s*$", r"\1", out).strip()
    if out and out[-1] not in ".!?":
        out += "."
    return out


def sanitize_clip_text(text: str, lang: str) -> str:
    t = strip_prefix(text or "")
    t = _collapse_spaces(t)
    t = _remove_list_prefixes(t)
    t = _force_max_sentences(t, max_sentences=2)
    t = _dedupe_repeated_words(t, max_repeat=2)
    return t


def _extract_first_sentences(text: str, n: int) -> str:
    t = _collapse_spaces(text or "")
    parts = _SENT_SPLIT_RE.split(t)
    parts = [p.strip() for p in parts if p.strip()]
    if not parts:
        out = t.strip()
    else:
        out = " ".join(parts[:n]).strip()
    if out and out[-1] not in ".!?":
        out += "."
    return out


def _pick_lines_starting(lines: List[str], prefix: str) -> Optional[str]:
    p = prefix.lower()
    for l in lines:
        if l.lower().startswith(p):
            return l
    return None


def _extract_after_prefix(line: str, prefix: str) -> str:
    if line.lower().startswith(prefix.lower()):
        return line[len(prefix) :].strip()
    return line.strip()


def _normalize_timed_anomalies(raw: str, lang: str) -> str:
    if not raw:
        return "нет" if lang == "ru" else ("жоқ" if lang == "kz" else "none")

    low = raw.strip().lower()
    if lang == "ru" and low in {"нет", "нет.", "отсутствуют", "не обнаружены"}:
        return "нет"
    if lang == "kz" and low in {"жоқ", "жоқ.", "анықталмады"}:
        return "жоқ"
    if lang == "en" and low in {"none", "none.", "no"}:
        return "none"

    items = _TIMED_ITEM_RE.findall(raw)
    if not items:
        return "нет" if lang == "ru" else ("жоқ" if lang == "kz" else "none")

    cleaned: List[str] = []
    for label, s1, s2 in items[:3]:
        lab = _collapse_spaces(label)
        lab = re.sub(r"[;]+$", "", lab).strip()
        lab = re.sub(r"^[\-\•\*]+\s*", "", lab).strip()
        if not lab:
            continue
        cleaned.append(f"{lab} ({s1}-{s2})")

    if not cleaned:
        return "нет" if lang == "ru" else ("жоқ" if lang == "kz" else "none")

    return "; ".join(cleaned)


def sanitize_global_summary(text: str, lang: str) -> str:
    raw = text or ""
    lines = [l.strip() for l in raw.splitlines() if l.strip()]

    if lang == "ru":
        short_label = "Краткое описание:"
        legacy_timed_label = "Аномалии по времени:"
        keys = [
            ("- Тип сцены:", "неизвестно"),
            ("- Плотность людей:", "низкая"),
            ("- Тип движения:", "стабильное"),
            ("- Аномалии:", "нет"),
            ("- Класс безопасности:", "норма"),
        ]
        none_anom = "нет"
    elif lang == "kz":
        short_label = "Қысқаша сипаттама:"
        legacy_timed_label = "Уақыт бойынша аномалиялар:"
        keys = [
            ("- Сахна түрі:", "анық емес"),
            ("- Адамдар тығыздығы:", "төмен"),
            ("- Қозғалыс түрі:", "тұрақты"),
            ("- Аномалиялар:", "жоқ"),
            ("- Қауіп классы:", "норма"),
        ]
        none_anom = "жоқ"
    else:
        short_label = "Short summary:"
        legacy_timed_label = "Timed anomalies:"
        keys = [
            ("- Scene type:", "unknown"),
            ("- People density:", "low"),
            ("- Motion type:", "stable"),
            ("- Anomalies:", "none"),
            ("- Risk class:", "normal"),
        ]
        none_anom = "none"

    short_line = _pick_lines_starting(lines, short_label)
    short_text = _extract_after_prefix(short_line, short_label) if short_line else ""
    if not short_text:
        short_text = _extract_first_sentences(raw, 2)
    else:
        short_text = _collapse_spaces(short_text)
        if short_text and short_text[-1] not in ".!?":
            short_text += "."

    legacy_timed_line = _pick_lines_starting(lines, legacy_timed_label)
    legacy_timed_raw = _extract_after_prefix(legacy_timed_line, legacy_timed_label) if legacy_timed_line else ""
    legacy_timed_norm = _normalize_timed_anomalies(legacy_timed_raw, lang=lang) if legacy_timed_raw else ""

    found: Dict[str, str] = {}
    for l in lines:
        if not l.startswith("-"):
            continue
        for k, _d in keys:
            if l.lower().startswith(k.lower()):
                found[k] = _collapse_spaces(l[len(k) :].strip()) or _d

    def pick_density(v: str) -> str:
        vlow = (v or "").lower()
        if lang == "ru":
            if "нет" in vlow:
                return "нет людей"
            if "выс" in vlow or "много" in vlow:
                return "высокая"
            if "сред" in vlow:
                return "средняя"
            return "низкая"
        if lang == "kz":
            if "жоқ" in vlow:
                return "жоқ"
            if "жоғ" in vlow or "көп" in vlow:
                return "жоғары"
            if "орта" in vlow:
                return "орташа"
            return "төмен"
        if "none" in vlow or vlow == "no":
            return "none"
        if "high" in vlow or "many" in vlow:
            return "high"
        if "med" in vlow:
            return "medium"
        return "low"

    def pick_motion(v: str) -> str:
        vlow = (v or "").lower()
        if lang == "ru":
            if "хаот" in vlow:
                return "хаотичное"
            if "усил" in vlow or "раст" in vlow:
                return "усиливающееся"
            if "слаб" in vlow:
                return "слабое"
            if "нет" in vlow:
                return "нет"
            return "стабильное"
        if lang == "kz":
            if "хаот" in vlow:
                return "хаотикалық"
            if "күш" in vlow:
                return "күшейіп жатыр"
            if "әлсіз" in vlow:
                return "әлсіз"
            if "жоқ" in vlow:
                return "жоқ"
            return "тұрақты"
        if "chaot" in vlow:
            return "chaotic"
        if "increas" in vlow:
            return "increasing"
        if "weak" in vlow:
            return "weak"
        if "none" in vlow or vlow == "no":
            return "none"
        return "stable"

    def pick_risk(v: str) -> str:
        vlow = (v or "").lower()
        if lang == "ru":
            if "опас" in vlow:
                return "опасно"
            if "подоз" in vlow or "күм" in vlow:
                return "подозрительно"
            return "норма"
        if lang == "kz":
            if "қауіп" in vlow:
                return "қауіпті"
            if "күм" in vlow:
                return "күмәнді"
            return "норма"
        if "danger" in vlow:
            return "dangerous"
        if "susp" in vlow:
            return "suspicious"
        return "normal"

    def pick_anom(v: str) -> str:
        v = _collapse_spaces(v or "")
        if not v:
            return none_anom
        vlow = v.lower()
        if lang == "ru" and vlow in {"нет", "нет.", "не обнаружены", "отсутствуют"}:
            return "нет"
        if lang == "kz" and vlow in {"жоқ", "жоқ.", "анықталмады"}:
            return "жоқ"
        if lang == "en" and vlow in {"none", "none.", "no"}:
            return "none"
        v = re.sub(r"^\s*[\-\•\*]+\s*", "", v).strip()
        return (v[:200] if v else none_anom)

    out_lines: List[str] = []
    out_lines.append(f"{short_label} {short_text}".strip())

    for i, (k, d) in enumerate(keys):
        v = found.get(k, d)
        if i == 1:
            v = pick_density(v)
        elif i == 2:
            v = pick_motion(v)
        elif i == 3:
            v = pick_anom(v)
            if (v.lower() == none_anom) and legacy_timed_norm and legacy_timed_norm != none_anom:
                v = legacy_timed_norm
        elif i == 4:
            v = pick_risk(v)
        out_lines.append(f"{k} {v}".strip())

    return "\n".join(out_lines).strip()


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

            lang = self.cfg.model.language
            texts = [sanitize_clip_text(t, lang=lang) for t in texts]

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

        t_post_start = time.perf_counter()
        merged = smooth_annotations(annotations)
        post_time = time.perf_counter() - t_post_start

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
                if global_summary:
                    global_summary = sanitize_global_summary(global_summary, lang=self.cfg.model.language)
        except Exception:
            global_summary = None

        extra: Dict[str, Any] = {}
        if global_summary:
            extra["global_summary"] = global_summary

        total_time = float(preprocess_time_sec) + float(model_time) + float(post_time)

        metrics = RunMetrics(
            video_id=video_id,
            video_duration_sec=float(video_duration_sec),
            num_frames=int(num_frames),
            num_clips=int(num_clips),
            avg_clip_duration_sec=float(avg_clip_duration),
            preprocess_time_sec=float(preprocess_time_sec),
            model_time_sec=float(model_time),
            postprocess_time_sec=float(post_time),
            total_time_sec=float(total_time),
            extra=extra or None,
        )

        return merged, metrics

    def run_for_ui(self, *args, **kwargs):
        annotations, metrics = self.run(*args, **kwargs)
        rows = [{"start_sec": a.start_sec, "end_sec": a.end_sec, "description": a.description} for a in annotations]
        return rows, metrics
