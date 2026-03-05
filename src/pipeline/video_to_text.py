# src/pipeline/video_to_text.py
"""
Video-to-text pipeline for SmartCampus V2T.

Purpose:
- Builds per-clip descriptions (RU/KZ/EN) with Qwen3-VL.
- Performs temporal smoothing (merge similar adjacent segments).
- Produces optional global summary (1вЂ“2 human sentences + 5 strict metric lines).
- Ensures descriptions are factual, human-readable, and non-duplicative.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import cv2
except Exception:
    cv2 = None

from src.core.qwen_vl_backend import QwenVLBackend
from src.core.types import Annotation, RunMetrics
from src.pipeline.pipeline_config import PipelineConfig


def build_prompt(lang: str) -> str:
    if lang == "ru":
        return (
            "РўС‹ РѕРїРёСЃС‹РІР°РµС€СЊ CCTV-С„СЂР°РіРјРµРЅС‚. РќР°РїРёС€Рё 1вЂ“2 РєРѕСЂРѕС‚РєРёС… РїСЂРµРґР»РѕР¶РµРЅРёСЏ РµСЃС‚РµСЃС‚РІРµРЅРЅС‹Рј СЏР·С‹РєРѕРј. "
            "РќР°С‡РёРЅР°Р№ СЃСЂР°Р·Сѓ СЃ РґРµР№СЃС‚РІРёСЏ (Р±РµР· В«РќР° РєР°РґСЂРµВ», В«РќР° СЃРЅРёРјРєРµВ», В«Р’РёРґРЅРѕВ», В«РќР°Р±Р»СЋРґР°РµС‚СЃСЏВ», В«РќР° РІРёРґРµРѕВ»). "
            "РўРѕР»СЊРєРѕ РЅР°Р±Р»СЋРґР°РµРјС‹Рµ С„Р°РєС‚С‹: С‡С‚Рѕ РґРµР»Р°СЋС‚ Р»СЋРґРё РёР»Рё РѕР±СЉРµРєС‚С‹; РґРІРёР¶РµРЅРёРµ СѓРїРѕРјРёРЅР°Р№ С‚РѕР»СЊРєРѕ РµСЃР»Рё РѕРЅРѕ Р·Р°РјРµС‚РЅРѕ. "
            "РќРµ РїСЂРёРґСѓРјС‹РІР°Р№ С‚РёРї РјРµСЃС‚Р°, СЂРѕР»Рё Р»СЋРґРµР№, РїСЂРёС‡РёРЅС‹ РёР»Рё РєРѕРЅС‚РµРєСЃС‚. "
            "РќРµ СѓРєР°Р·С‹РІР°Р№ РІРѕР·СЂР°СЃС‚, РїРѕР», СЌРјРѕС†РёРё РёР»Рё РЅР°РјРµСЂРµРЅРёСЏ. "
            "РќРµ РёСЃРїРѕР»СЊР·СѓР№ РјРµС‚СЂРёРєРё Рё СЏСЂР»С‹РєРё. "
            "Р•СЃР»Рё РЅРёС‡РµРіРѕ РЅРµРѕР±С‹С‡РЅРѕРіРѕ РЅРµС‚ вЂ” РїСЂРѕСЃС‚Рѕ РѕРїРёС€Рё РѕР±С‹С‡РЅСѓСЋ Р°РєС‚РёРІРЅРѕСЃС‚СЊ."
        )
    if lang == "kz":
        return (
            "CCTV С„СЂР°РіРјРµРЅС‚С–РЅ СЃРёРїР°С‚С‚Р°. 1вЂ“2 Т›С‹СЃТ›Р° СЃУ©Р№Р»РµРјРґС– С‚Р°Р±РёТ“Рё С‚С–Р»РјРµРЅ Р¶Р°Р·. "
            "РЎУ©Р№Р»РµРјРґС– Р±С–СЂРґРµРЅ У™СЂРµРєРµС‚С‚РµРЅ Р±Р°СЃС‚Р° ( В«РљР°РґСЂРґР°В», В«РЎСѓСЂРµС‚С‚РµВ», В«РљУ©СЂС–РЅРµРґС–В», В«Р‘Р°Р№Т›Р°Р»Р°РґС‹В» РґРµРјРµ). "
            "РўРµРє РєУ©СЂС–РЅРµС‚С–РЅ С„Р°РєС‚С–Р»РµСЂРґС– Р¶Р°Р·: Р°РґР°РјРґР°СЂ РЅРµРјРµСЃРµ РЅС‹СЃР°РЅРґР°СЂ РЅРµ С–СЃС‚РµРї Р¶Р°С‚С‹СЂ; Т›РѕР·Т“Р°Р»С‹СЃ Р°Р№Т›С‹РЅ Р±РѕР»СЃР° Т“Р°РЅР° Р°Р№С‚. "
            "РћСЂС‹РЅРґС‹, Р°РґР°РјРґР°СЂРґС‹ТЈ СЂУ©Р»С–РЅ, СЃРµР±РµРї РїРµРЅ РєРѕРЅС‚РµРєСЃС‚С– РѕР№РґР°РЅ Т›РѕСЃРїР°. "
            "Р–Р°СЃ, Р¶С‹РЅС‹СЃ, СЌРјРѕС†РёСЏ, РЅРёРµС‚ РєУ©СЂСЃРµС‚РїРµ. "
            "РњРµС‚СЂРёРєР° РЅРµРјРµСЃРµ Р¶С–РєС‚РµСѓ СЃУ©Р·РґРµСЂС–РЅ Т›РѕР»РґР°РЅР±Р°. "
            "Р•СЂРµРєС€Рµ РЅУ™СЂСЃРµ Р±РѕР»РјР°СЃР° вЂ” Т›Р°Р»С‹РїС‚С‹ У™СЂРµРєРµС‚С‚С– СЃРёРїР°С‚С‚Р°."
        )
    return (
        "Describe a CCTV segment in 1вЂ“2 short natural sentences. "
        "Start directly with the action (avoid 'In the frame', 'In the image', 'We can see', 'There is'). "
        "Only observable facts: what people or objects are doing; mention motion only if clearly visible. "
        "Do not guess the location type, peopleвЂ™s roles, causes, or context. "
        "No age, gender, emotions, or intentions. "
        "Do not use metric or classification words. "
        "If nothing unusual is visible, describe normal activity."
    )


def build_global_summary_prompt(
    lang: str,
    video_id: str,
    duration_sec: float,
    merged_anns: List[Annotation],
) -> str:
    header: List[str] = []

    if lang == "ru":
        header.append("РўС‹ вЂ” Р°РЅР°Р»РёС‚РёРє CCTV. РќРёР¶Рµ РїСЂРёРІРµРґРµРЅС‹ РѕРїРёСЃР°РЅРёСЏ С„СЂР°РіРјРµРЅС‚РѕРІ РѕРґРЅРѕРіРѕ РІРёРґРµРѕ.")
        header.append("РСЃРїРѕР»СЊР·СѓР№ С‚РѕР»СЊРєРѕ С„Р°РєС‚С‹ РёР· СЌС‚РёС… РѕРїРёСЃР°РЅРёР№, РЅРёС‡РµРіРѕ РЅРµ РІС‹РґСѓРјС‹РІР°Р№.")
        header.append(
            "РЎРЅР°С‡Р°Р»Р° РЅР°РїРёС€Рё 1вЂ“2 РїСЂРµРґР»РѕР¶РµРЅРёСЏ РѕР±С‹С‡РЅС‹Рј С‡РµР»РѕРІРµС‡РµСЃРєРёРј СЏР·С‹РєРѕРј: "
            "С‡С‚Рѕ РІ С†РµР»РѕРј РїСЂРѕРёСЃС…РѕРґРёС‚ Рё РєР°Рє РјРµРЅСЏРµС‚СЃСЏ Р°РєС‚РёРІРЅРѕСЃС‚СЊ СЃРѕ РІСЂРµРјРµРЅРµРј. "
            "РќРµ РёСЃРїРѕР»СЊР·СѓР№ Рё РЅРµ РїРѕРІС‚РѕСЂСЏР№ СЃР»РѕРІР°-СЏСЂР»С‹РєРё: С‚РёРї СЃС†РµРЅС‹, РїР»РѕС‚РЅРѕСЃС‚СЊ, РґРІРёР¶РµРЅРёРµ, Р°РЅРѕРјР°Р»РёРё, РєР»Р°СЃСЃ Р±РµР·РѕРїР°СЃРЅРѕСЃС‚Рё."
        )
        header.append("Р—Р°С‚РµРј РІС‹РІРµРґРё Р РћР’РќРћ 5 СЃС‚СЂРѕРє, РєР°Р¶РґР°СЏ РЅР°С‡РёРЅР°РµС‚СЃСЏ СЃ '- ' Рё СЃС‚СЂРѕРіРѕ РїРѕ С€Р°Р±Р»РѕРЅСѓ:")
        header.append("- РўРёРї СЃС†РµРЅС‹: 1вЂ“3 СЃР»РѕРІР° РёР»Рё 'РЅРµРёР·РІРµСЃС‚РЅРѕ'")
        header.append("- РџР»РѕС‚РЅРѕСЃС‚СЊ Р»СЋРґРµР№: РЅРµС‚ Р»СЋРґРµР№ / РЅРёР·РєР°СЏ / СЃСЂРµРґРЅСЏСЏ / РІС‹СЃРѕРєР°СЏ")
        header.append("- РўРёРї РґРІРёР¶РµРЅРёСЏ: РЅРµС‚ / СЃР»Р°Р±РѕРµ / СЃС‚Р°Р±РёР»СЊРЅРѕРµ / СѓСЃРёР»РёРІР°СЋС‰РµРµСЃСЏ / С…Р°РѕС‚РёС‡РЅРѕРµ")
        header.append("- РђРЅРѕРјР°Р»РёРё: РµСЃР»Рё РµСЃС‚СЊ вЂ” 1вЂ“3 РїСѓРЅРєС‚Р° С‡РµСЂРµР· '; ' РІ С„РѕСЂРјР°С‚Рµ В«СЃРѕР±С‹С‚РёРµ (0:01-0:10)В»; РµСЃР»Рё РЅРµС‚ вЂ” 'РЅРµС‚'")
        header.append("- РљР»Р°СЃСЃ Р±РµР·РѕРїР°СЃРЅРѕСЃС‚Рё: РЅРѕСЂРјР° / РїРѕРґРѕР·СЂРёС‚РµР»СЊРЅРѕ / РѕРїР°СЃРЅРѕ")
        header.append(f"\nР’РёРґРµРѕ: {video_id}, РґР»РёС‚РµР»СЊРЅРѕСЃС‚СЊ ~{int(duration_sec)} СЃРµРєСѓРЅРґ.\n")
        header.append("Р¤СЂР°РіРјРµРЅС‚С‹:")

    elif lang == "kz":
        header.append("РЎРµРЅ CCTV Р°РЅР°Р»РёС‚РёРіС–СЃС–ТЈ. РўУ©РјРµРЅРґРµ Р±С–СЂ Р±РµР№РЅРµРЅС–ТЈ С„СЂР°РіРјРµРЅС‚ СЃРёРїР°С‚С‚Р°РјР°Р»Р°СЂС‹ Р±РµСЂС–Р»РіРµРЅ.")
        header.append("ТљРѕСЂС‹С‚С‹РЅРґС‹РЅС‹ С‚РµРє РѕСЃС‹ СЃРёРїР°С‚С‚Р°РјР°Р»Р°СЂРґР°Т“С‹ С„Р°РєС‚С–Р»РµСЂРіРµ СЃТЇР№РµРЅС–Рї Р¶Р°СЃР°, РѕР№РґР°РЅ Т›РѕСЃРїР°.")
        header.append(
            "РђР»РґС‹РјРµРЅ 1вЂ“2 СЃУ©Р№Р»РµРјРјРµРЅ Т›Р°СЂР°РїР°Р№С‹Рј С‚С–Р»РґРµ Р¶Р°Р»РїС‹ РЅРµ Р±РѕР»С‹Рї Р¶Р°С‚Т›Р°РЅС‹РЅ Р¶У™РЅРµ СѓР°Т›С‹С‚ Р±РѕР№С‹РЅС€Р° У©Р·РіРµСЂС–СЃС‚С– СЃРёРїР°С‚С‚Р°. "
            "РљРµР»РµСЃС– РјРµС‚СЂРёРєР° Р°С‚Р°СѓР»Р°СЂС‹РЅ Т›РѕР»РґР°РЅР±Р° Р¶У™РЅРµ Т›Р°Р№С‚Р°Р»Р°РјР°: СЃР°С…РЅР° С‚ТЇСЂС–, С‚С‹Т“С‹Р·РґС‹Т›, Т›РѕР·Т“Р°Р»С‹СЃ С‚ТЇСЂС–, Р°РЅРѕРјР°Р»РёСЏР»Р°СЂ, Т›Р°СѓС–Рї РєР»Р°СЃСЃС‹."
        )
        header.append("РЎРѕРґР°РЅ РєРµР№С–РЅ Р”УР› 5 Р¶РѕР» Р±РµСЂ, У™СЂ Р¶РѕР» '- ' РґРµРї Р±Р°СЃС‚Р°Р»СЃС‹РЅ Р¶У™РЅРµ С€Р°Р±Р»РѕРЅТ“Р° СЃР°Р№ Р±РѕР»СЃС‹РЅ:")
        header.append("- РЎР°С…РЅР° С‚ТЇСЂС–: 1вЂ“3 СЃУ©Р· РЅРµРјРµСЃРµ 'Р°РЅС‹Т› РµРјРµСЃ'")
        header.append("- РђРґР°РјРґР°СЂ С‚С‹Т“С‹Р·РґС‹Т“С‹: Р¶РѕТ› / С‚У©РјРµРЅ / РѕСЂС‚Р°С€Р° / Р¶РѕТ“Р°СЂС‹")
        header.append("- ТљРѕР·Т“Р°Р»С‹СЃ С‚ТЇСЂС–: Р¶РѕТ› / У™Р»СЃС–Р· / С‚Т±СЂР°Т›С‚С‹ / РєТЇС€РµР№С–Рї Р¶Р°С‚С‹СЂ / С…Р°РѕС‚РёРєР°Р»С‹Т›")
        header.append("- РђРЅРѕРјР°Р»РёСЏР»Р°СЂ: Р±Р°СЂ Р±РѕР»СЃР° вЂ” 1вЂ“3 РїСѓРЅРєС‚ '; ' Р°СЂТ›С‹Р»С‹ В«РѕТ›РёТ“Р° (0:01-0:10)В» С„РѕСЂРјР°С‚С‹РјРµРЅ; Р¶РѕТ› Р±РѕР»СЃР° вЂ” 'Р¶РѕТ›'")
        header.append("- ТљР°СѓС–Рї РєР»Р°СЃСЃС‹: РЅРѕСЂРјР° / РєТЇРјУ™РЅРґС– / Т›Р°СѓС–РїС‚С–")
        header.append(f"\nР‘РµР№РЅРµ: {video_id}, Т±Р·Р°Т›С‚С‹Т“С‹ ~{int(duration_sec)} СЃРµРєСѓРЅРґ.\n")
        header.append("Р¤СЂР°РіРјРµРЅС‚С‚РµСЂ:")

    else:
        header.append("You are a CCTV analyst. Below are descriptions of segments from the same video.")
        header.append("Use only facts from these descriptions. Do not invent anything.")
        header.append(
            "First, write 1вЂ“2 natural sentences describing what is happening overall and how activity changes over time. "
            "Do not use or repeat metric labels: scene type, people density, motion type, anomalies, risk class."
        )
        header.append("Then output EXACTLY 5 lines, each starting with '- ', strictly following this template:")
        header.append("- Scene type: 1вЂ“3 words or 'unknown'")
        header.append("- People density: none / low / medium / high")
        header.append("- Motion type: none / weak / stable / increasing / chaotic")
        header.append("- Anomalies: if any вЂ” 1вЂ“3 items separated by '; ' in the form 'event (0:01-0:10)'; if none вЂ” 'none'")
        header.append("- Risk class: normal / suspicious / dangerous")
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
    return re.sub(r"^\[[0-9:\s\-]+\]\s*", "", text or "")


def _collapse_spaces(s: str) -> str:
    s = (s or "").replace("\n", " ").replace("\r", " ").replace("\t", " ")
    return _WS_RE.sub(" ", s).strip()


def _remove_list_prefixes(s: str) -> str:
    s = re.sub(r"^\s*[-вЂў*вЂ”]+\s*", "", s or "")
    s = re.sub(r"^\s*\(?\d+\)?[.)]\s*", "", s)
    return s.strip()


def _strip_generic_openers(s: str) -> str:
    return re.sub(
        r"^(РІРёРґРЅРѕ|РЅР°Р±Р»СЋРґР°РµС‚СЃСЏ|РЅР° РІРёРґРµРѕ|we can see|there is|there are|РєУ©СЂС–РЅРµРґС–|Р±Р°Р№Т›Р°Р»Р°РґС‹)\s*[:,\-]?\s*",
        "",
        s,
        flags=re.IGNORECASE,
    )


def _force_max_sentences(s: str, max_sentences: int = 2) -> str:
    s = (s or "").strip()
    if not s:
        return s
    parts = [p.strip() for p in _SENT_SPLIT_RE.split(s) if p.strip()]
    kept = parts[: max(1, int(max_sentences))] if parts else [s]
    out = _collapse_spaces(" ".join(kept))
    if out and out[-1] not in ".!?":
        out += "."
    return out


def _dedupe_repeated_words(text: str, max_repeat: int = 2) -> str:
    words = (text or "").split()
    out: List[str] = []
    run_word: Optional[str] = None
    run_len = 0
    for w in words:
        key = w.lower()
        if key == run_word:
            run_len += 1
            if run_len >= max_repeat:
                continue
        else:
            run_word = key
            run_len = 0
        out.append(w)
    t = " ".join(out).strip()
    if t and t[-1] not in ".!?":
        t += "."
    return t


def sanitize_clip_text(text: str, lang: str) -> str:
    t = strip_prefix(text)
    t = _collapse_spaces(t)
    t = _strip_generic_openers(t)
    t = _remove_list_prefixes(t)
    t = _force_max_sentences(t, max_sentences=2)
    t = _dedupe_repeated_words(t, max_repeat=2)
    return t


def _extract_first_sentences(text: str, n: int) -> str:
    t = _collapse_spaces(text)
    parts = [p.strip() for p in _SENT_SPLIT_RE.split(t) if p.strip()]
    out = " ".join(parts[:n]).strip() if parts else t.strip()
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
        return line[len(prefix):].strip()
    return line.strip()


def _normalize_timed_anomalies(raw: str, lang: str) -> str:
    if not raw:
        return "РЅРµС‚" if lang == "ru" else ("Р¶РѕТ›" if lang == "kz" else "none")
    low = raw.strip().lower()
    if lang == "ru" and low in {"РЅРµС‚", "РЅРµС‚.", "РѕС‚СЃСѓС‚СЃС‚РІСѓСЋС‚", "РЅРµ РѕР±РЅР°СЂСѓР¶РµРЅС‹"}:
        return "РЅРµС‚"
    if lang == "kz" and low in {"Р¶РѕТ›", "Р¶РѕТ›.", "Р°РЅС‹Т›С‚Р°Р»РјР°РґС‹"}:
        return "Р¶РѕТ›"
    if lang == "en" and low in {"none", "none.", "no"}:
        return "none"

    items = _TIMED_ITEM_RE.findall(raw)
    if not items:
        return "РЅРµС‚" if lang == "ru" else ("Р¶РѕТ›" if lang == "kz" else "none")

    cleaned: List[str] = []
    for label, s1, s2 in items[:3]:
        lab = _collapse_spaces(label)
        lab = re.sub(r"[;]+$", "", lab).strip()
        lab = re.sub(r"^[\-\вЂў\*]+\s*", "", lab).strip()
        if lab:
            cleaned.append(f"{lab} ({s1}-{s2})")

    return "; ".join(cleaned) if cleaned else ("РЅРµС‚" if lang == "ru" else ("Р¶РѕТ›" if lang == "kz" else "none"))


def _strip_metric_fragments(short_text: str, lang: str) -> str:
    t = _collapse_spaces(short_text or "")
    if not t:
        return t

    if lang == "ru":
        cuts = [
            " - С‚РёРї СЃС†РµРЅС‹:",
            "- С‚РёРї СЃС†РµРЅС‹:",
            "С‚РёРї СЃС†РµРЅС‹:",
            " - РїР»РѕС‚РЅРѕСЃС‚СЊ Р»СЋРґРµР№:",
            "- РїР»РѕС‚РЅРѕСЃС‚СЊ Р»СЋРґРµР№:",
            "РїР»РѕС‚РЅРѕСЃС‚СЊ Р»СЋРґРµР№:",
            " - С‚РёРї РґРІРёР¶РµРЅРёСЏ:",
            "- С‚РёРї РґРІРёР¶РµРЅРёСЏ:",
            "С‚РёРї РґРІРёР¶РµРЅРёСЏ:",
            " - Р°РЅРѕРјР°Р»РёРё:",
            "- Р°РЅРѕРјР°Р»РёРё:",
            "Р°РЅРѕРјР°Р»РёРё:",
            " - РєР»Р°СЃСЃ Р±РµР·РѕРїР°СЃРЅРѕСЃС‚Рё:",
            "- РєР»Р°СЃСЃ Р±РµР·РѕРїР°СЃРЅРѕСЃС‚Рё:",
            "РєР»Р°СЃСЃ Р±РµР·РѕРїР°СЃРЅРѕСЃС‚Рё:",
        ]
    elif lang == "kz":
        cuts = [
            " - СЃР°С…РЅР° С‚ТЇСЂС–:",
            "- СЃР°С…РЅР° С‚ТЇСЂС–:",
            "СЃР°С…РЅР° С‚ТЇСЂС–:",
            " - Р°РґР°РјРґР°СЂ С‚С‹Т“С‹Р·РґС‹Т“С‹:",
            "- Р°РґР°РјРґР°СЂ С‚С‹Т“С‹Р·РґС‹Т“С‹:",
            "Р°РґР°РјРґР°СЂ С‚С‹Т“С‹Р·РґС‹Т“С‹:",
            " - Т›РѕР·Т“Р°Р»С‹СЃ С‚ТЇСЂС–:",
            "- Т›РѕР·Т“Р°Р»С‹СЃ С‚ТЇСЂС–:",
            "Т›РѕР·Т“Р°Р»С‹СЃ С‚ТЇСЂС–:",
            " - Р°РЅРѕРјР°Р»РёСЏР»Р°СЂ:",
            "- Р°РЅРѕРјР°Р»РёСЏР»Р°СЂ:",
            "Р°РЅРѕРјР°Р»РёСЏР»Р°СЂ:",
            " - Т›Р°СѓС–Рї РєР»Р°СЃСЃС‹:",
            "- Т›Р°СѓС–Рї РєР»Р°СЃСЃС‹:",
            "Т›Р°СѓС–Рї РєР»Р°СЃСЃС‹:",
        ]
    else:
        cuts = [
            " - scene type:",
            "- scene type:",
            "scene type:",
            " - people density:",
            "- people density:",
            "people density:",
            " - motion type:",
            "- motion type:",
            "motion type:",
            " - anomalies:",
            "- anomalies:",
            "anomalies:",
            " - risk class:",
            "- risk class:",
            "risk class:",
        ]

    low = t.lower()
    cut_pos: Optional[int] = None
    for c in cuts:
        p = low.find(c)
        if p != -1:
            cut_pos = p if cut_pos is None else min(cut_pos, p)

    if cut_pos is not None:
        t = t[:cut_pos].strip()

    if lang == "kz":
        t = re.sub(r"(Р¶Р°Р»РїС‹\s+С‚ТЇСЃС–РЅС–РєС‚С–\s+Т›РѕСЂС‹С‚С‹РЅРґС‹[:\-]?\s*)", "", t, flags=re.IGNORECASE).strip()
        t = re.sub(r"(Т›РѕСЂС‹С‚С‹РЅРґС‹(РЅС‹ТЈ)?\s+РЅРµРіС–Р·С–РЅРґРµ[^.]*\.)", "", t, flags=re.IGNORECASE).strip()

    t = re.sub(r"\s*-\s*$", "", t).strip()
    t = _strip_generic_openers(t)
    t = _force_max_sentences(t, max_sentences=2)
    return t


def sanitize_global_summary(text: str, lang: str) -> str:
    raw = text or ""
    lines = [l.strip() for l in raw.splitlines() if l.strip()]

    if lang == "ru":
        short_label = "РљСЂР°С‚РєРѕРµ РѕРїРёСЃР°РЅРёРµ:"
        timed_label = "РђРЅРѕРјР°Р»РёРё РїРѕ РІСЂРµРјРµРЅРё:"
        keys = [
            ("- РўРёРї СЃС†РµРЅС‹:", "РЅРµРёР·РІРµСЃС‚РЅРѕ"),
            ("- РџР»РѕС‚РЅРѕСЃС‚СЊ Р»СЋРґРµР№:", "РЅРёР·РєР°СЏ"),
            ("- РўРёРї РґРІРёР¶РµРЅРёСЏ:", "СЃС‚Р°Р±РёР»СЊРЅРѕРµ"),
            ("- РђРЅРѕРјР°Р»РёРё:", "РЅРµС‚"),
            ("- РљР»Р°СЃСЃ Р±РµР·РѕРїР°СЃРЅРѕСЃС‚Рё:", "РЅРѕСЂРјР°"),
        ]
        none_anom = "РЅРµС‚"
    elif lang == "kz":
        short_label = "ТљС‹СЃТ›Р°С€Р° СЃРёРїР°С‚С‚Р°РјР°:"
        timed_label = "РЈР°Т›С‹С‚ Р±РѕР№С‹РЅС€Р° Р°РЅРѕРјР°Р»РёСЏР»Р°СЂ:"
        keys = [
            ("- РЎР°С…РЅР° С‚ТЇСЂС–:", "Р°РЅС‹Т› РµРјРµСЃ"),
            ("- РђРґР°РјРґР°СЂ С‚С‹Т“С‹Р·РґС‹Т“С‹:", "С‚У©РјРµРЅ"),
            ("- ТљРѕР·Т“Р°Р»С‹СЃ С‚ТЇСЂС–:", "С‚Т±СЂР°Т›С‚С‹"),
            ("- РђРЅРѕРјР°Р»РёСЏР»Р°СЂ:", "Р¶РѕТ›"),
            ("- ТљР°СѓС–Рї РєР»Р°СЃСЃС‹:", "РЅРѕСЂРјР°"),
        ]
        none_anom = "Р¶РѕТ›"
    else:
        short_label = "Short summary:"
        timed_label = "Timed anomalies:"
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
    short_text = _extract_first_sentences(raw, 2) if not short_text else _collapse_spaces(short_text)
    short_text = _strip_metric_fragments(short_text, lang=lang)
    short_text = _collapse_spaces(short_text)
    short_text = _strip_generic_openers(short_text)
    short_text = _force_max_sentences(short_text, max_sentences=2)

    timed_line = _pick_lines_starting(lines, timed_label)
    timed_raw = _extract_after_prefix(timed_line, timed_label) if timed_line else ""
    timed_norm = _normalize_timed_anomalies(timed_raw, lang=lang) if timed_raw else ""

    found: Dict[str, str] = {}
    for l in lines:
        if not l.startswith("-"):
            continue
        for k, d in keys:
            if l.lower().startswith(k.lower()):
                found[k] = _collapse_spaces(l[len(k):].strip()) or d

    def pick_density(v: str) -> str:
        vlow = (v or "").lower()
        if lang == "ru":
            if "РЅРµС‚" in vlow:
                return "РЅРµС‚ Р»СЋРґРµР№"
            if "РІС‹СЃ" in vlow or "РјРЅРѕРіРѕ" in vlow:
                return "РІС‹СЃРѕРєР°СЏ"
            if "СЃСЂРµРґ" in vlow:
                return "СЃСЂРµРґРЅСЏСЏ"
            return "РЅРёР·РєР°СЏ"
        if lang == "kz":
            if "Р¶РѕТ›" in vlow:
                return "Р¶РѕТ›"
            if "Р¶РѕТ“" in vlow or "РєУ©Рї" in vlow:
                return "Р¶РѕТ“Р°СЂС‹"
            if "РѕСЂС‚Р°" in vlow:
                return "РѕСЂС‚Р°С€Р°"
            return "С‚У©РјРµРЅ"
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
            if "С…Р°РѕС‚" in vlow:
                return "С…Р°РѕС‚РёС‡РЅРѕРµ"
            if "СѓСЃРёР»" in vlow or "СЂР°СЃС‚" in vlow:
                return "СѓСЃРёР»РёРІР°СЋС‰РµРµСЃСЏ"
            if "СЃР»Р°Р±" in vlow:
                return "СЃР»Р°Р±РѕРµ"
            if "РЅРµС‚" in vlow:
                return "РЅРµС‚"
            return "СЃС‚Р°Р±РёР»СЊРЅРѕРµ"
        if lang == "kz":
            if "С…Р°РѕС‚" in vlow:
                return "С…Р°РѕС‚РёРєР°Р»С‹Т›"
            if "РєТЇС€" in vlow:
                return "РєТЇС€РµР№С–Рї Р¶Р°С‚С‹СЂ"
            if "У™Р»СЃС–Р·" in vlow:
                return "У™Р»СЃС–Р·"
            if "Р¶РѕТ›" in vlow:
                return "Р¶РѕТ›"
            return "С‚Т±СЂР°Т›С‚С‹"
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
            if "РѕРїР°СЃ" in vlow:
                return "РѕРїР°СЃРЅРѕ"
            if "РїРѕРґРѕР·" in vlow or "РєТЇРј" in vlow:
                return "РїРѕРґРѕР·СЂРёС‚РµР»СЊРЅРѕ"
            return "РЅРѕСЂРјР°"
        if lang == "kz":
            if "Т›Р°СѓС–Рї" in vlow:
                return "Т›Р°СѓС–РїС‚С–"
            if "РєТЇРј" in vlow:
                return "РєТЇРјУ™РЅРґС–"
            return "РЅРѕСЂРјР°"
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
        if lang == "ru" and vlow in {"РЅРµС‚", "РЅРµС‚.", "РЅРµ РѕР±РЅР°СЂСѓР¶РµРЅС‹", "РѕС‚СЃСѓС‚СЃС‚РІСѓСЋС‚"}:
            return "РЅРµС‚"
        if lang == "kz" and vlow in {"Р¶РѕТ›", "Р¶РѕТ›.", "Р°РЅС‹Т›С‚Р°Р»РјР°РґС‹"}:
            return "Р¶РѕТ›"
        if lang == "en" and vlow in {"none", "none.", "no"}:
            return "none"
        v = re.sub(r"^\s*[\-\вЂў\*]+\s*", "", v).strip()
        return (v[:200] if v else none_anom)

    out_lines: List[str] = [f"{short_label} {short_text}".strip()]

    for i, (k, d) in enumerate(keys):
        v = found.get(k, d)
        if i == 1:
            v = pick_density(v)
        elif i == 2:
            v = pick_motion(v)
        elif i == 3:
            v = pick_anom(v)
            if (v.lower() == none_anom) and timed_norm and timed_norm != none_anom:
                v = timed_norm
        elif i == 4:
            v = pick_risk(v)
        out_lines.append(f"{k} {v}".strip())

    return "\n".join(out_lines).strip()


def norm_tokens(text: str) -> List[str]:
    text = strip_prefix((text or "").lower())
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
            current.extra = current.extra or {}
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


def _bucket_indices_by_len(nested: List[List[str]]) -> Dict[int, List[int]]:
    buckets: Dict[int, List[int]] = {}
    for i, seq in enumerate(nested):
        buckets.setdefault(len(seq), []).append(i)
    return buckets


@dataclass(frozen=True)
class _FrameLike:
    path: str
    timestamp_sec: float


@dataclass(frozen=True)
class _VideoMetaLike:
    video_id: str
    duration_sec: float
    frames: List[_FrameLike]


def build_clips_from_video_meta(
    video_meta: Any,
    window_sec: float,
    stride_sec: float,
    min_clip_frames: int,
    max_clip_frames: int,
    keyframe_policy: str = "middle",
    return_keyframes: bool = False,
):
    def _pick_keyframe_path(paths_list: List[str], policy: str) -> Optional[str]:
        if not paths_list:
            return None
        mode = str(policy or "middle").strip().lower()
        if mode in {"first", "start"}:
            return str(paths_list[0])
        if mode in {"last", "end"}:
            return str(paths_list[-1])
        if mode in {"middle", "mid", ""}:
            return str(paths_list[len(paths_list) // 2])
        if mode in {"sharpest", "max_sharpness"} and cv2 is not None:
            best_idx = len(paths_list) // 2
            best_score = -1.0
            for idx, frame_path in enumerate(paths_list):
                try:
                    image = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
                    if image is None:
                        continue
                    score = float(cv2.Laplacian(image, cv2.CV_64F).var())
                    if score > best_score:
                        best_score = score
                        best_idx = idx
                except Exception:
                    continue
            return str(paths_list[best_idx])
        return str(paths_list[len(paths_list) // 2])

    frames_raw = getattr(video_meta, "frames", None) or []
    if not frames_raw:
        if return_keyframes:
            return [], [], []
        return [], []

    frames: List[_FrameLike] = []
    for f in frames_raw:
        frames.append(
            _FrameLike(
                path=str(getattr(f, "path", "")),
                timestamp_sec=float(getattr(f, "timestamp_sec", 0.0)),
            )
        )

    frames = sorted(frames, key=lambda x: x.timestamp_sec)
    ts = [float(x.timestamp_sec) for x in frames]
    paths = [str(x.path) for x in frames]
    duration = float(getattr(video_meta, "duration_sec", 0.0) or 0.0)

    clips: List[List[str]] = []
    clip_timestamps: List[Tuple[float, float]] = []
    clip_keyframes: List[Optional[str]] = []

    if window_sec <= 0 or stride_sec <= 0 or duration <= 0:
        if return_keyframes:
            return clips, clip_timestamps, clip_keyframes
        return clips, clip_timestamps

    n = len(frames)
    l = 0
    r = 0
    t = 0.0

    while t < duration + 1e-6:
        t_end = min(t + float(window_sec), duration)

        while l < n and ts[l] < t:
            l += 1
        if r < l:
            r = l
        while r < n and ts[r] <= t_end:
            r += 1

        count = r - l
        if count >= int(min_clip_frames):
            win_paths = paths[l:r]
            if len(win_paths) > int(max_clip_frames):
                step = len(win_paths) / float(max_clip_frames)
                idxs = [min(len(win_paths) - 1, int(i * step)) for i in range(int(max_clip_frames))]
                win_paths = [win_paths[i] for i in idxs]
            last_ts = ts[r - 1] if r - 1 >= l else t_end
            clips.append(win_paths)
            clip_timestamps.append((float(t), float(last_ts)))
            clip_keyframes.append(_pick_keyframe_path(win_paths, keyframe_policy))

        t += float(stride_sec)

    if return_keyframes:
        return clips, clip_timestamps, clip_keyframes
    return clips, clip_timestamps


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
        clip_lengths = [len(c) for c in clips]

        annotations: List[Annotation] = []
        clip_prompt = build_prompt(self.cfg.model.language)
        batch_size = max(1, int(getattr(self.cfg.model, "batch_size", 1)))
        max_batch_frames = int(getattr(self.cfg.model, "max_batch_frames", 0) or 0)

        t_gen_start = time.perf_counter()
        buckets = _bucket_indices_by_len(clips)

        for L in sorted(buckets.keys()):
            idxs = buckets[L]
            effective_batch = batch_size
            if max_batch_frames > 0:
                effective_batch = max(1, min(batch_size, max_batch_frames // max(1, int(L))))
            for batch_start in range(0, len(idxs), effective_batch):
                batch_idxs = idxs[batch_start: batch_start + effective_batch]
                batch_paths: List[List[Path]] = [[Path(p) for p in clips[i]] for i in batch_idxs]
                batch_ts = [clip_timestamps[i] for i in batch_idxs]

                texts = self.backend.describe_clips_batch(
                    batch_frame_paths=batch_paths,
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
                subset = merged[:10]
                summary_prompt = build_global_summary_prompt(
                    lang=self.cfg.model.language,
                    video_id=video_id,
                    duration_sec=float(video_duration_sec),
                    merged_anns=subset,
                )
                global_summary = self.backend.generate_text(summary_prompt)
                if global_summary:
                    global_summary = sanitize_global_summary(global_summary, lang=self.cfg.model.language)
        except Exception:
            global_summary = None

        extra: Dict[str, Any] = {}
        extra["clip_stats"] = {
            "num_clips": int(num_clips),
            "num_frames": int(num_frames),
            "frames_min": int(min(clip_lengths)) if clip_lengths else 0,
            "frames_max": int(max(clip_lengths)) if clip_lengths else 0,
            "frames_avg": float(sum(clip_lengths) / len(clip_lengths)) if clip_lengths else 0.0,
        }
        extra["pipeline"] = {
            "batch_size": int(batch_size),
            "max_batch_frames": int(getattr(self.cfg.model, "max_batch_frames", 0) or 0),
            "attn_implementation": str(getattr(self.cfg.model, "attn_implementation", "auto")),
            "torch_compile": bool(getattr(self.cfg.runtime, "torch_compile", False)),
            "torch_compile_mode": str(getattr(self.cfg.runtime, "torch_compile_mode", "")),
            "autocast_infer": bool(getattr(self.cfg.runtime, "autocast_infer", True)),
            "dtype": str(getattr(self.cfg.model, "dtype", "")),
        }
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

