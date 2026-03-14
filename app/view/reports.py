# app/view/reports.py
"""
Reports Streamlit page logic.

Purpose:
- Render the reporting surface and supporting evidence for selected videos.
- Keep report-building helpers and page state in one module.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

import streamlit as st

from app.api_client import BackendClient
from app.lib.formatters import E, clip_text, mmss, variant_from_token, variant_label
from app.view.shared import _loc, _page_title, _resolve_video_context, _section, _ui_lang, _video_items


def _bind_reports_state() -> None:
    """Initialize reports page state."""

    defaults = {
        "reports_video_id": "",
        "reports_lang": "",
        "reports_variant": "__base__",
        "reports_focus": "",
        "reports_payload": {},
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _report_overview_rows(video_id: str, lang_code: str, variant_token: str, outputs: Dict[str, Any], lang: str) -> List[tuple[str, str]]:
    """Build the report snapshot rows."""

    metrics = outputs.get("metrics") if isinstance(outputs.get("metrics"), dict) else {}
    annotations = outputs.get("annotations") if isinstance(outputs.get("annotations"), list) else []
    total_time = metrics.get("total_time_sec")
    return [
        (_loc("Видео", "Бейне", "Video", lang=lang), video_id or "-"),
        (_loc("Язык", "Тіл", "Language", lang=lang), str(lang_code or "-").upper()),
        (_loc("Вариант", "Нұсқа", "Variant", lang=lang), variant_label(variant_token)),
        (_loc("Сегменты", "Сегменттер", "Segments", lang=lang), str(len(annotations))),
        (_loc("Общее время", "Жалпы уақыт", "Total time", lang=lang), f"{float(total_time or 0.0):.1f}s" if total_time is not None else "-"),
    ]


def _render_info_rows(rows: Iterable[tuple[str, str]]) -> None:
    """Render key-value rows for the side report panel."""

    html = []
    for label, value in rows:
        html.append(
            f"""
            <div class="info-row">
                <div class="info-label">{E(label)}</div>
                <div class="info-value">{E(value)}</div>
            </div>
            """
        )
    st.markdown("".join(html), unsafe_allow_html=True)


def _render_supporting_evidence(hits: List[Dict[str, Any]], lang: str) -> None:
    """Render supporting evidence as clean divider rows."""

    _section(_loc("Опорные эпизоды", "Тірек эпизодтар", "Supporting evidence", lang=lang))
    if not hits:
        st.markdown(
            f"<div class='section-caption'>{E(_loc('Эпизоды появятся после построения отчёта.', 'Эпизодтар есеп құрылғаннан кейін көрінеді.', 'Supporting evidence appears after the report is built.', lang=lang))}</div>",
            unsafe_allow_html=True,
        )
        return

    visible_hits = [hit for hit in hits[:8] if isinstance(hit, dict)]
    for idx, hit in enumerate(visible_hits):
        headline = f"{str(hit.get('video_id') or '-')} · {mmss(float(hit.get('start_sec', 0.0) or 0.0))} - {mmss(float(hit.get('end_sec', 0.0) or 0.0))}"
        st.markdown(f"<div class='row-title'>{E(headline)}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='row-copy'>{E(clip_text(str(hit.get('description') or ''), 220))}</div>", unsafe_allow_html=True)
        if idx < len(visible_hits) - 1:
            st.markdown("<div class='timeline-divider'></div>", unsafe_allow_html=True)


def reports_metrics_tab(client: BackendClient, cfg: Any, ui_text: Dict[str, Dict[str, Any]]) -> None:
    """Render the reports page."""

    _ = ui_text
    _bind_reports_state()
    lang = _ui_lang(cfg)

    videos = _video_items(client)
    context = _resolve_video_context(
        videos,
        video_key="reports_video_id",
        variant_key="reports_variant",
        lang_key="reports_lang",
        base_lang=lang,
    )
    video_ids = context["video_ids"]
    variant_tokens = context["variant_tokens"]
    selected_video_id = context["selected_video_id"]
    current_variant = context["current_variant"]
    available_languages = context["available_languages"]
    current_lang = context["current_lang"]

    try:
        outputs = client.get_video_outputs(
            selected_video_id,
            str(st.session_state.get("reports_lang") or lang),
            variant=variant_from_token(st.session_state.get("reports_variant")),
        )
    except Exception:
        outputs = {}

    _page_title(_loc("Отчёты", "Есептер", "Reports", lang=lang))

    controls = st.columns([1.8, 0.95, 0.95], gap="small")
    with controls[0]:
        st.selectbox(_loc("Видео", "Бейне", "Video", lang=lang), options=video_ids, key="reports_video_id")
    with controls[1]:
        st.selectbox(_loc("Вариант", "Нұсқа", "Variant", lang=lang), options=variant_tokens, key="reports_variant", format_func=variant_label)
    with controls[2]:
        st.selectbox(_loc("Язык", "Тіл", "Language", lang=lang), options=available_languages or [lang], key="reports_lang")

    top = st.columns([1.52, 1.08], gap="large")
    with top[0]:
        _section(_loc("Демо-отчёт", "Демо-есеп", "Demo report", lang=lang))
        st.text_area(
            _loc("Фокус отчёта", "Есеп бағыты", "Report focus", lang=lang),
            key="reports_focus",
            height=120,
            placeholder=_loc(
                "Например: что происходило у входа и были ли сигналы риска?",
                "Мысалы: кіреберісте не болды және тәуекел белгілері болды ма?",
                "For example: what happened near the entrance and were there risk signals?",
                lang=lang,
            ),
        )

        report_build_notice = ""
        action_cols = st.columns([0.85, 3.15], gap="small")
        with action_cols[0]:
            if st.button("▶", key="reports_build_btn", use_container_width=True, help=_loc("Построить отчёт", "Есеп құрастыру", "Build report", lang=lang)):
                if not selected_video_id:
                    st.session_state["reports_payload"] = {}
                    report_build_notice = _loc(
                        "Нельзя построить отчёт: сначала выберите видео.",
                        "Есепті құрастыру мүмкін емес: алдымен бейнені таңдаңыз.",
                        "The report cannot be built until a video is selected.",
                        lang=lang,
                    )
                elif not outputs:
                    st.session_state["reports_payload"] = {}
                    report_build_notice = _loc(
                        "Нельзя построить отчёт для видео {video_id}: результаты обработки пока недоступны.",
                        "{video_id} бейнесі үшін есепті құрастыру мүмкін емес: өңдеу нәтижелері әзірге қолжетімсіз.",
                        "The report cannot be built for video {video_id}: processed results are not available yet.",
                        lang=lang,
                    ).format(video_id=selected_video_id)
                else:
                    try:
                        st.session_state["reports_payload"] = client.build_report(
                            video_id=selected_video_id or None,
                            language=str(st.session_state.get("reports_lang") or lang),
                            variant=variant_from_token(st.session_state.get("reports_variant")),
                            query=str(st.session_state.get("reports_focus") or "").strip() or None,
                            top_k=8,
                        )
                    except Exception:
                        report_build_notice = _loc(
                            "Не удалось построить отчёт для видео {video_id}. Проверьте, что обработка завершена и данные доступны.",
                            "{video_id} бейнесі үшін есепті құрастыру мүмкін болмады. Өңдеу аяқталғанын және деректердің қолжетімді екенін тексеріңіз.",
                            "The report could not be built for video {video_id}. Check that processing is complete and the data is available.",
                            lang=lang,
                        ).format(video_id=selected_video_id)

        with action_cols[1]:
            if report_build_notice:
                st.markdown(
                    f"<div class='bootstrap-alert bootstrap-alert--warning' role='alert'>{E(report_build_notice)}</div>",
                    unsafe_allow_html=True,
                )

        report_payload = st.session_state.get("reports_payload") or {}
        report_text = str(report_payload.get("report") or "").strip()
        if report_text:
            st.markdown(f"<div class='report-body'>{E(report_text)}</div>", unsafe_allow_html=True)

    with top[1]:
        _section(_loc("Снимок запуска", "Іске қосу көрінісі", "Run snapshot", lang=lang))
        _render_info_rows(_report_overview_rows(selected_video_id, current_lang, current_variant, outputs, lang))
        if outputs:
            summary_text = str(outputs.get("global_summary") or "").strip()
            if summary_text:
                st.markdown(f"<div class='analytics-copy'>{E(clip_text(summary_text, 260))}</div>", unsafe_allow_html=True)

    _render_supporting_evidence(list((st.session_state.get("reports_payload") or {}).get("supporting_hits") or []), lang)
