# app/view/overview.py
"""
Overview Streamlit page logic.

Purpose:
- Render the demo landing page, feature cards, and system-stage walkthrough.
- Keep presentation-oriented overview content separate from operational pages.
"""

from __future__ import annotations

import math
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import streamlit as st

from app.api_client import BackendClient
from app.lib.formatters import (
    E,
    available_variant_ids,
    clip_text,
    collect_available_languages,
    first_sentence,
    fmt_bytes,
    hms,
    humanize_token,
    mmss,
    variant_from_token,
    variant_label,
    video_variant_tokens,
)
from app.lib.media import ensure_browser_video, get_video_meta, img_to_data_uri, mtime
from app.view.shared import (
    PAGE_SIZE,
    ICON_CLEAR,
    ICON_CONFIRM,
    ICON_DELETE,
    ICON_DOWN,
    ICON_NEXT,
    ICON_OPEN,
    ICON_PAUSE,
    ICON_PREV,
    ICON_REFRESH,
    ICON_RESUME,
    ICON_START,
    ICON_UP,
    _caption,
    _error_prefix,
    _loc,
    _mark,
    _page_title,
    _resolve_video_context,
    _section,
    _session_choice,
    _ui_lang,
    _video_items,
    soft_note,
)

def _overview_stages(lang: str) -> List[Dict[str, str]]:
    """Return more detailed overview stages in plain language."""

    return [
        {
            "emoji": "📥",
            "title": _loc("Загрузка видео", "Бейнені жүктеу", "Video upload", lang=lang),
            "short": _loc("Видео попадает в систему и становится доступным для дальнейшей работы.", "Бейне жүйеге түсіп, кейінгі жұмысқа қолжетімді болады.", "The video enters the system and becomes available for further work.", lang=lang),
            "copy": _loc(
                "Сначала оператор добавляет ролик в библиотеку. Система принимает исходный файл, сохраняет базовые параметры и создает для него понятную рабочую запись. С этого момента видео уже можно отправлять в обработку, отслеживать в очереди и использовать в других разделах интерфейса.",
                "Алдымен оператор роликті кітапханаға қосады. Жүйе бастапқы файлды қабылдап, негізгі параметрлерін сақтап, оған түсінікті жұмыс жазбасын жасайды. Осы сәттен бастап бейнені өңдеуге жіберуге, кезекте бақылауға және интерфейстің басқа бөлімдерінде пайдалануға болады.",
                "First, the operator adds the video to the library. The system stores the source file, captures its basic metadata, and creates a clear working record that can be processed and tracked.",
                lang=lang,
            ),
            "impact": _loc("Это стартовая точка для всего дальнейшего сценария.", "Бұл бүкіл келесі сценарийдің бастапқы нүктесі.", "This is the starting point for the rest of the workflow.", lang=lang),
        },
        {
            "emoji": "⚙️",
            "title": _loc("Подготовка материала", "Материалды дайындау", "Preparation", lang=lang),
            "short": _loc("Система приводит ролик к стабильному рабочему виду перед анализом.", "Жүйе талдау алдында роликті тұрақты жұмыс форматына келтіреді.", "The system normalizes the video before analysis.", lang=lang),
            "copy": _loc(
                "Дальше ролик подготавливается к устойчивой аналитике. Система выравнивает технические свойства, чтобы разные файлы обрабатывались по одному сценарию. Пользователю не нужно думать о форматах и параметрах: эта работа скрыта внутри пайплайна.",
                "Одан кейін ролик тұрақты аналитикаға дайындалады. Жүйе әртүрлі файл бір сценариймен өңделуі үшін техникалық қасиеттерін теңестіреді. Пайдаланушыға формат пен параметр туралы ойлаудың қажеті жоқ: бұл жұмыс пайплайн ішінде жасалады.",
                "Next, the video is normalized for stable analytics. The system aligns technical properties so different files can be processed in a consistent way.",
                lang=lang,
            ),
            "impact": _loc("Этот этап снижает риск технических ошибок в следующих шагах.", "Бұл кезең келесі қадамдардағы техникалық қателер қаупін азайтады.", "This stage reduces the risk of technical issues later in the flow.", lang=lang),
        },
        {
            "emoji": "🔎",
            "title": _loc("Анализ сцен и событий", "Сахна мен оқиғаны талдау", "Scene and event analysis", lang=lang),
            "short": _loc("Кадры превращаются в наблюдения, события и понятные описания.", "Кадрлар бақылауларға, оқиғаларға және түсінікті сипаттамаларға айналады.", "Frames turn into observations, events, and readable descriptions.", lang=lang),
            "copy": _loc(
                "На этом этапе система уже работает со смыслом происходящего. Она рассматривает временные фрагменты ролика, выделяет важные моменты, формирует описания и готовит данные для таймлайна. Именно здесь видео перестает быть просто набором кадров и становится понятной последовательностью событий.",
                "Бұл кезеңде жүйе болып жатқанның мағынасымен жұмыс істейді. Ол роликтің уақыт фрагменттерін қарайды, маңызды сәттерді бөледі, сипаттамалар жасайды және таймлайнға дерек дайындайды. Дәл осы жерде бейне жай кадр жиыны емес, түсінікті оқиғалар тізбегіне айналады.",
                "At this stage, the system starts working with the meaning of what is happening. It reviews time segments, highlights important moments, creates descriptions, and prepares the timeline.",
                lang=lang,
            ),
            "impact": _loc("Именно здесь появляется основа для просмотра, поиска и объяснений.", "Дәл осы жерде қарау, іздеу және түсіндіру үшін негіз пайда болады.", "This is where the foundation for review, search, and explanation is created.", lang=lang),
        },
        {
            "emoji": "🗂️",
            "title": _loc("Индексация результатов", "Нәтижелерді индекстеу", "Indexing", lang=lang),
            "short": _loc("Найденные эпизоды попадают в поисковую основу системы.", "Табылған эпизодтар жүйенің іздеу негізіне түседі.", "Detected episodes are written into the system index.", lang=lang),
            "copy": _loc(
                "После анализа результаты нужно не просто сохранить, а сделать пригодными для быстрого поиска. Система складывает эпизоды и описания в индекс, на который потом опираются поиск, ассистент и отчеты. За счет этого нужный момент можно находить не вручную, а по смыслу запроса.",
                "Талдаудан кейін нәтижені жай сақтау жеткіліксіз, оны жылдам іздеуге ыңғайлы ету керек. Жүйе эпизодтар мен сипаттамаларды кейін іздеу, ассистент және есептер сүйенетін индекске салады. Соның арқасында керекті сәтті қолмен емес, сұраудың мағынасы бойынша табуға болады.",
                "After analysis, the results are stored in a retrieval-ready index. This is the base later used by search, the assistant, and reports.",
                lang=lang,
            ),
            "impact": _loc("Этот шаг превращает обработанные данные в рабочий поисковый инструмент.", "Бұл қадам өңделген деректі жұмыс істейтін іздеу құралына айналдырады.", "This step turns processed output into a usable retrieval tool.", lang=lang),
        },
        {
            "emoji": "🧾",
            "title": _loc("Поиск, ответы и отчёты", "Іздеу, жауап және есеп", "Search, answers, and reports", lang=lang),
            "short": _loc("Пользователь находит нужный эпизод и получает понятный итоговый вывод.", "Пайдаланушы керекті эпизодты тауып, түсінікті қорытынды алады.", "Users retrieve the right episode and get a clear final conclusion.", lang=lang),
            "copy": _loc(
                "Финальный этап объединяет все предыдущие результаты. Пользователь задает запрос, находит нужный эпизод, открывает его в аналитике, уточняет вопрос через ассистента и при необходимости формирует короткий отчет. Важно, что весь итоговый ответ остается связанным с конкретными фрагментами видео, поэтому его легко показать и проверить.",
                "Соңғы кезең алдыңғы нәтижелердің бәрін біріктіреді. Пайдаланушы сұрау береді, керекті эпизодты табады, оны аналитикада ашады, ассистент арқылы сұрағын нақтылайды және қажет болса қысқа есеп жасайды. Маңыздысы, соңғы жауап нақты бейне фрагменттерімен байланыста қалады, сондықтан оны көрсету де, тексеру де оңай.",
                "The final stage brings everything together. Users search, open the relevant episode, refine their question through the assistant, and build a short report while staying grounded in concrete video evidence.",
                lang=lang,
            ),
            "impact": _loc("Так система становится не просто пайплайном, а понятным инструментом для анализа и демонстрации.", "Осылайша жүйе жай пайплайн емес, талдау мен демонстрацияға арналған түсінікті құралға айналады.", "This is where the pipeline becomes a practical tool for analysis and demonstration.", lang=lang),
        },
    ]

def _overview_features(lang: str) -> List[Dict[str, str]]:
    """Return the refreshed Overview feature cards."""

    return [
        {
            "tone": "feature-card tone-a",
            "label": _loc("VLM-слой", "VLM қабаты", "VLM layer", lang=lang),
            "title": _loc("Генерация видеоописаний", "Бейнесипаттамаларды генерациялау", "Video description generation", lang=lang),
            "copy": _loc(
                "Система проходит по фрагментам видео и формирует нейтральные описания сцен, действий и объектов.",
                "Жүйе бейненің фрагменттері бойынша өтіп, көріністердің, әрекеттердің және объектілердің бейтарап сипаттамаларын жасайды.",
                "The system moves through video fragments and produces neutral descriptions of scenes, actions, and objects.",
                lang=lang,
            ),
            "detail": _loc(
                "Это создает основу для таймлайна, сводки и последующего анализа без ручного просмотра каждого отрезка.",
                "Бұл таймлайнға, қорытындыға және әр бөлікті қолмен қарамай-ақ кейінгі талдауға негіз дайындайды.",
                "This creates the base for the timeline, summary, and later analysis without manually reviewing every segment.",
                lang=lang,
            ),
        },
        {
            "tone": "feature-card tone-b",
            "label": _loc("Структурирование", "Құрылымдау", "Structuring", lang=lang),
            "title": _loc("Аналитика", "Аналитика", "Analytics", lang=lang),
            "copy": _loc(
                "Из видеоописаний собираются эпизоды, сигналы внимания, краткая сводка и метрики обработки.",
                "Бейнесипаттамалардан эпизодтар, назар аударту сигналдары, қысқа қорытынды және өңдеу метрикалары жиналады.",
                "Video descriptions are turned into episodes, attention signals, a short summary, and processing metrics.",
                lang=lang,
            ),
            "detail": _loc(
                "Оператор сразу видит, что произошло, когда это было и какие участки ролика важны для проверки.",
                "Оператор бірден не болғанын, оның қашан болғанын және роликтің қай бөліктері тексеруге маңызды екенін көреді.",
                "Operators immediately see what happened, when it happened, and which parts of the video matter for review.",
                lang=lang,
            ),
        },
        {
            "tone": "feature-card tone-c",
            "label": _loc("Навигация", "Навигация", "Navigation", lang=lang),
            "title": _loc("Поиск событий", "Оқиғаларды іздеу", "Event search", lang=lang),
            "copy": _loc(
                "Поиск находит нужные эпизоды по смыслу запроса, временным ориентирам и содержанию наблюдений.",
                "Іздеу сұраудың мағынасы, уақыт белгілері және бақылау мазмұны бойынша керек эпизодтарды табады.",
                "Search retrieves the right episodes from the query meaning, time cues, and observation content.",
                lang=lang,
            ),
            "detail": _loc(
                "Это ускоряет разбор больших архивов и помогает сразу перейти к нужному моменту в аналитике.",
                "Бұл үлкен архивтерді талдауды жылдамдатып, аналитикадағы керек сәтке бірден өтуге көмектеседі.",
                "This speeds up archive review and helps jump straight to the relevant moment in analytics.",
                lang=lang,
            ),
        },
        {
            "tone": "feature-card tone-d",
            "label": _loc("Диалог по данным", "Дерекке негізделген диалог", "Grounded dialogue", lang=lang),
            "title": _loc("Ассистент", "Ассистент", "Assistant", lang=lang),
            "copy": _loc(
                "Ассистент отвечает на вопросы по найденным эпизодам и опирается на те же фрагменты, что открыты в системе.",
                "Ассистент табылған эпизодтар бойынша сұрақтарға жауап береді және жүйеде ашық тұрған сол фрагменттерге сүйенеді.",
                "The assistant answers questions about retrieved episodes and relies on the same fragments opened in the system.",
                lang=lang,
            ),
            "detail": _loc(
                "Так проще уточнять детали, собирать краткие выводы и удерживать ответ привязанным к видеодоказательствам.",
                "Осылайша детальдарды нақтылау, қысқа қорытынды жинау және жауапты бейне-дәлелдерге байлап ұстау жеңілдейді.",
                "This makes it easier to refine details, build short conclusions, and keep the answer tied to video evidence.",
                lang=lang,
            ),
        },
    ]

def _stage_nav_button_label(stage: Dict[str, str], stage_number: int, lang: str) -> str:
    """Build a compact multiline stage-switch button label without emoji."""

    stage_label = _loc("\u042d\u0442\u0430\u043f", "\u041a\u0435\u0437\u0435\u04a3", "Stage", lang=lang)
    return "\n".join((f"{stage_label} {stage_number}", stage["title"]))

def _render_overview_carousel(stages: List[Dict[str, str]], session_key: str, lang: str) -> None:
    """Render overview stages with one main card and two compact native stage switch buttons."""

    if not stages:
        return

    idx = int(st.session_state.get(session_key) or 0)
    idx = max(0, min(idx, len(stages) - 1))
    st.session_state[session_key] = idx
    current = stages[idx]
    stage_label = _loc("\u042d\u0442\u0430\u043f", "\u041a\u0435\u0437\u0435\u04a3", "Stage", lang=lang)

    progress = "".join(
        f"<span class='stage-dot{' active' if i == idx else ''}'>{i + 1}</span>"
        for i in range(len(stages))
    )
    st.markdown(f"<div class='stage-progress'>{progress}</div>", unsafe_allow_html=True)

    adjacent: List[int] = []
    if idx > 0:
        adjacent.append(idx - 1)
    if idx + 1 < len(stages):
        adjacent.append(idx + 1)

    with st.container():
        _mark("stage-layout-marker")
        main_col, side_col = st.columns([2.02, 0.98], gap="medium")

        with main_col:
            st.markdown(
                f"""
                <div class="stage-card">
                    <div class="stage-card-head">
                        <div class="stage-badge">{E(current['emoji'])}</div>
                        <div>
                            <div class="stage-kicker">{E(stage_label)} {idx + 1}</div>
                            <div class="stage-title">{E(current['title'])}</div>
                        </div>
                    </div>
                    <div class="stage-short">{E(current['short'])}</div>
                    <div class="stage-copy">{E(current['copy'])}</div>
                    <div class="stage-impact">{E(current['impact'])}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with side_col:
            for target_idx in adjacent:
                stage = stages[target_idx]
                with st.container():
                    _mark("stage-nav-marker")
                    if st.button(
                        _stage_nav_button_label(stage, target_idx + 1, lang),
                        key=f"{session_key}_{target_idx}",
                        use_container_width=True,
                        help=stage["title"],
                    ):
                        st.session_state[session_key] = target_idx
                        st.rerun()
            if len(adjacent) < 2:
                st.markdown("<div class='stage-nav-spacer'></div>", unsafe_allow_html=True)

def overview_tab(client: BackendClient, cfg: Any, ui_text: Dict[str, Dict[str, Any]]) -> None:
    """Render the overview page with cleaner spacing."""

    _ = client
    _ = ui_text
    lang = _ui_lang(cfg)

    st.markdown(
        f"""
        <div class="intro-card">
            <div class="intro-title">{E(_loc('SmartCampus V2T для поиска событий и понятного анализа видео', 'SmartCampus V2T бейне оқиғаларын іздеуге және түсінікті талдауға арналған', 'SmartCampus V2T for event retrieval and clear video review', lang=lang))}</div>
            <div class="intro-copy">{E(_loc('Демо-интерфейс показывает полный путь: от загрузки видео и запуска обработки до просмотра результата, поиска нужного эпизода и подготовки итогового отчёта.', 'Демо-интерфейс толық жолды көрсетеді: бейнені жүктеу мен өңдеуді іске қосудан бастап нәтижені қарауға, керекті эпизодты іздеуге және қорытынды есеп дайындауға дейін.', 'This demo interface shows the full path from upload and processing to review, event retrieval, and final reporting.', lang=lang))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='overview-section-gap'></div>", unsafe_allow_html=True)
    _section(_loc("Основные функции системы", "Жүйенің негізгі функциялары", "Main system functions", lang=lang))
    feature_rows = _overview_features(lang)
    for start in range(0, len(feature_rows), 2):
        columns = st.columns(2, gap="large")
        for col, item in zip(columns, feature_rows[start : start + 2]):
            with col:
                st.markdown(
                    f"""
                    <div class="{E(item['tone'])}">
                        <div class="feature-label">{E(item['label'])}</div>
                        <div class="feature-title">{E(item['title'])}</div>
                        <div class="feature-copy">{E(item['copy'])}</div>
                        <div class="feature-detail">{E(item['detail'])}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        if start + 2 < len(feature_rows):
            st.markdown("<div class='overview-grid-gap'></div>", unsafe_allow_html=True)

    st.markdown("<div class='overview-section-gap'></div>", unsafe_allow_html=True)
    _section(_loc("Этапы работы системы", "Жүйе жұмысының кезеңдері", "System stages", lang=lang))
    _render_overview_carousel(_overview_stages(lang), "overview_stage_index", lang)
