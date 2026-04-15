# Руководство по проекту SmartCampus V2T

English version: [docs/project_guide.md](project_guide.md)

## 1. Область документа

Это руководство описывает репозиторий SmartCampus V2T в его текущем кодовом состоянии. Документ задуман как основной технический reference по следующим темам:

- назначение системы,
- runtime-поверхности приложения,
- функциональные роли frontend, backend, worker и основного pipeline,
- модель конфигурации,
- очередь и система jobs,
- схема хранения артефактов,
- поиск, grounded generation и многоязычное поведение,
- локальные эксплуатационные и исследовательские инструменты, которые уже есть в репозитории.

Этот репозиторий нельзя корректно свести ни к «обертке над моделью», ни к «демо на Streamlit», ни к «набору скриптов». Это локальная end-to-end система видеоаналитики, которая принимает сохраненное видео, превращает его в структурированные семантические артефакты, строит производные многоязычные представления, собирает retrieval-индексы и открывает поверх них grounded search/report/QA/RAG сценарии.

Система изначально спроектирована как local-first. В ней приоритет отдан прозрачному состоянию на файловой системе, явной конфигурации runtime и стабильным локальным точкам входа, а не внешней распределенной инфраструктуре.

## 2. Позиционирование системы

SmartCampus V2T состоит из трех основных runtime-поверхностей:

| Runtime-поверхность | Точка входа | Основная роль | Типичный пользователь или потребитель |
| --- | --- | --- | --- |
| Streamlit UI | `app/main.py` | интерактивная операторская консоль | локальный оператор, демонстратор, проверяющий |
| FastAPI backend | `backend/api.py` | стабильный HTTP-слой над библиотекой, jobs, queue, retrieval и reports | Streamlit UI, smoke-скрипты, локальная автоматизация |
| Background worker | `backend/worker.py` | асинхронное выполнение process, translate и index jobs | внутренняя очередь выполнения |

Все три поверхности используют одну конфигурационную модель и одно локальное хранилище артефактов. UI не запускает тяжелый pipeline напрямую. Он обращается к backend, backend хранит и отдает состояние, а worker выполняет длительные этапы обработки в фоне.

## 3. Функциональное покрытие

На текущий момент проект реализует следующие крупные возможности:

| Возможность | Что делает система | Основные результаты | Ключевые зоны кода |
| --- | --- | --- | --- |
| Управление библиотекой видео | сохраняет загруженные ролики как управляемые локальные объекты | raw-видео, manifest по видео | `app/view/storage.py`, `backend/api.py`, `src/utils/video_store.py` |
| Нормализация видео | подготавливает ролик к стабильному анализу | подготовленные кадры, cache-метаданные | `src/video/io.py` |
| Генерация видеоописаний | запускает vision-language слой по временным клипам | clip observations на английском | `src/video/describe.py`, `src/core/vlm_backend.py` |
| Семантическое структурирование | превращает сырые наблюдения в структурированные сегменты | segment rows с семантическими полями | `src/llm/analyze.py`, `src/guard/*` |
| Генерация summary | строит сводку по видео | summary payload по языкам | `src/llm/summary.py` |
| Перевод | создает производные представления на русском и казахском | переведенные сегменты и summary | `backend/jobs/translate_runtime.py`, `src/translation/service.py` |
| Индексация | строит sparse и dense retrieval-состояние | индексные файлы, manifests, status | `src/search/builder.py`, `backend/jobs/index_runtime.py` |
| Поиск | находит релевантные события по тексту и фильтрам | ранжированные search hits | `src/search/engine.py`, `backend/retrieval_runtime.py` |
| Grounded reports | формирует короткие отчеты с опорой на доказательства | report text, citations, supporting hits | `backend/http/grounded.py` |
| Grounded QA | отвечает на вопросы по индексированным доказательствам | answer text, citations, supporting hits | `backend/http/grounded.py` |
| Grounded RAG assistant | поддерживает ассистентный режим с привязкой к найденным эпизодам | answer text, context, citations, supporting hits | `backend/http/grounded.py`, `app/view/search.py` |
| Метрики и эксперименты | экспортирует метрики, запускает матрицы экспериментов, считает relevance | CSV, JSON, zip bundles, comparison reports | `scripts/*` |

Есть одно особенно важное архитектурное правило:

| Правило | Практический смысл |
| --- | --- |
| English-first canonical storage | канонические структурированные артефакты сначала строятся на английском |
| Производные многоязычные слои | `ru` и `kz` являются производными от английских исходных артефактов |
| Evidence-based generation | отчеты, QA и assistant-ответы должны оставаться привязанными к реальным найденным фрагментам видео |

## 4. Сквозной жизненный цикл

Обычный end-to-end сценарий выглядит так:

1. Видео загружается в локальную библиотеку.
2. Backend создает `process` job и ставит ее в файловую очередь.
3. Worker берет job в lease и разрешает эффективный profile и variant.
4. Видео нормализуется и подготавливается в analysis-ready представление.
5. Формируются clip windows и keyframes.
6. VLM генерирует сырые английские clip observations.
7. Guard и schema helpers выравнивают и очищают payload.
8. Семантический слой превращает наблюдения в структурированные сегменты.
9. Генерируется video-level summary.
10. Артефакты, метрики и manifests записываются в `data/videos/<video_id>/`.
11. При необходимости создаются translation jobs для `ru` и `kz`.
12. При необходимости обновляется retrieval-индекс.
13. UI и API используют результаты в разделах analytics, search, reports, QA и RAG.

Тот же pipeline можно показать так:

```text
загрузка видео
  -> создание job
  -> lease в worker
  -> подготовка видео
  -> построение клипов
  -> VLM-наблюдения
  -> guard/schema repair
  -> структурирование сегментов
  -> summary
  -> сохранение артефактов
  -> перевод
  -> индексация
  -> search / reports / QA / RAG
```

Этот жизненный цикл изначально асинхронный. Upload и создание job быстрые, а тяжелая обработка выполняется отдельно в worker.

## 5. Структура репозитория

### 5.1 Верхнеуровневая структура

| Каталог | Ответственность | Примечание |
| --- | --- | --- |
| `app/` | Streamlit frontend | страницы, общие UI helpers, HTTP client |
| `backend/` | FastAPI backend и orchestration worker | API entrypoint, retrieval runtime, job runtime helpers |
| `src/` | доменная логика ядра | runtime config, video pipeline, LLM, translation, search |
| `configs/` | runtime profiles и variants | YAML-профили и экспериментальные overrides |
| `scripts/` | локальные эксплуатационные и исследовательские инструменты | smoke checks, metrics, evaluation, experiment matrix |
| `data/` | runtime-состояние и артефакты | videos, jobs, queue, indexes, research outputs |
| `docs/` | документация | текущее руководство и связанные документы |
| `.agent/` | приватная рабочая память проекта | notes, decisions, state snapshots |

### 5.2 Структура frontend

| Путь | Назначение |
| --- | --- |
| `app/main.py` | единая Streamlit-точка входа и верхнеуровневая маршрутизация |
| `app/api_client.py` | UI-клиент для backend HTTP routes |
| `app/state.py` | общие session defaults |
| `app/view/overview.py` | обзорная страница системы |
| `app/view/storage.py` | библиотека, загрузка, очередь и запуск jobs |
| `app/view/analytics.py` | просмотр видео, summary, metrics и timeline |
| `app/view/search.py` | поиск, результаты и grounded assistant |
| `app/view/reports.py` | grounded reporting |
| `app/view/shared.py` | общие UI helpers, localization helpers, shared components |
| `app/lib/i18n.py` | загрузка UI text и translator helpers |
| `app/lib/media.py` | CSS и media helpers |
| `app/assets/` | CSS, logo, UI text assets |

### 5.3 Структура backend

| Путь | Назначение |
| --- | --- |
| `backend/api.py` | регистрация стабильных HTTP routes |
| `backend/worker.py` | основной worker loop и dispatch |
| `backend/deps.py` | общие config/path/dependency helpers для backend entrypoints |
| `backend/retrieval_runtime.py` | retrieval, fallback, grounding, citations и metrics helpers |
| `backend/schemas.py` | request/response contracts |
| `backend/http/common.py` | общие API helpers, upload normalization, grounded-response helpers |
| `backend/http/grounded.py` | построение report, QA и RAG responses |
| `backend/jobs/*` | queue, persistence, execution, worker support, metrics, experimental fan-out |

### 5.4 Структура доменной логики

| Путь | Назначение |
| --- | --- |
| `src/core/*` | typed config, config loading, runtime flags, artifact types, интеграция с VLM backend |
| `src/video/*` | подготовка видео, генерация клипов, observation pipeline |
| `src/llm/*` | семантическое структурирование и summary generation |
| `src/guard/*` | safety gates и schema cleanup |
| `src/translation/*` | translation routing, cache и post-edit |
| `src/search/*` | builder, query engine, ranking, embeddings, corpus helpers |
| `src/utils/video_store.py` | каноническая схема хранения артефактов по видео |

## 6. Frontend / операторская консоль

### 6.1 Путь инициализации UI

`app/main.py` является единственной Streamlit-точкой входа. Во время запуска модуль:

1. загружает effective config через `src/core/runtime.py`,
2. применяет CSS из `app/assets/styles.css`,
3. поднимает локализованные строки из `app/assets/ui_text.json`,
4. привязывает session defaults,
5. собирает backend base URL из typed config,
6. проверяет backend health,
7. рисует общий header,
8. отправляет пользователя в нужный page-module.

### 6.2 Верхнеуровневые страницы

| Страница | Модуль | Пользовательская роль | Типичные backend-зависимости |
| --- | --- | --- | --- |
| Overview | `app/view/overview.py` | объясняет систему, ее функции и этапы обработки | health only |
| Storage | `app/view/storage.py` | загрузка видео, просмотр библиотеки, создание jobs, управление queue | videos, jobs, queue |
| Video Analytics | `app/view/analytics.py` | разбор конкретного обработанного видео, summary, metrics и timeline | outputs, metrics |
| Search | `app/view/search.py` | hybrid search, просмотр hits, grounded assistant | search, index rebuild, rag |
| Reports | `app/view/reports.py` | генерация коротких grounded reports по видео или query | reports, outputs |

### 6.3 Функциональность страниц

| Страница | Основные controls | Что получает пользователь |
| --- | --- | --- |
| Overview | feature cards, stage switcher | презентация системы, ключевые функции, пошаговое описание pipeline |
| Storage | фильтры библиотеки, upload widget, profile selector, variant selector, queue controls | карточки видео, состояние загрузки, очередь, действия запуска обработки |
| Video Analytics | выбор видео, языка, варианта, timeline seek buttons | playback, summary, readiness status, metrics rows, event timeline |
| Search | query box, language selector, variant selector, event/risk filters, anomaly-only, dedupe, assistant input | search hits, metadata chips, evidence-linked assistant replies |
| Reports | выбранное видео, язык, вариант, report query | report text, citations, supporting evidence, metrics summary rows |

### 6.4 Связь UI и backend

| Действие в UI | Backend route |
| --- | --- |
| загрузить библиотеку | `GET /v1/videos` |
| загрузить видео | `POST /v1/videos/upload` |
| удалить видео | `DELETE /v1/videos/{video_id}` |
| получить outputs для analytics/reports | `GET /v1/videos/{video_id}/outputs` |
| получить metrics summary | `GET /v1/videos/{video_id}/metrics-summary` |
| создать process или translate job | `POST /v1/jobs` |
| прочитать состояние job | `GET /v1/jobs/{job_id}` |
| отменить job | `POST /v1/jobs/{job_id}/cancel` |
| посмотреть queue | `GET /v1/queue` |
| поставить queue на паузу или снять с паузы | `POST /v1/queue/pause`, `POST /v1/queue/resume` |
| поменять порядок или удалить queued job | `POST /v1/queue/move`, `DELETE /v1/queue/{job_id}` |
| пересобрать индекс | `POST /v1/index/rebuild` |
| выполнить поиск | `POST /v1/search` |
| построить grounded report | `POST /v1/reports` |
| получить grounded QA или assistant answer | `POST /v1/qa`, `POST /v1/rag` |

## 7. Backend HTTP surface

### 7.1 Health routes

| Метод | Route | Назначение |
| --- | --- | --- |
| `GET` | `/healthz` | базовая проверка liveness |
| `GET` | `/v1/health` | versioned health |
| `GET` | `/v1/healthz` | versioned liveness |

### 7.2 Video routes

| Метод | Route | Назначение | Основной ответ |
| --- | --- | --- | --- |
| `GET` | `/v1/videos` | список управляемых видео | `List[VideoItem]` |
| `POST` | `/v1/videos/upload` | загрузка и нормализация исходного файла | `VideoItem` |
| `DELETE` | `/v1/videos/{video_id}` | удаление локального video tree | `{ ok: true }` |
| `GET` | `/v1/videos/{video_id}/outputs` | чтение output bundle для языка и варианта | `VideoOutputs` |
| `GET` | `/v1/videos/{video_id}/batch-manifest` | чтение experimental parent/child manifest | raw JSON payload |
| `GET` | `/v1/videos/{video_id}/metrics-summary` | нормализованная сводка processing metrics | `MetricsSummaryResponse` |

### 7.3 Job и queue routes

| Метод | Route | Назначение | Основной ответ |
| --- | --- | --- | --- |
| `POST` | `/v1/jobs` | создать и поставить job в очередь | `JobCreateResponse` |
| `GET` | `/v1/jobs/{job_id}` | прочитать состояние job | `JobStatus` |
| `POST` | `/v1/jobs/{job_id}/cancel` | запросить отмену | `JobCancelResponse` |
| `GET` | `/v1/queue` | snapshot очереди | `QueueListResponse` |
| `POST` | `/v1/queue/pause` | поставить queue на паузу | `QueueStatus` payload |
| `POST` | `/v1/queue/resume` | возобновить queue | `QueueStatus` payload |
| `POST` | `/v1/queue/move` | изменить порядок queued job | `QueueMoveResponse` |
| `DELETE` | `/v1/queue/{job_id}` | удалить queued job и пометить его canceled | raw JSON payload |

### 7.4 Retrieval и grounded generation routes

| Метод | Route | Назначение | Основной ответ |
| --- | --- | --- | --- |
| `GET` | `/v1/index/status` | текущее состояние индекса | `IndexStatus` |
| `POST` | `/v1/index/rebuild` | пересобрать или обновить индекс | `IndexRebuildResponse` |
| `POST` | `/v1/search` | hybrid search по обработанным артефактам | `SearchResponse` |
| `POST` | `/v1/reports` | grounded report generation | `ReportResponse` |
| `POST` | `/v1/qa` | grounded question answering | `QaResponse` |
| `POST` | `/v1/rag` | grounded assistant response | `RagResponse` |

### 7.5 Ключевые request contracts

#### Search request

| Поле | Смысл |
| --- | --- |
| `query` | основной естественно-языковой поисковый запрос |
| `top_k` | лимит результатов |
| `video_id` | ограничение поиском по одному видео |
| `language` | язык retrieval |
| `variant` | namespace экспериментального варианта |
| `dedupe` | дедупликация пересекающихся hits |
| `start_sec`, `end_sec` | фильтр по временному диапазону |
| `min_duration_sec`, `max_duration_sec` | фильтр по длительности сегмента |
| `event_type` | фильтр по типу события |
| `risk_level` | фильтр по уровню риска |
| `tags` | фильтр по тегам |
| `objects` | фильтр по объектам |
| `people_count_bucket` | грубый фильтр по количеству людей |
| `motion_type` | фильтр по типу движения |
| `anomaly_only` | требование только anomaly-marked hits |

#### Search hit payload

| Поле | Смысл |
| --- | --- |
| `video_id` | видео-владелец сегмента |
| `language` | язык индексированного артефакта |
| `start_sec`, `end_sec` | временные границы |
| `description` | человекочитаемое описание сегмента |
| `score`, `sparse_score`, `dense_score` | значения ранжирования |
| `segment_id` | идентификатор сегмента, если он есть |
| `event_type`, `risk_level` | структурированная семантика |
| `tags`, `objects` | семантические списки |
| `people_count_bucket`, `motion_type` | дополнительные структурированные поля |
| `anomaly_flag` | признак аномалии |
| `variant` | namespace варианта, если это не base |

#### Grounded generation requests

| Request model | Обязательный ввод | Дополнительное ограничение |
| --- | --- | --- |
| `ReportRequest` | `query` или `video_id` | `language`, `variant`, `top_k` |
| `QaRequest` | `question` | `video_id`, `language`, `variant`, `top_k` |
| `RagRequest` | `query` | `video_id`, `language`, `variant`, `top_k` |

#### Output bundle response

`VideoOutputs` представляет один языково-вариантный набор результатов. В нем могут находиться:

| Поле | Смысл |
| --- | --- |
| `manifest` | payload готовности и статуса |
| `run_manifest` | config и execution metadata |
| `batch_manifest` | метаданные experimental fan-out |
| `annotations` | структурированные сегменты, которые получает UI |
| `metrics` | сырой metrics payload |
| `global_summary` | итоговый summary text |

## 8. Queue, jobs и worker model

### 8.1 Почему используется файловая очередь

Проект не зависит от внешнего брокера вроде Redis, RabbitMQ или Celery. Состояние очереди хранится в локальных файлах. Это упрощает запуск на одной машине, облегчает отладку и делает систему удобной для демонстрации без дополнительной инфраструктуры.

### 8.2 Основные backend-модули job-слоя

| Модуль | Роль |
| --- | --- |
| `backend/jobs/store.py` | низкоуровневые примитивы jobs и queue files |
| `backend/jobs/control.py` | locking, leasing, state transitions, cancellation checks |
| `backend/jobs/queue_runtime.py` | queue snapshot, reorder, running-job discovery |
| `backend/jobs/worker_runtime.py` | разрешение effective config и service bundle для job |
| `backend/jobs/process_runtime.py` | полное выполнение process job |
| `backend/jobs/translate_runtime.py` | выполнение translation job |
| `backend/jobs/runtime_common.py` | общие finalize и index helpers |
| `backend/jobs/index_runtime.py` | хранение index status |
| `backend/jobs/experimental.py` | helpers для experiment fan-out |

### 8.3 Поддерживаемые типы jobs

| Job type | Назначение | Типичный источник |
| --- | --- | --- |
| `process` | полный pipeline для одного видео | UI или backend API |
| `translate` | построение target-language view из английских артефактов | process runtime или API |
| `index` | пересборка retrieval-индекса без повторной обработки видео | API или локальное обслуживание |

### 8.4 Состояния job

| State | Значение |
| --- | --- |
| `queued` | job ожидает в очереди |
| `running` | worker взял job и исполняет ее |
| `done` | job завершилась успешно |
| `failed` | job завершилась с unrecovered error |
| `cancel_requested` | отмена запрошена, но исполнение могло еще не завершиться |
| `canceled` | job отменена и финализирована |

### 8.5 Роли worker

Переменная `SMARTCAMPUS_WORKER_ROLE` может ограничивать тип jobs, которые worker имеет право исполнять.

| Роль worker | Разрешенная работа |
| --- | --- |
| `all` | все типы jobs |
| `gpu` | `process` jobs |
| `mt` | `translate` jobs |
| `cpu` | `index` jobs |

Это позволяет локально разделять GPU-нагруженную обработку, перевод и индексную работу.

### 8.6 Experimental fan-out

Worker может развернуть одну родительскую experimental job в набор дочерних jobs, если активная конфигурация указывает experiment mode. На практике это выглядит так:

1. родительская job создается под experiment-capable profile,
2. worker видит, что job должна быть expanded,
3. создаются child jobs по каждому настроенному variant,
4. пишется experimental batch manifest,
5. родительская job помечается как done именно как шаг expand, а не как обычный process run.

Так реализуется сравнительный прогон профилей и вариантов без усложнения внешнего API.

## 9. Модель конфигурации

### 9.1 Сборка effective config

Effective runtime configuration строится в `src/core/runtime.py`, а typed-модель конфигурации определена в `src/core/config.py`.

Разрешение конфигурации поддерживает:

- один активный profile,
- ноль или один active variant,
- YAML-наследование через `extends`,
- выбор через environment variables,
- post-processing в strongly typed config object.

### 9.2 Основные selectors

| Переменная | Роль |
| --- | --- |
| `SMARTCAMPUS_PROFILE` | выбрать активный profile |
| `SMARTCAMPUS_VARIANT` | выбрать активный variant |
| `SMARTCAMPUS_WORKER_ROLE` | ограничить роль worker |

### 9.3 Typed config blocks

| Config block | Что контролирует |
| --- | --- |
| `paths` | project root, `data/`, `models/`, `indexes/`, assets и config folders |
| `ui` | языки UI, asset paths, CSS path, UI text path |
| `backend` | host, scheme и port backend |
| `search` | embeddings, reranking, fusion, dedupe, fallback languages, dense input mode |
| `video` | decode, resize, frame gating, face blur, anonymization, save policy |
| `clips` | clip windowing, stride, keyframe policy |
| `model` | VLM model path, device, dtype, batch policy, inference limits |
| `llm` | backend семантического и summary generation |
| `guard` | query и output safety gating |
| `runtime` | seed, threading, TF32, matmul precision, compile flags, metrics sampling |
| `translation` | MT backend, routes, batch size, cache, query-time translation, post-edit policy |
| `jobs` | каталог job records |
| `queue` | каталог queue |
| `locks` | каталог worker locks |
| `worker` | poll interval, lease interval, concurrency policy |
| `index` | политика auto-update индекса |
| `webhook` | внешние уведомления |
| `experiment` | compare mode и variant ids |

### 9.4 Активные profiles в репозитории

| Profile | Роль |
| --- | --- |
| `configs/profiles/main.yaml` | основной рабочий runtime |
| `configs/profiles/experimental.yaml` | экспериментальный profile с overwrite и repeated metrics |

### 9.5 Variants

Файлы в `configs/variants/` задают явные overrides. Они используются для контролируемых сравнений и могут иметь собственные output namespaces и собственное индексное состояние.

## 10. Основной processing pipeline

`process` job является главным pipeline-сценарием. Он оркестрируется в `backend/jobs/process_runtime.py`, но использует сервисы и логику из `src/`.

### 10.1 Карта этапов

| Этап | Основной код | Вход | Выход |
| --- | --- | --- | --- |
| разрешение контекста | `backend/jobs/worker_runtime.py` | job record, active profile/variant | effective config и service bundle |
| подготовка видео | `src/video/io.py` | raw video file | подготовленные кадры и video metadata |
| построение клипов | `src/video/clips.py` | подготовленные кадры | clip windows и keyframes |
| VLM-наблюдение | `src/video/describe.py`, `src/core/vlm_backend.py` | clip/keyframe payloads | сырые английские observations |
| schema repair и guard | `src/guard/service.py`, `src/guard/schemas.py` | сырой generated payload | нормализованный и безопасный payload |
| структурирование сегментов | `src/llm/analyze.py` | observations | structured segment rows |
| summary generation | `src/llm/summary.py` | structured segments | video summary |
| persistence | `src/utils/video_store.py` | все артефакты pipeline | сохраненные manifests, rows, summary, metrics |
| опциональная индексация | `src/search/builder.py` | сохраненные сегменты | обновленный index state |
| опциональный перевод | `backend/jobs/translate_runtime.py` | английские outputs | производные `ru`/`kz` outputs |

### 10.2 Что делает этап подготовки видео

`src/video/io.py` — один из самых важных доменных модулей. Он отвечает за:

- разрешение пути к исходному видео,
- нормализацию fps и decode behavior,
- resize и letterbox кадров,
- фильтрацию темных и низкоинформативных кадров,
- измерение blur и motion,
- optional face anonymization,
- сохранение prepared frames и вспомогательных кадров,
- запись и повторное использование cache metadata.

Именно этот этап обеспечивает стабильное и однородное представление ролика для последующих VLM и semantic stages.

### 10.3 Observation layer и semantic layer

В системе сознательно разделены два уровня:

| Уровень | Назначение |
| --- | --- |
| VLM observation | описать то, что визуально присутствует в clip windows |
| semantic structuring | превратить observation layer в сегменты с event type, tags, risk, people bucket, motion и anomaly cues |

Это принципиально важно, потому что retrieval и reporting работают не только по сырым captions, а по структурированным сегментам.

### 10.4 Summary stage

Summary stage строит video-level summary по структурированным сегментам. Summary хранится отдельно от segment rows, поэтому ее можно использовать в analytics и reports без повторного прохода по всему набору сегментов.

## 11. Translation layer

Перевод находится downstream от английского pipeline. Сначала строятся канонические английские артефакты, потом из них выводятся локализованные представления.

### 11.1 Translation flow

| Шаг | Описание |
| --- | --- |
| загрузка source artifacts | чтение английских segments и summaries |
| route selection | выбор MT route для языковой пары |
| перевод сегментов | перевод описаний сегментов и связанных полей |
| selective post-edit | точечное улучшение summaries, reports, QA или selected segments |
| сохранение outputs | запись translated segments и summaries |
| optional target-language indexing | перестроение retrieval state для target language |

### 11.2 Активная translation-конфигурация

В основном profile сейчас используются следующие принципы:

| Область | Текущий смысл |
| --- | --- |
| backend | `ctranslate2` |
| source language | английский |
| target languages | русский и казахский |
| query-time translation | включен |
| offline translation | включен |
| post-edit targets | summaries, reports, QA, selected segments |

### 11.3 Почему английский остается каноническим

Такой дизайн не дает системе расползтись на несколько первичных источников правды, сохраняет цельность индексации и упрощает experiment comparison, потому что многоязычные слои можно пересобирать из одной английской базы.

## 12. Индексация и retrieval

### 12.1 За что отвечает index builder

`src/search/builder.py` отвечает за:

- сканирование сохраненных segment sources,
- нормализацию searchable text,
- подготовку sparse и dense retrieval inputs,
- обновление manifests и config fingerprints,
- запись index metadata и embeddings,
- поддержку no-change detection и логики частичного обновления.

### 12.2 За что отвечает query engine

`src/search/engine.py` отвечает за:

- загрузку активного индекса и config,
- выполнение sparse retrieval,
- выполнение dense retrieval,
- объединение результатов через fusion,
- optional reranking,
- дедупликацию overlapping hits,
- возврат ранжированных hits вместе со структурированными полями.

### 12.3 Retrieval stack

| Модуль | Ответственность |
| --- | --- |
| `src/search/corpus.py` | построение searchable и dense text из segment artifacts |
| `src/search/embed.py` | выбор embedding backend и embedding cache behavior |
| `src/search/rank.py` | fusion, reranking, dedupe |
| `src/search/store.py` | persistence helpers retrieval-слоя |
| `src/search/types.py` | внутренние типы retrieval |

### 12.4 Какие фильтры поддерживает поиск

| Семейство фильтров | Поля |
| --- | --- |
| область поиска | `video_id`, `language`, `variant` |
| ранжирование | `top_k`, `dedupe` |
| время | `start_sec`, `end_sec`, `min_duration_sec`, `max_duration_sec` |
| семантика | `event_type`, `risk_level`, `tags`, `objects`, `people_count_bucket`, `motion_type` |
| аномалии | `anomaly_only` |

### 12.5 Язык запроса и fallback-поведение

`backend/retrieval_runtime.py` держит request-side retrieval logic. В его зоне ответственности:

- нормализация request language,
- variant-aware разрешение index directory,
- query translation, если она включена конфигурацией,
- fallback search по альтернативным языкам,
- загрузка LLM и guard services для grounded generation,
- преобразование hits в API schema objects,
- сборка citations и supporting hits.

## 13. Grounded reports, QA и RAG

Grounded-response слой реализован в `backend/http/grounded.py` поверх `backend/retrieval_runtime.py`.

### 13.1 Режимы работы

| Режим | Route | Назначение |
| --- | --- | --- |
| report | `/v1/reports` | построить короткий доказательный report |
| QA | `/v1/qa` | ответить на прямой вопрос по найденным evidence |
| RAG assistant | `/v1/rag` | вернуть assistant-style answer и context |

### 13.2 Общее grounded-поведение

Все grounded-режимы используют одну и ту же базовую схему:

- query guarding до retrieval,
- получение supporting evidence,
- сборка context block из найденных hits,
- optional LLM generation, если она доступна,
- deterministic fallback text generation, если полноформатная генерация недоступна,
- перевод ответа в нужный язык,
- output guarding,
- возврат citations и supporting hits вместе с текстом.

### 13.3 Главный принцип grounding

Смысл этого слоя не просто в генерации текста. Его задача — создавать текст, который остается привязанным к конкретным найденным видеодоказательствам. Поэтому API-ответы всегда несут не только текст, но и citations/supporting hits.

## 14. Модель артефактов и хранения

Все runtime-состояние хранится под `data/`.

### 14.1 Структура данных по одному видео

```text
data/videos/<video_id>/
  manifest.json
  raw/
    <исходный или нормализованный видеофайл>
  cache/
    ...
  outputs/
    clip_observations.json
    manifest.json
    metrics.json
    run_manifest.json
    experimental_manifest.json
    segments/
      en.jsonl.zst или en.jsonl.gz
      ru.jsonl.zst или ru.jsonl.gz
      kz.jsonl.zst или kz.jsonl.gz
    summaries/
      en.json
      ru.json
      kz.json
    variants/
      <variant_id>/
        clip_observations.json
        manifest.json
        metrics.json
        run_manifest.json
        segments/
        summaries/
```

### 14.2 Важные файлы по видео

| Файл | Смысл |
| --- | --- |
| `manifest.json` в корне видео | базовые метаданные library entry |
| `raw/*` | загруженный или нормализованный source video |
| `outputs/clip_observations.json` | raw VLM output до финального structuring |
| `outputs/segments/<lang>.*` | structured segment rows для конкретного языка |
| `outputs/summaries/<lang>.json` | summary payload для языка |
| `outputs/metrics.json` | timings, counters и execution metadata |
| `outputs/manifest.json` | состояние готовности outputs |
| `outputs/run_manifest.json` | profile, variant, config fingerprint и ссылки на результаты |
| `outputs/experimental_manifest.json` | tracking parent/child experiment jobs |

### 14.3 Глобальное runtime-состояние

| Путь | Назначение |
| --- | --- |
| `data/jobs/` | persisted job records |
| `data/queue/` | queue files, задающие порядок исполнения |
| `data/locks/` | worker lease и lock state |
| `data/indexes/` | retrieval indexes |
| `data/cache/` | shared cache вне per-video trees |
| `data/thumbs/` | thumbnails для UI |
| `data/research/` | experiment outputs и metrics bundles |

## 15. Метрики, исследования и experiment tooling

В репозитории уже есть набор выделенных скриптов для эксплуатации и исследования системы за пределами UI.

### 15.1 Скрипты

| Скрипт | Назначение |
| --- | --- |
| `scripts/runtime/smoke_services.py` | быстрый smoke check доступности API и UI |
| `scripts/runtime/inspect_runtime.py` | печать разрешённого profile, variant и ключевых runtime-настроек |
| `scripts/runtime/profile_vlm_path.py` | профилирование preprocess, clip build и packed VLM inference на одном видео |
| `scripts/collect_metrics.py` | экспорт нормализованных metrics bundles из сохраненных outputs |
| `scripts/run_experiment_matrix.py` | оркестрация повторных прогонов по profiles, variants и выбранным видео |
| `scripts/eval_relevance.py` | offline evaluation качества retrieval |
| `scripts/build_comparison_report.py` | сборка компактных comparison summaries |
| `scripts/check_ui_sanity.py` | ручная структурная проверка UI-слоя |
| `scripts/check_backend_sanity.py` | ручная структурная проверка backend imports и duplicate defs |

### 15.2 Типовые исследовательские outputs

Метрики и experiment flows могут производить:

| Output | Значение |
| --- | --- |
| `metrics_runs.csv` | overview по отдельным runs |
| `metrics_stage_runs.csv` | построчные метрики по этапам для каждого run |
| `metrics_stage_aggregate.csv` | агрегированные stage metrics |
| `metrics_snapshot.json` | нормализованный snapshot метрик |
| `metrics_by_video.json` | группировка метрик по видео |
| `metrics_by_profile_variant.json` | группировка метрик по конфигурациям |
| `system_metrics.json` | общая сводка по системе |
| comparison CSV и JSON | компактные cross-run comparison outputs |

## 16. Запуск и deployment model

### 16.1 Ручной локальный старт

```powershell
python -m uvicorn backend.api:app --host 127.0.0.1 --port 8000
python -m backend.worker
python -m streamlit run app/main.py --server.port 8501
```

### 16.2 One-command local start

```powershell
run_all.bat
```

`run_all.bat` — это convenience launcher для локального старта. Sanity-скрипты остаются в проекте как ручные инструменты, но не являются обязательной частью запуска.

### 16.3 Docker-старт

```powershell
docker compose up --build
```

Поддерживаются и дополнительные compose profiles:

```powershell
docker compose --profile with_vllm up --build
docker compose --profile with_ct2 up --build
```

## 17. Архитектурные характеристики

У системы есть несколько определяющих характеристик, которые важно понимать при чтении и развитии кода:

| Характеристика | Практическое следствие |
| --- | --- |
| local-first design | runtime state можно напрямую инспектировать на диске |
| filesystem queue | orchestration jobs остается простой и demo-friendly |
| English-first canonical artifacts | перевод — производный слой, а не первичный источник правды |
| grounded retrieval layer | generated responses должны оставаться привязанными к evidence |
| единая config model для UI, backend и worker | один profile/variant управляет всем локальным стеком |
| page-oriented UI layout | frontend сгруппирован по пользовательским workflow, а не по типам виджетов |
| domain-oriented `src/` layout | video, LLM, translation, search и runtime config разведены по доменам |

## 18. Практическое резюме

SmartCampus V2T — это полноценный локальный стек видеоаналитики. В прикладном смысле он:

- ведет локальную библиотеку видео,
- обрабатывает видео в структурированные семантические артефакты,
- строит summary и многоязычные представления,
- создает retrieval-индексы,
- открывает analytics, search, reports, QA и assistant workflows,
- поддерживает export метрик и experiment comparison,
- хранит runtime-состояние в локальном и проверяемом виде.

Если новому разработчику нужен короткий и точный mental model, то он такой:

1. `app/` — операторская консоль.
2. `backend/` — HTTP и orchestration слой jobs.
3. `src/` — реальная логика video, LLM, translation и search.
4. `data/` — runtime source of truth для артефактов и очереди.
5. Обработка начинается в queue, превращается в structured artifacts, а затем — в searchable и explainable evidence.
