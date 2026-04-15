# SmartCampus V2T Project Guide

Russian version: [docs/project_guide.ru.md](project_guide.ru.md)

## 1. Document Scope

This guide describes the SmartCampus V2T repository as it exists in code today. It is meant to be the main technical reference for:

- the overall purpose of the system,
- the runtime surfaces that make up the application,
- the functional responsibilities of the frontend, backend, worker, and core pipeline,
- the configuration model,
- the queue and job system,
- the storage and artifact layout,
- retrieval, grounded generation, and multilingual behavior,
- the local operations and research tooling that ship with the repository.

This repository is not just a model wrapper, not just a Streamlit demo, and not just a set of scripts. It is a local end-to-end video analytics system that ingests stored video, turns it into structured semantic artifacts, derives multilingual views, builds retrieval indexes, and exposes grounded search and reporting workflows on top of those artifacts.

The current system is intentionally local-first. It favors inspectable filesystem state, explicit runtime configuration, and stable local entrypoints over distributed infrastructure.

## 2. System Positioning

SmartCampus V2T is composed of three main runtime surfaces:

| Runtime surface | Entrypoint | Main role | Typical user or caller |
| --- | --- | --- | --- |
| Streamlit operator UI | `app/main.py` | interactive operator console | local user, demo operator, reviewer |
| FastAPI backend | `backend/api.py` | stable HTTP surface over library, jobs, queue, retrieval, and reports | Streamlit UI, smoke scripts, local automation |
| Background worker | `backend/worker.py` | asynchronous execution of process, translate, and index jobs | queue-driven internal runtime |

These surfaces share one configuration model and one local artifact layout. The UI does not execute the heavy pipeline directly. Instead, it talks to the backend, the backend persists and exposes state, and the worker performs long-running processing in the background.

## 3. Functional Coverage

The repository currently implements the following major capabilities:

| Capability | What it does | Main outputs | Primary code areas |
| --- | --- | --- | --- |
| Video library management | stores uploaded videos as managed local library entries | raw video files, per-video manifest | `app/view/storage.py`, `backend/api.py`, `src/utils/video_store.py` |
| Video normalization | prepares videos for stable analysis | prepared frames, cached metadata | `src/video/io.py` |
| Clip observation | runs the vision-language stage over temporal clips | clip observations in English | `src/video/describe.py`, `src/core/vlm_backend.py` |
| Semantic structuring | converts raw clip observations into structured segments | segment rows with semantic fields | `src/llm/analyze.py`, `src/guard/*` |
| Summary generation | produces one video-level summary | summary payload per language | `src/llm/summary.py` |
| Translation | creates derived Russian and Kazakh views | translated segments and summaries | `backend/jobs/translate_runtime.py`, `src/translation/service.py` |
| Index building | builds retrieval-ready sparse and dense state | index files, manifests, status | `src/search/builder.py`, `backend/jobs/index_runtime.py` |
| Search | retrieves relevant events using text plus metadata filters | ranked search hits | `src/search/engine.py`, `backend/retrieval_runtime.py` |
| Grounded reports | produces evidence-backed short reports | report text, citations, supporting hits | `backend/http/grounded.py` |
| Grounded QA | answers questions against indexed evidence | answer text, citations, supporting hits | `backend/http/grounded.py` |
| Grounded RAG assistant | supports assistant-style answers tied to retrieved evidence | answer text, context, citations, supporting hits | `backend/http/grounded.py`, `app/view/search.py` |
| Metrics and experiment tooling | exports metrics, runs experiment matrices, evaluates relevance | CSV, JSON, zip bundles, comparison reports | `scripts/*` |

One architectural rule is especially important:

| Rule | Meaning |
| --- | --- |
| English-first canonical storage | the canonical structured artifacts are produced in English first |
| Derived multilingual views | `ru` and `kz` outputs are derived from the English source artifacts |
| Retrieval and grounding stay evidence-based | reports, QA, and assistant answers are expected to remain tied to retrieved video evidence rather than free-form generation |

## 4. End-to-End Lifecycle

The normal end-to-end lifecycle looks like this:

1. A video is uploaded into the local library.
2. The backend creates a `process` job and puts it into the filesystem queue.
3. The worker leases the job and resolves the effective profile and variant.
4. The video is normalized and prepared into analysis-ready frames.
5. Clip windows and keyframes are built.
6. The VLM generates raw English clip observations.
7. Guard and schema helpers repair and sanitize generated payloads.
8. The semantic layer turns the observations into structured segments.
9. A video-level summary is generated.
10. Artifacts, metrics, and manifests are persisted under `data/videos/<video_id>/`.
11. Optional translation jobs derive `ru` and `kz` outputs.
12. Optional index updates refresh the retrieval layer.
13. The UI and API expose analytics, search, reports, QA, and RAG over the stored results.

The same flow can be summarized as:

```text
video upload
  -> job creation
  -> queue lease by worker
  -> video preparation
  -> clip building
  -> VLM observation
  -> guard/schema repair
  -> semantic segment analysis
  -> summary generation
  -> artifact persistence
  -> translation fan-out
  -> index build/update
  -> search / reports / QA / RAG
```

The lifecycle is deliberately asynchronous. Upload and job creation are quick UI/backend operations. Heavy processing happens later in the worker.

## 5. Repository Structure

### 5.1 Top-level layout

| Directory | Responsibility | Notes |
| --- | --- | --- |
| `app/` | Streamlit frontend | page modules, shared UI helpers, client logic |
| `backend/` | FastAPI backend and worker orchestration | API entrypoint, retrieval runtime, job runtime helpers |
| `src/` | core domain logic | runtime config, video pipeline, LLM logic, translation, search |
| `configs/` | runtime profiles and variants | profile YAML, experimental overrides |
| `scripts/` | local operations and research tooling | smoke checks, metrics, evaluation, experiment matrix |
| `data/` | runtime state and artifacts | videos, jobs, queue, indexes, research outputs |
| `docs/` | repository documentation | this guide and related docs |
| `.agent/` | private project memory | working notes, decisions, state snapshots |

### 5.2 Frontend layout

| Path | Responsibility |
| --- | --- |
| `app/main.py` | one Streamlit entrypoint and top-level routing |
| `app/api_client.py` | UI-facing client for backend HTTP routes |
| `app/state.py` | shared UI session defaults |
| `app/view/overview.py` | product and system overview page |
| `app/view/storage.py` | library, upload, queue, and launch workflows |
| `app/view/analytics.py` | playback, summary, metrics, and timeline surface |
| `app/view/search.py` | search filters, results, and grounded assistant |
| `app/view/reports.py` | grounded reporting surface |
| `app/view/shared.py` | shared UI helpers, localized copy helpers, common components |
| `app/lib/i18n.py` | UI text loading and translator helpers |
| `app/lib/media.py` | CSS loading and local media helpers |
| `app/assets/` | CSS, logo, and UI text assets |

### 5.3 Backend layout

| Path | Responsibility |
| --- | --- |
| `backend/api.py` | stable HTTP route registration |
| `backend/worker.py` | main worker loop and dispatch |
| `backend/deps.py` | shared config/path/dependency helpers for backend entrypoints |
| `backend/retrieval_runtime.py` | retrieval, fallback, grounding, citation, and metrics helpers |
| `backend/schemas.py` | request and response contracts |
| `backend/http/common.py` | shared API helpers, upload normalization, grounded-response helpers |
| `backend/http/grounded.py` | report, QA, and RAG response construction |
| `backend/jobs/*` | queue, persistence, job execution, worker support, metrics, experimental fan-out |

### 5.4 Core domain layout

| Path | Responsibility |
| --- | --- |
| `src/core/*` | typed config, config loading, runtime flags, artifact types, VLM backend integration |
| `src/video/*` | video preparation, clip generation, observation pipeline prompts and logic |
| `src/llm/*` | semantic structuring and summary generation |
| `src/guard/*` | safety gates and schema cleanup |
| `src/translation/*` | translation routing, caching, and post-edit flow |
| `src/search/*` | index builder, query engine, ranking, embedding, corpus helpers |
| `src/utils/video_store.py` | canonical per-video storage layout and artifact persistence helpers |

## 6. Frontend / Operator Console

### 6.1 UI bootstrap path

`app/main.py` is the only Streamlit entrypoint. At startup it:

1. loads the effective config through `src/core/runtime.py`,
2. applies CSS from `app/assets/styles.css`,
3. loads localized UI text from `app/assets/ui_text.json`,
4. binds default session state,
5. builds the backend base URL from typed config,
6. pings backend health,
7. renders the global header,
8. dispatches to the selected page module.

### 6.2 Top-level pages

| Page | Module | User-facing purpose | Typical backend dependencies |
| --- | --- | --- | --- |
| Overview | `app/view/overview.py` | explains the system, its features, and processing stages | health only |
| Storage | `app/view/storage.py` | upload videos, browse library, create jobs, manage queue | videos, jobs, queue |
| Video Analytics | `app/view/analytics.py` | inspect one processed video, review summary, metrics, and timeline | outputs, metrics |
| Search | `app/view/search.py` | run hybrid search, inspect hits, use the assistant | search, index rebuild, rag |
| Reports | `app/view/reports.py` | generate concise grounded reports for one video or query | reports, outputs |

### 6.3 Page-by-page functional detail

| Page | Main controls | Main outputs shown to user |
| --- | --- | --- |
| Overview | localized feature cards, stage switcher | product description, feature summary, stage walkthrough |
| Storage | library filters, upload widget, profile selector, variant selector, queue controls | video cards, upload state, queue snapshot, processing actions |
| Video Analytics | selected video, selected language, selected variant, timeline seek buttons | video playback, summary, readiness status, metrics rows, event timeline |
| Search | query box, language selector, variant selector, event and risk filters, anomaly-only toggle, dedupe toggle, assistant input | search hits, metadata chips, evidence-linked assistant replies |
| Reports | selected video, language, variant, report query | report text, citations, supporting evidence rows, metrics summary rows |

### 6.4 UI-to-backend interaction model

| UI action | Backend route |
| --- | --- |
| load library | `GET /v1/videos` |
| upload video | `POST /v1/videos/upload` |
| delete video | `DELETE /v1/videos/{video_id}` |
| fetch outputs for analytics/reports | `GET /v1/videos/{video_id}/outputs` |
| fetch metrics summary | `GET /v1/videos/{video_id}/metrics-summary` |
| create process or translate job | `POST /v1/jobs` |
| read job status | `GET /v1/jobs/{job_id}` |
| cancel job | `POST /v1/jobs/{job_id}/cancel` |
| inspect queue | `GET /v1/queue` |
| pause or resume queue | `POST /v1/queue/pause`, `POST /v1/queue/resume` |
| reorder or remove queued jobs | `POST /v1/queue/move`, `DELETE /v1/queue/{job_id}` |
| rebuild retrieval index | `POST /v1/index/rebuild` |
| search | `POST /v1/search` |
| grounded reports | `POST /v1/reports` |
| grounded QA or assistant | `POST /v1/qa`, `POST /v1/rag` |

## 7. Backend HTTP Surface

### 7.1 Health routes

| Method | Route | Purpose |
| --- | --- | --- |
| `GET` | `/healthz` | non-versioned liveness |
| `GET` | `/v1/health` | versioned health |
| `GET` | `/v1/healthz` | versioned liveness |

### 7.2 Video routes

| Method | Route | Purpose | Main response |
| --- | --- | --- | --- |
| `GET` | `/v1/videos` | list managed videos | `List[VideoItem]` |
| `POST` | `/v1/videos/upload` | upload and normalize one source file | `VideoItem` |
| `DELETE` | `/v1/videos/{video_id}` | delete the local video tree | `{ ok: true }` |
| `GET` | `/v1/videos/{video_id}/outputs` | read one language/variant output bundle | `VideoOutputs` |
| `GET` | `/v1/videos/{video_id}/batch-manifest` | read experimental parent/child manifest | raw JSON payload |
| `GET` | `/v1/videos/{video_id}/metrics-summary` | read normalized processing summary | `MetricsSummaryResponse` |

### 7.3 Job and queue routes

| Method | Route | Purpose | Main response |
| --- | --- | --- | --- |
| `POST` | `/v1/jobs` | create and enqueue a job | `JobCreateResponse` |
| `GET` | `/v1/jobs/{job_id}` | read job state | `JobStatus` |
| `POST` | `/v1/jobs/{job_id}/cancel` | request cancellation | `JobCancelResponse` |
| `GET` | `/v1/queue` | return queue snapshot | `QueueListResponse` |
| `POST` | `/v1/queue/pause` | pause execution | `QueueStatus` payload |
| `POST` | `/v1/queue/resume` | resume execution | `QueueStatus` payload |
| `POST` | `/v1/queue/move` | reorder queued job | `QueueMoveResponse` |
| `DELETE` | `/v1/queue/{job_id}` | remove a queued job and mark it canceled | raw JSON payload |

### 7.4 Retrieval and grounded-generation routes

| Method | Route | Purpose | Main response |
| --- | --- | --- | --- |
| `GET` | `/v1/index/status` | read current index build state | `IndexStatus` |
| `POST` | `/v1/index/rebuild` | rebuild or refresh index | `IndexRebuildResponse` |
| `POST` | `/v1/search` | hybrid search over processed artifacts | `SearchResponse` |
| `POST` | `/v1/reports` | grounded report generation | `ReportResponse` |
| `POST` | `/v1/qa` | grounded question answering | `QaResponse` |
| `POST` | `/v1/rag` | grounded assistant response | `RagResponse` |

### 7.5 Key request contracts

#### Search request

| Field | Meaning |
| --- | --- |
| `query` | main natural-language search input |
| `top_k` | result limit |
| `video_id` | restrict search to one video |
| `language` | retrieval language |
| `variant` | experimental variant namespace |
| `dedupe` | deduplicate overlapping hits |
| `start_sec`, `end_sec` | filter by time range |
| `min_duration_sec`, `max_duration_sec` | filter by segment duration |
| `event_type` | semantic event filter |
| `risk_level` | risk filter |
| `tags` | tag filter |
| `objects` | object filter |
| `people_count_bucket` | coarse people-count filter |
| `motion_type` | motion filter |
| `anomaly_only` | require anomaly-marked hits |

#### Search hit payload

| Field | Meaning |
| --- | --- |
| `video_id` | owning video |
| `language` | language of the indexed artifact |
| `start_sec`, `end_sec` | temporal boundaries |
| `description` | human-readable segment description |
| `score`, `sparse_score`, `dense_score` | ranking values |
| `segment_id` | segment identifier when present |
| `event_type`, `risk_level` | structured semantics |
| `tags`, `objects` | semantic lists |
| `people_count_bucket`, `motion_type` | additional structured metadata |
| `anomaly_flag` | anomaly marker |
| `variant` | variant namespace if not base |

#### Grounded generation requests

| Request model | Required input | Optional scoping |
| --- | --- | --- |
| `ReportRequest` | `query` or `video_id` | `language`, `variant`, `top_k` |
| `QaRequest` | `question` | `video_id`, `language`, `variant`, `top_k` |
| `RagRequest` | `query` | `video_id`, `language`, `variant`, `top_k` |

#### Output bundle response

`VideoOutputs` provides one language/variant-scoped output bundle. It can contain:

| Field | Meaning |
| --- | --- |
| `manifest` | readiness/status payload |
| `run_manifest` | config and execution metadata |
| `batch_manifest` | experiment fan-out metadata |
| `annotations` | structured segments returned to the UI |
| `metrics` | raw metrics payload |
| `global_summary` | final summary text |

## 8. Queue, Jobs, and Worker Model

### 8.1 Why the system uses a filesystem queue

The project does not depend on an external broker such as Redis, RabbitMQ, or Celery. Queue state is persisted in local files. This keeps the system easy to run on a single machine, easy to inspect during development, and easy to demonstrate without extra infrastructure.

### 8.2 Main backend job modules

| Module | Role |
| --- | --- |
| `backend/jobs/store.py` | low-level job record and queue file primitives |
| `backend/jobs/control.py` | locking, leasing, state transitions, cancellation checks |
| `backend/jobs/queue_runtime.py` | queue snapshot, reorder, running-job discovery |
| `backend/jobs/worker_runtime.py` | resolve effective config and service bundle per job |
| `backend/jobs/process_runtime.py` | full process job execution |
| `backend/jobs/translate_runtime.py` | translation job execution |
| `backend/jobs/runtime_common.py` | shared finalize/index helpers |
| `backend/jobs/index_runtime.py` | index status persistence |
| `backend/jobs/experimental.py` | experimental fan-out helpers |

### 8.3 Supported job types

| Job type | Purpose | Typical creator |
| --- | --- | --- |
| `process` | run the full pipeline for one video | UI or backend API |
| `translate` | derive a target-language view from English artifacts | process runtime or API |
| `index` | rebuild retrieval indexes without reprocessing video | API or local maintenance |

### 8.4 Job lifecycle states

| State | Meaning |
| --- | --- |
| `queued` | job is waiting in the queue |
| `running` | worker has leased the job and is executing it |
| `done` | job completed successfully |
| `failed` | job ended with an unrecovered error |
| `cancel_requested` | cancellation was requested while work may still be in progress |
| `canceled` | job was canceled and finalized |

### 8.5 Worker roles

`SMARTCAMPUS_WORKER_ROLE` can limit what a worker is allowed to execute.

| Worker role | Allowed work |
| --- | --- |
| `all` | all job types |
| `gpu` | `process` jobs |
| `mt` | `translate` jobs |
| `cpu` | `index` jobs |

This allows local separation of heavy GPU-bound work, translation work, and index work when needed.

### 8.6 Experimental fan-out behavior

The worker can expand one parent experimental job into multiple child jobs when the active config indicates experiment mode. In practice:

1. a parent job is created under an experiment-capable profile,
2. the worker detects that the job should expand,
3. child jobs are created for each configured variant,
4. an experimental batch manifest is written,
5. the parent job is marked done as an expansion step rather than as a normal process run.

This is how comparative profile or variant runs are orchestrated without changing the public API surface.

## 9. Configuration Model

### 9.1 Effective config selection

The effective runtime configuration is built in `src/core/runtime.py` and typed in `src/core/config.py`.

Config resolution supports:

- one active profile,
- zero or one active variant,
- YAML `extends` inheritance,
- environment-level selection,
- typed post-processing into a strongly typed config object.

### 9.2 Main selectors

| Variable | Role |
| --- | --- |
| `SMARTCAMPUS_PROFILE` | select the active profile |
| `SMARTCAMPUS_VARIANT` | select the active variant |
| `SMARTCAMPUS_WORKER_ROLE` | restrict worker execution role |

### 9.3 Typed config blocks

| Config block | What it controls |
| --- | --- |
| `paths` | project root, `data/`, `models/`, `indexes/`, assets, config folders |
| `ui` | UI languages, asset paths, CSS path, UI text path |
| `backend` | backend host, scheme, and port |
| `search` | embedding, reranking, fusion, dedupe, fallback languages, dense input mode |
| `video` | decode, resize, frame gating, face blur, anonymization, save policy |
| `clips` | clip windowing, stride, keyframe policy |
| `model` | VLM model path, device, dtype, batch policy, inference limits |
| `llm` | semantic/summarization backend and generation settings |
| `guard` | query and output safety gating |
| `runtime` | seed, threading, TF32, matmul precision, compile flags, metrics sampling |
| `translation` | MT backend, routes, batch size, cache, query-time translation, post-edit policy |
| `jobs` | job record directory |
| `queue` | queue directory |
| `locks` | worker lock directory |
| `worker` | poll interval, lease interval, concurrency policy |
| `index` | index auto-update policy |
| `webhook` | outbound notification behavior |
| `experiment` | compare mode and known variant ids |

### 9.4 Active profiles in the repository

| Profile | Role |
| --- | --- |
| `configs/profiles/main.yaml` | default working runtime |
| `configs/profiles/experimental.yaml` | experiment-oriented profile with overwrite and repeated metrics enabled |

### 9.5 Variants

Variant files under `configs/variants/` provide explicit overrides. They are used for controlled comparisons and can receive their own output namespaces and index state.

## 10. Core Processing Pipeline

The `process` job is the main pipeline path. It is orchestrated in `backend/jobs/process_runtime.py` but uses services and logic from `src/`.

### 10.1 Stage map

| Stage | Main code | Input | Output |
| --- | --- | --- | --- |
| context resolution | `backend/jobs/worker_runtime.py` | job record, active profile/variant | effective config and service bundle |
| video preparation | `src/video/io.py` | raw video file | prepared frames and video metadata |
| clip building | `src/video/clips.py` | prepared video frames | clip windows and keyframes |
| clip observation | `src/video/describe.py`, `src/core/vlm_backend.py` | clip/keyframe payloads | raw English observations |
| schema repair and guarding | `src/guard/service.py`, `src/guard/schemas.py` | raw generated payloads | normalized, safe payloads |
| segment structuring | `src/llm/analyze.py` | raw observations | structured segment rows |
| summary generation | `src/llm/summary.py` | structured segments | video summary |
| persistence | `src/utils/video_store.py` | all generated artifacts | saved manifests, rows, summary, metrics |
| optional indexing | `src/search/builder.py` | stored segments | updated index state |
| optional translation | `backend/jobs/translate_runtime.py` | English outputs | `ru`/`kz` derived outputs |

### 10.2 Video preparation behavior

`src/video/io.py` is one of the most important domain modules. It is responsible for:

- resolving the source video path,
- normalizing fps and decode behavior,
- resizing and letterboxing frames,
- filtering dark or low-information frames,
- measuring blur and motion,
- applying optional face anonymization,
- saving prepared frames and optionally smaller helper frames,
- writing and reusing cache metadata.

This stage is what lets later VLM and semantic steps operate on a more stable and consistent representation of the source video.

### 10.3 Observation and semantic stages

The system separates two conceptual layers:

| Layer | Purpose |
| --- | --- |
| VLM observation | describe what is visually present in clip windows |
| semantic structuring | turn the observation layer into segments with event types, tags, risk, people buckets, motion, and anomaly cues |

This separation is important because the retrieval and reporting layers depend on structured segment data, not only on raw captions.

### 10.4 Summary stage

The summary stage creates a video-level summary from the structured segments. The summary is stored independently from the segment rows so it can be used in analytics and report views without re-running the entire segment list.

## 11. Translation Layer

Translation is intentionally downstream of the English pipeline. The system first builds canonical English artifacts, then derives localized views.

### 11.1 Translation flow

| Step | Description |
| --- | --- |
| load source artifacts | read English segments and summaries |
| route selection | resolve the MT route for the requested language pair |
| segment translation | translate segment descriptions and related fields |
| selective post-edit | optionally improve specific outputs such as summaries, reports, QA, or selected segments |
| output persistence | write translated segment and summary artifacts |
| optional target-language indexing | rebuild or update retrieval state for the target language |

### 11.2 Translation runtime settings

The active main profile currently uses:

| Setting area | Current intent |
| --- | --- |
| backend | `ctranslate2` |
| source language | English |
| target languages | Russian and Kazakh |
| query-time translation | enabled |
| offline translation | enabled |
| post-edit targets | summaries, reports, QA, selected segments |

### 11.3 Why English remains canonical

This design avoids having multiple primary sources of truth and keeps the indexing path coherent. It also makes experiment comparison easier because translated views can be regenerated from the same English base artifacts.

## 12. Indexing and Retrieval

### 12.1 Index builder responsibilities

`src/search/builder.py` is responsible for:

- scanning the stored segment sources,
- normalizing searchable text,
- producing sparse and dense retrieval inputs,
- updating manifests and config fingerprints,
- writing index metadata and embeddings,
- supporting no-change detection and incremental-style refresh logic.

### 12.2 Query engine responsibilities

`src/search/engine.py` is responsible for:

- loading the active index and config,
- executing sparse retrieval,
- executing dense retrieval,
- combining results through fusion,
- optionally reranking,
- deduplicating overlapping hits,
- returning ranked hits with structured metadata.

### 12.3 Retrieval stack

| Module | Responsibility |
| --- | --- |
| `src/search/corpus.py` | derive searchable and dense text from segment artifacts |
| `src/search/embed.py` | embedding backend selection and embedding cache behavior |
| `src/search/rank.py` | fusion, reranking, dedupe |
| `src/search/store.py` | retrieval persistence helpers |
| `src/search/types.py` | internal retrieval types |

### 12.4 Search filters exposed by the system

| Filter family | Fields |
| --- | --- |
| scope | `video_id`, `language`, `variant` |
| ranking | `top_k`, `dedupe` |
| temporal | `start_sec`, `end_sec`, `min_duration_sec`, `max_duration_sec` |
| semantics | `event_type`, `risk_level`, `tags`, `objects`, `people_count_bucket`, `motion_type` |
| anomaly behavior | `anomaly_only` |

### 12.5 Language and fallback behavior

`backend/retrieval_runtime.py` owns the request-side retrieval logic. It handles:

- request language normalization,
- variant-aware index resolution,
- query translation when configured,
- fallback search in alternate languages,
- LLM and guard service loading for grounded generation,
- hit conversion into API schema objects,
- citation and supporting-hit construction.

## 13. Grounded Reports, QA, and RAG

Grounded response generation is implemented by `backend/http/grounded.py` on top of `backend/retrieval_runtime.py`.

### 13.1 Modes

| Mode | Entry route | Purpose |
| --- | --- | --- |
| report | `/v1/reports` | produce a short evidence-backed report |
| QA | `/v1/qa` | answer a direct question over retrieved evidence |
| RAG assistant | `/v1/rag` | return an assistant-style grounded answer plus context |

### 13.2 Shared grounded behavior

All grounded modes share the following core behavior:

- query guarding before retrieval,
- retrieval of supporting evidence,
- context assembly from retrieved hits,
- optional LLM generation when available,
- deterministic fallback text generation when full generation is unavailable,
- output translation into the requested language,
- output guarding,
- citation and supporting-hit payload assembly.

### 13.3 Grounding principle

The point of this layer is not only to generate text. The point is to generate text that stays tied to concrete retrieved video evidence. That is why the API responses return citations and supporting hits alongside the final text.

## 14. Artifact and Storage Model

All runtime state is stored under `data/`.

### 14.1 Per-video layout

```text
data/videos/<video_id>/
  manifest.json
  raw/
    <original-or-normalized-video-file>
  cache/
    ...
  outputs/
    clip_observations.json
    manifest.json
    metrics.json
    run_manifest.json
    experimental_manifest.json
    segments/
      en.jsonl.zst or en.jsonl.gz
      ru.jsonl.zst or ru.jsonl.gz
      kz.jsonl.zst or kz.jsonl.gz
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

### 14.2 Important per-video files

| File | Meaning |
| --- | --- |
| `manifest.json` at video root | basic library entry metadata |
| `raw/*` | uploaded or normalized source video |
| `outputs/clip_observations.json` | raw VLM output before final structuring |
| `outputs/segments/<lang>.*` | structured segment rows for one language |
| `outputs/summaries/<lang>.json` | summary payload for one language |
| `outputs/metrics.json` | timings, counters, and execution metadata |
| `outputs/manifest.json` | output readiness and status information |
| `outputs/run_manifest.json` | run metadata such as profile, variant, config fingerprint, and references |
| `outputs/experimental_manifest.json` | experiment parent/child tracking |

### 14.3 Global runtime state

| Path | Purpose |
| --- | --- |
| `data/jobs/` | persisted job records |
| `data/queue/` | queue files that define execution order |
| `data/locks/` | worker lease and lock state |
| `data/indexes/` | retrieval indexes |
| `data/cache/` | shared caches outside per-video trees |
| `data/thumbs/` | UI thumbnails |
| `data/research/` | experiment outputs and metrics bundles |

## 15. Metrics, Research, and Experiment Tooling

The repository contains dedicated scripts for operating and evaluating the system beyond the UI.

### 15.1 Scripts

| Script | Purpose |
| --- | --- |
| `scripts/runtime/smoke_services.py` | quick smoke check for API and UI availability |
| `scripts/runtime/inspect_runtime.py` | print the resolved runtime profile, variant, and key execution knobs |
| `scripts/runtime/profile_vlm_path.py` | profile preprocess, clip build, and packed VLM inference on one video |
| `scripts/collect_metrics.py` | export normalized metrics bundles from stored outputs |
| `scripts/run_experiment_matrix.py` | orchestrate repeated runs across profiles, variants, and selected videos |
| `scripts/eval_relevance.py` | evaluate retrieval quality offline from experiment outputs |
| `scripts/build_comparison_report.py` | produce compact comparison summaries for experiment runs |
| `scripts/check_ui_sanity.py` | manual structural validation of the UI layer |
| `scripts/check_backend_sanity.py` | manual structural validation of backend imports and duplicate definitions |

### 15.2 Typical research outputs

Metrics and experiment flows can produce:

| Output | Meaning |
| --- | --- |
| `metrics_runs.csv` | per-run metrics overview |
| `metrics_stage_runs.csv` | per-stage metrics rows for each run |
| `metrics_stage_aggregate.csv` | aggregated stage metrics |
| `metrics_snapshot.json` | normalized metrics snapshot |
| `metrics_by_video.json` | metrics grouped by video |
| `metrics_by_profile_variant.json` | metrics grouped by config combination |
| `system_metrics.json` | overall system summary |
| comparison CSV and JSON | compact cross-run comparison outputs |

## 16. Startup and Deployment Model

### 16.1 Manual local startup

```powershell
python -m uvicorn backend.api:app --host 127.0.0.1 --port 8000
python -m backend.worker
python -m streamlit run app/main.py --server.port 8501
```

### 16.2 One-command local startup

```powershell
run_all.bat
```

`run_all.bat` is the convenience local launcher. Manual sanity scripts remain available, but they are not mandatory for startup.

### 16.3 Docker startup

```powershell
docker compose up --build
```

Optional compose profiles currently include:

```powershell
docker compose --profile with_vllm up --build
docker compose --profile with_ct2 up --build
```

The recommended text-LLM-through-`vLLM` stack is:

```powershell
docker compose -f docker-compose.yml -f docker-compose.vllm.yml up --build -d
```

`.env` is optional and should stay untracked. Runtime behavior should stay in `configs/profiles/*.yaml` and `configs/variants/*.yaml`; Docker-level overrides like `VLLM_*` can be set directly in the shell when needed.

## 17. Architectural Characteristics

The current system has several defining characteristics that matter when reading or extending the codebase:

| Characteristic | Practical consequence |
| --- | --- |
| local-first design | runtime state is inspectable directly on disk |
| filesystem queue | job orchestration stays simple and demo-friendly |
| English-first canonical artifacts | translation is derived rather than primary |
| grounded retrieval layer | generated responses are designed to stay tied to evidence |
| shared config for UI, backend, and worker | one profile/variant model drives the whole local stack |
| page-oriented UI layout | frontend logic is grouped by operator workflow rather than by widget type |
| domain-oriented `src/` layout | video, LLM, translation, search, and config logic are kept in separate domains |

## 18. Practical Summary

SmartCampus V2T is a complete local video analytics stack. In concrete terms, it:

- manages a local video library,
- processes video into structured semantic artifacts,
- generates summaries and multilingual views,
- builds retrieval indexes,
- exposes analytics, search, reports, QA, and assistant workflows,
- supports experiment comparison and metrics export,
- keeps runtime state local and inspectable.

If a new developer needs a starting mental model, the shortest accurate description is:

1. `app/` is the operator console.
2. `backend/` is the HTTP and job orchestration layer.
3. `src/` is the actual video, LLM, translation, and search logic.
4. `data/` is the runtime source of truth for artifacts and queue state.
5. Processing starts in the queue, becomes structured artifacts, then becomes searchable and explainable evidence.
