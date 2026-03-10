# SmartCampus V2T Project Guide

## 1. Project Summary

SmartCampus V2T is a local end-to-end video analytics system that converts stored video into structured textual artifacts and retrieval-ready indexes. The system is designed around a practical operator workflow:

1. upload or register a video,
2. run the processing pipeline,
3. inspect summaries and per-segment results,
4. search indexed segments,
5. generate grounded reports,
6. query the processed corpus through QA and RAG,
7. export metrics and experiment results for research.

The repository contains three main runtime surfaces:

- a Streamlit operator interface,
- a FastAPI backend,
- a worker-based filesystem queue for asynchronous processing.

The pipeline is already assembled end to end. The codebase includes preprocessing, clip building, VLM captioning, structuring, summary generation, machine translation, selective post-edit, hybrid retrieval, experiment utilities, and research metrics export.

## 2. High-Level Architecture

```text
Streamlit UI
  -> FastAPI API
  -> filesystem queue + persisted jobs
  -> worker runtime
  -> preprocessing
  -> clip builder + keyframe policy
  -> VLM captioning
  -> segment structuring
  -> video summary generation
  -> translation + selective post-edit
  -> hybrid index build
  -> search / reports / QA / RAG
```

The repository is local-first. Data, indexes, manifests, metrics, and experiment artifacts are persisted under `data/`.

## 3. Repository Structure

### `app/`

Streamlit UI layer.

- `main.py`: Streamlit entrypoint and page routing.
- `ui.py`: facade over the modular UI.
- `api_client.py`: REST client for the backend.
- `state.py`: session defaults for UI state.
- `pages/`: page-level renderers for storage, analytics, and assistant surfaces.
- `components/`: reusable UI components.
- `lib/`: formatting, i18n, and media helpers.
- `assets/`: CSS, translations, and brand assets.

### `backend/`

Backend HTTP and worker runtime.

- `api.py`: FastAPI endpoints for videos, jobs, queue, index, search, reports, QA, and RAG.
- `job_control.py`: filesystem job records, queue helpers, and state transitions.
- `job_executors.py`: execution logic for `process`, `translate`, and `index`.
- `worker.py`: polling loop and job dispatch.
- `worker_runtime.py`: context loading, service construction, and failure handling.
- `experimental.py`: experimental fan-out helpers and batch manifest support.
- `schemas.py`: request and response contracts.
- `deps.py`: shared backend helpers.

### `configs/`

Runtime configuration source of truth.

- `profiles/`: base runtime profiles.
- `variants/`: experiment-specific overrides.

Runtime selection uses one profile plus an optional variant override.

### `src/`

Core application logic.

- `preprocessing/`: FFmpeg decode normalization and frame preparation.
- `pipeline/`: clip building, captioning orchestration, segment schema, and summaries.
- `translation/`: MT routing, cache, and selective post-edit.
- `search/`: hybrid indexing and querying.
- `guard/`: input/output guard hooks.
- `llm/`: shared text-generation client.
- `core/`: runtime types and model-specific backend logic.
- `utils/`: config loading and video artifact path helpers.

### `scripts/`

Operational and research tooling.

- `smoke_services.py`: API/UI smoke checks.
- `collect_metrics.py`: export research-ready metrics bundles.
- `run_experiment_matrix.py`: run profile/variant experiment matrices.
- `eval_relevance.py`: offline retrieval evaluation with query-label datasets.
- `build_comparison_report.py`: latest-per-config comparison reports.

### `data/`

Local runtime artifacts.

- uploaded videos,
- derived outputs,
- queue and job records,
- locks,
- indexes,
- caches,
- research bundles and experiment artifacts.

### `docs/`

Project documentation.

- `project_guide.md`: detailed architecture and operations guide.
- `local_progress.md`: local-only progress notes, ignored by git.

## 4. Runtime Model

### Profiles and Variants

The runtime is controlled by:

- one active profile,
- zero or one active variant override.

Profiles:

- `main`: default runtime profile.
- `experimental`: experiment-oriented profile with variant fan-out support.

Variants:

- `exp_a`
- `exp_b`
- `exp_c`

The effective configuration is the merged result of:

1. selected profile,
2. optional `extends` chain inside YAML,
3. optional variant override.

Typed config objects are defined in `src/pipeline/pipeline_config.py`. Loading and merging logic lives in `src/utils/config_loader.py`.

### Worker Roles

Workers can be restricted by role:

- `gpu` -> `process`
- `mt` -> `translate`
- `cpu` -> `index`
- `all` -> all job types

Role selection is controlled through `SMARTCAMPUS_WORKER_ROLE`.

### Canonical Language Model

The system stores and indexes a canonical English artifact view. Additional `ru` and `kz` outputs are generated as derived views. This keeps indexing and storage normalized while still exposing multilingual operator outputs.

## 5. Processing Pipeline

### 5.1 Process Job

The `process` job is the main end-to-end path.

1. The worker resolves the effective config for the job.
2. The source video is located in `data/videos/<video_id>/raw/`.
3. Decode normalization is applied with explicit fps, resolution, and pixel format handling.
4. Frame quality filters are applied for dark frames, low-motion frames, and blur flags.
5. Clip windows are produced from the processed frame stream.
6. Keyframes are selected according to the configured keyframe policy.
7. The VLM generates English clip captions.
8. Captions are sanitized and smoothed.
9. Segment schema payloads are built and enriched.
10. A video summary is generated.
11. Artifacts and manifests are written to disk.
12. Optional index updates and translation fan-out are triggered.
13. Metrics are persisted.

Key implementation points:

- preprocessing: `src/preprocessing/video_io.py`
- orchestration: `src/pipeline/video_to_text.py`
- structuring: `src/pipeline/structuring.py`
- summary generation: `src/pipeline/summary_service.py`

### 5.2 Translate Job

The `translate` job consumes existing source outputs.

1. Source segments and summary are loaded.
2. MT routes are selected from config.
3. Segment and summary translation is executed.
4. Selective LLM post-edit is applied to configured targets.
5. Translated artifacts are persisted.
6. Optional target-language index updates are performed.
7. Translation metadata and timings are stored in metrics.

Implementation point:

- `src/translation/service.py`

### 5.3 Index Job

The `index` job rebuilds or refreshes retrieval indexes from stored outputs.

1. Segment files are discovered for the target language and variant.
2. Searchable text and dense input text are built.
3. BM25 and dense representations are stored.
4. Metadata and manifest files are updated.
5. Index state is written for API and UI inspection.

Implementation point:

- `src/search/index_builder.py`

## 6. Key Runtime Concepts

### Decode Normalization

The system explicitly controls:

- fps,
- resize,
- pixel format.

This stabilizes downstream behavior across runs and source files.

### Keyframe Policy

Clip keyframes are selectable through config. Current policies include:

- `first`
- `middle`
- `last`
- `sharpest`

### Selective Post-Edit

Translation does not send every segment through LLM post-edit. Instead, configured subsets are post-edited. This keeps quality improvements focused on the outputs that matter most while controlling cost and latency.

### Hybrid Retrieval

Search combines:

- sparse BM25,
- dense embedding search,
- metadata filters,
- reranking.

The search stack supports dense input mode `text_keyframe`, which augments text retrieval with lightweight keyframe-derived visual tokens.

## 7. Main Files And Their Roles

### API and Job Surfaces

- `backend/api.py`
  Central HTTP surface for ingest, queue control, outputs, metrics summary, search, reports, QA, and RAG.

- `backend/job_executors.py`
  The central execution layer for `process`, `translate`, and `index`.

- `backend/worker.py`
  The queue poller and dispatcher.

- `backend/worker_runtime.py`
  Runtime context resolver for profile/variant-specific service bundles.

### Core Pipeline

- `src/pipeline/video_to_text.py`
  Main process-job orchestration and clip captioning pipeline.

- `src/preprocessing/video_io.py`
  Decode normalization and frame preparation.

- `src/pipeline/structuring.py`
  Structured segment enrichment.

- `src/pipeline/summary_service.py`
  Video-level summary generation.

### Search

- `src/search/index_builder.py`
  Index build/update logic and embedder integration.

- `src/search/query_engine.py`
  Query-time hybrid retrieval and rerank flow.

### Translation

- `src/translation/service.py`
  Translation routing, cache handling, and selective post-edit.

## 8. Artifact Model

The local artifact layout is centered under `data/`.

```text
data/
  videos/<video_id>/
    raw/
    cache/
    outputs/
      segments/
      summaries/
      metrics.json
      manifest.json
      run_manifest.json
      variants/<variant_id>/...
  indexes/
  cache/
  jobs/
  queue/
  locks/
  research/
```

### Important Output Files

- `manifest.json`
  Language-level output status and artifact availability.

- `run_manifest.json`
  Effective profile, variant, config fingerprint, and artifact paths.

- `metrics.json`
  Runtime metrics for the process or translation path, including stage timings.

- `batch_manifest.json`
  Experimental parent/child fan-out manifest for variant runs.

## 9. Metrics And Research Exports

Metrics now expose:

- stage-level timings,
- throughput-style derived metrics,
- translation totals,
- indexing totals,
- stage aggregates across runs,
- config-level comparison outputs.

### Metrics Bundle

Generate a research bundle:

```powershell
python scripts/collect_metrics.py
```

Bundle outputs are written under `data/research/metrics_bundle/`.

Important files:

- `metrics_runs.csv`
- `metrics_stage_runs.csv`
- `metrics_stage_aggregate.csv`
- `metrics_snapshot.json`
- `metrics_by_video.json`
- `metrics_by_profile_variant.json`
- `system_metrics.json`

### Experiment Matrix

Run a local experiment matrix:

```powershell
python scripts/run_experiment_matrix.py --video-ids 20 --profiles main,experimental
```

The script:

- submits jobs,
- waits for terminal outputs,
- snapshots run artifacts,
- refreshes the metrics bundle,
- optionally runs relevance evaluation,
- builds a comparison report.

### Relevance Evaluation

Run offline retrieval evaluation:

```powershell
python scripts/eval_relevance.py --labels data/research/relevance/queries_labels.json --profiles main --video-ids 20
```

The evaluator computes:

- `P@K`
- `Recall@K`
- `nDCG@K`
- `MRR`

### Config-Level Comparison

Build a comparison report:

```powershell
python scripts/build_comparison_report.py --experiment-manifest data/research/experiments/<label>/experiment.json
```

Storage behavior:

- the same config key overwrites its previous latest result,
- a new config key creates a new file,
- a global comparison report is built over the latest saved config rows.

## 10. Frontend Surface

The UI remains on Streamlit and is structured as an operator console rather than a public product site.

Main tabs:

- storage: video library, upload, process execution, output inspection
- analytics: retrieval, filtering, search result inspection
- assistant: grounded reports, QA, RAG, runtime metrics inspection

The UI is now organized internally into:

- page renderers,
- reusable components,
- shared helper libraries.

This keeps Streamlit while avoiding a single monolithic UI module.

## 11. Configuration Areas

The most important config surfaces are:

- `paths`
  Filesystem roots for runtime artifacts and models.

- `video`
  Decode resolution, fps, filters, anonymization, JPEG settings.

- `clips`
  Windowing and keyframe selection policy.

- `model`
  VLM model path, dtype, batch size, decode settings.

- `llm`
  Summary and structuring backend settings.

- `translation`
  MT backends, language routes, cache behavior, post-edit targets.

- `search`
  Dense model, reranker, fusion weights, dedupe, dense input mode.

- `runtime`
  overwrite behavior, metrics controls, reproducibility flags.

- `worker`
  queue polling, leasing, and concurrency behavior.

## 12. Running The System

### Local Multi-Process Run

Terminal 1:

```powershell
python -m uvicorn backend.api:app --host 127.0.0.1 --port 8000
```

Terminal 2:

```powershell
python -m backend.worker
```

Terminal 3:

```powershell
python -m streamlit run app/main.py --server.port 8501
```

### Convenience Start

```powershell
run_all.bat
```

### Docker

```powershell
docker compose up --build
```

Optional profiles:

```powershell
docker compose --profile with_vllm up --build
docker compose --profile with_ct2 up --build
```

## 13. API Surface Summary

### Health

- `/healthz`
- `/v1/health`
- `/v1/healthz`

### Videos

- `GET /v1/videos`
- `POST /v1/videos/upload`
- `DELETE /v1/videos/{video_id}`
- `GET /v1/videos/{video_id}/outputs`
- `GET /v1/videos/{video_id}/batch-manifest`
- `GET /v1/videos/{video_id}/metrics-summary`

### Jobs And Queue

- `POST /v1/jobs`
- `GET /v1/jobs/{job_id}`
- `POST /v1/jobs/{job_id}/cancel`
- `GET /v1/queue`
- `POST /v1/queue/pause`
- `POST /v1/queue/resume`
- `POST /v1/queue/move`
- `DELETE /v1/queue/{job_id}`

### Index

- `GET /v1/index/status`
- `POST /v1/index/rebuild`

### Retrieval And Grounded Generation

- `POST /v1/search`
- `POST /v1/reports`
- `POST /v1/qa`
- `POST /v1/rag`

## 14. Operational Notes

- The repository is local-first and artifact-heavy by design.
- `data/` is ignored by git and acts as the runtime workspace.
- `.agent/` is local Codex memory and is ignored by git.
- `docs/local_progress.md` is local-only and ignored by git.
- `__pycache__/` is ignored by git and should not be treated as source structure.

## 15. Current Architectural Pressure Points

The codebase is functional, but several modules carry too much responsibility:

- `backend/api.py`
- `backend/job_executors.py`
- `src/search/index_builder.py`
- `src/search/query_engine.py`
- `src/pipeline/video_to_text.py`

These are the main candidates for internal reorganization without changing runtime behavior.

## 16. Technology Stack

- API: FastAPI, Pydantic, Uvicorn
- UI: Streamlit
- Models: PyTorch, Transformers, Qwen family
- Video: FFmpeg, OpenCV
- Translation: CTranslate2, sentencepiece, sacremoses
- Retrieval: BM25 plus dense embeddings plus rerank
- Storage: JSON, local filesystem, SQLite-based caches where applicable
- Containers: Docker, Docker Compose

## 17. Development Guidance

- Treat the current profile and variant system as the runtime source of truth.
- Keep runtime behavior stable during structural refactors.
- Prefer explicit, typed, low-magic code.
- Keep manifests and metrics consistent with artifact writes.
- Use the scripts in `scripts/` for research and operational work instead of creating parallel ad hoc entrypoints.
