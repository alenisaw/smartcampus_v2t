# SmartCampus V2T

SmartCampus V2T is a reproducible video-to-text analytics system for urban surveillance workflows.  
It processes raw camera video into structured segment artifacts, multilingual summaries, searchable indexes, and grounded answers with citations.

## Project Overview

The system runs an end-to-end pipeline:

1. ingest a video,
2. normalize and preprocess frames,
3. generate clip-level captions with a vision-language model,
4. extract structured fields and video summary with a text LLM,
5. produce multilingual views through MT + selective post-edit,
6. build hybrid retrieval indexes,
7. serve search/report/QA/RAG endpoints.

Canonical storage and indexing language is English (`en`), while `ru` and `kz` are generated as additional views.

## Architecture

```text
[UI]
  -> [FastAPI API]
  -> [Guard Query Gate, optional]
  -> [FS Queue + Worker Manager]
  -> [Ingest + Artifact Storage]
  -> [Preprocessing: FFmpeg normalize + OpenCV filtering + cache]
  -> [Clip Builder: window/stride + keyframe policy]
  -> [VLM Captioning: Qwen3-VL-2B]
  -> [Text Postprocess]
  -> [LLM Structuring + Video Summary]
  -> [Guard Output Gate, optional]
  -> [Translation v2 + selective post-edit]
  -> [Structured Storage]
  -> [Index Builder: BM25 + Dense + Metadata]
  -> [Search / Reports / QA / RAG]
```

## Core Components

### Backend (`backend/`)

| Module | Responsibility |
| --- | --- |
| `api.py` | REST API for videos, jobs, queue, index, search, reports, QA, RAG |
| `worker.py` | worker loop, leasing/locks, role-aware job routing |
| `job_executors.py` | `process`, `translate`, and `index` job execution |
| `worker_runtime.py` | profile context resolution, service bootstrapping, failure handling |

### Pipeline (`src/`)

| Module | Responsibility |
| --- | --- |
| `preprocessing/video_io.py` | decode normalization (`fps/resize/pix_fmt`), quality filters, frame cache |
| `pipeline/video_to_text.py` | clip building and VLM inference |
| `pipeline/structuring.py` | Segment Schema v2 field extraction (LLM + fallback) |
| `pipeline/summary_service.py` | Video Summary v2 generation with citations |
| `translation/service.py` | MT routing, cache, selective LLM post-edit |
| `search/index_builder.py` | BM25 + dense + metadata index build |
| `search/query_engine.py` | hybrid retrieval, fusion, reranking, filters |
| `llm/client.py` | unified text LLM backend (`transformers` or `vllm`) |

## Worker Roles

The queue supports role-scoped workers:

| Role | Allowed job types |
| --- | --- |
| `gpu` | `process` |
| `mt` | `translate` |
| `cpu` | `index` |
| `all` | all job types |

Set role with `SMARTCAMPUS_WORKER_ROLE`.

## Runtime Profiles and Configuration

Profiles are stored in `configs/profiles/`.

| Profile | Typical usage |
| --- | --- |
| `main` | default runtime |
| `experimental` | experimental comparisons |

Set active profile with `SMARTCAMPUS_PROFILE`.

Experimental overrides are stored in `configs/variants/` and selected with `SMARTCAMPUS_VARIANT`.

| Variant | Intent | Main differences |
| --- | --- | --- |
| `exp_a` | balanced baseline | standard fps/window/candidates, dense mode `text` |
| `exp_b` | quality-focused | higher fps and candidate pools, dense mode `text_keyframe`, stronger rerank |
| `exp_c` | speed-focused | lower fps/resolution, smaller candidate pools, reduced rerank depth |

### Most relevant config sections

| Section | What it controls |
| --- | --- |
| `paths` | data/models/index/cache locations |
| `video` | decode policy, sampling, filtering, anonymization |
| `clips` | window/stride limits and keyframe policy |
| `model` | VLM model and decoding |
| `llm` | text LLM backend and generation parameters |
| `guard` | query/output policy gates |
| `translation` | MT backend, route models, cache, post-edit targets |
| `search` | hybrid weights, rerank settings, dense input mode |
| `runtime` | overwrite behavior and metrics collection settings |

## Processing Flows

### Process job

1. Decode normalization and frame preprocessing.
2. Clip generation by `window_sec`/`stride_sec`.
3. EN caption generation by VLM.
4. Text postprocessing and Segment Schema v2 construction.
5. Structuring and video summary generation.
6. Artifact persistence (`segments`, `summaries`, `metrics`, manifests).
7. Optional index update and translation job fan-out.

### Translate job

1. Source artifacts load.
2. MT translation by configured language routes.
3. Selective LLM post-edit for configured targets.
4. Target artifact save + translation metadata update.
5. Optional target-language index update.

### Search / grounded generation

1. Query guard (optional).
2. Hybrid retrieval (BM25 + dense + metadata filters).
3. Reranking over top candidates.
4. Grounded response generation for reports/QA/RAG with citations.
5. Optional output translation + post-edit + output guard.

## Artifacts and Data Layout

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
  research/
```

## Local Run

```powershell
python -m uvicorn backend.api:app --host 127.0.0.1 --port 8000
python -m backend.worker
python -m streamlit run app/main.py --server.port 8501
```

Alternative:

```powershell
run_all.bat
```

## Docker Run

Default stack:

```powershell
docker compose up --build
```

With optional vLLM service:

```powershell
docker compose --profile with_vllm up --build
```

With optional CT2 profile service:

```powershell
docker compose --profile with_ct2 up --build
```

Service smoke check:

```powershell
python scripts/smoke_services.py --api http://127.0.0.1:8000 --ui http://127.0.0.1:8501
```

## API Surface

| Group | Endpoints |
| --- | --- |
| Health | `/healthz`, `/v1/health`, `/v1/healthz` |
| Videos | `/v1/videos`, `/v1/videos/upload`, `/v1/videos/{video_id}`, `/v1/videos/{video_id}/outputs` |
| Jobs | `/v1/jobs`, `/v1/jobs/{job_id}`, `/v1/jobs/{job_id}/cancel` |
| Queue | `/v1/queue`, `/v1/queue/pause`, `/v1/queue/resume`, `/v1/queue/move`, `/v1/queue/{job_id}` |
| Index | `/v1/index/status`, `/v1/index/rebuild` |
| Retrieval | `/v1/search`, `/v1/reports`, `/v1/qa`, `/v1/rag` |

## Technology Stack

| Category | Stack |
| --- | --- |
| API/UI | FastAPI, Pydantic, Uvicorn, Streamlit |
| ML | PyTorch, Transformers, Qwen3 family |
| Video | FFmpeg, OpenCV |
| Translation | CTranslate2, sentencepiece, sacremoses |
| Retrieval | NumPy, BM25 + dense embeddings + rerank |
| Storage | JSON/JSONL.ZST, SQLite |
| Containers | Docker, Docker Compose |
