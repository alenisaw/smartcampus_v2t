# SmartCampus V2T

SmartCampus V2T is a local video analytics system for surveillance-style and operator-facing workflows. It processes stored video into clip observations, structured segment artifacts, multilingual summaries, hybrid search indexes, and grounded report/QA/RAG responses.

## What The System Does

- ingests raw videos into the local library,
- normalizes and preprocesses frames,
- builds temporal clips and keyframes,
- generates English clip observations with a vision-language model,
- structures segments and video summaries with a text LLM,
- produces `ru` and `kz` views through MT plus selective post-edit,
- builds hybrid retrieval indexes,
- serves search, grounded reports, QA, and RAG through FastAPI and Streamlit.

Canonical storage and indexing language is English. Additional language views are generated from the English source artifacts.

## Documentation Map

- Project guide: [docs/project_guide.md](docs/project_guide.md)
- Project guide, Russian version: [docs/project_guide.ru.md](docs/project_guide.ru.md)

## Repository Layout

```text
app/       Streamlit operator UI
backend/   FastAPI entrypoints, retrieval runtime, HTTP helpers, worker runtime, and job execution layers
configs/   Runtime profiles and experimental variants
scripts/   Local operations, metrics, experiments, evaluation
src/       Core runtime, video, semantic LLM, guard, translation, and search layers
data/      Local runtime artifacts, indexes, caches, research outputs
docs/      Project documentation
```

## Quick Start

### Local processes

```powershell
python -m uvicorn backend.api:app --host 127.0.0.1 --port 8000
python -m backend.worker
python -m streamlit run app/main.py --server.port 8501
```

### One-command local start

```powershell
run_all.bat
```

### Docker

```powershell
docker compose up --build
```

Optional compose profiles:

```powershell
docker compose --profile with_vllm up --build
docker compose --profile with_ct2 up --build
```

Service-oriented text LLM stack through `vLLM`:

```powershell
docker compose -f docker-compose.yml -f docker-compose.vllm.yml up --build
```

This stack uses the `container_vllm` runtime profile and expects local assets under:

```text
./data
./models
```

Compose already has safe defaults for this stack.
If you need Docker-level overrides such as `VLLM_*`, set them in the shell or a local untracked `.env`.
Application behavior should live in `configs/profiles/*.yaml` and `configs/variants/*.yaml`.
The Docker image also installs `faiss-cpu` from `requirements-faiss.txt`, so `search.ann_backend: auto` resolves to a real FAISS index inside containers.

The `vLLM` integration is Docker-oriented. The supported path is the compose override stack above rather than a native Windows launcher.

### Docker Bring-Up

1. Confirm local directories exist:

```text
./data
./models
```

2. Optional: set Docker override variables in the shell only when needed, for example:

```powershell
$env:VLLM_MODEL_ID="/app/models/qwen3-4b-instruct-2507"
$env:VLLM_GPU_MEMORY_UTILIZATION="0.86"
$env:VLLM_MAX_MODEL_LEN="512"
$env:VLLM_MAX_NUM_SEQS="2"
$env:VLLM_MAX_NUM_BATCHED_TOKENS="1024"
```

3. Inspect the resolved runtime before startup:

```powershell
python scripts/runtime/inspect_runtime.py --profile container_vllm
```

4. Start the full service stack:

```powershell
docker compose -f docker-compose.yml -f docker-compose.vllm.yml up --build -d
```

5. Watch logs while the text model warms up:

```powershell
docker compose -f docker-compose.yml -f docker-compose.vllm.yml logs -f vllm_text api worker_gpu worker_cpu worker_mt ui
```

The `vllm_text` service now starts from the local `./models/qwen3-4b-instruct-2507` path by default.
On a first run under Docker Desktop / WSL, `vLLM` may spend several minutes loading weights and compiling before `/v1/models` becomes ready.

6. Run the smoke check:

```powershell
python scripts/runtime/smoke_services.py --api http://127.0.0.1:8000 --ui http://127.0.0.1:8501 --llm http://127.0.0.1:8001
```

7. Rebuild the index if needed:

```powershell
curl -X POST http://127.0.0.1:8000/v1/index/rebuild
```

8. Inspect the resolved backend:

```powershell
curl http://127.0.0.1:8000/v1/index/status
```

The language payload now includes `ann_backend`, `ann_index_type`, `dense_valid_count`, and `num_docs`.

### Docker Debugging

Useful checks when the stack is up:

```powershell
docker compose -f docker-compose.yml -f docker-compose.vllm.yml ps
docker compose -f docker-compose.yml -f docker-compose.vllm.yml logs --tail=200 api worker_gpu worker_cpu worker_mt ui vllm_text
python scripts/runtime/inspect_runtime.py --profile container_vllm --variant throughput
python scripts/runtime/smoke_services.py --api http://127.0.0.1:8000 --ui http://127.0.0.1:8501 --llm http://127.0.0.1:8001
```

For the VLM path itself:

```powershell
python scripts/runtime/profile_vlm_path.py --video .\data\videos\sample.mp4 --profile main --variant throughput
```

The profiling report now includes packed-batch statistics, padding ratio, resolved attention backend, and OOM retry count.

### Service smoke check

```powershell
python scripts/runtime/smoke_services.py --api http://127.0.0.1:8000 --ui http://127.0.0.1:8501
```

With `vLLM` endpoint verification:

```powershell
python scripts/runtime/smoke_services.py --api http://127.0.0.1:8000 --ui http://127.0.0.1:8501 --llm http://127.0.0.1:8001
```

## Main Runtime Surfaces

- API: `backend/api.py`
- Worker loop: `backend/worker.py`
- Retrieval/runtime glue: `backend/retrieval_runtime.py`
- API support: `backend/http/common.py`, `backend/http/grounded.py`
- Job execution: `backend/jobs/process_runtime.py`, `backend/jobs/translate_runtime.py`, `backend/jobs/runtime_common.py`
- Video observation pipeline: `src/video/describe.py`
- Semantic analysis and summary: `src/llm/analyze.py`, `src/llm/summary.py`
- Guard layer: `src/guard/service.py`, `src/guard/schemas.py`
- Translation: `src/translation/service.py`
- Search build/query: `src/search/builder.py`, `src/search/engine.py`
- UI entrypoint: `app/main.py`

## Research And Evaluation Utilities

- Metrics bundle export: `scripts/collect_metrics.py`
- Experiment matrix orchestration: `scripts/run_experiment_matrix.py`
- Offline relevance evaluation: `scripts/eval_relevance.py`
- Config-level comparison report: `scripts/build_comparison_report.py`
- Runtime inspection: `scripts/runtime/inspect_runtime.py`
- Service smoke checks: `scripts/runtime/smoke_services.py`
- Single-video VLM path profiling: `scripts/runtime/profile_vlm_path.py`

## Runtime Profiles

- `configs/profiles/main.yaml`: default runtime
- `configs/profiles/experimental.yaml`: experiment-oriented runtime
- `configs/variants/fast.yaml`, `throughput.yaml`, `max_quality.yaml`: explicit experimental overrides

`fast` and `throughput` now use a lower `video.analysis_fps` than decode `target_fps`, tighter clip stride/frame limits, shorter VLM generations, and adaptive packed batching with OOM backoff. The worker keeps stable preprocessing while reducing the number of frames passed into the VLM.

### VLM Path Profiling

Use this when comparing `main`, `fast`, and `throughput` on the same video:

```powershell
python scripts/runtime/profile_vlm_path.py --video .\data\videos\sample.mp4 --profile main --variant fast
```

Optional JSON export:

```powershell
python scripts/runtime/profile_vlm_path.py --video .\data\videos\sample.mp4 --profile main --variant throughput --output .\data\research\vlm_profile.json
```

## API Surface

- Health: `/healthz`, `/v1/health`, `/v1/healthz`
- Videos: `/v1/videos`, `/v1/videos/upload`, `/v1/videos/{video_id}`, `/v1/videos/{video_id}/outputs`
- Jobs: `/v1/jobs`, `/v1/jobs/{job_id}`, `/v1/jobs/{job_id}/cancel`
- Queue: `/v1/queue`, `/v1/queue/pause`, `/v1/queue/resume`, `/v1/queue/move`, `/v1/queue/{job_id}`
- Index: `/v1/index/status`, `/v1/index/rebuild`
- Retrieval and grounded generation: `/v1/search`, `/v1/reports`, `/v1/qa`, `/v1/rag`
