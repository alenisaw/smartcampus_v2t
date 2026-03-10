# SmartCampus V2T

SmartCampus V2T is a local video-to-text analytics system for surveillance-style and operator-facing workflows. It processes stored video into structured segment artifacts, multilingual summaries, hybrid search indexes, and grounded report/QA/RAG responses.

## What The System Does

- ingests raw videos into the local library,
- normalizes and preprocesses frames,
- builds temporal clips and keyframes,
- generates English captions with a vision-language model,
- enriches segments with structured fields and video summaries,
- produces `ru` and `kz` views through MT plus selective post-edit,
- builds hybrid retrieval indexes,
- serves search, grounded reports, QA, and RAG through FastAPI and Streamlit.

Canonical storage and indexing language is English. Additional language views are generated from the English source artifacts.

## Documentation Map

- Project guide: [docs/project_guide.md](docs/project_guide.md)
- Local-only worklog: `docs/local_progress.md` is ignored by git and not part of the public project docs.

## Repository Layout

```text
app/       Streamlit operator UI
backend/   FastAPI API, queue control, worker runtime, job executors
configs/   Runtime profiles and experimental variants
scripts/   Local operations, metrics, experiments, evaluation
src/       Core pipeline, translation, search, config, and utilities
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

### Service smoke check

```powershell
python scripts/smoke_services.py --api http://127.0.0.1:8000 --ui http://127.0.0.1:8501
```

## Main Runtime Surfaces

- API: `backend/api.py`
- Worker loop: `backend/worker.py`
- Job execution: `backend/job_executors.py`
- Main processing pipeline: `src/pipeline/video_to_text.py`
- Translation: `src/translation/service.py`
- Search build/query: `src/search/index_builder.py`, `src/search/query_engine.py`
- UI entrypoint: `app/main.py`

## Research And Evaluation Utilities

- Metrics bundle export: `scripts/collect_metrics.py`
- Experiment matrix orchestration: `scripts/run_experiment_matrix.py`
- Offline relevance evaluation: `scripts/eval_relevance.py`
- Config-level comparison report: `scripts/build_comparison_report.py`

## Runtime Profiles

- `configs/profiles/main.yaml`: default runtime
- `configs/profiles/experimental.yaml`: experiment-oriented runtime
- `configs/variants/exp_a.yaml`, `exp_b.yaml`, `exp_c.yaml`: explicit experimental overrides

## API Surface

- Health: `/healthz`, `/v1/health`, `/v1/healthz`
- Videos: `/v1/videos`, `/v1/videos/upload`, `/v1/videos/{video_id}`, `/v1/videos/{video_id}/outputs`
- Jobs: `/v1/jobs`, `/v1/jobs/{job_id}`, `/v1/jobs/{job_id}/cancel`
- Queue: `/v1/queue`, `/v1/queue/pause`, `/v1/queue/resume`, `/v1/queue/move`, `/v1/queue/{job_id}`
- Index: `/v1/index/status`, `/v1/index/rebuild`
- Retrieval and grounded generation: `/v1/search`, `/v1/reports`, `/v1/qa`, `/v1/rag`

## Notes

- The repository is built around local filesystem artifacts under `data/`.
- `.agent/` and `docs/local_progress.md` are local-only and ignored by git.
- The detailed architecture, runtime flows, configs, artifact model, and operations guide live in [docs/project_guide.md](docs/project_guide.md).
