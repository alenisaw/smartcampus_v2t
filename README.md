# SmartCampus V2T

SmartCampus V2T is a a video-to-text analytics pipeline aimed at urban surveillance scenarios. The codebase is organized around a reproducible processing flow that turns raw video into structured multilingual artifacts, searchable indexes, and grounded operator-facing outputs.

The repository is intentionally split by responsibility: interface code lives separately from backend orchestration, reusable pipeline logic is isolated in `src/`, and experiment notes are kept outside the main repository page.

---

## Repository Layout

The project is structured as a layered application rather than a single script-driven prototype.

| Path | Role |
| --- | --- |
| [`app/`](/C:/Users/alenk/Downloads/smartcampus_v2t/app) | Streamlit UI, assets, and frontend-facing helpers |
| [`backend/`](/C:/Users/alenk/Downloads/smartcampus_v2t/backend) | FastAPI API, queue control, worker runtime, and job execution |
| [`src/`](/C:/Users/alenk/Downloads/smartcampus_v2t/src) | Core pipeline modules: preprocessing, structuring, summaries, guard, search, translation, storage |
| [`configs/`](/C:/Users/alenk/Downloads/smartcampus_v2t/configs) | Runtime profiles and experimental variants |
| [`data/`](/C:/Users/alenk/Downloads/smartcampus_v2t/data) | Local videos, generated outputs, caches, indexes, research exports |
| [`models/`](/C:/Users/alenk/Downloads/smartcampus_v2t/models) | Local model directories used by the runtime |
| [`scripts/`](/C:/Users/alenk/Downloads/smartcampus_v2t/scripts) | Utility script for research-oriented metrics export |
| [`docs/`](/C:/Users/alenk/Downloads/smartcampus_v2t/docs) | Supporting project documentation such as status tracking and experiment notes |

The repository root also contains the main entrypoints and operational files:

| File | Purpose |
| --- | --- |
| [`README.md`](/C:/Users/alenk/Downloads/smartcampus_v2t/README.md) | project-facing repository overview |
| [`run_all.bat`](/C:/Users/alenk/Downloads/smartcampus_v2t/run_all.bat) | local convenience launcher |
| [`Dockerfile`](/C:/Users/alenk/Downloads/smartcampus_v2t/Dockerfile) | container image definition |
| [`docker-compose.yml`](/C:/Users/alenk/Downloads/smartcampus_v2t/docker-compose.yml) | local multi-service stack |
| [`requirements.txt`](/C:/Users/alenk/Downloads/smartcampus_v2t/requirements.txt) | Python dependencies |

---

## Application Layers

The repository is built around a clear separation of concerns.

### UI Layer

The Streamlit interface in [`app/`](/C:/Users/alenk/Downloads/smartcampus_v2t/app) provides the operator-facing experience. It handles video browsing, playback, scene inspection, retrieval views, and grounded response presentation. Assets such as styles and multilingual UI text are stored alongside the UI code.

### API and Worker Layer

The HTTP API and job-processing logic live in [`backend/`](/C:/Users/alenk/Downloads/smartcampus_v2t/backend). This layer exposes endpoints for uploads, jobs, outputs, indexing, search, reports, QA, and RAG, while the worker handles queued processing and background tasks.

### Pipeline Layer

Reusable implementation modules are kept in [`src/`](/C:/Users/alenk/Downloads/smartcampus_v2t/src). This is where the actual processing pipeline is implemented: video preprocessing, clip building, caption structuring, summary generation, search indexing, translation, guard logic, and storage helpers.

### Configuration Layer

Profiles and variants are isolated in [`configs/`](/C:/Users/alenk/Downloads/smartcampus_v2t/configs). This keeps runtime behavior declarative and makes it possible to switch between the primary path and controlled experiment variants without editing code.

### Data and Artifact Layer

All local runtime artifacts are written under [`data/`](/C:/Users/alenk/Downloads/smartcampus_v2t/data). This includes uploaded videos, per-video outputs, index data, cache data, and research-oriented metrics exports.

---

## Runtime Profiles

The repository uses two top-level runtime modes.

| Profile | Description |
| --- | --- |
| [`main`](/C:/Users/alenk/Downloads/smartcampus_v2t/configs/profiles/main.yaml) | the default operational and demonstration profile |
| [`experimental`](/C:/Users/alenk/Downloads/smartcampus_v2t/configs/profiles/experimental.yaml) | the comparison profile that expands runs into variants |

The experimental mode is backed by dedicated variant configs:

| Variant | Config |
| --- | --- |
| `exp_a` | [`exp_a.yaml`](/C:/Users/alenk/Downloads/smartcampus_v2t/configs/variants/exp_a.yaml) |
| `exp_b` | [`exp_b.yaml`](/C:/Users/alenk/Downloads/smartcampus_v2t/configs/variants/exp_b.yaml) |
| `exp_c` | [`exp_c.yaml`](/C:/Users/alenk/Downloads/smartcampus_v2t/configs/variants/exp_c.yaml) |

This layout keeps the operational path simple while preserving a dedicated place for side-by-side evaluation work.

---

## Data Layout

Each video is stored in its own folder under `data/videos/<video_id>/`. The repository keeps raw input and generated artifacts together so that runs remain inspectable and reproducible.

```text
data/videos/<video_id>/
  raw/
  cache/
  manifest.json
  outputs/
    segments/
    summaries/
    metrics.json
    manifest.json
    run_manifest.json
    experimental_manifest.json
    variants/
      <variant_id>/
        segments/
        summaries/
        metrics.json
        manifest.json
        run_manifest.json
```

Beyond per-video data, the repository also keeps:

| Path | Contents |
| --- | --- |
| [`data/indexes/`](/C:/Users/alenk/Downloads/smartcampus_v2t/data/indexes) | search indexes |
| [`data/cache/`](/C:/Users/alenk/Downloads/smartcampus_v2t/data/cache) | translation, embedding, and converted runtime caches |
| [`data/research/`](/C:/Users/alenk/Downloads/smartcampus_v2t/data/research) | exported metrics snapshots for analysis |

---

## Operational Script

The `scripts` directory is intentionally kept minimal. It currently contains a single repository-level utility:

| Script | Purpose |
| --- | --- |
| [`collect_metrics.py`](/C:/Users/alenk/Downloads/smartcampus_v2t/scripts/collect_metrics.py) | exports a zipped metrics bundle with per-run, per-video, and system-wide summaries |

Index rebuilds remain available through the backend and worker flow, while the standalone script is reserved for research and reporting exports.

---

## Local Models

The repository is designed to work with local model directories. The current workspace keeps its model inventory under [`models/`](/C:/Users/alenk/Downloads/smartcampus_v2t/models), while converted runtime caches live under [`data/cache/ct2_models/`](/C:/Users/alenk/Downloads/smartcampus_v2t/data/cache/ct2_models).

This keeps the runtime local-first and makes the main profile usable without depending on external online model resolution during normal operation.

---

## Supporting Documentation

The repository keeps non-primary documentation in the `docs` directory so that the main page remains focused on the codebase itself.

| Document | Purpose |
| --- | --- |
| [`docs/STATUS.md`](/C:/Users/alenk/Downloads/smartcampus_v2t/docs/STATUS.md) | detailed local implementation status and remaining work |
| [`docs/EXPERIMENTS.md`](/C:/Users/alenk/Downloads/smartcampus_v2t/docs/EXPERIMENTS.md) | experiment planning, comparisons, and evaluation notes |

The intended separation is simple: `README` explains the repository, `STATUS` preserves development context, and `EXPERIMENTS` captures research-oriented comparison work.
