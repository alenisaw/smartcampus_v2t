# SmartCampus V2T

**SmartCampus V2T** is a practical, end-to-end **video-to-text analytics system** for long-form CCTV and surveillance footage.
The project focuses on transforming raw video streams into **structured, searchable textual representations** using modern multimodal models and efficient indexing, with a strong emphasis on usability through a Streamlit-based UI.

![Semantic Pipeline Overview](docs/figure1_main.png)

---

## Overview

The system processes long, untrimmed surveillance videos and converts them into interpretable text that can be explored, searched, and analyzed.
Instead of operating on low-level detections alone, SmartCampus V2T prioritizes **semantic understanding** and **human-readable outputs**.

Core ideas:
- Video understanding expressed directly as text
- Deterministic, inspectable processing stages
- UI-first interaction model for experimentation and analysis

---

## System Architecture

![System Architecture](docs/figure2_main.png)

The pipeline is organized into modular components:

- **Preprocessing layer**
  Normalizes FPS, filters redundant or dark frames, applies optional face anonymization, and caches prepared frames.

- **Clip generation**
  Videos are segmented into overlapping temporal windows (clips) suitable for multimodal inference.

- **Video-to-Text inference**
  Each clip is described using Qwen3-VL with concise, factual prompts (RU / KZ / EN).

- **Temporal post-processing**
  Adjacent clips with similar semantics are merged to reduce redundancy.

- **Global summarization**
  A structured summary is generated strictly from clip-level descriptions.

- **Indexing and search**
  Hybrid sparse+dense indexing enables fast semantic retrieval over time intervals.

---

## Repository Structure

```text
app/                    # Streamlit application (main control plane)
config/pipeline.yaml    # Unified configuration
src/
  core/                 # Qwen3-VL backend wrapper
  preprocessing/        # Video preparation and caching
  pipeline/             # Clip V2T, merging, global summary
  search/               # Index builder and query engine
data/
  raw/ 
  prepared/ 
  runs/ 
  indexes/
docs/                  #Files related with README.md and user guides
  figure1_main.png
  figure2_main.png
```

---

## Installation & Configuration

### Environment setup

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate   # Linux / macOS
# .venv\Scripts\activate  # Windows

pip install -U pip
pip install -r requirements.txt
```

### Configuration

All system parameters are defined in a single file:

```
config/pipeline.yaml
```

This includes:
- Paths for videos, runs, prepared frames, and indexes
- Video preprocessing parameters
- Model and inference settings
- Search and indexing configuration

---
## Running the project

The entire workflow is driven from the Streamlit interface.

```bash
streamlit run app/streamlit_app.py
```

UI will be opened automatically after running command or you may manually open it by next link:
```
http://localhost:8501
```

From the UI you can:
- Preprocess videos
- Run video-to-text inference
- Inspect runs and metrics
- Build or update indexes
- Perform semantic search over timelines

---
## Component Status

| Component                       | Status              |
|---------------------------------|---------------------|
| Video preprocessing             | Implemented         |
| Clip V2T inference              | Implemented         |
| Temporal merging                | Implemented         |
| Global summary                  | Implemented         |
| Hybrid indexing                 | Implemented         |
| Semantic search                 | Implemented         |
| Streamlit UI                    | Implemented         |
