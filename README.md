# smartcampus_v2t
Smart campus video-to-text pipeline

smartcampus_v2t/
│
├── analysis/
│   ├── __init__.py
│   └── timeline_builder.py
│
├── app/
│   └── streamlit_app.py
│
├── config/
│   ├── pipeline.yaml
│   ├── qlora.yaml
│   └── rag.yaml
│
├── data/
│   ├── annotations/
│   ├── indexes/
│   ├── metrics/
│   ├── prepared/
│   │   └── test/
│   │       ├── frames/
│   │       ├── small/
│   │       └── meta.json
│   └── raw/
│       └── test.mp4
│
├── experiments/
│   ├── run_query_demo.py
│   └── run_single_video.py
│
├── models/
│   ├── qwen3_vl_4b_fp8/
│   └── haarcascade_frontalface_default.xml
│
├── results/
│   ├── qlora_runs/
│   └── runs/
│
├── src/
│   ├── core/
│   │   ├── __init__.py
│   │   ├── qwen_vl_backend.py
│   │   └── types.py
│   │
│   ├── few_shot/
│   │   ├── __init__.py
│   │   └── few_shot_templates.py
│   │
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── eval_metrics.py
│   │   ├── runtime.py
│   │   └── tracker.py
│   │
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── pipeline_config.py
│   │   └── video_to_text.py
│   │
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   └── video_io.py
│   │
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── faiss_index.py
│   │   ├── query_engine.py
│   │   ├── retrieval_prompt.py
│   │   └── semantic_db.py
│   │
│   ├── search/
│   │   ├── __init__.py
│   │   ├── index_builder.py
│   │   └── query_engine.py
│   │
│   ├── streaming/
│   │   ├── stream_reader.py
│   │   └── streaming_pipeline.py
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── qlora_dataset.py
│   │   └── qlora_train.py
│   │
│   └── utils/
│       ├── __init__.py
│       ├── config.py
│       └── utils.py
│
├── Dockerfile
└── README.md


