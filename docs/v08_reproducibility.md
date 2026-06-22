# SmartCampus V2T v0.8 Reproducibility Guide

This guide ensures any researcher can reproduce the exact results of the SmartCampus V2T v0.8 experiment.

## Prerequisites
- Windows OS (tested on Windows 10/11)
- Python 3.10+
- NVIDIA GPU with CUDA drivers configured (e.g. RTX 5000 Ada)
- FFmpeg installed in system path

## Reproduction Workflow
Run the pipeline steps sequentially:
1. `python scripts/runtime/detect_hardware.py --out data/research/v08/hardware.json`
2. `python scripts/runtime/autotune_hardware.py --profile-template configs/profiles/v08_base.yaml --out configs/generated/v08_auto_server.yaml --target balanced`
3. Prepare datasets using prepare scripts.
4. Run the pipeline: `python scripts/experiments/run_v08_pipeline.py --manifest data/manifests/v08_combined.csv --profile configs/generated/v08_auto_server.yaml --variants base,no_merge --out data/research/v08`
5. Run evaluations and ablations.
6. Export charts and bundle the results ZIP.
