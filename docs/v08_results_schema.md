# SmartCampus V2T v0.8 Output Schema

This document details the file tree and JSON/CSV schemas for the v0.8 reproducible run artifacts.

## Results Folder Structure
Refer to `smartcampus_v08_full_reproducible_experiment_plan.md` for the full directory tree structure inside the final ZIP.

## Manifest Schema (v08_combined.csv)
Columns:
- `dataset_id`: (string) "avenue_full" or "shanghaitech_full"
- `video_id`: (string) Unique video identifier
- `source_path`: (string) Raw video file/frame path
- `prepared_video_path`: (string) Clean video file path used by worker
- `split`: (string) "train" or "test"
- `scene_id`: (string) Scene identifier
- `label_type`: (string) Anomaly label type
- `has_anomaly`: (boolean) Anomaly presence flag
- `duration_sec`: (float) Video duration in seconds
- `fps`: (float) Video FPS
- `width`: (int) Video width in pixels
- `height`: (int) Video height in pixels
- `num_frames`: (int) Total frame count
- `notes`: (string) Optional remarks
