# SmartCampus V2T v0.8 Experiment Protocol

This document outlines the standard protocol to execute the reproducible experiments for SmartCampus V2T version 0.8.

## Protocol Steps
1. **Hardware Detection & Autotuning**: Run the hardware inspection script to generate hardware configurations optimal for the workstation's system resources and GPU capabilities.
2. **Dataset Auditing & Preparation**: Audit the available video datasets and generate standard video manifests. Convert frame sequences to normalized video files.
3. **Pipeline Ingestion**: Describe all temporality-sliced clips using the visual-language model. Merge clips semantically into final searchable video segments.
4. **Index Building**: Construct dense (Vector/Faiss) and sparse (BM25) search indices over the final video segments.
5. **Retrieval Evaluation**: Evaluate retrieval quality across lexical, semantic, and hybrid search configurations using English and multilingual queries.
6. **Ablation Studies**: Ablate semantic merging thresholds, dense-to-sparse weights, and reranking parameters.
7. **Paper Tables & Plots Generation**: Export tables and figures directly from the execution results.
8. **Final Packaging**: Bundle all configs, reports, logs, figures, and databases into a single ZIP file for reproducibility.
