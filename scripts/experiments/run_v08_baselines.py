# scripts/experiments/run_v08_baselines.py
import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from statistics import mean
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.deps import load_cfg_and_raw, get_backend_paths
from src.search.builder import load_index, resolve_index_dir, search_config_fingerprint
from src.search.engine import QueryEngine
from scripts.eval_relevance import _match_hit, _metrics_for_query, _normalize_relevance_item

# Baseline configuration templates
BASELINE_CONFIGS = {
    "sparse_only": {
        "w_bm25": 1.0,
        "w_dense": 0.0,
        "candidate_k_sparse": 200,
        "candidate_k_dense": 0,
        "rerank_enabled": False,
        "fusion": "rrf"
    },
    "dense_only": {
        "w_bm25": 0.0,
        "w_dense": 1.0,
        "candidate_k_sparse": 0,
        "candidate_k_dense": 200,
        "rerank_enabled": False,
        "fusion": "rrf"
    },
    "hybrid_rrf": {
        "w_bm25": 0.45,
        "w_dense": 0.55,
        "candidate_k_sparse": 200,
        "candidate_k_dense": 200,
        "fusion": "rrf",
        "rrf_k": 60,
        "rerank_enabled": False
    },
    "hybrid_minmax": {
        "w_bm25": 0.45,
        "w_dense": 0.55,
        "candidate_k_sparse": 200,
        "candidate_k_dense": 200,
        "fusion": "minmax",
        "rerank_enabled": False
    },
    "rerank_on": {
        "w_bm25": 0.45,
        "w_dense": 0.55,
        "candidate_k_sparse": 200,
        "candidate_k_dense": 200,
        "fusion": "rrf",
        "rrf_k": 60,
        "rerank_enabled": True,
        "rerank_top_k": 30
    },
    "rerank_off": {
        "w_bm25": 0.45,
        "w_dense": 0.55,
        "candidate_k_sparse": 200,
        "candidate_k_dense": 200,
        "fusion": "rrf",
        "rrf_k": 60,
        "rerank_enabled": False
    }
}

def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval baselines on v0.8 index")
    parser.add_argument("--experiment", type=str, default="data/research/v08/experiment_manifest.json")
    parser.add_argument("--labels", type=str, default="data/relevance/v08_queries_en.json")
    parser.add_argument("--retrieval", type=str, default="sparse_only,dense_only,hybrid_rrf,hybrid_minmax,rerank_on,rerank_off")
    parser.add_argument("--variants", type=str, default="base,no_merge")
    parser.add_argument("--out", type=str, default="data/research/v08/retrieval")
    args = parser.parse_args()

    out_dir = Path(args.out).resolve()
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(args.labels):
        print(f"Error: Relevance labels not found: {args.labels}")
        sys.exit(1)

    with open(args.labels, "r", encoding="utf-8") as f:
        relevance_data = json.load(f)
    queries = relevance_data.get("queries", [])
    defaults = relevance_data.get("defaults", {})
    iou_threshold = float(defaults.get("iou_threshold", 0.5))

    retrieval_methods = [r.strip() for r in args.retrieval.split(",") if r.strip()]
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]

    # Load baseline profile (v08_auto_server)
    cfg, raw_cfg = load_cfg_and_raw(profile="v08_auto_server")
    indexes_dir = Path(cfg.paths.indexes_dir).resolve()

    by_query_rows = []
    summary_rows = []
    latency_rows = []
    failures_rows = []

    for variant in variants:
        # Resolve index path for variant
        # Note: variant "base" is represented as None in resolve_index_dir
        var_name = variant
        var_param = variant
        
        cfg_fp = search_config_fingerprint(cfg)
        index_dir = resolve_index_dir(
            base_index_dir=indexes_dir,
            config_fingerprint=cfg_fp,
            language="en",
            variant=var_param
        )
        
        if not index_dir.exists() or not (index_dir / "doc_ids.json").exists():
            # Fallback to general index folders
            lang_root = indexes_dir / "en"
            if var_param:
                lang_root = lang_root / "variants" / var_param
            if lang_root.exists() and lang_root.is_dir():
                dirs = [d for d in lang_root.iterdir() if d.is_dir() and (d / "doc_ids.json").exists()]
                if dirs:
                    index_dir = sorted(dirs, key=lambda x: x.stat().st_mtime, reverse=True)[0]
                    print(f"Fallback selected index: {index_dir}")

        if not index_dir.exists() or not (index_dir / "doc_ids.json").exists():
            print(f"Warning: Index directory not found or empty for variant={var_name} at {index_dir}. Skipping variant.")
            failures_rows.append([f"ALL_QUERIES_{var_name}", var_name, "all", f"Index not found at {index_dir}"])
            continue

        print(f"Loading index from {index_dir} for variant {var_name}...")
        loaded_index = load_index(index_dir)
        
        for method in retrieval_methods:
            if method not in BASELINE_CONFIGS:
                print(f"Warning: Unknown retrieval method '{method}', skipping.")
                continue

            print(f"Running evaluation for variant={var_name}, method={method}...")
            overrides = BASELINE_CONFIGS[method]
            
            import torch
            device_val = getattr(cfg.search, "device", None) or getattr(cfg.model, "device", None) or ("cuda" if torch.cuda.is_available() else "cpu")

            # Setup QueryEngine
            engine = QueryEngine(
                index=loaded_index,
                config_path=Path(cfg.config_path),
                index_dir=indexes_dir,
                config_fingerprint=cfg_fp,
                language="en",
                variant=var_param,
                w_bm25=float(overrides.get("w_bm25", 0.45)),
                w_dense=float(overrides.get("w_dense", 0.55)),
                candidate_k_sparse=int(overrides.get("candidate_k_sparse", 200)),
                candidate_k_dense=int(overrides.get("candidate_k_dense", 200)),
                embedding_backend="auto",
                fallback_embed_model_name=str(cfg.search.embed_model_name),
                rerank_enabled=bool(overrides.get("rerank_enabled", False)),
                rerank_top_k=int(overrides.get("rerank_top_k", 30)),
                reranker_model_name=str(cfg.search.reranker_model_id),
                reranker_backend="auto",
                fusion=str(overrides.get("fusion", "rrf")),
                rrf_k=int(overrides.get("rrf_k", 60)),
                dedupe_mode=str(cfg.search.dedupe_mode),
                dedupe_tol_sec=float(cfg.search.dedupe_tol_sec),
                dedupe_overlap_thr=float(cfg.search.dedupe_overlap_thr),
                normalize_text=True,
                lemmatize=True,
                device=str(device_val),
                embed_model_name=str(cfg.search.embedding_model_id)
            )

            latencies = []
            metrics_accum = {
                "p@1": [], "p@5": [], "recall@5": [], "mrr": [], "ndcg@5": []
            }

            for query_idx, query_obj in enumerate(queries, start=1):
                query_id = query_obj.get("query_id", f"q_{query_idx:03d}")
                query_text = str(query_obj.get("query_text") or query_obj.get("query") or "").strip()
                search_video_id = query_obj.get("video_id")
                dataset_id = str(query_obj.get("dataset_id") or "").strip()
                if not dataset_id:
                    target_videos = [str(item.get("video_id") or "") for item in query_obj.get("relevant", [])]
                    if target_videos and all(video.startswith("avenue_") for video in target_videos):
                        dataset_id = "avenue_full"
                    elif target_videos and all(video.startswith("shanghaitech_") for video in target_videos):
                        dataset_id = "shanghaitech_full"
                    else:
                        dataset_id = "cross_dataset"
                
                filters = query_obj.get("filters", {})
                if var_param:
                    filters = dict(filters)
                    filters["variant"] = var_param

                t0 = time.perf_counter()
                try:
                    hits, search_stats = engine.search(
                        query=query_text,
                        top_k=20,
                        video_id=search_video_id,
                        filters=filters,
                        dedupe=True,
                        return_stats=True
                    )
                    latency = (time.perf_counter() - t0) * 1000.0
                    latencies.append(latency)

                    # Compute relevance metrics
                    relevant_raw = query_obj.get("relevant", [])
                    labels = [
                        _normalize_relevance_item(item, idx + 1, str(search_video_id or ""))
                        for idx, item in enumerate(relevant_raw)
                    ]
                    labels = [x for x in labels if float(x.get("grade") or 0.0) > 0.0]

                    matched_ids = []
                    hit_grades = []
                    for hit in hits:
                        hit_payload = {
                            "video_id": hit.video_id,
                            "segment_id": hit.segment_id,
                            "start_sec": hit.start_sec,
                            "end_sec": hit.end_sec
                        }
                        label_id, grade = _match_hit(hit_payload, labels, iou_threshold=iou_threshold)
                        matched_ids.append(label_id)
                        hit_grades.append(float(grade))

                    metrics = _metrics_for_query(hit_grades, matched_ids, labels, ks=[1, 5])
                    
                    p1 = float(metrics.get("p@1", 0.0))
                    p5 = float(metrics.get("p@5", 0.0))
                    r5 = float(metrics.get("recall@5", 0.0))
                    mrr = float(metrics.get("mrr@5", 0.0) or metrics.get("mrr", 0.0))
                    ndcg5 = float(metrics.get("ndcg@5", 0.0))

                    metrics_accum["p@1"].append(p1)
                    metrics_accum["p@5"].append(p5)
                    metrics_accum["recall@5"].append(r5)
                    metrics_accum["mrr"].append(mrr)
                    metrics_accum["ndcg@5"].append(ndcg5)

                    top = hits[0] if hits else None
                    by_query_rows.append([
                        dataset_id, str(search_video_id or ""), query_id, query_text,
                        str(query_obj.get("language") or defaults.get("language") or "en"),
                        method, var_name, p1, p5, r5, mrr, ndcg5, round(latency, 2),
                        str(top.video_id if top else ""), str(top.segment_id or "") if top else "",
                        float(top.start_sec) if top else "", float(top.end_sec) if top else "",
                        str(top.description or "") if top else "", str(matched_ids[0] or "") if matched_ids else "",
                    ])

                except Exception as exc:
                    print(f"Error executing query {query_id} for variant={var_name}, method={method}: {exc}")
                    failures_rows.append([query_id, var_name, method, str(exc)])

            if latencies:
                mean_p1 = round(mean(metrics_accum["p@1"]), 4)
                mean_p5 = round(mean(metrics_accum["p@5"]), 4)
                mean_r5 = round(mean(metrics_accum["recall@5"]), 4)
                mean_mrr = round(mean(metrics_accum["mrr"]), 4)
                mean_ndcg5 = round(mean(metrics_accum["ndcg@5"]), 4)

                summary_rows.append([
                    var_name, method, mean_p1, mean_p5, mean_r5, mean_mrr, mean_ndcg5
                ])

                mean_lat = round(mean(latencies), 2)
                p95_lat = round(np.percentile(latencies, 95), 2)
                latency_rows.append([
                    var_name, method, mean_lat, p95_lat
                ])

    # Write CSV files
    with open(out_dir / "retrieval_results_by_query.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "dataset_id", "video_id", "query_id", "query_text", "language", "method", "variant",
            "P@1", "P@5", "Recall@5", "MRR", "nDCG@5", "latency_ms", "top_result_video",
            "top_result_segment_id", "top_result_start_sec", "top_result_end_sec",
            "top_result_description", "matched_label_id",
        ])
        writer.writerows(by_query_rows)

    with open(out_dir / "retrieval_results_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["variant", "method", "mean_p@1", "mean_p@5", "mean_recall@5", "mean_mrr", "mean_ndcg@5"])
        writer.writerows(summary_rows)

    with open(out_dir / "retrieval_latency_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["variant", "method", "mean_latency_ms", "p95_latency_ms"])
        writer.writerows(latency_rows)

    with open(out_dir / "retrieval_failures.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["query_id", "variant", "method", "error"])
        writer.writerows(failures_rows)

    print("Retrieval baseline evaluation complete.")

if __name__ == "__main__":
    main()
