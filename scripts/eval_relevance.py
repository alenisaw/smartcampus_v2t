"""
Offline retrieval relevance evaluator for SmartCampus experiment runs.

Features:
- Loads a local query+labels dataset.
- Evaluates retrieval on one or more targets (profile/variant/video).
- Computes P@K, Recall@K, nDCG@K, and MRR.
- Supports targets from an experiment manifest produced by run_experiment_matrix.py.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.deps import get_backend_paths, load_cfg_and_raw, now_ts
from src.search.index_builder import load_index, resolve_index_dir, search_config_fingerprint
from src.search.query_engine import QueryEngine
from src.utils.video_store import (
    list_videos,
    read_segments,
    segments_path,
)


def _resolve_path(arg: str) -> Path:
    path = Path(arg)
    if path.is_absolute():
        return path
    return (Path.cwd() / path).resolve()


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        if not path.exists():
            return {}
        text = path.read_text(encoding="utf-8-sig")
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _parse_csv_tokens(value: str) -> List[str]:
    items: List[str] = []
    for part in str(value or "").split(","):
        token = str(part).strip()
        if token:
            items.append(token)
    return items


def _parse_ks(value: str) -> List[int]:
    out: List[int] = []
    for part in _parse_csv_tokens(value):
        try:
            k = int(part)
        except Exception:
            continue
        if k > 0:
            out.append(k)
    out = sorted(set(out))
    return out or [1, 3, 5, 10]


def _slug(text: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(text or "").strip())
    cleaned = cleaned.strip("._-")
    return cleaned or "relevance_eval"


def _interval_iou(a0: float, a1: float, b0: float, b1: float) -> float:
    a_start, a_end = min(a0, a1), max(a0, a1)
    b_start, b_end = min(b0, b1), max(b0, b1)
    inter = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    union = max(a_end, b_end) - min(a_start, b_start)
    if union <= 0:
        return 0.0
    return float(inter / union)


def _dcg(grades: Sequence[float], k: int) -> float:
    score = 0.0
    for idx, grade in enumerate(grades[: max(0, int(k))], start=1):
        denom = math.log2(float(idx) + 1.0)
        score += float((2.0 ** float(grade) - 1.0) / denom)
    return float(score)


def _mean_or_none(values: Iterable[Optional[float]]) -> Optional[float]:
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return None
    return float(mean(clean))


def _coerce_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    try:
        text = str(value or "").strip()
        if not text:
            return default
        return float(text)
    except Exception:
        return default


def _query_matches_target(query: Dict[str, Any], target: Dict[str, Any]) -> bool:
    q_profile = str(query.get("profile") or "").strip().lower()
    q_variant = str(query.get("variant") or "").strip().lower()
    q_video = str(query.get("video_id") or "").strip()

    t_profile = str(target.get("profile") or "").strip().lower()
    t_variant = str(target.get("variant") or "").strip().lower()
    t_video = str(target.get("video_id") or "").strip()

    if q_profile and q_profile != t_profile:
        return False
    if q_variant and q_variant != t_variant:
        return False
    if q_video and q_video != t_video:
        return False
    return True


def _normalize_relevance_item(item: Dict[str, Any], idx: int, query_video_id: str) -> Dict[str, Any]:
    label_id = str(item.get("label_id") or f"rel_{idx:03d}")
    segment_id = str(item.get("segment_id") or "").strip() or None
    video_id = str(item.get("video_id") or query_video_id or "").strip() or None
    start_sec = _coerce_float(item.get("start_sec"))
    end_sec = _coerce_float(item.get("end_sec"))
    grade = _coerce_float(item.get("grade"), default=1.0)
    if grade is None:
        grade = 1.0
    return {
        "label_id": label_id,
        "segment_id": segment_id,
        "video_id": video_id,
        "start_sec": start_sec,
        "end_sec": end_sec,
        "grade": max(0.0, float(grade)),
    }


def _match_hit(
    hit: Dict[str, Any],
    labels: Sequence[Dict[str, Any]],
    *,
    iou_threshold: float,
) -> Tuple[Optional[str], float]:
    hit_segment_id = str(hit.get("segment_id") or "").strip() or None
    hit_video_id = str(hit.get("video_id") or "").strip() or None
    hit_start = _coerce_float(hit.get("start_sec"), default=0.0) or 0.0
    hit_end = _coerce_float(hit.get("end_sec"), default=0.0) or 0.0

    best_label: Optional[str] = None
    best_grade = 0.0
    best_iou = 0.0

    for label in labels:
        label_video = str(label.get("video_id") or "").strip() or None
        if label_video and hit_video_id and label_video != hit_video_id:
            continue

        grade = float(label.get("grade") or 0.0)
        if grade <= 0:
            continue

        segment_id = label.get("segment_id")
        if segment_id and hit_segment_id and str(segment_id) == str(hit_segment_id):
            if grade > best_grade:
                best_label = str(label.get("label_id") or "")
                best_grade = grade
                best_iou = 1.0
            continue

        label_start = _coerce_float(label.get("start_sec"))
        label_end = _coerce_float(label.get("end_sec"))
        if label_start is None or label_end is None:
            continue
        iou = _interval_iou(hit_start, hit_end, float(label_start), float(label_end))
        if iou < float(iou_threshold):
            continue
        if (grade > best_grade) or (grade == best_grade and iou > best_iou):
            best_label = str(label.get("label_id") or "")
            best_grade = grade
            best_iou = iou

    return best_label, float(best_grade)


def _metrics_for_query(
    hit_grades: Sequence[float],
    matched_label_ids: Sequence[Optional[str]],
    labels: Sequence[Dict[str, Any]],
    ks: Sequence[int],
) -> Dict[str, Any]:
    total_relevant_labels = len([x for x in labels if float(x.get("grade") or 0.0) > 0.0])
    ideal_grades = sorted([float(x.get("grade") or 0.0) for x in labels if float(x.get("grade") or 0.0) > 0.0], reverse=True)

    metrics: Dict[str, Any] = {
        "relevant_total": int(total_relevant_labels),
        "hits_total": int(len(hit_grades)),
    }

    first_rel_rank: Optional[int] = None
    for rank, grade in enumerate(hit_grades, start=1):
        if float(grade) > 0.0:
            first_rel_rank = rank
            break
    metrics["mrr"] = float(1.0 / first_rel_rank) if first_rel_rank is not None else 0.0

    for k in ks:
        kk = int(k)
        top_grades = [float(x) for x in hit_grades[:kk]]
        top_label_ids = [x for x in matched_label_ids[:kk] if x]

        relevant_hits = sum(1 for g in top_grades if float(g) > 0.0)
        precision = float(relevant_hits / float(kk)) if kk > 0 else 0.0

        unique_rel = len(set(str(x) for x in top_label_ids))
        recall = float(unique_rel / float(total_relevant_labels)) if total_relevant_labels > 0 else 0.0

        dcg = _dcg(top_grades, kk)
        idcg = _dcg(ideal_grades, kk)
        ndcg = float(dcg / idcg) if idcg > 0 else 0.0

        mrr_k = 0.0
        for rank, grade in enumerate(top_grades, start=1):
            if grade > 0.0:
                mrr_k = float(1.0 / rank)
                break

        metrics[f"p@{kk}"] = precision
        metrics[f"recall@{kk}"] = recall
        metrics[f"ndcg@{kk}"] = ndcg
        metrics[f"mrr@{kk}"] = mrr_k

    return metrics


def _targets_from_experiment(path: Path) -> List[Dict[str, Any]]:
    state = _read_json(path)
    out: List[Dict[str, Any]] = []
    seen: set[Tuple[str, str, str, int, str]] = set()

    for item in state.get("results", []) if isinstance(state.get("results"), list) else []:
        if not isinstance(item, dict):
            continue
        if str(item.get("status") or "").strip().lower() != "done":
            continue
        profile = str(item.get("profile") or "").strip().lower() or "main"
        video_id = str(item.get("video_id") or "").strip()
        run_idx = int(item.get("run_idx") or 1)
        unit_key = str(item.get("unit_key") or "")

        variant_results = item.get("variant_results")
        if isinstance(variant_results, dict) and variant_results:
            for variant in variant_results.keys():
                variant_token = str(variant or "").strip().lower()
                key = (video_id, profile, variant_token, run_idx, unit_key)
                if key in seen:
                    continue
                seen.add(key)
                out.append(
                    {
                        "target_id": f"{unit_key}__{variant_token}",
                        "video_id": video_id,
                        "profile": profile,
                        "variant": variant_token,
                        "run_idx": run_idx,
                        "source": "experiment_manifest",
                    }
                )
            continue

        variant = str(item.get("variant") or "").strip().lower()
        key = (video_id, profile, variant, run_idx, unit_key)
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "target_id": unit_key or f"{profile}:{variant or 'base'}:{video_id}:run{run_idx}",
                "video_id": video_id,
                "profile": profile,
                "variant": variant,
                "run_idx": run_idx,
                "source": "experiment_manifest",
            }
        )

    return out


def _targets_from_matrix(
    *,
    profiles: Sequence[str],
    variants: Sequence[str],
    video_ids: Sequence[str],
    runs: int,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for run_idx in range(1, max(1, int(runs)) + 1):
        for video_id in video_ids:
            for profile in profiles:
                profile_token = str(profile).strip().lower() or "main"
                variants_list = list(variants) if variants else [""]
                for variant in variants_list:
                    variant_token = str(variant).strip().lower()
                    out.append(
                        {
                            "target_id": f"{profile_token}:{variant_token or 'base'}:{video_id}:run{run_idx}",
                            "video_id": str(video_id),
                            "profile": profile_token,
                            "variant": variant_token,
                            "run_idx": run_idx,
                            "source": "manual_matrix",
                        }
                    )
    return out


def _build_template(path: Path) -> Dict[str, Any]:
    cfg, raw = load_cfg_and_raw()
    paths = get_backend_paths(cfg, raw)
    videos = list_videos(Path(paths.videos_dir))

    sample_video_id = ""
    sample_segment_id = "seg_000001"
    if videos:
        sample_video_id = str(videos[0].get("video_id") or "")
    if sample_video_id:
        seg_path = segments_path(Path(paths.videos_dir), sample_video_id, "en", variant=None)
        rows = read_segments(seg_path)
        if rows and isinstance(rows[0], dict):
            sample_segment_id = str(rows[0].get("segment_id") or sample_segment_id)

    payload = {
        "version": "1.0",
        "defaults": {
            "language": "en",
            "top_k": 20,
            "iou_threshold": 0.5,
        },
        "queries": [
            {
                "query_id": "q_example_01",
                "query": "person walking near building",
                "video_id": sample_video_id,
                "language": "en",
                "filters": {},
                "relevant": [
                    {
                        "label_id": "rel_001",
                        "segment_id": sample_segment_id,
                        "grade": 3,
                    }
                ],
            }
        ],
    }
    _write_json(path, payload)
    return payload


def _engine_key(profile: str, variant: str, language: str) -> Tuple[str, str, str]:
    return (str(profile).strip().lower(), str(variant).strip().lower(), str(language).strip().lower())


def _select_index_dir(
    *,
    indexes_dir: Path,
    config_fingerprint: str,
    language: str,
    variant: str,
) -> Tuple[Path, str, bool]:
    """Select an index directory, falling back to the latest available one."""

    exact = resolve_index_dir(
        base_index_dir=Path(indexes_dir),
        config_fingerprint=config_fingerprint,
        language=language,
        variant=(variant or None),
    )
    if (exact / "doc_ids.json").exists():
        return exact, str(config_fingerprint), False

    lang_root = Path(indexes_dir) / str(language).strip().lower()
    if variant:
        lang_root = lang_root / "variants" / str(variant).strip().lower()
    candidates = []
    if lang_root.exists() and lang_root.is_dir():
        for item in lang_root.iterdir():
            if not item.is_dir():
                continue
            if not (item / "doc_ids.json").exists():
                continue
            try:
                mt = float(item.stat().st_mtime)
            except Exception:
                mt = 0.0
            candidates.append((mt, item))
    if not candidates:
        raise RuntimeError(
            f"Index directory not found for language={language}, variant={variant or 'base'}. "
            f"Expected {exact}"
        )
    candidates.sort(key=lambda pair: pair[0], reverse=True)
    chosen = candidates[0][1]
    return chosen, str(chosen.name), True


def evaluate_relevance(
    *,
    labels_path: Path,
    out_dir: Path,
    ks: Sequence[int],
    experiment_manifest: Optional[Path],
    profiles: Sequence[str],
    variants: Sequence[str],
    video_ids: Sequence[str],
    runs: int,
    default_top_k: int,
) -> Dict[str, Any]:
    dataset = _read_json(labels_path)
    queries_raw = dataset.get("queries")
    if not isinstance(queries_raw, list) or not queries_raw:
        raise RuntimeError(f"Dataset {labels_path} must contain non-empty 'queries'")

    defaults = dataset.get("defaults") if isinstance(dataset.get("defaults"), dict) else {}
    default_lang = str(defaults.get("language") or "en").strip().lower() or "en"
    iou_threshold = float(_coerce_float(defaults.get("iou_threshold"), default=0.5) or 0.5)
    base_top_k = int(_coerce_float(defaults.get("top_k"), default=float(default_top_k)) or default_top_k)

    targets: List[Dict[str, Any]]
    if experiment_manifest is not None:
        targets = _targets_from_experiment(experiment_manifest)
    else:
        inferred_videos = [str(v).strip() for v in video_ids if str(v).strip()]
        if not inferred_videos:
            inferred_videos = sorted(
                {
                    str(item.get("video_id") or "").strip()
                    for item in queries_raw
                    if isinstance(item, dict) and str(item.get("video_id") or "").strip()
                }
            )
        if not inferred_videos:
            raise RuntimeError("No target videos resolved. Provide --video-ids or include video_id in queries.")
        targets = _targets_from_matrix(
            profiles=profiles or ["main"],
            variants=variants,
            video_ids=inferred_videos,
            runs=runs,
        )

    if not targets:
        raise RuntimeError("No evaluation targets resolved.")

    engine_cache: Dict[Tuple[str, str, str], QueryEngine] = {}
    rows: List[Dict[str, Any]] = []

    for target in targets:
        target_profile = str(target.get("profile") or "main").strip().lower() or "main"
        target_variant = str(target.get("variant") or "").strip().lower()
        target_video = str(target.get("video_id") or "").strip()

        for query_idx, query_obj in enumerate(queries_raw, start=1):
            if not isinstance(query_obj, dict):
                continue
            if not _query_matches_target(query_obj, target):
                continue

            query_id = str(query_obj.get("query_id") or f"q_{query_idx:03d}")
            query_text = str(query_obj.get("query") or "").strip()
            if not query_text:
                continue
            language = str(query_obj.get("language") or default_lang).strip().lower() or "en"
            top_k = int(_coerce_float(query_obj.get("top_k"), default=float(base_top_k)) or base_top_k)
            top_k = max(top_k, max(int(k) for k in ks))

            key = _engine_key(target_profile, target_variant, language)
            engine = engine_cache.get(key)
            if engine is None:
                cfg, _raw = load_cfg_and_raw(profile=target_profile, variant=(target_variant or None))
                cfg_fp = search_config_fingerprint(cfg)
                selected_index_dir, selected_fp, fp_fallback = _select_index_dir(
                    indexes_dir=Path(cfg.paths.indexes_dir),
                    config_fingerprint=cfg_fp,
                    language=language,
                    variant=target_variant,
                )
                loaded_index = load_index(selected_index_dir)
                engine = QueryEngine(
                    index=loaded_index,
                    config_path=Path(cfg.config_path),
                    index_dir=Path(cfg.paths.indexes_dir),
                    config_fingerprint=selected_fp,
                    language=language,
                    variant=(target_variant or None),
                    w_bm25=float(getattr(cfg.search, "w_bm25", 0.45)),
                    w_dense=float(getattr(cfg.search, "w_dense", 0.55)),
                    candidate_k_sparse=int(getattr(cfg.search, "candidate_k_sparse", 200)),
                    candidate_k_dense=int(getattr(cfg.search, "candidate_k_dense", 200)),
                    embedding_backend=str(getattr(cfg.search, "embedding_backend", "auto")),
                    fallback_embed_model_name=str(getattr(cfg.search, "embed_model_name", "")),
                    rerank_enabled=bool(getattr(cfg.search, "rerank_enabled", True)),
                    rerank_top_k=int(getattr(cfg.search, "rerank_top_k", 30)),
                    reranker_model_name=str(getattr(cfg.search, "reranker_model_id", "")),
                    reranker_backend=str(getattr(cfg.search, "reranker_backend", "auto")),
                    fusion=str(getattr(cfg.search, "fusion", "rrf")),
                    rrf_k=int(getattr(cfg.search, "rrf_k", 60)),
                    dedupe_mode=str(getattr(cfg.search, "dedupe_mode", "overlap")),
                    dedupe_tol_sec=float(getattr(cfg.search, "dedupe_tol_sec", 1.0)),
                    dedupe_overlap_thr=float(getattr(cfg.search, "dedupe_overlap_thr", 0.7)),
                    normalize_text=bool(getattr(cfg.search, "normalize_text", True)),
                    lemmatize=bool(getattr(cfg.search, "lemmatize", False)),
                    embed_model_name=str(getattr(cfg.search, "embedding_model_id", "")),
                )
                setattr(engine, "_resolved_index_dir", str(selected_index_dir))
                setattr(engine, "_index_fp_fallback", bool(fp_fallback))
                engine_cache[key] = engine

            search_video_id = str(query_obj.get("video_id") or target_video or "").strip() or None
            filters = query_obj.get("filters")
            if not isinstance(filters, dict):
                filters = {}
            if target_variant:
                filters = dict(filters)
                filters.setdefault("variant", target_variant)

            t0 = time.perf_counter()
            hits, search_stats = engine.search(
                query=query_text,
                top_k=top_k,
                video_id=search_video_id,
                filters=filters,
                dedupe=bool(query_obj.get("dedupe", True)),
                return_stats=True,
            )
            latency_ms = float((time.perf_counter() - t0) * 1000.0)

            relevant_raw = query_obj.get("relevant")
            if not isinstance(relevant_raw, list):
                relevant_raw = []
            labels = [
                _normalize_relevance_item(item, idx + 1, str(search_video_id or ""))
                for idx, item in enumerate(relevant_raw)
                if isinstance(item, dict)
            ]
            labels = [x for x in labels if float(x.get("grade") or 0.0) > 0.0]

            matched_ids: List[Optional[str]] = []
            hit_grades: List[float] = []
            for hit in hits:
                hit_payload = {
                    "video_id": str(getattr(hit, "video_id", "") or ""),
                    "segment_id": str(getattr(hit, "segment_id", "") or ""),
                    "start_sec": float(getattr(hit, "start_sec", 0.0) or 0.0),
                    "end_sec": float(getattr(hit, "end_sec", 0.0) or 0.0),
                }
                label_id, grade = _match_hit(hit_payload, labels, iou_threshold=iou_threshold)
                matched_ids.append(label_id)
                hit_grades.append(float(grade))

            metrics = _metrics_for_query(hit_grades, matched_ids, labels, ks=ks)
            row: Dict[str, Any] = {
                "target_id": str(target.get("target_id") or ""),
                "source": str(target.get("source") or ""),
                "video_id": target_video,
                "profile": target_profile,
                "variant": target_variant,
                "run_idx": int(target.get("run_idx") or 1),
                "query_id": query_id,
                "query": query_text,
                "language": language,
                "top_k": int(top_k),
                "filters": filters,
                "search_video_id": search_video_id,
                "latency_ms": latency_ms,
                "search_stats": search_stats if isinstance(search_stats, dict) else {},
                "index_dir": str(getattr(engine, "_resolved_index_dir", "")),
                "index_fp_fallback": bool(getattr(engine, "_index_fp_fallback", False)),
                **metrics,
            }
            rows.append(row)

    if not rows:
        raise RuntimeError("No query/target pairs were evaluated. Check labels and target filters.")

    grouped_by_target: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        grouped_by_target.setdefault(str(row.get("target_id") or ""), []).append(row)

    def aggregate(block: List[Dict[str, Any]]) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "queries": len(block),
            "mrr": _mean_or_none(row.get("mrr") for row in block),
            "latency_ms_mean": _mean_or_none(row.get("latency_ms") for row in block),
        }
        for k in ks:
            out[f"p@{k}"] = _mean_or_none(row.get(f"p@{k}") for row in block)
            out[f"recall@{k}"] = _mean_or_none(row.get(f"recall@{k}") for row in block)
            out[f"ndcg@{k}"] = _mean_or_none(row.get(f"ndcg@{k}") for row in block)
            out[f"mrr@{k}"] = _mean_or_none(row.get(f"mrr@{k}") for row in block)
        return out

    by_target = [
        {
            "target_id": target_id,
            "profile": str(items[0].get("profile") or ""),
            "variant": str(items[0].get("variant") or ""),
            "video_id": str(items[0].get("video_id") or ""),
            "run_idx": int(items[0].get("run_idx") or 1),
            "metrics": aggregate(items),
        }
        for target_id, items in sorted(grouped_by_target.items())
    ]

    overall = aggregate(rows)
    created_at = now_ts()
    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"relevance_eval_{ts}.json"
    csv_path = out_dir / f"relevance_eval_{ts}.csv"

    payload = {
        "created_at": created_at,
        "dataset_path": str(labels_path),
        "experiment_manifest": str(experiment_manifest) if experiment_manifest else None,
        "ks": [int(x) for x in ks],
        "rows_total": len(rows),
        "targets_total": len(by_target),
        "by_target": by_target,
        "overall": overall,
        "rows": rows,
    }
    _write_json(json_path, payload)

    csv_fields = [
        "target_id",
        "profile",
        "variant",
        "video_id",
        "run_idx",
        "query_id",
        "language",
        "top_k",
        "mrr",
        "latency_ms",
    ]
    for k in ks:
        csv_fields.extend([f"p@{k}", f"recall@{k}", f"ndcg@{k}", f"mrr@{k}"])

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in csv_fields})

    return {
        "json_path": str(json_path),
        "csv_path": str(csv_path),
        "overall": overall,
        "rows_total": len(rows),
        "targets_total": len(by_target),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline relevance evaluator for SmartCampus retrieval.")
    parser.add_argument("--labels", type=str, default="data/research/relevance/queries_labels.json", help="Path to query+labels dataset.")
    parser.add_argument("--out-dir", type=str, default="data/research/relevance", help="Directory for evaluation reports.")
    parser.add_argument("--ks", type=str, default="1,3,5,10", help="Comma-separated K values, for example '1,3,5,10'.")
    parser.add_argument("--default-top-k", type=int, default=20, help="Default search depth when query does not override top_k.")
    parser.add_argument("--experiment-manifest", type=str, default="", help="Path to experiment.json produced by run_experiment_matrix.py.")
    parser.add_argument("--profiles", type=str, default="main", help="Comma-separated profiles for manual target matrix.")
    parser.add_argument("--variants", type=str, default="", help="Comma-separated variants for manual target matrix.")
    parser.add_argument("--video-ids", nargs="*", default=[], help="Video ids for manual target matrix.")
    parser.add_argument("--runs", type=int, default=1, help="Number of run repetitions in manual target matrix.")
    parser.add_argument("--init-template", action="store_true", help="Create a labels template file and exit.")
    parser.add_argument("--force-template", action="store_true", help="Allow overwriting labels template when --init-template is used.")
    args = parser.parse_args()

    labels_path = _resolve_path(args.labels)
    out_dir = _resolve_path(args.out_dir)

    if args.init_template:
        if labels_path.exists() and not bool(args.force_template):
            raise RuntimeError(f"Labels file already exists: {labels_path}. Use --force-template to overwrite.")
        payload = _build_template(labels_path)
        print(f"Template written: {labels_path}")
        print(f"Queries: {len(payload.get('queries') or [])}")
        return

    experiment_manifest = _resolve_path(args.experiment_manifest) if str(args.experiment_manifest or "").strip() else None
    if experiment_manifest is not None and not experiment_manifest.exists():
        raise RuntimeError(f"Experiment manifest not found: {experiment_manifest}")

    result = evaluate_relevance(
        labels_path=labels_path,
        out_dir=out_dir,
        ks=_parse_ks(args.ks),
        experiment_manifest=experiment_manifest,
        profiles=_parse_csv_tokens(args.profiles),
        variants=_parse_csv_tokens(args.variants),
        video_ids=[str(x).strip() for x in args.video_ids if str(x).strip()],
        runs=max(1, int(args.runs)),
        default_top_k=max(1, int(args.default_top_k)),
    )
    print(f"Rows evaluated: {int(result['rows_total'])}")
    print(f"Targets evaluated: {int(result['targets_total'])}")
    print(f"JSON report: {result['json_path']}")
    print(f"CSV report: {result['csv_path']}")
    print(f"Overall metrics: {json.dumps(result['overall'], ensure_ascii=False)}")


if __name__ == "__main__":
    main()
