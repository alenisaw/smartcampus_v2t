import csv
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.experiments import rebuild_v08_indexes as indexes
from scripts.experiments import run_v08_ablations as ablations


def _write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def test_rebuild_indexes_writes_status_for_each_variant_language(tmp_path, monkeypatch):
    def loader(profile, variant):
        return SimpleNamespace(active_variant=variant, marker=f"{profile}:{variant}"), {}

    def builder(cfg, cfg_fp, language):
        return {"num_docs": 3, "dense_valid_count": 3, "ann_backend": "faiss", "ann_index_type": "hnsw",
                "embed_dim": 8, "time_sec": 1.25, "index_dir": str(tmp_path / language / str(cfg.active_variant))}

    monkeypatch.setattr(indexes, "search_config_fingerprint", lambda cfg: cfg.marker)
    rows = indexes.rebuild_indexes(profile="p", variants=["base", "no_merge"], languages=["en"],
                                   out_dir=tmp_path / "status", loader=loader, builder=builder)
    assert len(rows) == 2
    assert (tmp_path / "status" / "index_status.csv").is_file()
    assert rows[1]["config_fingerprint"] == "p:no_merge"


def test_rebuild_indexes_rejects_empty_index(tmp_path, monkeypatch):
    monkeypatch.setattr(indexes, "search_config_fingerprint", lambda cfg: "fp")
    with pytest.raises(indexes.IndexBuildError, match="Empty index"):
        indexes.rebuild_indexes(
            profile="p", variants=["base"], languages=["en"], out_dir=tmp_path,
            loader=lambda **kwargs: (SimpleNamespace(active_variant=None), {}),
            builder=lambda **kwargs: {"num_docs": 0, "index_dir": ""},
        )


def test_ablations_fail_before_running_tau_when_measurements_are_missing(tmp_path, monkeypatch):
    source = tmp_path / "retrieval.csv"
    _write_csv(source, [{"dataset_id": "avenue_full", "profile": "p", "variant": "base",
                         "config_fingerprint": "fp", "p@5": "1", "mrr": "1", "ndcg@5": "1", "latency_ms": "1"}])
    monkeypatch.setattr(ablations.subprocess, "run", lambda *args, **kwargs: pytest.fail("tau must not run"))
    with pytest.raises(ablations.AblationError, match="alpha column"):
        ablations.run_ablations(videos_dir=tmp_path, retrieval_by_query=source, out_dir=tmp_path / "out", taus=[0.9])


def test_ablations_aggregate_only_measured_rows(tmp_path, monkeypatch):
    source = tmp_path / "retrieval.csv"
    rows = []
    for alpha in ablations.ALPHAS:
        for rerank in ("false", "true"):
            rows.append({"dataset_id": "avenue_full", "profile": "p", "variant": "base", "config_fingerprint": "fp",
                         "alpha": str(alpha), "rerank_enabled": rerank, "p@5": "0.5", "mrr": "0.4",
                         "ndcg@5": "0.3", "latency_ms": "2.0"})
    _write_csv(source, rows)

    def fake_run(command, cwd):
        out = Path(command[command.index("--output") + 1])
        _write_csv(out / "tau_ablation_per_run.csv", [{"run_id": "r", "video_id": "avenue_1", "profile": "p",
                   "variant": "", "tau": "0.9", "final_segments": "2", "compression_ratio": "0.5"}])
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(ablations.subprocess, "run", fake_run)
    monkeypatch.setattr(ablations, "load_cfg_and_raw", lambda **kwargs: (SimpleNamespace(marker="p"), {}))
    monkeypatch.setattr(ablations, "search_config_fingerprint", lambda cfg: "fp")
    out = tmp_path / "out"
    ablations.run_ablations(videos_dir=tmp_path, retrieval_by_query=source, out_dir=out, taus=[0.9])
    assert (out / "threshold_ablation.csv").is_file()
    with (out / "alpha_ablation.csv").open(newline="", encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == 5
    with (out / "rerank_ablation.csv").open(newline="", encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == 2
