import gzip
import json
from pathlib import Path

import pytest

from scripts.experiments.generate_relevance_labels import (
    LabelError,
    build_template,
    export_labels,
    load_catalog,
    validate_annotations,
)


def _videos(tmp_path: Path) -> Path:
    videos = tmp_path / "videos"
    path = videos / "video-1" / "outputs" / "segments" / "en.jsonl.gz"
    path.parent.mkdir(parents=True)
    rows = [
        {"segment_id": "seg_000001", "start_sec": 0.0, "end_sec": 2.5, "description": "person walks"},
        {"segment_id": "seg_000002", "start_sec": 2.5, "end_sec": 5.0, "description": "person runs"},
    ]
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return videos


def _reviewed():
    return {
        "reviewed": True,
        "queries": [
            {
                "query_id": "q_en_1", "query": "person walking", "language": "en",
                "relevant": [{"label_id": "rel_1", "video_id": "video-1", "segment_id": "seg_000001", "grade": 3}],
            },
            {
                "query_id": "q_ru_1", "query": "человек идет", "language": "ru",
                "relevant": [{"label_id": "rel_2", "video_id": "video-1", "start_sec": 0.5, "end_sec": 2.0, "grade": 2}],
            },
        ],
    }


def test_template_is_grounded_and_assigns_no_grades(tmp_path):
    catalog = load_catalog(_videos(tmp_path))
    template = build_template(catalog, "en", "base")
    assert template["reviewed"] is False
    assert template["queries"] == []
    assert [row["segment_id"] for row in template["segment_catalog"]] == ["seg_000001", "seg_000002"]
    assert all("grade" not in row for row in template["segment_catalog"])


def test_reviewed_annotations_split_into_english_and_multilingual(tmp_path):
    catalog = load_catalog(_videos(tmp_path))
    annotations = tmp_path / "reviewed.json"
    annotations.write_text(json.dumps(_reviewed()), encoding="utf-8")
    out_en, out_multi = tmp_path / "en.json", tmp_path / "multi.json"
    export_labels(annotations, catalog, {}, out_en, out_multi)
    assert [q["query_id"] for q in json.loads(out_en.read_text())["queries"]] == ["q_en_1"]
    assert [q["query_id"] for q in json.loads(out_multi.read_text())["queries"]] == ["q_ru_1"]


def test_unreviewed_or_missing_segment_is_rejected(tmp_path):
    catalog = load_catalog(_videos(tmp_path))
    payload = _reviewed(); payload["reviewed"] = False
    with pytest.raises(LabelError, match="reviewed=true"):
        validate_annotations(payload, catalog, {})
    payload = _reviewed(); payload["queries"][0]["relevant"][0]["segment_id"] = "fabricated"
    with pytest.raises(LabelError, match="missing segment"):
        validate_annotations(payload, catalog, {})


def test_timestamp_must_lie_inside_a_real_segment(tmp_path):
    catalog = load_catalog(_videos(tmp_path))
    payload = _reviewed()
    label = payload["queries"][1]["relevant"][0]
    label["start_sec"], label["end_sec"] = 1.0, 4.0
    with pytest.raises(LabelError, match="outside all processed segments"):
        validate_annotations(payload, catalog, {})
