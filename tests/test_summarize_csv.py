"""summarize_csv.py: row counts, column stats and the files it writes, on a small
CSV shaped like the phenovision annotation exports."""

import json

import pandas as pd
import pytest

import summarize_csv


@pytest.fixture
def annotations(tmp_path, monkeypatch):
    # Small chunks so the streaming path crosses chunk boundaries.
    monkeypatch.setattr(summarize_csv, "CHUNK", 7)
    monkeypatch.setattr(summarize_csv, "SAMPLE_ROWS", 10)
    rows = [dict(observation_id=i // 2, file_name=f"data/imgs/{i}.jpg",
                 trait=["flowers", "fruits", "leaves"][i % 3],
                 score=f"{0.5 + i / 100:.2f}", note="" if i % 4 else "x")
            for i in range(50)]
    path = tmp_path / "ann.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    (tmp_path / "root" / "imgs").mkdir(parents=True)
    for i in range(0, 50, 5):
        (tmp_path / "root" / "imgs" / f"{i}.jpg").touch()
    return path, tmp_path


def test_counts_and_columns(annotations):
    path, tmp = annotations
    s, report = summarize_csv.summarize(path, tmp / "out", image_root=str(tmp / "root"))
    assert s["rows"] == 50
    assert s["column_names"] == ["observation_id", "file_name", "trait", "score", "note"]
    col = {c["column"]: c for c in s["columns"]}
    assert col["observation_id"]["distinct"] == 25
    assert col["trait"]["distinct"] == 3
    assert dict(col["trait"]["top_values"])["flowers"] == 17
    assert col["score"]["type"] == "numeric"
    assert col["score"]["min"] == pytest.approx(0.5)
    assert col["score"]["max"] == pytest.approx(0.99)
    assert col["trait"]["type"] == "text"
    assert col["note"]["nulls"] == 37
    assert "**rows (observations):** 50" in report


def test_writes_files(annotations):
    path, tmp = annotations
    out = tmp / "out"
    summarize_csv.summarize(path, out, image_root=str(tmp / "root"))
    for f in ["summary.md", "summary.json", "columns.csv", "head.csv", "sample.csv"]:
        assert (out / f).exists()
    assert len(pd.read_csv(out / "sample.csv")) == 10
    check = json.loads((out / "summary.json").read_text())["path_check"]["file_name"]
    assert check["checked"] == 10
    assert 0 <= check["exist"] <= 10


def test_high_cardinality_is_capped(annotations, monkeypatch):
    path, tmp = annotations
    monkeypatch.setattr(summarize_csv, "DISTINCT_CAP", 5)
    s, _ = summarize_csv.summarize(path, tmp / "out")
    col = {c["column"]: c for c in s["columns"]}
    assert col["file_name"]["distinct"] == ">5"
    assert col["observation_id"]["distinct"] == 25     # ids stay exact
