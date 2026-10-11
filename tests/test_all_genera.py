"""
The all-genera run without a GPU: build_genus_index.py picks N flower-detected
photos per angiosperm genus and finds their files; all_genera_report.py turns
the run's results into the metadata CSV and the low / median / high figures;
plot_conf_vs_probability.py draws the scatter.
"""

import csv
import gzip
import sys
from argparse import Namespace

import numpy as np
import pandas as pd
import pytest
from PIL import Image

import build_genus_index
import segment

HEADER = ("dataSource,scientificName,trait,family,year,dayOfYear,latitude,longitude,"
          "observedMetadataUrl,occurrenceID,genus,date,recordedBy,"
          "coordinateUncertaintyInMeters,modelUri,accuracyExcludingUncertainFamily,"
          "observedImageUrl,predictionClass,countImages,countFamily,certainty,"
          "predictionProbability,proportionCertaintyFamily,accuracyFamily,"
          "observedImageGuid,basisOfRecord,annotationID,verbatimTrait,"
          "annotationMethod").split(",")

TAXA = [  # taxon_id, ancestry, rank_level, rank, name, active
    (47125, "48460/47126/211194", 57, "subphylum", "Angiospermae", "true"),
    (500, "48460/47126/211194/47125/47124", 30, "family", "Asteraceae", "true"),
    (501, "48460/47126/211194/47125/47124/500", 20, "genus", "Rudbeckia", "true"),
    (502, "48460/47126/211194/47125/47124/500", 20, "genus", "Bellis", "true"),
    (600, "48460/47126/211194/136329", 30, "family", "Pinaceae", "true"),
    (601, "48460/47126/211194/136329/600", 20, "genus", "Pinus", "true"),
]


def _row(photo_id, genus, family, p, trait="flower present", cls="Detected"):
    values = dict.fromkeys(HEADER, "")
    values.update(scientificName=f"{genus} sp", trait=trait, family=family, genus=genus,
                  occurrenceID=f"obs-{photo_id}", predictionClass=cls, certainty="High",
                  predictionProbability=str(p), observedImageGuid=str(photo_id))
    return values


@pytest.fixture
def world(tmp_path):
    root = tmp_path / "share"
    medium = root / "phenobase_inat_data" / "images" / "medium"
    rows = []
    # Rudbeckia: 6 flower photos on disk (cap is 4); Bellis: 2 (one not on disk);
    # Newgenus: not in iNat by name, but its family is an angiosperm family;
    # Pinus: not an angiosperm.
    spec = ([(100 + i, "Rudbeckia", "Asteraceae") for i in range(6)]
            + [(200, "Bellis", "Asteraceae"), (201, "Bellis", "Asteraceae")]
            + [(300, "Newgenus", "Asteraceae"), (400, "Pinus", "Pinaceae")])
    for n, (pid, genus, family) in enumerate(spec):
        rows.append(_row(pid, genus, family, round(0.6 + 0.04 * n, 2)))
        if pid != 201:
            folder = medium / f"batch_{pid % 3 + 1}"
            folder.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (60, 40), (200, 160, 20)).save(folder / f"{pid}.webp")
    rows.append(_row(105, "Rudbeckia", "Asteraceae", 0.1, cls="Not Detected"))
    rows.append(_row(900, "Rudbeckia", "Asteraceae", 0.9, trait="fruit present"))

    annotations = tmp_path / "annotations.csv"
    with open(annotations, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADER)
        writer.writeheader()
        writer.writerows(rows)
    taxa = tmp_path / "data" / "taxa.csv.gz"
    taxa.parent.mkdir()
    with gzip.open(taxa, "wt", encoding="utf-8") as handle:
        handle.write("taxon_id\tancestry\trank_level\trank\tname\tactive\n")
        for t in TAXA:
            handle.write("\t".join(str(v) for v in t) + "\n")
    return Namespace(tmp=tmp_path, root=root, annotations=annotations, taxa=taxa,
                     index=tmp_path / "data" / "all_genera_index.parquet")


def _build_index(world, monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "build_genus_index.py", "--annotations", str(world.annotations),
        "--image-root", str(world.root), "--taxa", str(world.taxa),
        "--path-cache", str(world.tmp / "data" / "image_paths_all.parquet"),
        "--per-genus", "4", "--out", str(world.index), "--chunksize", "5"])
    build_genus_index.main()
    return pd.read_parquet(world.index)


def test_build_genus_index(world, monkeypatch):
    index = _build_index(world, monkeypatch)
    per = index.groupby("genus").size().to_dict()
    assert per == {"Rudbeckia": 4, "Bellis": 1, "Newgenus": 1}      # no Pinus; cap 4
    assert set(index.loc[index.genus == "Rudbeckia", "photo_id"]) <= set(range(100, 106))
    assert index.loc[index.genus == "Rudbeckia", "taxon_id"].eq(501).all()
    assert index["file_name"].str.startswith("data/phenobase_inat_data/images/medium/").all()

    genera = pd.read_csv(world.index.with_name("all_genera_index_genera.csv")).set_index("genus")
    assert not genera.loc["Pinus", "angiosperm"] and genera.loc["Pinus", "selected"] == 0
    assert genera.loc["Bellis", "flower_detected"] == 2 and genera.loc["Bellis", "with_image"] == 1

    # A valid --local-index, and the cached listing is reused on a second build.
    photos = segment.load_local_photos(world.index, world.root, None, 0, 1)
    assert all(segment.resolve_local(p) for p in photos["local_path"])
    assert len(_build_index(world, monkeypatch)) == len(index)


class _Saver:
    def __init__(self, results):
        self.results = results

    def record(self, key, fields):
        self.results.write(**fields)


def test_report_and_plot(world, monkeypatch):
    index = _build_index(world, monkeypatch)
    out = world.tmp / "results" / "all_genera_flower"
    results = segment.Results(out / "results_shard_000.csv", out / "errors_shard_000.txt")
    args = Namespace(masks_only=True, discard_outputs=False, prompt="flower", mask_format="npz")
    photos = segment.load_local_photos(world.index, world.root, None, 0, 1).to_dict("records")
    expected = {}
    for n, row in enumerate(photos):
        image = Image.open(row["local_path"]).convert("RGB")
        # Seven masks with known scores; the last photo (Newgenus) gets none.
        count = 0 if n == len(photos) - 1 else 7
        masks = []
        for i in range(count):
            m = np.zeros((40, 60), np.uint8)
            m[2 + 5 * i:6 + 5 * i, 5:30] = 1
            masks.append(m)
        scores = [0.99 - 0.01 * i - 0.002 * n for i in range(count)]
        expected[row["photo_id"]] = np.mean(scores[:5]) if scores else None
        common = {"photo_id": row["photo_id"], "taxon_id": row["taxon_id"],
                  "taxon_name": row["taxon_name"], "row_index": row["row_index"], "shard": 0,
                  "batch": row["batch"], "image_path": row["local_path"],
                  "width": 60, "height": 40}
        stages = {"prep_s": "0.01", "infer_s": "0.02", "post_s": "0.001"}
        segment._save_photo(row, image, segment._image_path(out, row), masks, scores,
                            stages, common, out, args, _Saver(results))

    import all_genera_report
    export = world.tmp / "export" / "all_genera"
    monkeypatch.setattr(sys, "argv", ["all_genera_report.py", str(out), "--index",
                                      str(world.index), "--export", str(export),
                                      "--workers", "1"])
    all_genera_report.main()

    meta = pd.read_csv(export / "all_genera_segment_metadata.csv")
    assert len(meta) == len(index) and (out / "all_genera_segment_metadata.csv").exists()
    for _, r in meta.iterrows():                 # mean of the five highest scores
        want = expected[r.photo_id]
        assert (pd.isna(r.conf_top5_mean) if want is None
                else abs(r.conf_top5_mean - want) < 1e-4)
    assert meta["prediction_probability"].notna().all()

    rud = meta[(meta.genus == "Rudbeckia") & meta.conf_top5_mean.notna()].sort_values(
        "conf_top5_mean")
    picked = meta[meta.genus == "Rudbeckia"].set_index("selected")["photo_id"].to_dict()
    assert picked["low"] == rud.iloc[0].photo_id and picked["high"] == rud.iloc[-1].photo_id
    assert picked["median"] == rud.iloc[(len(rud) - 1) // 2].photo_id
    # A genus with one detected photo gets only 'high'.
    assert list(meta.loc[meta.genus == "Bellis", "selected"]) == ["high"]

    figures = sorted(p.name for p in (export / "figures" / "Asteraceae").glob("*.jpg"))
    assert len(figures) == 4                      # Rudbeckia x3, Bellis x1
    # Newgenus's only photo had no detection: in the CSV, but nothing to draw.
    assert list(meta.loc[meta.genus == "Newgenus", "selected"].fillna("")) == [""]
    assert any(f.startswith("Rudbeckia_median_") for f in figures)
    assert Image.open(export / "figures" / "Asteraceae" / figures[0]).format == "JPEG"

    import plot_conf_vs_probability
    monkeypatch.setattr(sys, "argv", ["plot_conf_vs_probability.py",
                                      str(export / "all_genera_segment_metadata.csv")])
    plot_conf_vs_probability.main()
    assert (export / "segment_conf_vs_prediction_probability.png").stat().st_size > 10_000
