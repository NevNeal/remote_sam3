"""
All-genera run 2 without a GPU: the tiered index, segment.py's --max-masks,
the report (tier scatters, family histograms), the coverage plots and CSV, and
the packed masks round trip.
"""

import csv
import gzip
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

import segment
from test_all_genera import HEADER, TAXA, _row

RUN2 = Path(__file__).resolve().parents[1] / "all_genera" / "run2"
sys.path.insert(0, str(RUN2))

import build_index  # noqa: E402
import coverage  # noqa: E402
import pack_masks  # noqa: E402
import report  # noqa: E402


@pytest.fixture
def world(tmp_path):
    root = tmp_path / "share"
    medium = root / "phenobase_inat_data" / "images" / "medium" / "batch_1"
    medium.mkdir(parents=True)
    # Rudbeckia: 40 photos, probability rising with photo_id; Bellis: 5;
    # Pinus: not an angiosperm.
    spec = ([(1000 + i, "Rudbeckia", 0.5 + i / 100) for i in range(40)]
            + [(2000 + i, "Bellis", 0.6 + i / 50) for i in range(5)]
            + [(3000, "Pinus", 0.9)])
    rows = []
    for pid, genus, p in spec:
        family = "Pinaceae" if genus == "Pinus" else "Asteraceae"
        rows.append(_row(pid, genus, family, p))
        Image.new("RGB", (60, 40), (200, 160, 20)).save(medium / f"{pid}.webp")
    annotations = tmp_path / "annotations.csv"
    with open(annotations, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADER)
        writer.writeheader()
        writer.writerows(rows)
    taxa = tmp_path / "data" / "taxa.csv.gz"
    taxa.parent.mkdir()
    with gzip.open(taxa, "wt", encoding="utf-8") as handle:
        handle.write("taxon_id\tancestry\trank_level\trank\tname\tactive\n")
        for t in TAXA + [(510, "48460/47126/211194/47125/47124/500/501", 10, "species",
                          "Rudbeckia hirta", "true")]:
            handle.write("\t".join(str(v) for v in t) + "\n")
    return Namespace(tmp=tmp_path, root=root, annotations=annotations, taxa=taxa,
                     index=tmp_path / "data" / "all_genera_run2_index.parquet")


def _build(world, monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "build_index.py", "--annotations", str(world.annotations),
        "--image-root", str(world.root), "--taxa", str(world.taxa),
        "--path-cache", str(world.tmp / "data" / "image_paths_all.parquet"),
        "--out", str(world.index), "--chunksize", "7"])
    build_index.main()
    return pd.read_parquet(world.index)


def test_tiers(world, monkeypatch):
    index = _build(world, monkeypatch)
    rud = index[index.genus == "Rudbeckia"].set_index("photo_id")["pv_tier"]
    assert len(rud) == 30
    assert set(rud[rud == "low"].index) == set(range(1000, 1010))
    assert set(rud[rud == "high"].index) == set(range(1030, 1040))
    assert set(rud[rud == "median"].index) == set(range(1015, 1025))
    bellis = index[index.genus == "Bellis"].sort_values("photo_id")["pv_tier"].tolist()
    assert bellis == ["low", "low", "median", "median", "high"]
    assert "Pinus" not in set(index.genus)
    genera = pd.read_csv(world.index.with_name("all_genera_run2_index_genera.csv"))
    assert genera.set_index("genus").loc["Rudbeckia", "taxon_id"] == 501


class _Saver:
    def __init__(self, results):
        self.results = results

    def record(self, key, fields):
        self.results.write(**fields)


def _segment(world, out):
    """What segment.py would write at --max-masks 5. Bellis and Rudbeckia's five
    lowest photos get no masks."""
    results = segment.Results(out / "results_shard_000.csv", out / "errors_shard_000.txt")
    args = Namespace(masks_only=True, discard_outputs=False, prompt="flower",
                     mask_format="npz", max_masks=5)
    for n, row in enumerate(segment.load_local_photos(world.index, world.root, None, 0, 1)
                            .to_dict("records")):
        count = 0 if row["taxon_name"].startswith("Bellis") or row["photo_id"] < 1005 else 7
        masks = []
        for i in range(count):
            m = np.zeros((40, 60), bool)
            m[2 + 5 * i:6 + 5 * i, 5 + i:30 + n % 9] = True
            masks.append(m)
        scores = [0.95 - 0.05 * i - 0.001 * n for i in range(count)]
        common = {"photo_id": row["photo_id"], "taxon_id": row["taxon_id"],
                  "taxon_name": row["taxon_name"], "row_index": row["row_index"],
                  "shard": 0, "batch": row["batch"], "image_path": row["local_path"],
                  "width": 60, "height": 40}
        stages = {"prep_s": "0.01", "infer_s": "0.02", "post_s": "0.001"}
        masks, scores = masks[:args.max_masks], scores[:args.max_masks]
        segment._save_photo(row, Image.open(row["local_path"]).convert("RGB"),
                            segment._image_path(out, row), masks, scores, stages,
                            common, out, args, _Saver(results))


def test_max_masks_trims_in_handle(monkeypatch):
    seen = {}

    class Saver:
        def submit(self, fn, row, image, path, masks, scores, *rest):
            seen["scores"] = scores

    args = Namespace(max_masks=5, shard=0, host="h", gpu="g", discard_outputs=False)
    row = {"photo_id": 1, "taxon_id": 1, "taxon_name": "A b", "quality_grade": "",
           "latitude": "", "longitude": "", "photo_url": "", "row_index": 0,
           "batch": "batch_00001", "local_path": "/x.webp"}
    masks = [np.ones((2, 2), bool)] * 7
    scores = [0.9, 0.8, 0.7, 0.6, 0.55, 0.52, 0.51]
    segment._handle(row, Image.new("RGB", (2, 2)), None, {}, Path("/tmp"), None, None,
                    "cpu", args, None, {}, None, Saver(), outcome=(masks, scores, {}))
    assert seen["scores"] == scores[:5]


def test_report_coverage_pack(world, monkeypatch):
    _build(world, monkeypatch)
    out = world.tmp / "results" / "run2"
    _segment(world, out)
    export = world.tmp / "export"

    monkeypatch.setattr(sys, "argv", ["report.py", str(out), "--index", str(world.index),
                                      "--export", str(export), "--workers", "1"])
    report.main()
    meta = pd.read_csv(export / "all_genera_segment_metadata.csv")
    assert {"pv_tier", "pv_rank"} <= set(meta.columns) and meta["num_masks"].max() == 5
    tiers = pd.read_csv(export / "tier_genus_summary.csv")
    rud = tiers[tiers.genus == "Rudbeckia"].set_index("pv_tier")
    assert rud.loc["low", "photos"] == 10 and rud.loc["low", "photos_with_mask"] == 5
    assert rud.loc["high", "prediction_probability_mean"] == pytest.approx(0.845)
    assert rud.loc["low", "conf_top5_mean_ci95"] > 0
    for tier in report.TIERS:
        assert (export / f"scatter_tier_{tier}.png").stat().st_size > 10_000
    assert (export / "family_histograms" / "Asteraceae.png").exists()

    inat = world.tmp / "data" / "inat_photos.parquet"
    pd.DataFrame({"taxon_id": [510] * 50 + [502] * 7, "photo_id": range(57)}).to_parquet(inat)
    monkeypatch.setattr(sys, "argv", [
        "coverage.py", "--genera", str(world.index.with_name("all_genera_run2_index_genera.csv")),
        "--metadata", str(export / "all_genera_segment_metadata.csv"),
        "--taxa", str(world.taxa), "--inat", str(inat), "--export", str(export)])
    coverage.main()
    low = pd.read_csv(export / "low_hit_taxa.csv").set_index("name")
    assert list(low.index) == ["Bellis"]                 # Asteraceae has 45 hits
    assert low.loc["Bellis", "category"] == "both"
    assert low.loc["Bellis", "inat_observations"] == 7
    assert low.loc["Bellis", "inat_url"] == "https://www.inaturalist.org/taxa/502"
    page = (export / "coverage_genera.html").read_text()
    # Rudbeckia: 40 hits but 25 masked photos -> plotted, under on masks only.
    assert "plotly_click" in page and "Bellis" in page and "Rudbeckia" in page
    assert "masked photos only (1)" in page
    # Asteraceae's count rolls up the species (50) and Bellis (7).
    families = (export / "coverage_families.html").read_text()
    assert "Asteraceae" in families and "iNat photos: 57" in families

    monkeypatch.setattr(sys, "argv", ["pack_masks.py", str(out), "--metadata",
                                      str(export / "all_genera_segment_metadata.csv"),
                                      "--out", str(export / "masks")])
    pack_masks.main()
    from maskio import load_photo_masks
    by_photo = meta.set_index("photo_id")
    n = 0
    for row, mask in pack_masks.read_pack(export / "masks"):
        stored = load_photo_masks(out, by_photo.loc[row["photo_id"], "mask_path"])
        assert np.array_equal(stored[row["mask"]], mask)
        n += 1
    assert n == 25 * 5


def test_varints_round_trip():
    values = np.array([0, 1, 127, 128, 300, 16383, 16384, 2_000_000])
    assert np.array_equal(pack_masks.unvarints(pack_masks.varints(values)), values)
