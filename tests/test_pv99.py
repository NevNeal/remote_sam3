"""
The pv99 run without a GPU: the >= 0.99 index and the metadata export.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import segment
from test_run2 import _Saver, world  # noqa: F401  (world is a fixture)

PV99 = Path(__file__).resolve().parents[1] / "all_genera" / "pv99"


def _load(name):
    # run2 also has a build_index.py, already imported by test_run2.
    spec = importlib.util.spec_from_file_location(f"pv99_{name}", PV99 / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pv99_index = _load("build_index")
pv99_export = _load("export")


def _build(world, monkeypatch, cut="0.99"):
    index = world.tmp / "data" / "pv99_index.parquet"
    monkeypatch.setattr(sys, "argv", [
        "build_index.py", "--annotations", str(world.annotations),
        "--image-root", str(world.root), "--taxa", str(world.taxa),
        "--path-cache", str(world.tmp / "data" / "image_paths_all.parquet"),
        "--min-probability", cut, "--out", str(index), "--chunksize", "7"])
    pv99_index.main()
    return index


def test_index_keeps_only_high_probability(world, monkeypatch):
    # Rudbeckia 1000..1039 at 0.50..0.89, Bellis 2000..2004 at 0.60..0.68.
    index = pd.read_parquet(_build(world, monkeypatch, "0.85"))
    assert sorted(index.photo_id) == list(range(1035, 1040))
    assert (index.prediction_probability >= 0.85).all()
    assert "Pinus" not in set(index.genus)
    genera = pd.read_csv(world.tmp / "data" / "pv99_index_genera.csv")
    row = genera.set_index("genus").loc["Rudbeckia"]
    assert (row.flower_detected, row.at_min_probability, row.selected) == (40, 5, 5)


def test_export_metadata(world, monkeypatch):
    from argparse import Namespace
    index = _build(world, monkeypatch, "0.6")
    out = world.tmp / "results" / "pv99_flower"
    results = segment.Results(out / "results_shard_000.csv", out / "errors_shard_000.txt")
    args = Namespace(masks_only=True, discard_outputs=False, prompt="flower",
                     mask_format="npz", max_masks=5)
    rows = segment.load_local_photos(index, world.root, None, 0, 1).to_dict("records")
    for n, row in enumerate(rows):
        count = n % 3
        masks = [np.pad(np.ones((4, 4), bool), ((i, 36 - i), (i, 56 - i)))
                 for i in range(count)]
        scores = [0.9 - 0.1 * i for i in range(count)]
        common = {"photo_id": row["photo_id"], "taxon_id": row["taxon_id"],
                  "taxon_name": row["taxon_name"], "row_index": row["row_index"],
                  "shard": 0, "batch": row["batch"], "image_path": row["local_path"],
                  "width": 60, "height": 40}
        segment._save_photo(row, Image.open(row["local_path"]).convert("RGB"),
                            segment._image_path(out, row), masks, scores,
                            {"prep_s": "0", "infer_s": "0", "post_s": "0"},
                            common, out, args, _Saver(results))

    export = world.tmp / "export" / "pv99"
    monkeypatch.setattr(sys, "argv", ["export.py", str(out), "--index", str(index),
                                      "--export", str(export), "--csv"])
    pv99_export.main()
    meta = pd.read_parquet(export / "pv99_segment_metadata.parquet")
    assert list(meta.columns) == pv99_export.COLUMNS
    assert len(meta) == len(rows)
    assert meta.num_masks.sum() == sum(n % 3 for n in range(len(rows)))
    assert meta.loc[meta.num_masks > 0, "mask_path"].str.endswith(".npz").all()
    assert (export / "pv99_segment_metadata.csv").exists()
    assert (export / "pv99_genus_summary.csv").exists()
    assert (export / "pv99_index_genera.csv").exists()
    assert not (out / "overlays").exists() and not (out / "segments").exists()


def test_pick_floor_then_fill():
    photos = pd.DataFrame({
        "photo_id": range(130),
        "genus": ["Big"] * 100 + ["Mid"] * 20 + ["Tiny"] * 10,
    })
    chosen = pv99_index.pick(photos, target=60, per_genus=15, seed=0)
    assert len(chosen) == 60
    per = chosen.groupby("genus").size()
    assert per["Mid"] >= 15 and per["Big"] >= 15 and per["Tiny"] == 10
    assert (chosen.pv_pick == "floor").sum() == 40
    assert chosen.photo_id.is_unique and chosen.photo_id.is_monotonic_increasing
    # The floor alone is over target: keep it whole, no fill.
    over = pv99_index.pick(photos, target=20, per_genus=15, seed=0)
    assert len(over) == 40 and set(over.pv_pick) == {"floor"}
    # target 0: everything.
    assert len(pv99_index.pick(photos, target=0, per_genus=15, seed=0)) == 130
