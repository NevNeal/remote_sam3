"""
review_figure.py: one quality-check figure per photo from a run's CSV, masks
and image - at most five masks, most confident first.
"""

import sys

import numpy as np
import pandas as pd
from PIL import Image

import review_figure


def test_review_figure(tmp_path, monkeypatch):
    run = tmp_path / "62741_flower"
    (run / "masks" / "batch_00001").mkdir(parents=True)
    (run / "images" / "batch_00001").mkdir(parents=True)
    stem = "Rudbeckia_hirta_7"
    Image.new("RGB", (120, 90), (200, 160, 20)).save(run / "images/batch_00001" / f"{stem}.jpg")
    paths, scores = [], []
    for i in range(7):                       # seven masks: only five are drawn
        mask = np.zeros((90, 120), np.uint8)
        mask[5 + 10 * i:15 + 10 * i, 10:60] = 1
        rel = f"masks/batch_00001/{stem}_instance_{i}.npy"
        np.save(run / rel, mask)
        paths.append(rel)
        scores.append(0.99 - 0.01 * i)
    pd.DataFrame([
        {"photo_id": 7, "taxon_name": "Rudbeckia hirta", "row_index": 0, "status": "ok",
         "num_masks": 7, "image_path": f"images/batch_00001/{stem}.jpg",
         "mask_paths": ";".join(paths), "mask_scores": ",".join(f"{s:.4f}" for s in scores)},
        {"photo_id": 8, "taxon_name": "Rudbeckia hirta", "row_index": 1, "status": "ok",
         "num_masks": 0, "image_path": "", "mask_paths": "", "mask_scores": ""},
    ]).to_csv(run / "results_shard_000.csv", index=False)

    monkeypatch.setattr(sys, "argv", ["review_figure.py", str(run), "--workers", "1"])
    review_figure.main()
    out = run / "review" / "batch_00001" / f"{stem}_review.png"
    figure = Image.open(out)
    assert figure.height == review_figure.HEADER_H + review_figure.PANEL_H + review_figure.GAP
    assert len(list((run / "review").rglob("*.png"))) == 1      # photo 8 has no masks

    stamp = out.stat().st_mtime_ns
    review_figure.main()                                        # resume: not redrawn
    assert out.stat().st_mtime_ns == stamp


def test_make_figure_draws_at_most_top_k_tiles():
    image = Image.new("RGB", (100, 80), (10, 120, 10))
    mask = np.zeros((80, 100), bool)
    mask[20:60, 30:70] = True
    one = review_figure.make_figure(image, [mask], [0.95], "t", 1)
    # With one mask, the other four tile slots stay background.
    x = review_figure.GAP * 2 + round(100 * review_figure.PANEL_H / 80) + 20
    tile_h = (review_figure.PANEL_H - review_figure.GAP * 4) // 5
    assert one.getpixel((x, review_figure.HEADER_H + tile_h + review_figure.GAP + 20)) \
        == review_figure.BG
    assert one.getpixel((x, review_figure.HEADER_H + 80)) != review_figure.BG
