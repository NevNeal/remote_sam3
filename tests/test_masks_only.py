"""
--masks-only end to end, without a GPU: segment.py's save step writes only the
.npy masks and records their size, render.py makes the overlays and cut-outs
from them afterwards, and run_metrics.py reports speed and mask size in
run_metrics.csv.
"""

import json
import sys
from argparse import Namespace

import numpy as np
import pandas as pd
from PIL import Image

import segment


class RecordingSaver:
    """The Saver's record() without the threads: straight into the CSV."""

    def __init__(self, results):
        self.results = results

    def record(self, key, fields):
        self.results.write(**fields)


def _segment_photo(out, results, photo_id, masks, size=(40, 30)):
    row = {"photo_id": photo_id, "stem": f"Rudbeckia_hirta_{photo_id}",
           "batch": "batch_00001"}
    image_path = out / "images" / "batch_00001" / f"{row['stem']}.jpg"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", size, (200, 160, 20))
    image.save(image_path)
    common = {"photo_id": photo_id, "taxon_id": 62741, "taxon_name": "Rudbeckia hirta",
              "quality_grade": "research", "row_index": photo_id, "shard": 0,
              "batch": "batch_00001", "image_path": segment._relative(image_path, out),
              "width": size[0], "height": size[1], "download_s": 0.2,
              "download_bytes": image_path.stat().st_size, "reused_image": 0,
              "wait_s": 0.01}
    stages = {"prep_s": "0.0100", "infer_s": "0.2000", "post_s": "0.0300"}
    args = Namespace(masks_only=True, discard_outputs=False, prompt="flower")
    segment._save_photo(row, image, image_path, masks, [0.95, 0.91][:len(masks)],
                        stages, common, out, args, RecordingSaver(results))


def _masks(count, size=(40, 30)):
    out = []
    for i in range(count):
        m = np.zeros((size[1], size[0]), dtype=np.uint8)
        m[5 + i:15 + i, 5:20] = 1
        out.append(m)
    return out


def test_masks_only_then_render_then_metrics(tmp_path, monkeypatch):
    out = tmp_path / "62741_flower"
    results = segment.Results(out / "results_shard_000.csv", out / "errors_shard_000.txt")
    _segment_photo(out, results, 1, _masks(2))
    _segment_photo(out, results, 2, [])                    # no detections

    # Only masks: no overlays, no cut-outs, and mask_bytes is the .npy total.
    npys = sorted((out / "masks").rglob("*.npy"))
    assert len(npys) == 2
    assert not (out / "overlays").exists() and not (out / "segments").exists()
    csv = pd.read_csv(out / "results_shard_000.csv")
    assert csv.loc[csv.photo_id == 1, "mask_bytes"].item() == sum(p.stat().st_size for p in npys)
    assert csv.loc[csv.photo_id == 1, "overlay_path"].isna().item()

    (out / "timing_shard_000.json").write_text(json.dumps({
        "shard": 0, "host": "h", "gpu": "NVIDIA B200", "gpu_uuid": "GPU-x",
        "started_at": 0.0, "finished_at": 10.0, "loop_s": 2.0, "model_load_s": 5.0,
        "workers": 48, "batch_size": 32, "save_workers": 10, "masks_only": True,
        "resumed_from": 0, "tally": {"ok": 1, "no_detections": 1}}))

    # The separate CPU job.
    import render
    monkeypatch.setattr(sys, "argv", ["render.py", str(out), "--workers", "1"])
    render.main()
    assert (out / "overlays" / "batch_00001" / "Rudbeckia_hirta_1_overlay.png").exists()
    assert len(list((out / "segments").rglob("*.png"))) == 2
    monkeypatch.setattr(sys, "argv", ["render.py", str(out), "--workers", "1"])
    render.main()                                          # resume: nothing to do
    assert len(pd.read_csv(out / "render_shard_000.csv")) == 1

    import run_metrics
    monkeypatch.setattr(sys, "argv", ["run_metrics.py", str(out), "--no-sacct",
                                      "--scan-masks"])
    run_metrics.main()
    m = pd.read_csv(out / "run_metrics.csv")
    get = lambda section, metric: m[(m.section == section) & (m.metric == metric)].value.item()
    total = sum(p.stat().st_size for p in npys)
    assert int(get("masks", "total_bytes")) == total
    assert int(get("masks", "disk_total_bytes")) == total
    assert float(get("masks", "avg_mb_per_mask")) == total / 2 / 1e6
    assert int(get("speed", "segments_total")) == 2
    assert float(get("speed", "avg_s_per_image_wall")) == 1.0      # 2 s loop / 2 photos
    per_seg = float(get("speed", "avg_s_per_segment_per_image"))
    assert 0.12 < per_seg < 1.0                            # (0.24 + save) / 2 masks
    assert int(get("render", "photos_rendered")) == 1
