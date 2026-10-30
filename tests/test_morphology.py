"""
Flower morphology without a GPU: picking the photos per genus, the crop around
the masks, the letterbox, the per-genus aggregation and the reference check.
The model step itself is exercised by a short local run, not here.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

MORPH = Path(__file__).resolve().parents[1] / "morphology"
sys.path.insert(0, str(MORPH))

import classify  # noqa: E402
import questions  # noqa: E402
from maskio import save_masks  # noqa: E402


def _meta(rows):
    cols = ["photo_id", "genus", "family", "scientific_name", "pv_tier",
            "prediction_probability", "num_masks", "conf_max", "image_path", "mask_path"]
    return pd.DataFrame(rows, columns=cols)


def test_select_prefers_masked_then_spreads_species():
    rows = []
    # Genus A: species x has 4 strong masked photos, y has 2 weak masked ones,
    # and 3 unmasked photos.
    for i, c in enumerate([0.99, 0.98, 0.97, 0.96]):
        rows.append([i, "A", "F", "A x", "high", 0.9, 1, c, f"{i}.jpg", f"m{i}.npz"])
    for i, c in enumerate([0.6, 0.5], 10):
        rows.append([i, "A", "F", "A y", "low", 0.9, 1, c, f"{i}.jpg", f"m{i}.npz"])
    for i, p in enumerate([0.7, 0.95, 0.8], 20):
        rows.append([i, "A", "F", "A x", "low", p, 0, None, f"{i}.jpg", None])
    out = classify.select(_meta(rows), per_genus=8)
    # One per species first (best x, best y), then the next of each, then the
    # remaining masked x photos, then unmasked by probability.
    assert out["photo_id"].tolist() == [0, 10, 1, 11, 2, 3, 21, 22]
    assert out["pick"].tolist() == list(range(8))
    assert out.loc[out["num_masks"] == 0, "mask_path"].isna().all()


def test_select_caps_per_genus_and_keeps_small_genera():
    rows = [[i, "Big", "F", "Big a", "high", 0.5, 1, 0.9, "x", "m"] for i in range(30)]
    rows += [[100, "Tiny", "G", "Tiny a", "high", 0.5, 0, None, "x", None]]
    out = classify.select(_meta(rows), per_genus=10)
    assert out.groupby("genus").size().to_dict() == {"Big": 10, "Tiny": 1}


def test_flower_box_margin_and_minimum():
    masks = np.zeros((1, 100, 200), dtype=bool)
    masks[0, 40:60, 90:110] = True                 # a 20 x 20 flower
    box = classify.flower_box(masks, 200, 100, margin=0.25, min_frac=0.0)
    assert box == (85, 35, 115, 65)
    # The minimum crop (half the short side = 50 px) wins over a tiny flower.
    box = classify.flower_box(masks, 200, 100, margin=0.0, min_frac=0.5)
    assert box == (75, 25, 125, 75)


def test_flower_box_slides_inside_at_the_edge_and_scales():
    masks = np.zeros((1, 50, 100), dtype=bool)     # half the photo's resolution
    masks[0, 0:5, 0:5] = True
    left, top, right, bottom = classify.flower_box(masks, 200, 100, margin=0.0, min_frac=0.5)
    assert (left, top) == (0, 0)
    assert right - left >= 50 and bottom - top >= 50
    assert classify.flower_box(np.zeros((2, 10, 10), bool), 10, 10) is None


def test_letterbox_is_square_and_keeps_aspect():
    image = Image.new("RGB", (300, 100), (255, 0, 0))
    box = classify.letterbox(image, 64)
    assert box.size == (64, 64)
    px = np.asarray(box)
    assert tuple(px[32, 32]) == (255, 0, 0)
    assert tuple(px[0, 32]) == classify.PAD


def test_load_crop_uses_masks_and_falls_back(tmp_path):
    image = Image.new("RGB", (200, 100), (0, 255, 0))
    image.save(tmp_path / "p.jpg")
    mask = np.zeros((100, 200), dtype=np.uint8)
    mask[40:60, 90:110] = 1
    (tmp_path / "masks").mkdir()
    save_masks(tmp_path / "masks" / "p_masks.npz", [mask], [0.9])
    row = {"image_path": str(tmp_path / "p.jpg"), "mask_path": "masks/p_masks.npz"}
    crop, cropped = classify.load_crop(row, tmp_path, 32)
    assert cropped and crop.size == (32, 32)
    crop, cropped = classify.load_crop({**row, "mask_path": None}, tmp_path, 32)
    assert not cropped
    crop, cropped = classify.load_crop({**row, "mask_path": "masks/gone.npz"}, tmp_path, 32)
    assert not cropped


def _photos(genus, unit_probs, error=""):
    """photos.csv rows for one genus, every question but `unit` uniform."""
    rows = []
    for i, probs in enumerate(unit_probs):
        rec = {"photo_id": f"{genus}{i}", "genus": genus, "family": "F",
               "cropped": True, "error": error}
        for q in questions.QUESTIONS:
            keys = questions.option_keys(q)
            vals = probs if q["key"] == "unit" else [1 / len(keys)] * len(keys)
            for k, p in zip(keys, vals):
                rec[f"{q['key']}__{k}"] = p
        rows.append(rec)
    return rows


def test_aggregate_ignores_abstain_options_and_counts_agreement():
    keys = questions.option_keys(questions.QUESTIONS[0])
    assert keys[:4] == ["solitary", "inflorescence", "head", "spadix"]
    none = keys.index("none")
    # Mostly "no flower visible", but among the real options "spadix" leads.
    p_abstain = np.zeros(len(keys)); p_abstain[none] = 0.7; p_abstain[3] = 0.3
    p_spadix = np.zeros(len(keys)); p_spadix[3] = 0.9; p_spadix[0] = 0.1
    p_solitary = np.zeros(len(keys)); p_solitary[0] = 0.8; p_solitary[3] = 0.2
    photos = pd.DataFrame(_photos("Arum", [p_abstain, p_spadix, p_spadix, p_solitary])
                          + _photos("Bad", [p_spadix], error="OSError: gone"))
    genus, probs = classify.aggregate(photos, "genus")
    assert genus["genus"].tolist() == ["Arum"]            # the unreadable photo is dropped
    row = genus.iloc[0]
    assert row["unit"] == "spadix"
    assert row["unit_agreement"] == pytest.approx(0.75)   # 3 of 4 photos pick spadix
    assert row["unit_abstain"] == pytest.approx(0.7 / 4)
    assert row["n_photos"] == 4
    assert probs.iloc[0]["unit__none"] == pytest.approx(0.175)
    assert "spadix" in row["description"]


def test_confidence_levels():
    assert classify.confidence(0.9, 0.9, 0.1) == "high"
    assert classify.confidence(0.5, 0.9, 0.1) == "medium"
    assert classify.confidence(0.9, 0.4, 0.1) == "low"
    assert classify.confidence(0.9, 0.9, 0.6) == "low"


def test_validate_scores_only_filled_cells():
    genus = pd.DataFrame([{"genus": "Arum", "unit": "spadix", "unit_share": 0.9,
                           "unit_agreement": 1.0, "unit_confidence": "high",
                           "symmetry": "actinomorphic", "symmetry_share": 0.5,
                           "symmetry_agreement": 0.5, "symmetry_confidence": "low"}])
    ref = pd.DataFrame([{"genus": "Arum", "unit": "spadix", "symmetry": None},
                        {"genus": "Absent", "unit": "head", "symmetry": None}])
    v = classify.validate(genus, ref)
    assert len(v) == 1 and bool(v.iloc[0]["correct"])


def test_reference_uses_real_options():
    ref = pd.read_csv(MORPH / "reference.csv", dtype=str)
    for q in questions.QUESTIONS:
        assert q["key"] in ref.columns
        used = set(ref[q["key"]].dropna())
        assert used <= set(questions.option_keys(q)), (q["key"], used)


def test_prompt_letters_match_options():
    for q in questions.QUESTIONS:
        text = questions.prompt_text(q)
        for letter, (_, option) in zip(questions.LETTERS, q["options"]):
            assert f"{letter}. {option}" in text
    assert len(classify.photo_columns()) == len(set(classify.photo_columns()))
