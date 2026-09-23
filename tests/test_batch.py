"""
The batched forward pass: collation, per-photo timing shares, and the
split-and-retry that keeps one bad batch from costing a shard.

No GPU and no model. Batching is control flow around `model(**inputs)`, so a
fake model that records what it was handed tests it exactly as the real one
would — and unlike the real one, it can be made to fail on demand.
"""

import time

import numpy as np
import pytest


class Sizes(list):
    """Stands in for the processor's original_sizes tensor."""

    def tolist(self):
        return list(self)


class Inputs(dict):
    """What the processor returns: a mapping the model is called with."""

    def to(self, device):
        self.device = device
        return self


class FakeProcessor:
    """Collates a list of images and hands back one detection dict per image."""

    def __init__(self, masks_per_image=2, score=0.95, reject_text_list=False):
        self.masks_per_image = masks_per_image
        self.score = score
        self.reject_text_list = reject_text_list
        self.calls = []

    def __call__(self, images=None, text=None, return_tensors=None):
        if self.reject_text_list and isinstance(text, list):
            raise ValueError("this processor broadcasts a single string")
        self.calls.append({"images": list(images), "text": text})
        return Inputs(pixel_values=[np.zeros((3, 8, 8)) for _ in images],
                      original_sizes=Sizes([list(reversed(i.size)) for i in images]))

    def post_process_instance_segmentation(self, outputs, threshold=None,
                                           mask_threshold=None, target_sizes=None):
        out = []
        for height, width in target_sizes:
            masks = [np.ones((height, width), dtype=np.uint8)
                     for _ in range(self.masks_per_image)]
            out.append({"masks": masks,
                        "scores": [self.score] * self.masks_per_image})
        return out


class FakeModel:
    """Records every batch it is given, and can be told to fail on big ones."""

    def __init__(self, fail_above=None, fail_on_size=None, seconds=0.01):
        self.fail_above = fail_above
        self.fail_on_size = fail_on_size      # an image size that always fails
        self.seconds = seconds
        self.batch_sizes = []

    def __call__(self, **inputs):
        images = inputs["original_sizes"]
        if self.fail_above is not None and len(images) > self.fail_above:
            raise RuntimeError(f"out of memory: batch of {len(images)}")
        if self.fail_on_size is not None and list(self.fail_on_size) in list(images):
            raise RuntimeError("corrupt image")
        self.batch_sizes.append(len(images))
        time.sleep(self.seconds)
        return {"logits": None}


def images(count, size=(64, 48)):
    from PIL import Image
    return [Image.new("RGB", size, (i * 7 % 256, 128, 64)) for i in range(count)]


# -- collation ----------------------------------------------------------------

def test_one_forward_pass_for_the_whole_batch(segment):
    model, processor = FakeModel(), FakeProcessor()
    results = segment._forward_batch(model, processor, images(8), "flower", 0.9, "cpu", False)

    assert model.batch_sizes == [8], "eight images should be one forward pass, not eight"
    assert len(results) == 8
    assert all(len(masks) == 2 for masks, _, _ in results)


def test_each_photo_gets_the_batch_share_of_prep_and_infer(segment):
    model, processor = FakeModel(seconds=0.05), FakeProcessor()
    results = segment._forward_batch(model, processor, images(4), "flower", 0.9, "cpu", False)

    infer = [float(stages["infer_s"]) for _, _, stages in results]
    assert len(set(infer)) == 1, "the batch is indivisible, so the share is equal"
    # The four shares have to add back up to the one forward pass that happened.
    assert sum(infer) == pytest.approx(0.05, abs=0.03)


def test_one_text_per_image_with_a_fallback(segment):
    processor = FakeProcessor()
    segment._forward_batch(FakeModel(), processor, images(3), "flower", 0.9, "cpu", False)
    assert processor.calls[-1]["text"] == ["flower", "flower", "flower"]

    # A processor that only takes one string still works.
    processor = FakeProcessor(reject_text_list=True)
    segment._forward_batch(FakeModel(), processor, images(3), "flower", 0.9, "cpu", False)
    assert processor.calls[-1]["text"] == "flower"


def test_a_short_result_list_is_an_error_not_a_misalignment(segment):
    processor = FakeProcessor()
    processor.post_process_instance_segmentation = (
        lambda *a, **k: [{"masks": [], "scores": []}])      # one result for four images
    with pytest.raises(RuntimeError, match="for a batch of 4"):
        segment._forward_batch(FakeModel(), processor, images(4), "flower", 0.9, "cpu", False)


def test_single_image_segment_is_the_batch_of_one(segment):
    model, processor = FakeModel(), FakeProcessor()
    masks, scores, stages = segment.segment(model, processor, images(1)[0],
                                            "flower", 0.9, "cpu", False)
    assert model.batch_sizes == [1]
    assert len(masks) == 2 and len(scores) == 2
    assert set(stages) == {"prep_s", "infer_s", "post_s"}


def test_detections_below_min_score_are_dropped(segment):
    model, processor = FakeModel(), FakeProcessor(score=0.5)
    results = segment._forward_batch(model, processor, images(2), "flower", 0.9, "cpu", False)
    assert all(masks == [] for masks, _, _ in results)


# -- split and retry ----------------------------------------------------------

def test_a_batch_that_does_not_fit_is_halved_not_lost(segment):
    segment.segment_batch.warned = True          # keep the test output quiet
    model, processor = FakeModel(fail_above=8), FakeProcessor()
    results = segment.segment_batch(model, processor, images(32), "flower", 0.9, "cpu", False)

    assert len(results) == 32, "every image still gets a result"
    assert not any(isinstance(r, Exception) for r in results)
    assert max(model.batch_sizes) <= 8
    assert sum(model.batch_sizes) == 32, "each image is segmented exactly once"


def test_results_stay_in_the_order_the_images_were_given(segment):
    segment.segment_batch.warned = True
    processor = FakeProcessor()
    model = FakeModel(fail_above=4)
    photos = images(16)
    photos[5] = photos[5].resize((32, 96))       # a distinguishable one
    results = segment.segment_batch(model, processor, photos, "flower", 0.9, "cpu", False)

    assert results[5][0][0].shape == (96, 32), "photo 5's masks came back in slot 5"


def test_one_unsegmentable_image_does_not_take_the_batch_with_it(segment):
    segment.segment_batch.warned = True
    model = FakeModel(fail_on_size=[48, 64])     # original_sizes is (height, width)
    processor = FakeProcessor()
    photos = images(8, size=(100, 80))
    photos[3] = photos[3].resize((64, 48))

    results = segment.segment_batch(model, processor, photos, "flower", 0.9, "cpu", False)
    assert isinstance(results[3], Exception)
    assert [i for i, r in enumerate(results) if isinstance(r, Exception)] == [3]


# -- reading ahead ------------------------------------------------------------

def test_prefetch_yields_in_order_and_keeps_the_depth_bounded(segment):
    import threading

    inflight = {"now": 0, "max": 0}
    lock = threading.Lock()

    def fetch_one(row):
        with lock:
            inflight["now"] += 1
            inflight["max"] = max(inflight["max"], inflight["now"])
        time.sleep(0.002)
        with lock:
            inflight["now"] -= 1
        return row, "image", None, {}

    rows = [{"n": i} for i in range(50)]
    seen = [row["n"] for row, _, _, _ in segment._prefetch(rows, fetch_one, 4, depth=6)]

    assert seen == list(range(50)), "order is what pairs a result with its CSV row"
    assert inflight["max"] <= 6, "the whole shard must not be decoded at once"


def test_prefetch_records_how_long_the_consumer_waited(segment):
    def fetch_one(row):
        time.sleep(0.02)
        return row, "image", None, {}

    _, _, _, info = next(segment._prefetch([{"n": 0}], fetch_one, 2, depth=2))
    assert "wait_s" in info and info["wait_s"] >= 0
