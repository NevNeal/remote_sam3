#!/usr/bin/env python3
"""
Bullseye or not, for the top segment of every masked pv99 photo.

    python all_genera/pv99/analysis/bullseye_classify.py --shard 0 --num-shards 4
    python all_genera/pv99/analysis/bullseye_classify.py --limit 200        # smoke test

A photo's top segment is its single most confident SAM3 mask (the same one
top_segments.py cuts out). For every photo in pv99_segment_metadata.parquet
with at least one mask (~270k of 291k), the context crop around that mask (the
mask's box plus a 25% margin, at least 35% of the photo's short side,
letterboxed to 512 px; top_segments.context_box) is built on the fly by
DataLoader workers from the photo on /blue and the .npz under
results/pv99_flower/masks/, so nothing is written per photo but one CSV row.

Model: Qwen/Qwen3-VL-8B-Instruct in bf16, zero-shot, answering the single
question in bullseye_question.py. As in morph_classify.py the question is asked
in both option orders, each answer is the softmax over the option letters'
next-token logits, and the two are averaged (letter-position bias cancels).
Also recorded: `bullseye__mass`, the share of the whole next-token
distribution on the option letters, and `bullseye__flip`, whether the two
orders picked different options.

Photos are split across shards by photo_id % num_shards. Each shard appends to
export/pv99/analysis/bullseye/photos_shard<k>.csv and skips photo_ids already
in it, so a killed or timed-out task resumes when resubmitted.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402
from bullseye_question import LETTERS, QUESTION, option_keys, prompt_text  # noqa: E402
from top_segments import context_box, letterbox, mask_box  # noqa: E402

MODEL = "Qwen/Qwen3-VL-8B-Instruct"
SIZE = 512
META_COLS = ["photo_id", "family", "genus", "scientific_name", "conf_max", "num_masks",
             "image_path", "mask_path"]


def top_segment_crop(project, image_path, mask_path, size=SIZE):
    """The context crop around a photo's most confident mask, letterboxed to
    size x size, plus the mask's share of the photo and its score."""
    image = Image.open(common.local_image(project, image_path)).convert("RGB")
    masks, scores = common.photo_masks(project, mask_path)
    mask = masks[0]
    if mask.shape != (image.height, image.width):
        mask = np.asarray(Image.fromarray(mask).resize(image.size, Image.NEAREST), bool)
    box = mask_box(mask)
    crop = letterbox(image.crop(context_box(box, image.width, image.height)), size)
    return crop, float(mask.mean()), float(scores[0])


class Crops:
    """torch Dataset over metadata rows: (photo_id, crop or None, error)."""

    def __init__(self, project, rows):
        self.project, self.rows = project, rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        try:
            crop, area, score = top_segment_crop(self.project, row["image_path"], row["mask_path"])
            return row, crop, dict(mask_area_frac=round(area, 5), seg_score=round(score, 5)), ""
        except Exception as exc:
            return row, None, {}, f"{type(exc).__name__}: {exc}"[:200]


class Collate:
    """Runs the Qwen processor inside the DataLoader workers: every photo twice,
    once per option order."""

    def __init__(self, processor, chats):
        self.processor, self.chats = processor, chats

    def __call__(self, batch):
        good = [b for b in batch if b[1] is not None]
        inputs = None
        if good:
            texts = [chat for _ in good for chat in self.chats]
            images = [b[1] for b in good for _ in self.chats]
            inputs = self.processor(text=texts, images=images, return_tensors="pt",
                                    padding=True)
        return batch, inputs


def chats_and_ids(processor):
    n = len(QUESTION["options"])
    orders = [list(range(n)), list(range(n))[::-1]]
    chats = []
    for order in orders:
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": prompt_text(order)}]}]
        chats.append(processor.apply_chat_template(messages, tokenize=False,
                                                   add_generation_prompt=True))
    ids = []
    for letter in LETTERS[:n]:
        tok = processor.tokenizer.encode(letter, add_special_tokens=False)
        if len(tok) != 1:
            raise RuntimeError(f"{letter!r} is not one token")
        ids.append(tok[0])
    return orders, chats, ids


def record(orders, probs, masses):
    """One photo: per option the probability averaged over both orders, the
    call among the non-abstain options, letter mass, and the order flip."""
    keys = option_keys()
    per_order = []
    for order, p in zip(orders, probs):
        per_option = np.zeros(len(order))
        per_option[order] = p                       # letter i showed option order[i]
        per_order.append(per_option)
    mean = np.mean(per_order, axis=0)
    rec = {f"bullseye__{k}": round(float(p), 5) for k, p in zip(keys, mean)}
    best = int(np.argmax(mean))
    rec["bullseye"] = keys[best]
    rec["bullseye_p"] = round(float(mean[best]), 5)
    rec["bullseye__mass"] = round(float(np.mean(masses)), 5)
    rec["bullseye__flip"] = int(np.argmax(per_order[0]) != np.argmax(per_order[1]))
    return rec


def out_columns():
    keys = option_keys()
    return (["photo_id", "family", "genus", "scientific_name", "conf_max", "num_masks",
             "seg_score", "mask_area_frac"]
            + [f"bullseye__{k}" for k in keys]
            + ["bullseye", "bullseye_p", "bullseye__mass", "bullseye__flip", "error"])


def load_model(name):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    processor = AutoProcessor.from_pretrained(name)
    processor.tokenizer.padding_side = "left"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"   # cpu: local smoke tests only
    model = AutoModelForImageTextToText.from_pretrained(
        name, dtype=torch.bfloat16, attn_implementation="sdpa", device_map=device).eval()
    return processor, model


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_args(parser)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--batch", type=int, default=32, help="photos per forward pass")
    parser.add_argument("--workers", type=int, default=common.n_workers())
    parser.add_argument("--limit", type=int, default=None, help="only this many photos")
    parser.add_argument("--out", default=None, help="CSV path (default bullseye/photos_shard<k>.csv)")
    args = parser.parse_args()

    import torch
    from torch.utils.data import DataLoader

    project = common.project_dir(args.project)
    out = (Path(args.out) if args.out else
           common.analysis_dir(project, "bullseye") / f"photos_shard{args.shard:02d}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)

    meta = common.load_metadata(project, META_COLS)
    meta = meta[meta["num_masks"].fillna(0) > 0]
    meta = meta[meta["photo_id"] % args.num_shards == args.shard].sort_values("photo_id")
    done = set(pd.read_csv(out, usecols=["photo_id"])["photo_id"]) if out.exists() else set()
    todo = meta[~meta["photo_id"].isin(done)]
    if args.limit:
        todo = todo.head(args.limit)
    print(f"[bullseye] shard {args.shard}/{args.num_shards}: {len(meta):,} masked photos, "
          f"{len(done):,} done, {len(todo):,} to go -> {out}", flush=True)
    if todo.empty:
        return

    processor, model = load_model(args.model)
    orders, chats, ids = chats_and_ids(processor)
    ids_t = torch.tensor(ids, device=model.device)
    gpu = torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"
    print(f"[bullseye] {args.model} on {gpu}, batch {args.batch} "
          f"photos x {len(chats)} orders, {args.workers} workers", flush=True)

    loader = DataLoader(Crops(project, todo.to_dict("records")), batch_size=args.batch,
                        num_workers=args.workers, collate_fn=Collate(processor, chats),
                        prefetch_factor=4 if args.workers else None,
                        persistent_workers=bool(args.workers))
    cols = out_columns()
    header = not out.exists()
    start, n_done, n_err, n_batches, buffer = time.time(), 0, 0, 0, []
    print(f"[bullseye] {time.strftime('%H:%M:%S')} model ready, starting", flush=True)
    for batch, inputs in loader:
        results = {}
        if inputs is not None:
            try:
                inputs = inputs.to(model.device)
                with torch.inference_mode():
                    logits = model(**inputs, logits_to_keep=1).logits[:, -1, :].float()
                picked = logits[:, ids_t]
                probs = torch.softmax(picked, -1).cpu().numpy()
                mass = torch.exp(torch.logsumexp(picked, -1)
                                 - torch.logsumexp(logits, -1)).cpu().numpy()
                k = len(chats)
                good = [b for b in batch if b[1] is not None]
                for j, (row, _, _, _) in enumerate(good):
                    results[row["photo_id"]] = record(orders, probs[j * k:(j + 1) * k],
                                                      mass[j * k:(j + 1) * k])
            except torch.OutOfMemoryError as exc:
                torch.cuda.empty_cache()
                results = {b[0]["photo_id"]: dict(error=f"OutOfMemoryError: {exc}"[:200])
                           for b in batch if b[1] is not None}
        for row, _, extra, err in batch:
            rec = {c: row.get(c) for c in cols if c in row}
            rec.update(extra)
            rec.update(results.get(row["photo_id"], {}))
            rec["error"] = rec.get("error") or err
            n_err += bool(rec["error"])
            buffer.append(rec)
        n_done += len(batch)
        n_batches += 1
        if n_batches <= 5 or n_batches % 25 == 0:
            rate = n_done / (time.time() - start)
            print(f"[bullseye] {time.strftime('%H:%M:%S')} batch {n_batches}: {n_done:,}/{len(todo):,}"
                  f" | {rate:.1f} photos/s since first batch requested", flush=True)
        if len(buffer) >= 2000 or n_done == len(todo):
            pd.DataFrame(buffer, columns=cols).to_csv(out, mode="a", header=header, index=False)
            header, buffer = False, []
            rate = n_done / (time.time() - start)
            print(f"[bullseye] {n_done:,}/{len(todo):,} | {rate:.1f} photos/s | "
                  f"{n_err} errors | eta {(len(todo) - n_done) / rate / 60:.1f} min", flush=True)
    if buffer:
        pd.DataFrame(buffer, columns=cols).to_csv(out, mode="a", header=header, index=False)
    print(f"[bullseye] wrote {out}")


if __name__ == "__main__":
    main()
