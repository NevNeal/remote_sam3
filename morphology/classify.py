#!/usr/bin/env python3
"""
Flower morphology per genus, zero-shot, from run 2's photos.

For every genus, 10 of its run 2 photos are cropped to their SAM3 flower masks
and shown to a vision-language model (Qwen3-VL-8B-Instruct by default) with
each multiple-choice question in questions.py. The model never writes an
answer: one forward pass per question gives its next-token probabilities over
the option letters, and those probabilities are the per-photo answer. A
genus's call is the option with the highest mean probability over its photos,
with how many of the photos agree.

    python morphology/classify.py select    --metadata .../all_genera_segment_metadata.csv --out data/morphology_index.parquet
    python morphology/classify.py classify  --index data/morphology_index.parquet --run-dir results/all_genera_run2_flower --out-dir results/morphology
    python morphology/classify.py summarize --out-dir results/morphology --reference morphology/reference.csv
    python morphology/classify.py sheets    --index data/morphology_index.parquet --run-dir results/all_genera_run2_flower --out-dir results/morphology
    python morphology/classify.py download  # the model, into $HF_HOME, from a login node

classify resumes: photos already in photos.csv are skipped, so a job cut off
by --max-seconds or the wall clock is finished by submitting it again.

Outputs in --out-dir:
    photos.csv              one row per photo: every option's probability
                            (<question>__<option>), the photo's call per question,
                            and how much probability fell on the letters at all
    genus_morphology.csv    one row per genus: call, share, agreement, abstain,
                            confidence per question, and a one-line description
    genus_probabilities.csv mean probability of every option, per genus (for ML)
    family_morphology.csv   the same calls, per family
    validation.csv          calls against morphology/reference.csv
    sheets/<Family>/<Genus>.jpg   the 10 crops with their calls, to check by eye
"""

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from questions import LETTERS, QUESTIONS, option_keys, prompt_text  # noqa: E402
from maskio import load_masks  # noqa: E402

DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
SIZE = 512            # every crop is letterboxed to SIZE x SIZE: 256 image tokens
MARGIN = 0.25         # context kept around the masks, as a share of the box's long side
MIN_FRAC = 0.35       # a crop is never smaller than this share of the photo's short side
PAD = (128, 128, 128)

INDEX_COLUMNS = ["photo_id", "genus", "family", "scientific_name", "pv_tier",
                 "prediction_probability", "num_masks", "conf_max", "image_path",
                 "mask_path", "pick"]


# ── select ───────────────────────────────────────────────────────────────────

def select(metadata, per_genus=10):
    """Up to per_genus photos per genus: photos with a SAM3 mask first, one per
    species before a second from any species, most confident mask first. A
    genus with too few masked photos is filled with its unmasked ones, which
    are then shown uncropped."""
    m = metadata.copy()
    m["has_mask"] = (m["num_masks"].fillna(0) > 0) & m["mask_path"].notna()
    m["scientific_name"] = m["scientific_name"].fillna("")
    m["_conf"] = m["conf_max"].fillna(-1.0)
    m["_prob"] = m["prediction_probability"].fillna(-1.0)
    m = m.sort_values(["genus", "has_mask", "_conf", "_prob", "photo_id"],
                      ascending=[True, False, False, False, True])
    m["_species_rank"] = m.groupby(["genus", "has_mask", "scientific_name"]).cumcount()
    m = m.sort_values(["genus", "has_mask", "_species_rank", "_conf", "_prob", "photo_id"],
                      ascending=[True, False, True, False, False, True])
    m = m.groupby("genus", sort=False).head(per_genus).copy()
    m["pick"] = m.groupby("genus").cumcount()
    m.loc[~m["has_mask"], "mask_path"] = None
    for col in INDEX_COLUMNS:
        if col not in m:
            m[col] = None
    return m[INDEX_COLUMNS].reset_index(drop=True)


# ── cropping ─────────────────────────────────────────────────────────────────

def flower_box(masks, width, height, margin=MARGIN, min_frac=MIN_FRAC):
    """The (left, top, right, bottom) box around every mask, with margin, in the
    photo's own pixels. None when the masks are empty."""
    union = np.asarray(masks, dtype=bool).any(axis=0)
    ys, xs = np.nonzero(union)
    if not len(xs):
        return None
    sx, sy = width / union.shape[1], height / union.shape[0]
    x0, x1 = xs.min() * sx, (xs.max() + 1) * sx
    y0, y1 = ys.min() * sy, (ys.max() + 1) * sy
    pad = margin * max(x1 - x0, y1 - y0)
    x0, x1, y0, y1 = x0 - pad, x1 + pad, y0 - pad, y1 + pad
    least = min_frac * min(width, height)
    for lo, hi, limit in ((0, 1, width), (2, 3, height)):
        box = [x0, x1, y0, y1]
        a, b = box[lo], box[hi]
        if b - a < least:
            mid = (a + b) / 2
            a, b = mid - least / 2, mid + least / 2
        # Slide back inside the photo rather than just clipping, so a flower at
        # the edge still gets its full minimum crop.
        if a < 0:
            a, b = 0, b - a
        if b > limit:
            a, b = a - (b - limit), limit
        box[lo], box[hi] = max(0.0, a), min(float(limit), b)
        x0, x1, y0, y1 = box
    return int(x0), int(y0), int(np.ceil(x1)), int(np.ceil(y1))


def letterbox(image, size=SIZE):
    """Scale to fit size x size (up or down) and centre on grey."""
    w, h = image.size
    scale = size / max(w, h)
    image = image.resize((max(1, round(w * scale)), max(1, round(h * scale))),
                         Image.BICUBIC)
    canvas = Image.new("RGB", (size, size), PAD)
    canvas.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    return canvas


def load_crop(row, run_dir, size=SIZE):
    """One index row -> (letterboxed RGB crop, cropped?). Raises on a missing image."""
    image = Image.open(row["image_path"]).convert("RGB")
    cropped = False
    mask_path = row.get("mask_path")
    if isinstance(mask_path, str) and mask_path:
        path = Path(mask_path)
        path = path if path.is_absolute() else Path(run_dir) / path
        if path.exists():
            masks, _ = load_masks(path)
            box = flower_box(masks, image.width, image.height)
            if box is not None:
                image = image.crop(box)
                cropped = True
    return letterbox(image, size), cropped


# ── classify ─────────────────────────────────────────────────────────────────

class Crops:
    """torch Dataset over index rows; workers do the disk reads and cropping."""

    def __init__(self, rows, run_dir):
        self.rows = rows
        self.run_dir = run_dir

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        try:
            image, cropped = load_crop(row, self.run_dir)
            return row, image, cropped, ""
        except Exception as exc:          # a bad file is recorded, not fatal
            return row, None, False, f"{type(exc).__name__}: {exc}"


class Collate:
    """Runs the image half of the processor in the DataLoader workers, once per
    batch, so the main process only tokenizes each question's text."""

    def __init__(self, image_processor):
        self.image_processor = image_processor

    def __call__(self, items):
        images = [image for _, image, _, _ in items if image is not None]
        pixels = self.image_processor(images=images, return_tensors="pt") if images else None
        return items, pixels


def letter_ids(tokenizer, n):
    ids = []
    for letter in LETTERS[:n]:
        tok = tokenizer.encode(letter, add_special_tokens=False)
        if len(tok) != 1:
            raise RuntimeError(f"letter {letter!r} is not a single token: {tok}")
        ids.append(tok[0])
    return ids


def build_prompts(processor):
    """Per question: (chat text with one image slot, option letter token ids)."""
    out = []
    for q in QUESTIONS:
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": prompt_text(q)}]}]
        chat = processor.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True)
        out.append((chat, letter_ids(processor.tokenizer, len(q["options"]))))
    return out


def text_inputs(processor, chat, pixels, torch):
    """What processor(text=..., images=...) returns, from image features already
    computed: each image placeholder expanded to that image's token count."""
    merge = processor.image_processor.merge_size
    counts = (pixels["image_grid_thw"].prod(-1) // merge ** 2).tolist()
    token = processor.image_token
    tok = processor.tokenizer([chat.replace(token, token * n, 1) for n in counts],
                              return_tensors="pt", padding=True)
    image_id = processor.tokenizer.convert_tokens_to_ids(token)
    tok["mm_token_type_ids"] = (tok["input_ids"] == image_id).to(torch.int64)
    return {**tok, **pixels}


def inputs_match(processor, prompts, images, pixels, torch):
    """True when text_inputs reproduces the processor exactly, for every question."""
    for chat, _ in prompts:
        ref = processor(text=[chat] * len(images), images=images,
                        return_tensors="pt", padding=True)
        ours = text_inputs(processor, chat, pixels, torch)
        if set(ref.keys()) != set(ours.keys()):
            return False
        for key in ref.keys():
            if not torch.equal(ref[key], ours[key].to(ref[key].dtype)):
                return False
    return True


def score(model, processor, chat, ids, images, torch, pixels=None):
    """Probabilities over the option letters for one question and a batch of
    images, plus the share of the full next-token distribution on those letters.
    Halves the batch and retries on out-of-memory."""
    try:
        if pixels is not None:
            inputs = text_inputs(processor, chat, pixels, torch)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
        else:
            inputs = processor(text=[chat] * len(images), images=images,
                               return_tensors="pt", padding=True).to(model.device)
        with torch.inference_mode():
            logits = model(**inputs, logits_to_keep=1).logits[:, -1, :].float()
        picked = logits[:, ids]
        probs = torch.softmax(picked, dim=-1)
        mass = torch.exp(torch.logsumexp(picked, -1) - torch.logsumexp(logits, -1))
        return probs.cpu().numpy(), mass.cpu().numpy()
    except torch.OutOfMemoryError:
        if len(images) == 1:
            raise
        torch.cuda.empty_cache()
        half = len(images) // 2
        a = score(model, processor, chat, ids, images[:half], torch)
        b = score(model, processor, chat, ids, images[half:], torch)
        return np.concatenate([a[0], b[0]]), np.concatenate([a[1], b[1]])


def photo_columns():
    cols = []
    for q in QUESTIONS:
        cols += [f"{q['key']}__{k}" for k in option_keys(q)]
    for q in QUESTIONS:
        cols += [q["key"], f"{q['key']}_p", f"{q['key']}__letter_mass"]
    return cols


def classify(args):
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForImageTextToText, AutoProcessor

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    photos_csv = out_dir / "photos.csv"

    index = pd.read_parquet(args.index)
    done = set()
    if photos_csv.exists():
        done = set(pd.read_csv(photos_csv, usecols=["photo_id"])["photo_id"])
    todo = index[~index["photo_id"].isin(done)]
    if args.limit:
        todo = todo.head(args.limit)
    print(f"[classify] {len(index)} photos in the index, {len(done)} already done, "
          f"{len(todo)} to go", flush=True)
    if todo.empty:
        return

    print(f"[classify] loading {args.model}", flush=True)
    processor = AutoProcessor.from_pretrained(args.model)
    processor.tokenizer.padding_side = "left"
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
    prompts = build_prompts(processor)
    print(f"[classify] gpu {torch.cuda.get_device_name()} | batch {args.batch_size} | "
          f"{len(QUESTIONS)} questions | workers {args.workers}", flush=True)

    loader = DataLoader(Crops(todo.to_dict("records"), args.run_dir),
                        batch_size=args.batch_size, num_workers=args.workers,
                        collate_fn=Collate(processor.image_processor),
                        prefetch_factor=4 if args.workers else None,
                        persistent_workers=False)
    fast = None                     # decided on the first batch
    meta_cols = ["photo_id", "genus", "family", "scientific_name", "pick"]
    columns = meta_cols + ["cropped", "error"] + photo_columns()
    header = not photos_csv.exists()
    start = time.time()
    n_done = 0
    for batch, pixels in loader:
        ok = [(row, image, cropped) for row, image, cropped, err in batch if image is not None]
        records = [{**{c: row.get(c) for c in meta_cols}, "cropped": False, "error": err}
                   for row, image, cropped, err in batch if image is None]
        if ok:
            images = [image for _, image, _ in ok]
            if fast is None:
                fast = inputs_match(processor, prompts, images, pixels, torch)
                print(f"[classify] precomputed image features "
                      f"{'match the processor' if fast else 'DIFFER from the processor: using the slow path'}",
                      flush=True)
            per_q = [score(model, processor, chat, ids, images, torch,
                           pixels if fast else None) for chat, ids in prompts]
            for i, (row, _, cropped) in enumerate(ok):
                rec = {**{c: row.get(c) for c in meta_cols}, "cropped": cropped, "error": ""}
                for q, (probs, mass) in zip(QUESTIONS, per_q):
                    keys = option_keys(q)
                    for k, p in zip(keys, probs[i]):
                        rec[f"{q['key']}__{k}"] = round(float(p), 5)
                    best = int(np.argmax(probs[i]))
                    rec[q["key"]] = keys[best]
                    rec[f"{q['key']}_p"] = round(float(probs[i][best]), 5)
                    rec[f"{q['key']}__letter_mass"] = round(float(mass[i]), 5)
                records.append(rec)
        pd.DataFrame(records, columns=columns).to_csv(
            photos_csv, mode="a", header=header, index=False)
        header = False
        n_done += len(batch)
        elapsed = time.time() - start
        rate = n_done / elapsed
        if n_done == len(batch) or (n_done // args.batch_size) % 20 == 0 or n_done == len(todo):
            eta = (len(todo) - n_done) / rate if rate else float("nan")
            print(f"[classify] {n_done}/{len(todo)} photos | {rate:.2f} photos/s | "
                  f"elapsed {elapsed / 60:.1f} min | eta {eta / 60:.1f} min", flush=True)
        if args.max_seconds and elapsed > args.max_seconds:
            print(f"[classify] stopping after {elapsed / 60:.1f} min (--max-seconds); "
                  f"submit again to finish the remaining {len(todo) - n_done}", flush=True)
            break
    print(f"[classify] wrote {photos_csv}", flush=True)


# ── summarize ────────────────────────────────────────────────────────────────

def aggregate(photos, by):
    """One row per `by` group: for each question the call among the non-abstain
    options, its share of the non-abstain probability, the fraction of photos
    whose own non-abstain choice agrees, and the mean abstain probability."""
    photos = photos[photos["error"].fillna("") == ""]
    rows, probs_rows = [], []
    for name, g in photos.groupby(by, sort=True):
        row = {by: name, "n_photos": len(g), "n_cropped": int(g["cropped"].astype(bool).sum())}
        if by == "genus":
            row["family"] = g["family"].iloc[0]
        prow = {by: name}
        described = []
        for q in QUESTIONS:
            keys = option_keys(q)
            P = g[[f"{q['key']}__{k}" for k in keys]].to_numpy(dtype=float)
            mean = P.mean(axis=0)
            for k, p in zip(keys, mean):
                prow[f"{q['key']}__{k}"] = round(float(p), 4)
            real = np.array([k not in q["abstain"] for k in keys])
            abstain = float(mean[~real].sum())
            r_idx = np.flatnonzero(real)
            best = r_idx[int(np.argmax(mean[r_idx]))]
            share = float(mean[best] / max(mean[r_idx].sum(), 1e-9))
            agree = float((r_idx[np.argmax(P[:, r_idx], axis=1)] == best).mean())
            conf = confidence(share, agree, abstain)
            row[q["key"]] = keys[best]
            row[f"{q['key']}_share"] = round(share, 3)
            row[f"{q['key']}_agreement"] = round(agree, 3)
            row[f"{q['key']}_abstain"] = round(abstain, 3)
            row[f"{q['key']}_confidence"] = conf
            if conf != "low":
                described.append(keys[best] + ("" if conf == "high" else "?"))
        row["description"] = " · ".join(described)
        rows.append(row)
        probs_rows.append(prow)
    return pd.DataFrame(rows), pd.DataFrame(probs_rows)


def confidence(share, agreement, abstain):
    if abstain >= 0.5 or agreement < 0.5:
        return "low"
    if share >= 0.6 and agreement >= 0.7 and abstain < 0.3:
        return "high"
    return "medium"


def validate(genus, reference):
    """Every non-blank reference cell against the genus call."""
    ref = reference.melt(id_vars="genus", var_name="question", value_name="expected").dropna()
    ref = ref[ref["expected"].astype(str).str.strip() != ""]
    rows = []
    calls = genus.set_index("genus")
    for r in ref.itertuples():
        if r.genus not in calls.index:
            continue
        g = calls.loc[r.genus]
        rows.append(dict(genus=r.genus, question=r.question, expected=r.expected,
                         called=g[r.question], correct=g[r.question] == r.expected,
                         share=g[f"{r.question}_share"],
                         agreement=g[f"{r.question}_agreement"],
                         confidence=g[f"{r.question}_confidence"]))
    return pd.DataFrame(rows)


def summarize(args):
    out_dir = Path(args.out_dir)
    photos = pd.read_csv(out_dir / "photos.csv")
    errors = int((photos["error"].fillna("") != "").sum())
    genus, genus_probs = aggregate(photos, "genus")
    family, _ = aggregate(photos, "family")
    genus.to_csv(out_dir / "genus_morphology.csv", index=False)
    genus_probs.to_csv(out_dir / "genus_probabilities.csv", index=False)
    family.to_csv(out_dir / "family_morphology.csv", index=False)
    print(f"[summarize] {len(photos)} photos ({errors} unreadable), "
          f"{len(genus)} genera, {len(family)} families -> {out_dir}")

    print("\n[summarize] mean probability on the option letters (should be near 1):")
    for q in QUESTIONS:
        print(f"    {q['key']:<10} {photos[q['key'] + '__letter_mass'].mean():.3f}")
    print("\n[summarize] genus calls by confidence:")
    print(pd.DataFrame({q["key"]: genus[f"{q['key']}_confidence"].value_counts()
                        for q in QUESTIONS}).fillna(0).astype(int).to_string())
    print("\n[summarize] genus calls:")
    for q in QUESTIONS:
        counts = genus[q["key"]].value_counts()
        print(f"    {q['key']:<10} " + ", ".join(f"{k} {v}" for k, v in counts.items()))

    if args.reference and Path(args.reference).exists():
        v = validate(genus, pd.read_csv(args.reference, dtype=str))
        v.to_csv(out_dir / "validation.csv", index=False)
        if len(v):
            print(f"\n[validate] {int(v['correct'].sum())}/{len(v)} reference calls correct "
                  f"({v['correct'].mean():.0%}) over {v['genus'].nunique()} genera")
            print(v.groupby("question")["correct"].agg(["sum", "count", "mean"])
                  .round(2).to_string())
            wrong = v[~v["correct"]]
            if len(wrong):
                print("\n[validate] misses:")
                print(wrong[["genus", "question", "expected", "called", "share",
                             "agreement"]].to_string(index=False))


# ── sheets ───────────────────────────────────────────────────────────────────

THUMB = 224
COLS = 5


def _sheet(job):
    from review_figure import _font
    genus, family, rows, calls, run_dir, out_dir = job
    path = Path(out_dir) / "sheets" / str(family) / f"{genus}.jpg"
    line = 15
    caption = line * len(QUESTIONS) + 22
    header = 26 + line * len(QUESTIONS)
    n_rows = max(1, -(-len(rows) // COLS))
    sheet = Image.new("RGB", (COLS * (THUMB + 8) + 8, header + n_rows * (THUMB + caption)),
                      "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 4), f"{genus}  ({family})  {len(rows)} photos", fill="black", font=_font(18))
    for j, q in enumerate(QUESTIONS):
        text = (f"{q['key']}: {calls[q['key']]}  share {calls[q['key'] + '_share']:.0%}  "
                f"agree {calls[q['key'] + '_agreement']:.0%}  "
                f"[{calls[q['key'] + '_confidence']}]") if calls is not None else q["key"]
        draw.text((8, 26 + j * line), text, fill="#333333", font=_font(13))
    for i, row in enumerate(rows):
        x = 8 + (i % COLS) * (THUMB + 8)
        y = header + (i // COLS) * (THUMB + caption)
        try:
            crop, _ = load_crop(row, run_dir, THUMB)
        except Exception:
            crop = Image.new("RGB", (THUMB, THUMB), PAD)
        sheet.paste(crop, (x, y))
        draw.text((x, y + THUMB + 2), f"{row['photo_id']}  {row.get('scientific_name') or ''}"[:34],
                  fill="black", font=_font(11))
        for j, q in enumerate(QUESTIONS):
            if row.get(q["key"]) is None or pd.isna(row.get(q["key"])):
                continue
            draw.text((x, y + THUMB + 16 + j * line),
                      f"{row[q['key']]} {float(row[q['key'] + '_p']):.0%}",
                      fill="#444444", font=_font(12))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=85)


def sheets(args):
    out_dir = Path(args.out_dir)
    index = pd.read_parquet(args.index)
    photos_csv = out_dir / "photos.csv"
    calls = {}
    if photos_csv.exists():
        photos = pd.read_csv(photos_csv)
        keep = ["photo_id"] + [c for q in QUESTIONS for c in (q["key"], q["key"] + "_p")]
        index = index.merge(photos[keep], on="photo_id", how="left")
        genus_csv = out_dir / "genus_morphology.csv"
        if genus_csv.exists():
            calls = pd.read_csv(genus_csv).set_index("genus").to_dict("index")
    jobs = [(genus, g["family"].iloc[0], g.sort_values("pick").to_dict("records"),
             calls.get(genus), args.run_dir, str(out_dir))
            for genus, g in index.groupby("genus")]
    with ProcessPoolExecutor(args.workers) as pool:
        for i, _ in enumerate(pool.map(_sheet, jobs, chunksize=16), 1):
            if i % 1000 == 0 or i == len(jobs):
                print(f"[sheets] {i}/{len(jobs)}", flush=True)
    print(f"[sheets] {out_dir / 'sheets'}")


# ── cli ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("select", help="pick the photos per genus")
    s.add_argument("--metadata", required=True, help="run 2's all_genera_segment_metadata.csv")
    s.add_argument("--out", required=True, help="index parquet to write")
    s.add_argument("--per-genus", type=int, default=10)

    c = sub.add_parser("classify", help="run the model (GPU)")
    c.add_argument("--index", required=True)
    c.add_argument("--run-dir", required=True, help="run 2's results dir; mask paths are relative to it")
    c.add_argument("--out-dir", required=True)
    c.add_argument("--model", default=os.environ.get("MORPH_MODEL", DEFAULT_MODEL))
    c.add_argument("--batch-size", type=int, default=64)
    c.add_argument("--workers", type=int, default=8)
    c.add_argument("--max-seconds", type=float, default=0, help="stop cleanly after this long")
    c.add_argument("--limit", type=int, default=0, help="only this many photos (testing)")

    m = sub.add_parser("summarize", help="per-genus and per-family calls (CPU)")
    m.add_argument("--out-dir", required=True)
    m.add_argument("--reference", default=str(HERE / "reference.csv"))

    h = sub.add_parser("sheets", help="contact sheet per genus (CPU)")
    h.add_argument("--index", required=True)
    h.add_argument("--run-dir", required=True)
    h.add_argument("--out-dir", required=True)
    h.add_argument("--workers", type=int, default=8)

    d = sub.add_parser("download", help="fetch the model into $HF_HOME")
    d.add_argument("--model", default=os.environ.get("MORPH_MODEL", DEFAULT_MODEL))

    args = p.parse_args()
    if args.cmd == "select":
        metadata = pd.read_csv(args.metadata, low_memory=False)
        index = select(metadata, args.per_genus)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        index.to_parquet(args.out, index=False)
        print(f"[select] {len(index)} photos over {index['genus'].nunique()} genera, "
              f"{int(index['mask_path'].notna().sum())} with masks -> {args.out}")
    elif args.cmd == "classify":
        classify(args)
    elif args.cmd == "summarize":
        summarize(args)
    elif args.cmd == "sheets":
        sheets(args)
    elif args.cmd == "download":
        from huggingface_hub import snapshot_download
        print(snapshot_download(args.model))


if __name__ == "__main__":
    main()
