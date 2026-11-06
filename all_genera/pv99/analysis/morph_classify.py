#!/usr/bin/env python3
"""
Zero-shot flower-type calls for the top segments, from a vision-language model.

    python all_genera/pv99/local/morph_classify.py  [--limit 40]

Input: export/pv99/analysis/top_segments/top_segments.csv and each segment's context crop
(the box around its mask plus a 25% margin, letterboxed to 512 px).
Model: Qwen/Qwen3-VL-8B-Instruct in bf16 on one B200 (--four-bit loads 4-bit
NF4 language-model weights instead, for an 8 GB GPU).

Each question in morph_questions.py is asked twice per photo, once with the
options in their written order and once reversed. For each order the answer is
the softmax over the option letters' next-token logits (the model never writes
text). The two are mapped back to the options and averaged, so a bias toward
"A" or toward the last letter cancels rather than voting for one option.
Also recorded per question: `__mass`, the share of the full next-token
distribution that fell on the option letters (near 1 = it answered the
question), and `__flip`, whether the two orders picked different options.

Writes export/pv99/analysis/morphology/photos.csv (resumable: photos already in it are
skipped).
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from morph_questions import LETTERS, QUESTIONS, option_keys, prompt_text  # noqa: E402

MODEL = "Qwen/Qwen3-VL-8B-Instruct"


def prompts(processor):
    """[(question, order, chat text, letter token ids)] for both orders."""
    out = []
    for q in QUESTIONS:
        n = len(q["options"])
        for order in (list(range(n)), list(range(n))[::-1]):
            messages = [{"role": "user", "content": [
                {"type": "image"}, {"type": "text", "text": prompt_text(q, order)}]}]
            chat = processor.apply_chat_template(messages, tokenize=False,
                                                 add_generation_prompt=True)
            ids = []
            for letter in LETTERS[:n]:
                tok = processor.tokenizer.encode(letter, add_special_tokens=False)
                if len(tok) != 1:
                    raise RuntimeError(f"{letter!r} is not one token")
                ids.append(tok[0])
            out.append((q, order, chat, ids))
    return out


def load_model(name, four_bit=False):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
    processor = AutoProcessor.from_pretrained(name)
    processor.tokenizer.padding_side = "left"
    kwargs = dict(dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda:0")
    if four_bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            llm_int8_skip_modules=["visual", "lm_head"])
    model = AutoModelForImageTextToText.from_pretrained(name, **kwargs).eval()
    return processor, model


CHUNK = 16  # prompts per forward pass: one photo's 8 questions x 2 orders (use 4 on an 8 GB GPU)


def score_image(model, processor, plist, image, torch, chunk=CHUNK):
    """Every (question, order) prompt for one image, `chunk` prompts per pass.
    Returns per prompt: (letter probabilities, letter mass)."""
    if len(plist) > chunk:
        return [r for i in range(0, len(plist), chunk)
                for r in score_image(model, processor, plist[i:i + chunk], image, torch, chunk)]
    inputs = processor(text=[chat for _, _, chat, _ in plist], images=[image] * len(plist),
                       return_tensors="pt", padding=True).to(model.device)
    with torch.inference_mode():
        logits = model(**inputs, logits_to_keep=1).logits[:, -1, :].float()
    out = []
    for row, (_, _, _, ids) in enumerate(plist):
        picked = logits[row, ids]
        probs = torch.softmax(picked, -1).cpu().numpy()
        mass = float(torch.exp(torch.logsumexp(picked, -1) - torch.logsumexp(logits[row], -1)))
        out.append((probs, mass))
    return out


def combine(plist, scored):
    """Per question: option probabilities averaged over both orders, mean
    letter mass, and whether the two orders disagreed."""
    rec, by_q = {}, {}
    for (q, order, _, _), (probs, mass) in zip(plist, scored):
        per_option = np.zeros(len(order))
        per_option[order] = probs                   # letter i showed option order[i]
        by_q.setdefault(q["key"], []).append((per_option, mass))
    for q in QUESTIONS:
        runs = by_q[q["key"]]
        mean = np.mean([p for p, _ in runs], axis=0)
        keys = option_keys(q)
        for k, p in zip(keys, mean):
            rec[f"{q['key']}__{k}"] = round(float(p), 5)
        best = int(np.argmax(mean))
        rec[q["key"]] = keys[best]
        rec[f"{q['key']}_p"] = round(float(mean[best]), 5)
        rec[f"{q['key']}__mass"] = round(float(np.mean([m for _, m in runs])), 5)
        rec[f"{q['key']}__flip"] = int(np.argmax(runs[0][0]) != np.argmax(runs[1][0]))
    return rec


def columns(meta_cols):
    cols = list(meta_cols)
    for q in QUESTIONS:
        cols += [f"{q['key']}__{k}" for k in option_keys(q)]
        cols += [q["key"], f"{q['key']}_p", f"{q['key']}__mass", f"{q['key']}__flip"]
    return cols + ["error"]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_args(parser)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--four-bit", action="store_true",
                        help="4-bit NF4 weights, for a small GPU (default bf16)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default=None, help="photos.csv path (default analysis/morphology)")
    args = parser.parse_args()

    import torch
    project = common.project_dir(args.project)
    segs = pd.read_csv(common.analysis_dir(project, "top_segments") / "top_segments.csv")
    out = Path(args.out) if args.out else common.analysis_dir(project, "morphology") / "photos.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set(pd.read_csv(out, usecols=["photo_id"])["photo_id"]) if out.exists() else set()
    todo = segs[~segs["photo_id"].isin(done)]
    if args.limit:
        todo = todo.head(args.limit)
    print(f"[morph] {len(segs)} segments, {len(done)} done, {len(todo)} to go", flush=True)
    if todo.empty:
        return

    processor, model = load_model(args.model, four_bit=args.four_bit)
    plist = prompts(processor)
    print(f"[morph] {args.model} on {torch.cuda.get_device_name()}, "
          f"{len(plist)} prompts per photo", flush=True)
    meta_cols = ["photo_id", "family", "genus", "scientific_name", "seg_rank", "seg_score"]
    header = not out.exists()
    start = time.time()
    buffer = []
    for n, row in enumerate(todo.to_dict("records"), 1):
        rec = {c: row[c] for c in meta_cols}
        try:
            image = Image.open(row["context"]).convert("RGB")
            rec.update(combine(plist, score_image(model, processor, plist, image, torch)))
            rec["error"] = ""
        except Exception as exc:
            rec["error"] = f"{type(exc).__name__}: {exc}"[:200]
            if isinstance(exc, torch.OutOfMemoryError):
                torch.cuda.empty_cache()
        buffer.append(rec)
        if len(buffer) >= 50 or n == len(todo):
            pd.DataFrame(buffer, columns=columns(meta_cols)).to_csv(out, mode="a", header=header, index=False)
            header, buffer = False, []
            rate = n / (time.time() - start)
            print(f"[morph] {n}/{len(todo)} | {rate:.2f} photos/s | eta "
                  f"{(len(todo) - n) / rate / 60:.1f} min", flush=True)
    print(f"[morph] wrote {out}")


if __name__ == "__main__":
    main()
