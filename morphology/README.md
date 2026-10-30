# Flower morphology per genus

Ten photos per genus from all-genera run 2. Each is cropped to its SAM3 flower
masks and shown to a vision-language model (Qwen3-VL-8B-Instruct) with six
multiple-choice questions (`questions.py`). The questions cover:

| key | options |
|---|---|
| `unit` | solitary · inflorescence · head (capitulum) · spadix · spikelet (glumes) · catkin · other · none |
| `symmetry` | actinomorphic · zygomorphic · asymmetric · unclear |
| `display` | petals · bracts · stamens · reduced (wind) · unclear |
| `fusion` | free · fused · no petals · unclear |
| `form` | open · tubular · bell · lipped · papilionaceous · orchid · spurred · unclear |
| `merosity` | 3/6 · 4/8 · 5/10 · many · unclear |

It is zero-shot: no training labels are needed. The model never writes an
answer. One forward pass per question gives its next-token probabilities over
the option letters, and those probabilities are the photo's answer. A genus's
call is the option with the highest mean probability over its photos, among
the options that aren't "can't tell". Each call comes with:

- `share`: that option's share of the non-abstain probability
- `agreement`: the fraction of the genus's photos whose own choice matches
- `abstain`: the mean probability on "can't tell"
- `confidence`: high / medium / low, from the three above

## Run it

From the project root (the `flower_sam3` checkout), after run 2:

```bash
git fetch && git checkout flower-morphology
python morphology/classify.py download   # once, on a login node: ~17 GB into hf_cache/
LIMIT=2000 bash morphology/run.sh        # a ~10 minute test: check the log's photos/s
bash morphology/run.sh                   # the whole thing
```

| Step | Script | Where | Output |
|---|---|---|---|
| 1 | `classify.sbatch` | 1 B200, < 8 h | `data/morphology_index.parquet`, `results/morphology/` |
| 2 | `sheets.sbatch` | CPU | `export/morphology/` (download this) |

That's about 90,600 photos (10,734 genera; 84,117 photos have masks, and the
rest are shown uncropped) times 6 questions. Each question is ~490 tokens, 256
of them the image. Scaling a measured RTX 4060 run up to a B200 projects about
2 hours. The job stops itself at 7.5 h anyway (`MAX_SECONDS`), writes its
summaries, and finishes inside the 8 h limit. If it was cut short, submit
`classify.sbatch` again and it resumes from `photos.csv`.

Inputs default to run 2's locations. Override any of them per run:
`METADATA` (the report's `all_genera_segment_metadata.csv`), `RUN_DIR` (where
the `.npz` masks live), `MORPH_OUT`, `MORPH_MODEL`, `PER_GENUS`, `BATCH_SIZE`.

## What lands in `export/morphology/`

- `genus_morphology.csv`: one row per genus. For each question it has the
  call, `_share`, `_agreement`, `_abstain` and `_confidence`, plus a
  `description` such as `inflorescence · zygomorphic · petals · fused · lipped?`
  (`?` = medium confidence; low-confidence calls are left out).
- `genus_probabilities.csv`: the mean probability of every option per genus.
  These are the features to train on later.
- `family_morphology.csv`: the same calls, per family.
- `photos.csv`: per photo, every option's probability, the photo's call, and
  `__letter_mass` (how much of the model's probability fell on the letters at
  all; near 1 means it was answering the question).
- `validation.csv`: the calls checked against `reference.csv`, ~60 genera
  whose traits are textbook (*Arum* spadix, *Poa* spikelets, *Salix* catkin,
  *Vicia* pea-like, *Campanula* fused bell...). The summary prints the
  accuracy per question.
- `sheets/<Family>/<Genus>.jpg`: the genus's 10 crops with each photo's calls
  under it and the genus's calls on top, for checking by eye.

## Caveats

- Zero-shot calls are a first pass, not ground truth. Read `validation.csv`
  and a sample of sheets before trusting a question. Petal count was the
  weakest question in the local test.
- The crop is the box around the SAM3 masks plus a 25% margin. When SAM3
  boxed one flower of an umbel, `unit` may see a solitary flower.
- Models have some letter-position bias. If a question looks skewed toward
  `A`, reorder its options in `questions.py` and rerun into a fresh
  `MORPH_OUT` (`photos.csv` columns follow the questions).
- Next step, if a question matters: hand-label a few hundred genera from the
  sheets, then train a small classifier on `genus_probabilities.csv`.

Tests: `pytest tests/test_morphology.py`.
