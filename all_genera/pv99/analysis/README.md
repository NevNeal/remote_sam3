# pv99 analysis

Everything runs on HPG against the finished pv99 run and writes into
`export/pv99/analysis/`. From the project root (the `flower_sam3` checkout):

```bash
git fetch && git checkout pv99-analysis
bash all_genera/pv99/analysis/run.sh
```

`run.sh` installs reportlab into the env if it is missing, caches
`Qwen/Qwen3-VL-8B-Instruct` (~17 GB, once), then submits two chained jobs:

| Job | Where | Scripts | Output in `export/pv99/analysis/` |
|---|---|---|---|
| `prep.sbatch` | CPU | `histogram.py` | `top_confidence_histogram.png` (+ `_bins.csv`): every photo's top mask score |
| | | `top_segments.py` | `top_segments/`: top 100 segments of each of the 30 biggest families as cut-outs, context crops, 10 x 10 contact sheets, `top_segments.csv` |
| | | `figures.py` | `figures/<Family>/`: run-2-style review figures, 30 photos per family (all a family has if fewer) |
| `morph.sbatch` | 1 B200 | `morph_classify.py` | `morphology/photos.csv`: per segment, the model's probability for every option of 8 botanical questions |
| | | `family_types.py` | `morphology/flower_types_by_family.png` and the family calls, shares, summary and calibration CSVs |
| | | `methodology_pdf.py` | `flower_types_methodology.pdf`: how each call was made |

## Choices

- **Figure sample**: per family, masked photos shuffled once and taken one per
  genus before any genus gives a second (`pack_select.figure_pool`), 10 spares.
- **Top segments**: a photo's segment is its best mask; per family the 200
  photos with the highest `conf_max`, taken in score order with at most 10 per
  genus (`--genus-cap`), filled without the cap if short.
- **Flower types**: three lines of evidence, kept separate. The textbook
  expectation per family is in `family_reference.csv`. The image evidence is
  zero-shot Qwen3-VL-8B answering the questions in `morph_questions.py`, with
  letter probabilities and both option orders averaged. The decision rules are
  in `family_types.py`. The PDF prints all three.

`SKIP_PREP=1 bash all_genera/pv99/analysis/run.sh` reruns only the GPU half.
`morph.sbatch` resumes from `photos.csv` if it is cut short.

## Bullseye pattern

Which genera and families have a visible bullseye (a target-like face, as in
black-eyed Susan)? One zero-shot Qwen3-VL-8B question (`bullseye_question.py`)
over the context crop of the top segment of **every** masked pv99 photo
(~270k), not just the 3,000 top segments above.

```bash
bash all_genera/pv99/analysis/bullseye_run.sh test   # 200 photos, prints photos/s
bash all_genera/pv99/analysis/bullseye_run.sh        # 4 B200 shards + report
```

Outputs in `export/pv99/analysis/bullseye/`: `photos.csv`, `genus_bullseye.csv`,
`family_bullseye.csv` (empirical-Bayes beta-binomial ranking with 90% credible
intervals), `controls.csv`, `top_genera.png`, `top_families.png` and contact
sheets under `sheets/` for checking precision by eye.
