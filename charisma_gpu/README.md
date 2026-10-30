# charisma on the GPU, for SAM3 runs

R charisma 1.0.0 re-implemented in PyTorch, plus a CLUT that separates
*Rudbeckia hirta* rays, red bullseye and disk. The general-purpose package
(tests, parity checks, methodology PDF) is its own repository, `charisma-gpu`.
This folder is the copy that reads this project's run outputs.

| File | What it is |
|---|---|
| `charisma.py` | the pipeline (same code as `charisma_gpu/core.py` in the standalone repo) |
| `run.py` | every segment of a run: `--run-dir` (photos + `.npz` masks) or `--png-dir` (cut-outs) |
| `clut_default.csv` | charisma's own CLUT, exported unchanged |
| `clut_rhirta.csv` | bullseye / green / disk / ray / pale / other |
| `petal.py` | red on the petals vs red on the disk: finds the disk by shape, counts bullseye only outside it |
| `parity.R`, `parity.py` | R charisma against this, on the same images |

```bash
# straight from a segment.py run directory (no cut-outs needed)
python charisma_gpu/run.py --run-dir results/62741_flower \
    --clut charisma_gpu/clut_rhirta.csv --out results/62741_flower/charisma_rhirta.csv

# or from rendered cut-outs (render.py's segments/)
python charisma_gpu/run.py --png-dir path/to/segments \
    --clut charisma_gpu/clut_rhirta.csv --out charisma_rhirta.csv
```

There's one row per segment: `px_<colour>` (per-pixel share), `px_<colour>_rgb`,
`ch_k` and `ch_<colour>` (charisma's answer). With the *R. hirta* CLUT it also
writes `petal_bullseye` (red on the petals only; use this one), `disk_red` and
`disk_area`. A reddish-brown disk is the same colour as a maroon bullseye, so
`px_bullseye` alone counts red disks too. `--png-dir` leaves `mask_score`
blank; join it from the run's `results.csv` on `photo_id` and `mask_index`.

Measured on an RTX 4060 laptop: all 126,653 taxon 62741 segments in ~2 minutes.
