# All-genera run 2

Thirty photos per angiosperm genus, chosen by the phenovision prediction
probability in the annotations CSV (`ANNOTATIONS` in `slurm/settings.sh`, the
same file run 1 used): the genus's 10 lowest, 10 median and 10 highest
flower-detected photos. A genus with fewer than 30 keeps all of them, split
into thirds. SAM3 runs with a detection threshold of 0.5 (run 1 used 0.9) and
keeps only each photo's 5 most confident masks (`segment.py --max-masks 5`).

From the project root (the `flower_sam3` checkout):

```bash
git pull
bash all_genera/run2/run2.sh     # index -> one B200 -> report, chained
```

| Step | Script | Where | Output |
|---|---|---|---|
| 1 | `index.sbatch` → `build_index.py` | CPU | `data/all_genera_run2_index.parquet`, `..._genera.csv` |
| 2 | `segment.sbatch` → `segment.py` | 1 B200, ~2 h | `results/all_genera_run2_flower/` |
| 3 | `report.sbatch` | CPU | `export/all_genera_run2/` (download this) |

What lands in `export/all_genera_run2/`:

- `all_genera_segment_metadata.csv`, `all_genera_genus_summary.csv`,
  `segment_conf_vs_prediction_probability.png`, `run_metrics.csv`,
  `figures/<Family>/...jpg`: as in run 1. The metadata gains `pv_tier`
  (low / median / high) and `pv_rank`.
- `scatter_tier_low.png`, `scatter_tier_median.png`, `scatter_tier_high.png`:
  one point per genus. x is the mean prediction probability of its 10 photos
  in that tier; y is the mean, over those photos with a mask, of each photo's
  mean top-5 mask score. Bars are each genus's 95% t intervals, cut at [0, 1];
  the line is least squares with its 95% confidence band.
- `tier_genus_summary.csv`, `tier_family_summary.csv`: the numbers behind
  those plots, with the CIs.
- `family_histograms/<Family>.png`: every kept mask score of the family's
  photos (up to 5 per photo), stacked by tier.
- `coverage_genera.html`, `coverage_families.html`: interactive (zoom, pan,
  hover); click a point to open the taxon on iNaturalist. Genera or families
  with fewer than 30 phenovision hits or fewer than 30 masked photos, plotted
  as phenovision hits against total iNat observations and coloured by which
  count is short. The iNat count comes from `data/observations.csv.gz` if it
  is there, else `data/inat_photos.parquet` (one row per research-grade photo,
  so the axis then says photos). Override with `INAT=... sbatch ...`.
- `low_hit_taxa.csv`: every genus and family with fewer than 30 phenovision
  hits, with its counts and iNat link.
- `masks/masks.rle.xz` + `masks/masks_index.parquet`: every mask, lossless, as
  small as we could make it (bounding-box crop, run-length varints, one xz
  stream for the whole run; about a fifth of the per-photo `.npz` files).
  Read with `pack_masks.read_pack()`.

Tests: `pytest tests/test_run2.py`.
