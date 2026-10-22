#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# All-genera run 2, as three chained jobs. Run from the project root (the
# flower_sam3 checkout):
#
#     bash all_genera/run2/run2.sh
#
#   1. index.sbatch     CPU   30 photos per angiosperm genus: the 10 lowest,
#                             10 median and 10 highest phenovision probabilities
#   2. segment.sbatch   B200  SAM3 at min score 0.5, top 5 masks per photo
#   3. report.sbatch    CPU   metadata, tier scatter plots, family histograms,
#                             coverage plots, low-hit CSV, packed masks
#
# Each starts only if the one before it succeeded. An existing index is reused;
# REBUILD_INDEX=1 builds it again.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
if [[ ! -f segment.py || ! -f all_genera/run2/run2.sh ]]; then
    echo "ERROR: run from the project root (the flower_sam3 checkout)" >&2
    exit 1
fi
mkdir -p logs

dep=""
if [[ -n "${REBUILD_INDEX:-}" || ! -f data/all_genera_run2_index.parquet ]]; then
    index=$(sbatch --parsable all_genera/run2/index.sbatch)
    echo "1. index   : job $index"
    dep="--dependency=afterok:$index"
else
    echo "1. index   : data/all_genera_run2_index.parquet exists, skipped"
fi
# $dep is deliberately unquoted: empty means no argument at all.
gpu=$(sbatch --parsable $dep all_genera/run2/segment.sbatch)
echo "2. segment : job $gpu   (hpg-b200, 1 GPU)"
report=$(sbatch --parsable --dependency="afterok:$gpu" all_genera/run2/report.sbatch)
echo "3. report  : job $report"
echo
echo "watch      : squeue -u \$USER"
echo "download   : export/all_genera_run2/   (when job $report has finished)"
