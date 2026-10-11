#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# The whole all-genera run, as three chained jobs. Run from the project root:
#
#     bash slurm/all_genera.sh
#
#   1. build_genus_index.sbatch   CPU   which photos, which files
#   2. all_genera.sbatch          B200  segment them, masks only
#   3. all_genera_report.sbatch   CPU   metadata CSV, 3 figures per genus, plot
#
# Each starts only if the one before it succeeded. If the index already exists
# (a rerun), step 1 is skipped; pass REBUILD_INDEX=1 to build it again.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
if [[ ! -f segment.py || ! -f slurm/settings.sh ]]; then
    echo "ERROR: run from the project root (the flower_sam3 checkout)" >&2
    exit 1
fi
mkdir -p logs

dep=""
if [[ -n "${REBUILD_INDEX:-}" || ! -f data/all_genera_index.parquet ]]; then
    index=$(sbatch --parsable slurm/build_genus_index.sbatch)
    echo "1. index   : job $index"
    dep="--dependency=afterok:$index"
else
    echo "1. index   : data/all_genera_index.parquet exists, skipped"
fi
# $dep is deliberately unquoted: empty means no argument at all.
gpu=$(sbatch --parsable $dep slurm/all_genera.sbatch)
echo "2. segment : job $gpu   (hpg-b200, 1 GPU)"
report=$(sbatch --parsable --dependency="afterok:$gpu" slurm/all_genera_report.sbatch)
echo "3. report  : job $report"
echo
echo "watch      : squeue -u \$USER"
echo "logs       : logs/sam3_genus_index-*.out  logs/sam3_all_genera-$gpu.out"
echo "download   : export/all_genera/   (when job $report has finished)"
