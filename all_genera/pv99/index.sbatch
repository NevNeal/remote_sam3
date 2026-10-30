#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# The pv99 run's index: every flower-detected angiosperm photo phenovision
# scored at >= MIN_PV (0.99). CPU only. See build_index.py next to this file.
# Reuses run 1's cached data/taxa.csv.gz and data/image_paths_all.parquet.
# The photo count is the last lines of the log.
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=sam3_pv99_index
#SBATCH --partition=hpg-default
#SBATCH --cpus-per-task=16
#SBATCH --mem=96gb
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/settings.sh

module purge
module load conda
set +u; conda activate "$CONDA_ENV"; set -u

python all_genera/pv99/build_index.py \
    --annotations     "$ANNOTATIONS" \
    --image-root      "$IMAGE_ROOT" \
    --taxa            "${PROJECT}/data/taxa.csv.gz" \
    --path-cache      "${PROJECT}/data/image_paths_all.parquet" \
    --min-probability "${MIN_PV:-0.99}" \
    --out             "${PROJECT}/data/pv99_index.parquet"
