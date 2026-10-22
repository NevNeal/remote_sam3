#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Run 2's index: 10 lowest / 10 median / 10 highest phenovision probability
# photos per angiosperm genus. CPU only. See build_index.py next to this file.
# Reuses run 1's cached data/taxa.csv.gz and data/image_paths_all.parquet.
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=sam3_run2_index
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

python all_genera/run2/build_index.py \
    --annotations "$ANNOTATIONS" \
    --image-root  "$IMAGE_ROOT" \
    --taxa        "${PROJECT}/data/taxa.csv.gz" \
    --path-cache  "${PROJECT}/data/image_paths_all.parquet" \
    --per-tier    "${PER_TIER:-10}" \
    --seed        "${SEED:-0}" \
    --out         "${PROJECT}/data/all_genera_run2_index.parquet"
