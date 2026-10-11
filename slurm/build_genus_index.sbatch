#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Index up to 100 flower-detected photos per angiosperm genus. CPU only.
# Normally submitted by slurm/all_genera.sh. See build_genus_index.py.
#
#     sbatch slurm/build_genus_index.sbatch
#     PER_GENUS=50 sbatch slurm/build_genus_index.sbatch
#
# Writes data/all_genera_index.parquet, data/all_genera_index_genera.csv (per
# genus counts), and caches data/taxa.csv.gz and data/image_paths_all.parquet.
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=sam3_genus_index
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

python build_genus_index.py \
    --annotations "$ANNOTATIONS" \
    --image-root  "$IMAGE_ROOT" \
    --taxa        "${PROJECT}/data/taxa.csv.gz" \
    --path-cache  "${PROJECT}/data/image_paths_all.parquet" \
    --per-genus   "${PER_GENUS:-100}" \
    --seed        "${SEED:-0}" \
    --out         "${PROJECT}/data/all_genera_index.parquet"
