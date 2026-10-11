#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Review figures for judging segment quality by eye. CPU only.
#
#     sbatch slurm/review.sbatch                       # 200 random photos, taxon 62741
#     SAMPLE=1000 sbatch slurm/review.sbatch           # a bigger sample
#     SAMPLE= sbatch --array=0-3 slurm/review.sbatch   # every photo, 4 tasks
#     RUN_DIR=results/160559_flower sbatch slurm/review.sbatch
#
# One PNG per photo in <run>/review/batch_NNNNN/: the image with its 5 most
# confident masks outlined and numbered on the left, those 5 cut-outs with their
# confidence on the right. See review_figure.py. A photo whose figure exists is
# skipped, so a rerun only draws what is missing; SEED picks a different sample.
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=sam3_review
#SBATCH --partition=hpg-default
#SBATCH --cpus-per-task=16
#SBATCH --mem=32gb
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
caller_taxon_id="${TAXON_ID:-}"
source slurm/settings.sh
TAXON_ID="${caller_taxon_id:-62741}"
RUN_DIR="${RUN_DIR:-${RESULTS}/${TAXON_ID}_${PROMPT// /_}}"
SAMPLE="${SAMPLE-200}"                       # SAMPLE= (empty) draws every photo

module purge
module load conda
set +u; conda activate "$CONDA_ENV"; set -u
export OMP_NUM_THREADS=1

args=("$RUN_DIR"
      --shard "${SLURM_ARRAY_TASK_ID:-0}"
      --num-shards "${NUM_SHARDS:-${SLURM_ARRAY_TASK_COUNT:-1}}"
      --workers "${SLURM_CPUS_PER_TASK:-16}"
      --seed "${SEED:-0}")
if [[ -n "$SAMPLE" ]]; then
    args+=(--sample "$SAMPLE")
fi
python review_figure.py "${args[@]}"
