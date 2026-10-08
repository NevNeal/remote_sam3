#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Overlays and cut-outs for a run segmented with MASKS_ONLY=1. CPU only.
#
#     sbatch --array=0-19 slurm/render.sbatch                     # taxon 62741
#     TAXON_ID=160559 sbatch --array=0-19 slurm/render.sbatch
#     RUN_DIR=results/62741_flower sbatch --array=0-19 slurm/render.sbatch
#
# Every task reads the same results CSVs and takes the photos where
# row_index % array size == its task id. A task that dies is resumed by
# resubmitting it: photos whose overlay already exists are skipped. Like
# array.sbatch, pass NUM_SHARDS when rerunning a subset:
#
#     NUM_SHARDS=20 sbatch --array=7,13 slurm/render.sbatch
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=sam3_render
#SBATCH --partition=hpg-default
#SBATCH --cpus-per-task=16
#SBATCH --mem=64gb
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --requeue
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
caller_taxon_id="${TAXON_ID:-}"
source slurm/settings.sh
TAXON_ID="${caller_taxon_id:-62741}"
RUN_DIR="${RUN_DIR:-${RESULTS}/${TAXON_ID}_${PROMPT// /_}}"

module purge
module load conda
set +u; conda activate "$CONDA_ENV"; set -u
export OMP_NUM_THREADS=1        # one PNG per process; no BLAS thread fan-out

python render.py "$RUN_DIR" --prompt "$PROMPT" \
    --shard "${SLURM_ARRAY_TASK_ID:-0}" \
    --num-shards "${NUM_SHARDS:-${SLURM_ARRAY_TASK_COUNT:-1}}" \
    --workers "${SLURM_CPUS_PER_TASK:-16}"
