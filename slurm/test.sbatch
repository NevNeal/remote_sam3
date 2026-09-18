#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Smoke test: one L4, 100 images. Run this before any real job.
#
#     sbatch slurm/test.sbatch
#     tail -f logs/sam3_test-<jobid>.out
#
# If this passes, the array will work — it exercises every moving part
# (conda env, GPU, parquet read, S3 downloads, SAM3 weights, output writes).
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=sam3_test
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --time=00:45:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/settings.sh

module purge
module load conda
conda activate "$CONDA_ENV"

export HF_HUB_OFFLINE=1          # weights are pre-cached by setup_env.sh
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
date

python segment.py \
    --taxon-id "$TAXON_ID" \
    --parquet  "$PARQUET" \
    --out      "${OUT_DIR}_test" \
    --prompt   "$PROMPT" \
    --min-score "$MIN_SCORE" \
    --workers  "$WORKERS" \
    --limit    100

date
python collect.py "${OUT_DIR}_test" --num-shards 1
