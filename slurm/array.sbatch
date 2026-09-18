#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# The real run. One GPU per array task, one shard of the taxon each.
#
#     sbatch --array=0-59 slurm/array.sbatch          # 60 shards, all at once
#     sbatch --array=0-299%40 slurm/array.sbatch      # 300 shards, 40 at a time
#
# num_shards is read from SLURM_ARRAY_TASK_COUNT, so the array size IS the shard
# count and the two can never drift apart. The one exception is rerunning a
# subset, where the count would be wrong — pass it explicitly then:
#
#     NUM_SHARDS=60 sbatch --array=7,23 slurm/array.sbatch
#
# collect.py prints that line for you when a shard is missing.
#
# Per-run overrides work the same way, because sbatch forwards your environment:
#
#     TAXON_ID=62741 PROMPT=petal sbatch --array=0-39 slurm/array.sbatch
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=sam3
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48gb
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --requeue
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/settings.sh

# SLURM_ARRAY_TASK_COUNT is the number of tasks in this array. Correct for
# 0-59 and for 0-299%40 alike; wrong only for a partial rerun, hence the override.
NUM_SHARDS="${NUM_SHARDS:-$SLURM_ARRAY_TASK_COUNT}"
SHARD="$SLURM_ARRAY_TASK_ID"

module purge
module load conda
conda activate "$CONDA_ENV"

export HF_HUB_OFFLINE=1          # weights are pre-cached by setup_env.sh
# Without this, every shard's BLAS spawns a thread per core on the node and
# three shards sharing an hpg-turin node thrash each other.
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

echo "shard $SHARD of $NUM_SHARDS on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
date

python segment.py \
    --taxon-id   "$TAXON_ID" \
    --parquet    "$PARQUET" \
    --out        "$OUT_DIR" \
    --prompt     "$PROMPT" \
    --min-score  "$MIN_SCORE" \
    --workers    "$WORKERS" \
    --shard      "$SHARD" \
    --num-shards "$NUM_SHARDS"

date
