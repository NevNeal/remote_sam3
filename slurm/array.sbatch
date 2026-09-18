#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# The real run. One GPU per array task, one shard of the taxon each.
#
#     sbatch --array=0-59 slurm/array.sbatch          # 60 shards, all at once
#     sbatch --array=0-299%40 slurm/array.sbatch      # 300 shards, 40 at a time
#
# Run slurm/test_2gpu.sbatch first — it is this same job at width 2.
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

# SLURM_ARRAY_TASK_COUNT is the number of tasks in this array. Correct for
# 0-59 and for 0-299%40 alike; wrong only for a partial rerun, hence the override.
export SHARD="$SLURM_ARRAY_TASK_ID"
export NUM_SHARDS="${NUM_SHARDS:-$SLURM_ARRAY_TASK_COUNT}"

bash "$SLURM_SUBMIT_DIR/slurm/run_shard.sh"
