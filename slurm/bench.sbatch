#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Benchmark array: a full taxon across N GPUs with everything timed and nothing
# kept. Don't submit this directly — use slurm/bench.sh, which also queues the
# report-and-cleanup job behind it.
#
# Identical to array.sbatch except:
#   DISCARD=1      every image/mask/overlay/cut-out is written (so save time and
#                  size are real), measured, then deleted — disk use stays ~0
#   TELEMETRY_S=5  GPU utilisation / memory / power sampled every 5 s
#   no --requeue   a requeued task would restart its clock and blur the timing
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=sam3_bench
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48gb
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

: "${OUT_DIR:?submit with slurm/bench.sh, which sets OUT_DIR}"
export SHARD="$SLURM_ARRAY_TASK_ID"
export NUM_SHARDS="${NUM_SHARDS:-$SLURM_ARRAY_TASK_COUNT}"
export DISCARD=1
export TELEMETRY_S="${TELEMETRY_S:-5}"

bash "$SLURM_SUBMIT_DIR/slurm/run_shard.sh"
