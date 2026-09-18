#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# THE WORKHORSE: a SLURM job array of independent single-L4 shards.
#
# This replaces the HTCondor `queue 5` model.  Each array task is one whole
# HTCondor-style job: 1 GPU, its own shard of the taxon, its own results CSV.
# Nothing is shared but the read-only parquet, the read-only HF cache, and the
# output directory — so the array scales to whatever the scheduler will give
# you, and a task that dies costs you only its own shard.
#
# Why L4 rather than B200 for this: there are ~600 L4s vs ~504 B200s, but the
# B200s live in a separate partition meant for large-model training and are far
# more contended.  SAM3 inference needs ~4 GB of VRAM and is latency-bound on
# image downloads, so 60 L4s finish sooner than 8 B200s you had to wait in line
# for.  See hpg/README.md → "Choosing a partition".
#
#   ./hpg/submit.sh array 60          # 60 shards
#   ./hpg/submit.sh array 60 4        # 60 shards, at most 4 running at once
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=sam3_l4
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48gb
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --requeue

set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/config.sh"

TAXON_ID="${TAXON_ID:-160559}"
PROMPT="${PROMPT:-flower}"
OUTPUT_DIR="${OUTPUT_DIR:-${RESULTS_ROOT}/${TAXON_ID}_${PROMPT// /_}}"

# NUM_SHARDS must equal the array width and must stay constant across reruns —
# it defines which rows belong to which shard.  submit.sh exports it.
NUM_SHARDS="${NUM_SHARDS:?NUM_SHARDS must be exported (use hpg/submit.sh)}"
SHARD="${SLURM_ARRAY_TASK_ID:-0}"

export WORKERS="${WORKERS:-8}"

date
echo "array task $SHARD of $NUM_SHARDS -> $OUTPUT_DIR"
"$HERE/run_shard.sh" "$TAXON_ID" "$OUTPUT_DIR" "$PROMPT" "$SHARD" "$NUM_SHARDS"
date
