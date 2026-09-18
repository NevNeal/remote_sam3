#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# One whole DGX B200 node: 8 B200 GPUs, 8 shards, one GPU pinned per shard.
#
# This is the "vertical" alternative to the L4 array.  A DGX B200 node is
# 8 × 180 GB Blackwell + 2 × Xeon Platinum 8570 (112 cores) + 2 TB RAM, so one
# node can host 8 independent SAM3 workers and still give each of them 12 CPU
# cores for image downloads and PNG encoding.
#
# Use this when:
#   * you want a single allocation instead of N queue positions, or
#   * you are benchmarking B200 vs L4 throughput per image, or
#   * you later batch multiple images per forward pass and actually need VRAM.
#
# Be aware: hpg-b200 is the contended partition and RC asks that it be used for
# jobs that genuinely need the VRAM or the FLOPs.  A single-image SAM3 forward
# pass uses ~4 GB of 180 GB, so running the L4 array is both faster in
# wall-clock (less queueing) and the better-neighbour choice.  This script
# exists for the benchmark and for the batched future.
#
#   ./hpg/submit.sh b200                 # shards 0-7  of 8
#   NUM_SHARDS=32 SHARD_BASE=8 ./hpg/submit.sh b200   # shards 8-15 of 32
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=sam3_b200
#SBATCH --partition=hpg-b200
#SBATCH --gpus=8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=768gb
#SBATCH --time=3-00:00:00
#SBATCH --requeue

set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/config.sh"

TAXON_ID="${TAXON_ID:-160559}"
PROMPT="${PROMPT:-flower}"
OUTPUT_DIR="${OUTPUT_DIR:-${RESULTS_ROOT}/${TAXON_ID}_${PROMPT// /_}}"

GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
# Total shards across the whole run, and this node's offset into that range.
NUM_SHARDS="${NUM_SHARDS:-$GPUS_PER_NODE}"
SHARD_BASE="${SHARD_BASE:-0}"

# 96 cores / 8 workers = 12 each; leave a little headroom for Lustre and the
# kernel rather than claiming all 112.
export WORKERS="${WORKERS:-12}"

date
nvidia-smi --query-gpu=index,name,memory.total --format=csv

pids=()
for gpu in $(seq 0 $((GPUS_PER_NODE - 1))); do
    shard=$((SHARD_BASE + gpu))
    if (( shard >= NUM_SHARDS )); then
        echo "shard $shard >= NUM_SHARDS ($NUM_SHARDS); leaving GPU $gpu idle"
        continue
    fi
    echo "launching shard $shard of $NUM_SHARDS on GPU $gpu"
    "$HERE/run_shard.sh" "$TAXON_ID" "$OUTPUT_DIR" "$PROMPT" \
        "$shard" "$NUM_SHARDS" "$gpu" \
        > "${LOG_DIR}/b200-${SLURM_JOB_ID}-gpu${gpu}.out" 2>&1 &
    pids+=($!)
done

# Wait for all 8, but do not let one failure abandon the others mid-run.
rc=0
for pid in "${pids[@]}"; do
    wait "$pid" || rc=1
done

date
echo "all shards exited (rc=$rc)"
exit "$rc"
