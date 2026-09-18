#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Parallel test: 2 shards on 2 L4 GPUs at the same time, 200 images total.
#
#     sbatch slurm/test_2gpu.sbatch
#
# This is the full multi-GPU design at its smallest size — the same job array
# and the same run_shard.sh that array.sbatch uses — so if it passes, the only
# thing that changes for a real run is the array width.
#
# Each array task is its own 1-GPU job. SLURM may put the two on the same node
# (an hpg-turin node has 3 L4s) or on two nodes; either is fine, because the
# shards never talk to each other. What the test proves is that they RUN AT THE
# SAME TIME on DIFFERENT GPUs and together cover every photo exactly once:
#
#     sacct -j <jobid> --format=JobID,NodeList,Start,End,Elapsed,State
#         -> two rows (_0 and _1), overlapping Start/End, both COMPLETED
#     grep -h '^gpu' logs/sam3_2gpu-<jobid>_*.out
#         -> two different GPU UUIDs
#     python collect.py results/160559_flower_test2gpu --num-shards 2
#         -> 200 photos, both shards present, no duplicates
#
# Overrides as usual: LIMIT=1000 TAXON_ID=62741 sbatch slurm/test_2gpu.sbatch
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=sam3_2gpu
#SBATCH --partition=hpg-turin
#SBATCH --array=0-1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48gb
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/settings.sh

export SHARD="$SLURM_ARRAY_TASK_ID"
export NUM_SHARDS="$SLURM_ARRAY_TASK_COUNT"
export LIMIT="${LIMIT:-200}"
export OUT_DIR="${OUT_DIR}_test2gpu"

bash slurm/run_shard.sh
