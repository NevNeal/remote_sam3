#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Segment images that are ALREADY on /blue — no taxon, no S3, no downloads.
#
#     sbatch --array=0-4 slurm/local.sbatch            # 5 GPUs, first 5,000 images
#     LIMIT= sbatch --array=0-59 slurm/local.sbatch    # no limit: the whole index
#
# The image list is data/local_image_paths.parquet: the 2,849,542 flowering rows
# of repro_annotations_full_2026-07-14.csv. Each row's `file_name` starts with
# `data/`, which IMAGE_ROOT replaces — see slurm/settings.sh.
#
# Identical to array.sbatch except for LOCAL_INDEX, IMAGE_ROOT, LIMIT and a
# separate OUT_DIR, so sharding, resume and logging all behave the same way.
# LIMIT takes the first N rows (ordered by photo_id) BEFORE sharding, so
# LIMIT=5000 across 5 shards is 1,000 images each.
#
# The images are read in place and never modified or deleted — only <out>/masks,
# overlays and segments are written.
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=sam3_local
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48gb
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --requeue
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/settings.sh

export SHARD="$SLURM_ARRAY_TASK_ID"
export NUM_SHARDS="${NUM_SHARDS:-$SLURM_ARRAY_TASK_COUNT}"
export LOCAL_INDEX="${LOCAL_INDEX:-${PROJECT}/data/local_image_paths.parquet}"
export IMAGE_ROOT="${IMAGE_ROOT}"
export LIMIT="${LIMIT-5000}"        # LIMIT= (empty) means the whole index
export OUT_DIR="${OUT_DIR:-${RESULTS}/local_${PROMPT// /_}}"

bash "$SLURM_SUBMIT_DIR/slurm/run_shard.sh"
