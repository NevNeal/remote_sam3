#!/bin/bash
# Entrypoint for each parallel A100 shard job.
set -euo pipefail

TAXON_IDS="${1:-49320}"
OUTPUT_DIR="${2:-output_shard_0}"
SHARD="${3:-0}"
NUM_SHARDS="${4:-5}"

export HF_HOME="$PWD/hf_cache"
mkdir -p "$HF_HOME" "$OUTPUT_DIR"

echo "[shard ${SHARD}/${NUM_SHARDS}] PWD=$PWD"
echo "[shard ${SHARD}/${NUM_SHARDS}] taxon_ids=$TAXON_IDS  output=$OUTPUT_DIR"
nvidia-smi || echo "(nvidia-smi not available)"

exec python segmentation_pipeline.py \
    --taxon-ids  "$TAXON_IDS"   \
    --output-dir "$OUTPUT_DIR"  \
    --shard      "$SHARD"       \
    --num-shards "$NUM_SHARDS"