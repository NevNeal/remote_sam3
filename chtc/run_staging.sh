#!/bin/bash
# Entrypoint for a single-GPU staging segmentation job on CHTC.
#
# The parquet index is delivered into the job scratch dir by the submit file's
# transfer_input_files (osdf:///chtc/staging/.../inat_photos.parquet), so it is
# simply ./inat_photos.parquet at runtime.
set -euo pipefail

TAXON_ID="${1:?usage: run_staging.sh <taxon_id> <output_dir> [prompt] [parquet]}"
OUTPUT_DIR="${2:?usage: run_staging.sh <taxon_id> <output_dir> [prompt] [parquet]}"
PROMPT="${3:-flower}"
PARQUET="${4:-inat_photos.parquet}"

export HF_HOME="$PWD/hf_cache"
mkdir -p "$HF_HOME" "$OUTPUT_DIR"

echo "[staging] PWD=$PWD"
echo "[staging] taxon_id=$TAXON_ID  output=$OUTPUT_DIR  prompt='$PROMPT'  parquet=$PARQUET"
ls -lh "$PARQUET" || { echo "ERROR: parquet not found at $PARQUET"; exit 1; }
nvidia-smi || echo "(nvidia-smi not available)"

exec python staging_pipeline.py \
    --taxon-id   "$TAXON_ID"  \
    --output-dir "$OUTPUT_DIR" \
    --parquet    "$PARQUET"   \
    --prompt     "$PROMPT"
