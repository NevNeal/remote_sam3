#!/bin/bash
# Container entrypoint: forwards args to test_on_chtc.py.
# Inside the Apptainer container, /app already has the pipeline code from the
# Dockerfile, but HTCondor's transfer_input_files lands fresh copies in $PWD,
# so we run from $PWD to pick those up.
set -euo pipefail

TAXON_ID="${1:-591507}"
OUTPUT_DIR="${2:-test_output}"
LIMIT="${3:-}"   # empty = no limit (process all images in the taxon)

export HF_HOME="$PWD/hf_cache"
mkdir -p "$HF_HOME"

echo "[run_tests.sh] PWD=$PWD"
echo "[run_tests.sh] HF_HOME=$HF_HOME"
echo "[run_tests.sh] taxon=$TAXON_ID output=$OUTPUT_DIR limit=${LIMIT:-<none>}"
echo "[run_tests.sh] nvidia-smi:"
nvidia-smi || echo "  (nvidia-smi not available in container)"

mkdir -p "$OUTPUT_DIR"

ARGS=(--taxon-id "$TAXON_ID" --output-dir "$OUTPUT_DIR" --results-file test_results.json)
if [[ -n "$LIMIT" ]]; then
    ARGS+=(--limit "$LIMIT")
fi

exec python test_on_chtc.py "${ARGS[@]}"
