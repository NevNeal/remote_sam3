#!/bin/bash
# Container entrypoint: forwards args to test_on_chtc.py.
# Inside the Apptainer container, /app already has the pipeline code from the
# Dockerfile, but HTCondor's transfer_input_files lands fresh copies in $PWD,
# so we run from $PWD to pick those up.
set -euo pipefail

TAXON_ID="${1:-591507}"
OUTPUT_DIR="${2:-test_output}"
LIMIT="${3:-1}"

echo "[run_tests.sh] PWD=$PWD"
echo "[run_tests.sh] taxon=$TAXON_ID output=$OUTPUT_DIR limit=$LIMIT"
echo "[run_tests.sh] nvidia-smi:"
nvidia-smi || echo "  (nvidia-smi not available in container)"

mkdir -p "$OUTPUT_DIR"

exec python test_on_chtc.py \
    --taxon-id "$TAXON_ID" \
    --output-dir "$OUTPUT_DIR" \
    --limit "$LIMIT" \
    --results-file test_results.json
