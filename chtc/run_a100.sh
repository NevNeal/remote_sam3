#!/bin/bash
# Container entrypoint for running the MAIN segmentation_pipeline.py on an A100.
#
# Unlike run_tests.sh (which wraps test_on_chtc.py), this calls the pipeline
# directly with its positional args:  taxon_id  output_folder  [prompt]
#
# It enables incremental Research Drive shipping: SHIP_BATCHES=1 makes the
# pipeline tar each completed 1000-image batch and push it off this compute
# node via `condor_chirp put` into the job spool, where chtc/watch_and_push.sh
# (running on an access/transfer node) forwards it to Research Drive.
#
# NOTE: condor_chirp must be available inside the container for shipping to
# fire. If it is not on PATH the pipeline logs a warning and disables shipping
# gracefully — results still come back via HTCondor's ON_EXIT transfer.
set -euo pipefail

TAXON_ID="${1:-336671}"
OUTPUT_DIR="${2:-output_336671}"
PROMPT="${3:-}"   # empty = pipeline default ("open flower")

export HF_HOME="$PWD/hf_cache"
mkdir -p "$HF_HOME" "$OUTPUT_DIR"

# Incremental shipping of completed batches off the compute node.
export SHIP_BATCHES=1
# export SHIP_POLL_SEC=30                    # how often to scan for completed batches
# export SHIP_CMD="condor_chirp put {src} {dst}"   # override the off-node transfer command

echo "[run_a100.sh] PWD=$PWD"
echo "[run_a100.sh] HF_HOME=$HF_HOME"
echo "[run_a100.sh] taxon=$TAXON_ID output=$OUTPUT_DIR prompt=${PROMPT:-<default>}"
echo "[run_a100.sh] SHIP_BATCHES=$SHIP_BATCHES"
echo "[run_a100.sh] condor_chirp: $(command -v condor_chirp || echo '<not found — shipping will disable>')"
echo "[run_a100.sh] nvidia-smi:"
nvidia-smi || echo "  (nvidia-smi not available in container)"

ARGS=("$TAXON_ID" "$OUTPUT_DIR")
if [[ -n "$PROMPT" ]]; then
    ARGS+=(--prompt "$PROMPT")
fi

exec python segmentation_pipeline.py "${ARGS[@]}"
