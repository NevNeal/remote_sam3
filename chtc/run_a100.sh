#!/bin/bash
# Container entrypoint for running the MAIN segmentation_pipeline.py on a GPU node.
#
# Calls the pipeline directly with its positional args:  taxon_id  output_folder  [prompt]
#
# Output is collected by HTCondor file transfer (when_to_transfer_output =
# ON_EXIT_OR_EVICT, see the .sub) and returned to the access point. After the
# job finishes, push the whole output folder to Research Drive once with
# chtc/push_to_researchdrive.sh from a transfer node. There is no incremental /
# per-batch shipping — the pipeline's mask_summary.csv makes the job resumable,
# so an evicted long run continues instead of restarting from zero.
set -euo pipefail

TAXON_ID="${1:-62741}"
OUTPUT_DIR="${2:-output_62741}"
PROMPT="${3:-}"   # empty = pipeline default ("flower")

export HF_HOME="$PWD/hf_cache"
mkdir -p "$HF_HOME" "$OUTPUT_DIR"

echo "[run_a100.sh] PWD=$PWD"
echo "[run_a100.sh] HF_HOME=$HF_HOME"
echo "[run_a100.sh] taxon=$TAXON_ID output=$OUTPUT_DIR prompt=${PROMPT:-<default 'flower'>}"
echo "[run_a100.sh] HF_TOKEN present: ${HF_TOKEN:+yes}"
echo "[run_a100.sh] nvidia-smi:"
nvidia-smi || echo "  (nvidia-smi not available in container)"

ARGS=("$TAXON_ID" "$OUTPUT_DIR")
if [[ -n "$PROMPT" ]]; then
    ARGS+=(--prompt "$PROMPT")
fi

exec python segmentation_pipeline.py "${ARGS[@]}"
