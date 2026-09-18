#!/bin/bash
# setup_blue.sh — one-time preparation of the /blue working area.
#
# Creates the directory layout, then warms the shared HuggingFace cache by
# downloading facebook/sam3 ONCE on a login node.  Array tasks then read the
# cache read-only with HF_HUB_OFFLINE=1, so 300 concurrent shards cause zero
# HuggingFace traffic.
#
# Usage:
#   export HF_TOKEN=hf_...        # facebook/sam3 is a gated repo
#   ./hpg/setup_blue.sh

set -euo pipefail
source "$(dirname "$0")/config.sh"

[[ "$HPG_GROUP" == "CHANGEME" ]] && { echo "ERROR: set HPG_GROUP in hpg/config.sh"; exit 1; }
: "${HF_TOKEN:?export HF_TOKEN=hf_... first (facebook/sam3 is gated)}"

mkdir -p "$BLUE"/{data,results,logs,sif,hf_cache}
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"
echo "[setup] layout under $BLUE:"
ls -la "$BLUE"

echo
echo "[setup] warming HF cache at $HF_HOME (facebook/sam3) ..."
module purge
module load apptainer
apptainer exec --bind "$BLUE:$BLUE" --env "HF_HOME=$HF_HOME" --env "HF_TOKEN=$HF_TOKEN" \
    "$SIF" python - <<'PY'
from transformers import Sam3Model, Sam3Processor
Sam3Model.from_pretrained("facebook/sam3")
Sam3Processor.from_pretrained("facebook/sam3")
print("cache warm")
PY

du -sh "$HF_HOME"
echo
echo "[setup] next: put the parquet index at $PARQUET"
echo "        from your workstation:"
echo "          scp inat_db/data/inat_photos.parquet ${HPG_USER}@hpg.rc.ufl.edu:${PARQUET}"
echo "        (3.9 GB — or use Globus / the 'hpg-xfer' data transfer endpoint)"
echo "        then:  sbatch hpg/sbatch_l4_test.sh"
