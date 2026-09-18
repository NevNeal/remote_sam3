#!/bin/bash
# build_sif.sh — convert the Docker Hub image into an Apptainer .sif on HPG.
#
# Run on a HiPerGator LOGIN node (not in a job).  The pull is network + CPU
# heavy but short; if it ever gets killed for resource use, run it inside a
# dev session instead:  srundev --time=60 --mem=16gb --cpus-per-task=4
#
# Usage:
#   ./hpg/build_sif.sh

set -euo pipefail
source "$(dirname "$0")/config.sh"

[[ "$HPG_GROUP" == "CHANGEME" ]] && { echo "ERROR: set HPG_GROUP in hpg/config.sh"; exit 1; }

module purge
module load apptainer

mkdir -p "$SIF_DIR" "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"

echo "[sif] building $SIF"
echo "[sif]   from  docker://$DOCKER_IMAGE"
echo "[sif]   cache $APPTAINER_CACHEDIR"

# --force so a re-run replaces a stale image rather than erroring out.
apptainer build --force "$SIF" "docker://$DOCKER_IMAGE"

ls -lh "$SIF"

echo
echo "[sif] sanity check — GPU arch list the image can actually target:"
apptainer exec "$SIF" python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda); print('arch_list', torch.cuda.get_arch_list())"
echo
echo "[sif] sm_100 must appear above for hpg-b200 to work."
