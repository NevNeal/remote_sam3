#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# archive_to_orange.sh — the HiPerGator stand-in for
# chtc/push_to_researchdrive.sh.
#
# On CHTC, results had to leave the cluster because job scratch was ephemeral
# and only transfer.chtc.wisc.edu could reach Research Drive.  On HiPerGator
# nothing has to move at all: /blue is already persistent, group-shared storage
# that the login nodes, compute nodes and Open OnDemand all see.
#
# What this script is for is the second step — getting a finished run off the
# high-performance (and expensive, and quota-limited) /blue filesystem onto
# /orange, which is the group's bulk/archival tier.  A full taxon run is tens to
# hundreds of GB of PNGs, so this matters quickly.
#
# Usage:
#   ./hpg/archive_to_orange.sh /blue/<group>/<user>/results/160559_flower
#   TAR=1 ./hpg/archive_to_orange.sh <run_dir>     # single tarball instead
#
# For copies off HiPerGator entirely (to a lab NAS, Dropbox, another
# university), use Globus via the UFRC endpoint rather than scp — it resumes.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/config.sh"

RUN_DIR="${1:?usage: archive_to_orange.sh <run_dir> [dest_subdir]}"
RUN_DIR="$(cd "$RUN_DIR" && pwd)"
RUN_NAME="$(basename "$RUN_DIR")"
DEST_SUBDIR="${2:-sam3/$(date +%Y-%m-%d)}"
DEST="/orange/${HPG_GROUP}/${HPG_USER}/${DEST_SUBDIR}"

[[ -d "/orange/${HPG_GROUP}" ]] || {
    echo "ERROR: /orange/${HPG_GROUP} does not exist."
    echo "       Orange is an investment tier — your group may not have one."
    echo "       Check with: orange_quota"
    exit 1
}

mkdir -p "$DEST"
echo "[archive] source : $RUN_DIR ($(du -sh "$RUN_DIR" | cut -f1))"
echo "[archive] dest   : $DEST/$RUN_NAME"
orange_quota || true

if [[ "${TAR:-0}" == "1" ]]; then
    # One big file is far kinder to Lustre metadata than a million small PNGs,
    # and much faster to move again later.
    echo "[archive] tarring (this is the slow part)..."
    tar -C "$(dirname "$RUN_DIR")" -cf "$DEST/${RUN_NAME}.tar" "$RUN_NAME"
    ls -lh "$DEST/${RUN_NAME}.tar"
else
    rsync -ah --info=progress2 "$RUN_DIR/" "$DEST/$RUN_NAME/"
fi

echo "[archive] done. /blue copy left in place — delete it yourself once verified:"
echo "            rm -rf $RUN_DIR"
