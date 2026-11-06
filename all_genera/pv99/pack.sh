#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# One tar of everything pv99 left behind, for download without Globus:
#
#   export/pv99/            metadata parquet, genus summary, run metrics
#   results/pv99_flower/    every .npz mask, the shard CSVs, telemetry
#   data/pv99_index*        the index the GPU job read
#   phenobase_inat_data/... only the source images the local figures need
#                           (pack_select.py: ~40 per family + 200 top-scoring
#                           photos for each of the 30 biggest families)
#
# From the project root, on a compute node (it reads ~300k small files):
#
#     srun -p hpg-default -c 2 --mem 8gb -t 2:00:00 bash all_genera/pv99/pack.sh
#
# Writes export/pv99_bundle.tar (uncompressed: masks and webp already are).
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
if [[ ! -f segment.py || ! -f all_genera/pv99/pack.sh ]]; then
    echo "ERROR: run from the project root (the flower_sam3 checkout)" >&2
    exit 1
fi
source slurm/settings.sh
RUN_DIR="${RUN_DIR:-results/pv99_${PROMPT// /_}}"
EXPORT="${EXPORT:-export/pv99}"
BUNDLE="${BUNDLE:-export/pv99_bundle.tar}"

module purge
module load conda
set +u; conda activate "$CONDA_ENV"; set -u

python all_genera/pv99/pack_select.py "$EXPORT/pv99_segment_metadata.parquet" \
    --image-root "$IMAGE_ROOT" --out "$EXPORT/pack_images.txt"

tar -cf "$BUNDLE.part" "$EXPORT" "$RUN_DIR" data/pv99_index* \
    -C "$IMAGE_ROOT" -T "$PWD/$EXPORT/pack_images.txt"
mv "$BUNDLE.part" "$BUNDLE"
echo "bundle     : $BUNDLE  ($(du -sh --apparent-size "$BUNDLE" | cut -f1))"
echo "download   : scp hpg.rc.ufl.edu:$PWD/$BUNDLE ."
