#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Index one taxon's flower-detected photos that are already on /blue. CPU only.
#
#     sbatch slurm/build_taxon_index.sbatch                  # taxon 62741
#     TAXON_ID=160559 sbatch slurm/build_taxon_index.sbatch
#
# Writes data/<taxon>_flower_index.parquet (what b200_taxon.sbatch reads) and
# data/<taxon>_photos.parquet (every research-grade photo of the taxon, from
# inat_photos.parquet). Photos the annotations call flower-detected but whose
# image is not on disk go to data/<taxon>_flower_index_missing.csv.
# See build_taxon_index.py for the joins.
#
# Chain the B200 run onto it:
#
#     jid=$(sbatch --parsable slurm/build_taxon_index.sbatch)
#     sbatch --dependency=afterok:$jid slurm/b200_taxon.sbatch
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=sam3_taxon_index
#SBATCH --partition=hpg-default
#SBATCH --cpus-per-task=2
#SBATCH --mem=48gb
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
caller_taxon_id="${TAXON_ID:-}"
source slurm/settings.sh
TAXON_ID="${caller_taxon_id:-62741}"

module purge
module load conda
set +u; conda activate "$CONDA_ENV"; set -u

python build_taxon_index.py \
    --taxon-id    "$TAXON_ID" \
    --parquet     "$PARQUET" \
    --annotations "$ANNOTATIONS" \
    --image-root  "$IMAGE_ROOT" \
    --path-index  "${PATH_INDEX:-${PROJECT}/data/local_image_paths.parquet}" \
    --out         "${PROJECT}/data/${TAXON_ID}_flower_index.parquet"
