#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# After run 2's B200 job. CPU only. Everything lands in export/all_genera_run2/,
# which is the download:
#
#   all_genera_segment_metadata.csv, all_genera_genus_summary.csv   as run 1,
#       plus pv_tier / pv_rank
#   segment_conf_vs_prediction_probability.png    as run 1, one point per photo
#   tier_genus_summary.csv, tier_family_summary.csv
#   scatter_tier_<low|median|high>.png            per genus, with 95% CIs
#   family_histograms/<Family>.png                mask scores stacked by tier
#   coverage_genera.html, coverage_families.html  interactive, click -> iNat
#   low_hit_taxa.csv                              under 30 phenovision hits
#   masks/masks.rle.xz + masks_index.parquet      every mask, packed
#   run_metrics.csv, all_genera_run2_index_genera.csv
#   figures/<Family>/<Genus>_<low|median|high>_<photo_id>.jpg   as run 1
#
# INAT: the iNat source for observation counts. Defaults to the observations
# dump if build_index.py left it in data/, else data/inat_photos.parquet.
# Resubmit if cut off: review figures that exist are not redrawn.
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=sam3_run2_report
#SBATCH --partition=hpg-default
#SBATCH --cpus-per-task=16
#SBATCH --mem=64gb
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/settings.sh
RUN_DIR="${RUN_DIR:-${RESULTS}/all_genera_run2_${PROMPT// /_}}"
EXPORT="${EXPORT:-${PROJECT}/export/all_genera_run2}"
INDEX="${PROJECT}/data/all_genera_run2_index.parquet"
if [[ -z "${INAT:-}" ]]; then
    INAT="${PROJECT}/data/inat_photos.parquet"
    if [[ -f "${PROJECT}/data/observations.csv.gz" ]]; then
        INAT="${PROJECT}/data/observations.csv.gz"
    fi
fi

module purge
module load conda
set +u; conda activate "$CONDA_ENV"; set -u
export OMP_NUM_THREADS=1
export MPLCONFIGDIR="${PROJECT}/.mplconfig"      # not the 40 GB $HOME
python -c "import matplotlib" 2>/dev/null \
    || pip install --no-cache-dir "matplotlib==3.10.9"

python all_genera/run2/report.py "$RUN_DIR" --index "$INDEX" --export "$EXPORT" \
    --workers "${SLURM_CPUS_PER_TASK:-16}"
python plot_conf_vs_probability.py "$EXPORT/all_genera_segment_metadata.csv"
python all_genera/run2/coverage.py \
    --genera   "${INDEX%.parquet}_genera.csv" \
    --metadata "$EXPORT/all_genera_segment_metadata.csv" \
    --taxa     "${PROJECT}/data/taxa.csv.gz" \
    --inat     "$INAT" \
    --export   "$EXPORT"
python all_genera/run2/pack_masks.py "$RUN_DIR" \
    --metadata "$EXPORT/all_genera_segment_metadata.csv" --out "$EXPORT/masks"

for f in "$RUN_DIR/run_metrics.csv" "${INDEX%.parquet}_genera.csv"; do
    if [[ -f "$f" ]]; then cp "$f" "$EXPORT/"; fi
done

echo
echo "export     : $EXPORT  ($(du -sh --apparent-size "$EXPORT" | cut -f1))"
echo "             path for Globus: $(realpath "$EXPORT")"
