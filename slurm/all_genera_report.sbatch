#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# After the all-genera B200 run: the metadata CSV, three review figures per
# genus (lowest / median / highest confidence) as JPEG, and the scatter plot.
# CPU only. Normally submitted by slurm/all_genera.sh.
#
# Everything lands in export/all_genera/ - that folder is the download:
#
#     all_genera_segment_metadata.csv
#     all_genera_genus_summary.csv
#     segment_conf_vs_prediction_probability.png
#     run_metrics.csv, all_genera_index_genera.csv
#     figures/<Family>/<Genus>_<low|median|high>_<photo_id>.jpg
#
# Resubmit it if it is cut off: figures that exist are not redrawn.
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=sam3_genera_report
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
RUN_DIR="${RUN_DIR:-${RESULTS}/all_genera_${PROMPT// /_}}"
EXPORT="${EXPORT:-${PROJECT}/export/all_genera}"
INDEX="${INDEX:-${PROJECT}/data/all_genera_index.parquet}"

module purge
module load conda
set +u; conda activate "$CONDA_ENV"; set -u
export OMP_NUM_THREADS=1
export MPLCONFIGDIR="${PROJECT}/.mplconfig"      # not the 40 GB $HOME

# matplotlib joined requirements.txt after the env was built; add it if absent.
python -c "import matplotlib" 2>/dev/null \
    || pip install --no-cache-dir "matplotlib==3.10.9"

python all_genera_report.py "$RUN_DIR" --index "$INDEX" --export "$EXPORT" \
    --workers "${SLURM_CPUS_PER_TASK:-16}"
python plot_conf_vs_probability.py "$EXPORT/all_genera_segment_metadata.csv"

for f in "$RUN_DIR/run_metrics.csv" "${INDEX%.parquet}_genera.csv"; do
    if [[ -f "$f" ]]; then cp "$f" "$EXPORT/"; fi
done

echo
echo "export     : $EXPORT"
echo "             $(find "$EXPORT/figures" -name '*.jpg' 2>/dev/null | wc -l) figures," \
     "$(du -sh --apparent-size "$EXPORT" | cut -f1) total"
echo "             path for Globus: $(realpath "$EXPORT")"
