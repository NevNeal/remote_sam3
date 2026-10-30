#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# After classify.sbatch. CPU only: one contact sheet per genus (its 10 crops,
# each photo's calls under it, the genus's calls on top), then everything into
# export/morphology/, which is the download. Every sheet is redrawn, so the
# calls on it are current after a resumed run. SHEETS=0 skips the sheets and only exports the CSVs.
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=morphology_sheets
#SBATCH --partition=hpg-default
#SBATCH --cpus-per-task=16
#SBATCH --mem=32gb
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/settings.sh
RUN_DIR="${RUN_DIR:-${RESULTS}/all_genera_run2_${PROMPT// /_}}"
INDEX="${INDEX:-${PROJECT}/data/morphology_index.parquet}"
MORPH_OUT="${MORPH_OUT:-${RESULTS}/morphology}"
EXPORT="${EXPORT:-${PROJECT}/export/morphology}"

module purge
module load conda
set +u; conda activate "$CONDA_ENV"; set -u
export OMP_NUM_THREADS=1

python morphology/classify.py summarize --out-dir "$MORPH_OUT"
if [[ "${SHEETS:-1}" != 0 ]]; then
    python morphology/classify.py sheets --index "$INDEX" --run-dir "$RUN_DIR" \
        --out-dir "$MORPH_OUT" --workers "${SLURM_CPUS_PER_TASK:-16}"
fi

mkdir -p "$EXPORT"
for f in photos.csv genus_morphology.csv genus_probabilities.csv \
         family_morphology.csv validation.csv; do
    if [[ -f "$MORPH_OUT/$f" ]]; then cp "$MORPH_OUT/$f" "$EXPORT/"; fi
done
cp morphology/questions.py "$EXPORT/"
if [[ -d "$MORPH_OUT/sheets" ]]; then
    rsync -a "$MORPH_OUT/sheets/" "$EXPORT/sheets/"
fi
echo
echo "export     : $EXPORT  ($(du -sh --apparent-size "$EXPORT" | cut -f1))"
echo "             path for Globus: $(realpath "$EXPORT")"
