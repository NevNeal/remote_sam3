#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# After every pv99 GPU task: collect the shards, run metrics, and the metadata
# table in export/pv99/. CPU only. The .npz masks are left in place under
# results/pv99_flower/masks/. CSV=1 also writes the metadata as CSV.
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=sam3_pv99_export
#SBATCH --partition=hpg-default
#SBATCH --cpus-per-task=8
#SBATCH --mem=192gb
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/settings.sh
RUN_DIR="${RUN_DIR:-${RESULTS}/pv99_${PROMPT// /_}}"
EXPORT="${EXPORT:-${PROJECT}/export/pv99}"

module purge
module load conda
set +u; conda activate "$CONDA_ENV"; set -u

python collect.py "$RUN_DIR" ${NUM_SHARDS:+--num-shards "$NUM_SHARDS"} \
    --retry-ids "$RUN_DIR/retry_ids.txt" || true
python run_metrics.py "$RUN_DIR" || true
python all_genera/pv99/export.py "$RUN_DIR" \
    --index "${PROJECT}/data/pv99_index.parquet" --export "$EXPORT" ${CSV:+--csv}

echo
echo "metadata   : $EXPORT  ($(du -sh --apparent-size "$EXPORT" | cut -f1))"
echo "masks      : $RUN_DIR/masks  ($(du -sh --apparent-size "$RUN_DIR/masks" | cut -f1))"
