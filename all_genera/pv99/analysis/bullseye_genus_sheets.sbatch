#!/bin/bash
# One bullseye summary sheet per genus (>= 10 bullseye photos; 360 genera) plus
# the ImageAnt script, cut from the source photos on /blue (CPU, ~10 min).
#     sbatch all_genera/pv99/analysis/bullseye_genus_sheets.sbatch
#     SHEET_ARGS="--min-bullseye 5" sbatch all_genera/pv99/analysis/bullseye_genus_sheets.sbatch
#SBATCH --job-name=pv99_bullseye_sheets
#SBATCH --partition=hpg-default
#SBATCH --cpus-per-task=16
#SBATCH --mem=32gb
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/settings.sh
module purge
module load conda
set +u; conda activate "$CONDA_ENV"; set -u
python all_genera/pv99/analysis/bullseye_genus_sheets.py --workers $((SLURM_CPUS_PER_TASK - 1)) ${SHEET_ARGS:-}
