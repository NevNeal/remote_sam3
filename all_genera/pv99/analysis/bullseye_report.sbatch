#!/bin/bash
# Bullseye genus/family tables, figures and contact sheets (CPU, minutes).
#SBATCH --job-name=pv99_bullseye_report
#SBATCH --partition=hpg-default
#SBATCH --cpus-per-task=16
#SBATCH --mem=48gb
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
export MPLBACKEND=Agg
python all_genera/pv99/analysis/bullseye_report.py
