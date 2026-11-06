#!/bin/bash
# pv99 analysis, CPU part: histogram, top segments, review figures.
# Submit through all_genera/pv99/analysis/run.sh.
#SBATCH --job-name=pv99_analysis_prep
#SBATCH --partition=hpg-default
#SBATCH --cpus-per-task=16
#SBATCH --mem=64gb
#SBATCH --time=6:00:00
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

A=all_genera/pv99/analysis
echo "[prep] $(date -Is) on $(hostname), $SLURM_CPUS_PER_TASK CPUs"
python $A/histogram.py
python $A/top_segments.py
python $A/figures.py
echo "[prep] $(date -Is) done"
