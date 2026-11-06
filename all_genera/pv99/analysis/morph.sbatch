#!/bin/bash
# pv99 analysis, GPU part: Qwen3-VL-8B flower-type calls on one B200, then the
# family comparison figure and the methodology PDF (CPU, a minute).
# Resumable: resubmit and photos already in morphology/photos.csv are skipped.
# Submit through all_genera/pv99/analysis/run.sh.
#SBATCH --job-name=pv99_analysis_morph
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96gb
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --requeue
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/settings.sh
module purge
module load conda
set +u; conda activate "$CONDA_ENV"; set -u
export HF_HUB_OFFLINE=1
export MPLBACKEND=Agg

A=all_genera/pv99/analysis
echo "[morph] $(date -Is) on $(hostname): $(nvidia-smi --query-gpu=name,uuid --format=csv,noheader)"
python $A/morph_classify.py
python $A/family_types.py
python $A/methodology_pdf.py
echo "[morph] $(date -Is) done"
