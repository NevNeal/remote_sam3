#!/bin/bash
# Full photo atlas of the bullseye genera: tiles cut from the source photos on
# /blue, then one PDF (CPU, ~10 min). Run after bullseye_report.sbatch.
#     sbatch all_genera/pv99/analysis/bullseye_atlas.sbatch
#     ATLAS_ARGS="--min-share 0.4" sbatch all_genera/pv99/analysis/bullseye_atlas.sbatch
#SBATCH --job-name=pv99_bullseye_atlas
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
python -c "import reportlab" 2>/dev/null || pip install --quiet reportlab
python all_genera/pv99/analysis/bullseye_atlas.py --workers $((SLURM_CPUS_PER_TASK - 1)) ${ATLAS_ARGS:-}
