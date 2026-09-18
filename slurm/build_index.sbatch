#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Build the parquet index on HiPerGator: download ~28 GB of iNat CSVs into
# /blue and join them there.
#
#     sbatch slurm/build_index.sbatch
#
# Only needed if you are NOT copying an existing inat_photos.parquet up from
# your workstation (scp is much faster — see README step 3). Run this for the
# monthly refresh, when iNat republishes the dumps, so the 28 GB never has to
# touch your laptop.
#
# No GPU. hpg-default is the general CPU partition and is easy to get.
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=inat_index
#SBATCH --partition=hpg-default
#SBATCH --cpus-per-task=8
#SBATCH --mem=24gb
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/settings.sh

module purge
module load conda
conda activate "$CONDA_ENV"

date
# --memory-limit stays under the 24 GB request so DuckDB spills to disk rather
# than being killed by the cgroup. --threads matches --cpus-per-task.
python build_index.py \
    --data-dir "$(dirname "$PARQUET")" \
    --memory-limit 16GB \
    --threads "$SLURM_CPUS_PER_TASK"
date

echo
echo "Index is at $PARQUET — next: sbatch slurm/test.sbatch"
