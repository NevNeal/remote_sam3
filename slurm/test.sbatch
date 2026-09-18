#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Smoke test: one L4, 100 images. Run this before anything else.
#
#     sbatch slurm/test.sbatch
#     tail -f logs/sam3_test-<jobid>.out
#
# Exercises every moving part (conda env, GPU, parquet read, S3 downloads, SAM3
# weights, output writes) on a single GPU. Then slurm/test_2gpu.sbatch checks
# the parallel side.
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=sam3_test
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --time=00:45:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/settings.sh

export SHARD=0
export NUM_SHARDS=1
export LIMIT="${LIMIT:-100}"
export OUT_DIR="${OUT_DIR}_test"

bash slurm/run_shard.sh

module load conda
set +u; conda activate "$CONDA_ENV"; set -u    # conda's activate scripts trip over set -u
python collect.py "$OUT_DIR" --num-shards 1
