#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# What batch size saturates one B200? Run this before slurm/b200.sbatch.
#
#     sbatch slurm/b200_sweep.sbatch
#     BATCH_SIZES=1,8,32,64,128 POOL=256 sbatch slurm/b200_sweep.sbatch
#
# Ten to twenty minutes, one GPU. It times the same pool of decoded images at
# each batch size in fp32 and bf16, prints images/s and peak VRAM for each, and
# ends with the BATCH_SIZE=... line to hand to slurm/b200.sbatch.
#
# Nothing is written and no images are saved, so this measures the GPU alone —
# deliberately, because an hour-long run at the wrong batch size costs an hour.
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=sam3_sweep
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=240gb
#SBATCH --time=00:45:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/settings.sh

LOCAL_INDEX="${LOCAL_INDEX:-${PROJECT}/data/local_image_paths.parquet}"
BATCH_SIZES="${BATCH_SIZES:-1,2,4,8,16,32,64}"
POOL="${POOL:-128}"
DTYPES="${DTYPES:-fp32,bf16}"
STAMP="$(date +%Y%m%d-%H%M%S)"

if [[ ! -f "$LOCAL_INDEX" ]]; then
    echo "ERROR: no local index at $LOCAL_INDEX" >&2
    exit 1
fi

module purge
module load conda
set +u; conda activate "$CONDA_ENV"; set -u
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

echo "node       : $(hostname)"
echo "gpu        : $(nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader)"

python sweep_batch.py \
    --local-index  "$LOCAL_INDEX" \
    --image-root   "$IMAGE_ROOT" \
    --prompt       "$PROMPT" \
    --min-score    "$MIN_SCORE" \
    --batch-sizes  "$BATCH_SIZES" \
    --dtypes       "$DTYPES" \
    --pool         "$POOL" \
    --workers      "$WORKERS" \
    --out          "benchmarks/b200_sweep_${STAMP}.json"
