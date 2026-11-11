#!/bin/bash
# Bullseye calls for the top segment of every masked pv99 photo (~270k),
# Qwen3-VL-8B on one B200 per array task. NUM_SHARDS must equal the array
# size; bullseye_run.sh keeps them in step. Resumable: resubmit a task and the
# photos already in its photos_shard<k>.csv are skipped.
#     NUM_SHARDS=4 sbatch --array=<task> all_genera/pv99/analysis/bullseye.sbatch
#SBATCH --job-name=pv99_bullseye
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128gb
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --requeue
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/settings.sh
module purge
module load conda
set +u; conda activate "$CONDA_ENV"; set -u
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

NUM_SHARDS="${NUM_SHARDS:-1}"
TASK="${SLURM_ARRAY_TASK_ID:-0}"
echo "[bullseye] $(date -Is) on $(hostname): task $TASK/$NUM_SHARDS, $(nvidia-smi --query-gpu=name --format=csv,noheader)"
python all_genera/pv99/analysis/bullseye_classify.py \
    --shard "$TASK" --num-shards "$NUM_SHARDS" \
    --workers $((SLURM_CPUS_PER_TASK - 2)) ${BULLSEYE_ARGS:-}
echo "[bullseye] $(date -Is) done"
