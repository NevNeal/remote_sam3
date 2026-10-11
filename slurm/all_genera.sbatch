#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# One B200 over the all-genera index: up to 100 flower-detected photos of every
# angiosperm genus, read in place on /blue, masks only, packed .npz masks.
# Normally submitted by slurm/all_genera.sh, after build_genus_index.sbatch.
#
# One GPU for as long as it takes: the walltime is 4 days and the job requeues;
# if it is ever killed, resubmitting resumes by photo_id.
#
#   MASK_FORMAT=npz  one bit-packed, compressed file per photo (maskio.py)
#                    instead of one full-frame .npy per mask: ~100x less disk
#                    and a third of the files.
#   MIN_SCORE        0.9, from settings.sh - only detections at or above it.
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=sam3_all_genera
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=240gb
#SBATCH --time=4-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --requeue
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
caller_out_dir="${OUT_DIR:-}"
caller_local_index="${LOCAL_INDEX:-}"
caller_batch_size="${BATCH_SIZE:-}"
caller_save_workers="${SAVE_WORKERS:-}"
caller_bf16="${BF16-1}"                      # unset -> 1; BF16= (empty) -> fp32
source slurm/settings.sh

export SHARD=0
export NUM_SHARDS=1
export LOCAL_INDEX="${caller_local_index:-${PROJECT}/data/all_genera_index.parquet}"
export IMAGE_ROOT="${IMAGE_ROOT}"
export LIMIT=
export MAX_SECONDS=
export BATCH_SIZE="${caller_batch_size:-32}"
export SAVE_WORKERS="${caller_save_workers:-12}"
export WORKERS="${WORKERS:-16}"
export BF16="$caller_bf16"
export MASKS_ONLY=1
export MASK_FORMAT=npz
export DISCARD=
export TELEMETRY_S="${TELEMETRY_S:-10}"
export OUT_DIR="${caller_out_dir:-${RESULTS}/all_genera_${PROMPT// /_}}"

if [[ ! -f "$LOCAL_INDEX" ]]; then
    echo "ERROR: no index at $LOCAL_INDEX - run slurm/build_genus_index.sbatch" >&2
    exit 1
fi

bash "$SLURM_SUBMIT_DIR/slurm/run_shard.sh"

# run_shard.sh activated the env in its own shell; this one needs it too.
module purge
module load conda
set +u; conda activate "$CONDA_ENV"; set -u

echo
echo "== all genera, one B200, masks only =="
python collect.py "$OUT_DIR" --num-shards 1 || true
python run_metrics.py "$OUT_DIR" --scan-masks || true
echo
echo "masks on disk : $(du -sh --apparent-size "$OUT_DIR/masks" 2>/dev/null | cut -f1) apparent," \
     "$(du -sh "$OUT_DIR/masks" 2>/dev/null | cut -f1) allocated"
