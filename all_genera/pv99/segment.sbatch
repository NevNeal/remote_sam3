#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# SAM3 over the pv99 index, one B200 per array task, read in place on /blue,
# masks only (one packed .npz per photo with a detection), no overlays or
# cut-outs.
#
#   MIN_SCORE=0.5   SAM3's detection threshold
#   MAX_MASKS=5     then only each photo's 5 most confident detections are kept
#
# NUM_SHARDS must equal the array size; run.sh keeps them in step. ~20M photos
# at ~119k photos/hour per B200 is ~170 GPU-hours, so 4 shards is ~2 days each.
# A killed or timed-out task resumes by photo_id when resubmitted:
#     NUM_SHARDS=4 sbatch --array=<task> all_genera/pv99/segment.sbatch
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=sam3_pv99
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=240gb
#SBATCH --time=4-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --requeue
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
caller_out_dir="${OUT_DIR:-}"
caller_bf16="${BF16-1}"                      # unset -> 1; BF16= (empty) -> fp32
source slurm/settings.sh

: "${NUM_SHARDS:?NUM_SHARDS not set - submit through all_genera/pv99/run.sh}"
export SHARD="${SLURM_ARRAY_TASK_ID:-0}"
export NUM_SHARDS
export LOCAL_INDEX="${PROJECT}/data/pv99_index.parquet"
export IMAGE_ROOT
export MIN_SCORE=0.5
export MAX_MASKS=5
export LIMIT="${LIMIT:-}"
export MAX_SECONDS=
export BATCH_SIZE="${BATCH_SIZE:-32}"
export SAVE_WORKERS="${SAVE_WORKERS:-12}"
export WORKERS="${WORKERS:-16}"
export BF16="$caller_bf16"
export MASKS_ONLY=1
export MASK_FORMAT=npz
export DISCARD=
export TELEMETRY_S="${TELEMETRY_S:-10}"
export OUT_DIR="${caller_out_dir:-${RESULTS}/pv99_${PROMPT// /_}}"

if [[ ! -f "$LOCAL_INDEX" ]]; then
    echo "ERROR: no index at $LOCAL_INDEX - run all_genera/pv99/index.sbatch" >&2
    exit 1
fi

bash "$SLURM_SUBMIT_DIR/slurm/run_shard.sh"
