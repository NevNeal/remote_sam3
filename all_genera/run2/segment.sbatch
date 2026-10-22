#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# One B200 over run 2's index (~30 photos per angiosperm genus), read in place
# on /blue, masks only, packed .npz masks.
#
#   MIN_SCORE=0.5   SAM3's detection threshold (run 1 used 0.9)
#   MAX_MASKS=5     then only each photo's 5 most confident detections are kept
#
# Run 1 did 119k photos/hour on one B200; this is ~235k photos. If the job is
# killed, resubmitting resumes by photo_id.
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=sam3_run2
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=240gb
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --requeue
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
caller_out_dir="${OUT_DIR:-}"
caller_bf16="${BF16-1}"                      # unset -> 1; BF16= (empty) -> fp32
source slurm/settings.sh

export SHARD=0
export NUM_SHARDS=1
export LOCAL_INDEX="${PROJECT}/data/all_genera_run2_index.parquet"
export IMAGE_ROOT
export MIN_SCORE=0.5
export MAX_MASKS=5
export LIMIT=
export MAX_SECONDS=
export BATCH_SIZE="${BATCH_SIZE:-32}"
export SAVE_WORKERS="${SAVE_WORKERS:-12}"
export WORKERS="${WORKERS:-16}"
export BF16="$caller_bf16"
export MASKS_ONLY=1
export MASK_FORMAT=npz
export DISCARD=
export TELEMETRY_S="${TELEMETRY_S:-10}"
export OUT_DIR="${caller_out_dir:-${RESULTS}/all_genera_run2_${PROMPT// /_}}"

if [[ ! -f "$LOCAL_INDEX" ]]; then
    echo "ERROR: no index at $LOCAL_INDEX - run all_genera/run2/index.sbatch" >&2
    exit 1
fi

bash "$SLURM_SUBMIT_DIR/slurm/run_shard.sh"

module purge
module load conda
set +u; conda activate "$CONDA_ENV"; set -u

echo
echo "== all genera run 2, one B200, min score 0.5, top 5 masks =="
python collect.py "$OUT_DIR" --num-shards 1 || true
python run_metrics.py "$OUT_DIR" --scan-masks || true
