#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# One B200, batched, every flower-detected photo of a taxon that is already on
# /blue, masks only.
#
#     jid=$(sbatch --parsable slurm/build_taxon_index.sbatch)       # CPU, first
#     sbatch --dependency=afterok:$jid slurm/b200_taxon.sbatch       # then this
#
#     TAXON_ID=160559 sbatch slurm/b200_taxon.sbatch                 # another taxon
#
# Then, as separate CPU jobs once this one has finished:
#
#     sbatch --array=0-9 slurm/render.sbatch        # overlays + cut-outs
#     sbatch slurm/pack_masks.sbatch                # masks -> .tar.zst for Globus
#
# What it reads: data/<taxon>_flower_index.parquet, from build_taxon_index.sbatch
# — the taxon's photos (inat_photos.parquet) that the phenovision annotations
# mark 'flower present' + Detected, each with its image path under IMAGE_ROOT.
# Images are opened where they lie; nothing is downloaded, copied or deleted.
#
# What differs from b200.sbatch (the one-hour benchmark over all of /blue):
#
#   LOCAL_INDEX  the one taxon's flower index, not the 2.8M-row flowering index.
#   MASKS_ONLY   only the .npy masks are written. Overlays and cut-outs are PNG
#                encoding — CPU work that would otherwise hold the B200 hostage.
#   no clock     every photo in the index, not an hour of them. If the walltime
#                runs out, resubmit: resume is by photo_id.
#   DISCARD      off — the masks are the point.
#
# The job ends by merging the shard CSV (collect.py) and writing the metrics
# (run_metrics.py): run_metrics.csv is the flat table of every number, including
# seconds per image, seconds per segment, and the size of every .npy on disk.
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=sam3_b200_taxon
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
# settings.sh fills these with L4-array defaults; capture what the caller passed
# first, so this job's own defaults apply only when nothing was.
caller_taxon_id="${TAXON_ID:-}"
caller_out_dir="${OUT_DIR:-}"
caller_local_index="${LOCAL_INDEX:-}"
caller_batch_size="${BATCH_SIZE:-}"
caller_save_workers="${SAVE_WORKERS:-}"
caller_bf16="${BF16-1}"                      # unset -> 1; BF16= (empty) -> fp32
source slurm/settings.sh

export SHARD=0
export NUM_SHARDS=1

export TAXON_ID="${caller_taxon_id:-62741}"
export LOCAL_INDEX="${caller_local_index:-${PROJECT}/data/${TAXON_ID}_flower_index.parquet}"
export IMAGE_ROOT="${IMAGE_ROOT}"
export LIMIT=                                # every photo in the index
export MAX_SECONDS=                          # the whole index, not a timed sample
export BATCH_SIZE="${caller_batch_size:-32}"
export SAVE_WORKERS="${caller_save_workers:-10}"
export WORKERS="${WORKERS:-16}"
export BF16="$caller_bf16"
export MASKS_ONLY=1
export DISCARD=
export TELEMETRY_S="${TELEMETRY_S:-5}"
export OUT_DIR="${caller_out_dir:-${RESULTS}/${TAXON_ID}_${PROMPT// /_}}"

if [[ ! -f "$LOCAL_INDEX" ]]; then
    echo "ERROR: no taxon index at $LOCAL_INDEX" >&2
    echo "       run: TAXON_ID=$TAXON_ID sbatch slurm/build_taxon_index.sbatch" >&2
    exit 1
fi

bash "$SLURM_SUBMIT_DIR/slurm/run_shard.sh"

# run_shard.sh activated the env in its own shell; this one needs it too.
module purge
module load conda
set +u; conda activate "$CONDA_ENV"; set -u

echo
echo "== taxon ${TAXON_ID}, one B200, masks only =="
python collect.py "$OUT_DIR" --num-shards 1 || true
python run_metrics.py "$OUT_DIR" --scan-masks || true
echo
echo "masks on disk : $(du -sh --apparent-size "$OUT_DIR/masks" 2>/dev/null | cut -f1) apparent," \
     "$(du -sh "$OUT_DIR/masks" 2>/dev/null | cut -f1) allocated"
echo "metrics       : $OUT_DIR/run_metrics.csv"
