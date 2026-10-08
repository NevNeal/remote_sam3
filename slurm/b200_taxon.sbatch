#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# One B200, batched, a whole taxon from S3, masks only.
#
#     sbatch slurm/b200_taxon.sbatch                        # taxon 62741, 'flower'
#     TAXON_ID=160559 sbatch slurm/b200_taxon.sbatch
#     LIMIT=2000 sbatch slurm/b200_taxon.sbatch             # smoke test first
#
# Then, as a separate CPU job once this one has finished:
#
#     sbatch --array=0-19 slurm/render.sbatch               # overlays + cut-outs
#
# What differs from b200.sbatch (the one-hour local benchmark):
#
#   TAXON_ID    photos come from the parquet index and are downloaded from iNat's
#               S3 bucket into <out>/images/, where render.sbatch finds them later.
#   MASKS_ONLY  only the .npy masks are written. Overlays and cut-outs are PNG
#               encoding — CPU work that would otherwise hold the B200 hostage.
#   no clock    the whole taxon, not an hour of it. If the walltime runs out,
#               resubmit the same command: resume is by photo_id and the images
#               already downloaded are reused.
#   WORKERS     48 download threads. These are network waits, not CPU, so they
#               go well past the 14 cores; back off if download_failed shows 429s.
#   DISCARD     off — the masks are the point.
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
#SBATCH --time=2-00:00:00
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
caller_batch_size="${BATCH_SIZE:-}"
caller_save_workers="${SAVE_WORKERS:-}"
caller_workers="${WORKERS:-}"
caller_bf16="${BF16-1}"                      # unset -> 1; BF16= (empty) -> fp32
source slurm/settings.sh

export SHARD=0
export NUM_SHARDS=1

export LOCAL_INDEX=                          # a taxon run: parquet + S3, not /blue
export TAXON_ID="${caller_taxon_id:-62741}"
export LIMIT="${LIMIT-}"
export MAX_SECONDS=                          # the whole taxon, not a timed sample
export BATCH_SIZE="${caller_batch_size:-32}"
export SAVE_WORKERS="${caller_save_workers:-10}"
export WORKERS="${caller_workers:-48}"
export BF16="$caller_bf16"
export MASKS_ONLY=1
export DISCARD=
export TELEMETRY_S="${TELEMETRY_S:-5}"
export OUT_DIR="${caller_out_dir:-${RESULTS}/${TAXON_ID}_${PROMPT// /_}}"

bash "$SLURM_SUBMIT_DIR/slurm/run_shard.sh"

echo
echo "== taxon ${TAXON_ID}, one B200, masks only =="
python collect.py "$OUT_DIR" --num-shards 1 || true
python run_metrics.py "$OUT_DIR" --scan-masks || true
echo
echo "masks on disk : $(du -sh --apparent-size "$OUT_DIR/masks" 2>/dev/null | cut -f1) apparent," \
     "$(du -sh "$OUT_DIR/masks" 2>/dev/null | cut -f1) allocated"
echo "metrics       : $OUT_DIR/run_metrics.csv"
