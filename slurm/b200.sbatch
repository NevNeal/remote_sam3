#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# One B200, batched, one hour on the clock.
#
#     sbatch slurm/b200.sbatch                     # 1 hour, batch 32, bf16
#     BATCH_SIZE=64 sbatch slurm/b200.sbatch       # after the sweep says 64
#     MAX_SECONDS=300 sbatch slurm/b200.sbatch     # 5 minutes, to check it runs
#
# The question this answers: how many of the phenovision images already on /blue
# can a single B200 segment in an hour? So it is one GPU, not an array — the
# per-GPU number is what a later array multiplies.
#
# Differences from local.sbatch, which are the whole point:
#
#   BATCH_SIZE  images per forward pass. local.sbatch runs 1, inherited from the
#               original pipeline; one image uses ~4 GB of a B200's 180 GB, so
#               at 1 the GPU is mostly idle. Run slurm/b200_sweep.sbatch first
#               to find the batch size that actually saturates it.
#   SAVE_WORKERS  threads writing masks, cut-outs and overlays. A batch of 32
#               arrives every second or so and each photo is a few hundred ms of
#               PNG encoding, so one writer would become the bottleneck and the
#               measurement would be of Pillow, not of the GPU.
#   MAX_SECONDS stops the loop on the clock instead of at the end of the list.
#   DISCARD     every output is written and sized, then deleted immediately, so
#               the timing and the storage estimate are real but the hour costs
#               almost nothing on /blue. Set DISCARD= to keep the results.
#
# A B200 node is 2 TB of RAM and 112 cores across 8 GPUs, so one GPU's fair
# share is 14 cores and ~256 GB. Asking for more makes the job queue longer
# without making it faster — the readers are waiting on Lustre, not on cores.
#
# The walltime is 1h30 for a 1h loop: model load is ~60-90s, the saver drains
# after the deadline, and a job killed mid-write loses that photo's CSV row.
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=sam3_b200
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=240gb
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
# settings.sh fills BATCH_SIZE, SAVE_WORKERS and OUT_DIR with the defaults that
# suit an L4 array. This job's defaults are different ones, so what the caller
# actually passed has to be captured before settings.sh hides it.
caller_out_dir="${OUT_DIR:-}"
caller_batch_size="${BATCH_SIZE:-}"
caller_save_workers="${SAVE_WORKERS:-}"
source slurm/settings.sh

# One GPU, one shard: this measures a single B200, so there is nothing to divide.
export SHARD=0
export NUM_SHARDS=1

export LOCAL_INDEX="${LOCAL_INDEX:-${PROJECT}/data/local_image_paths.parquet}"
export IMAGE_ROOT="${IMAGE_ROOT}"
export LIMIT="${LIMIT-}"                    # no limit: the clock is the limit
export MAX_SECONDS="${MAX_SECONDS:-3600}"
export BATCH_SIZE="${caller_batch_size:-32}"
export SAVE_WORKERS="${caller_save_workers:-10}"
export WORKERS="${WORKERS:-16}"
export BF16="${BF16:-1}"
export DISCARD="${DISCARD-1}"               # DISCARD= (empty) keeps the outputs
export TELEMETRY_S="${TELEMETRY_S:-5}"
export OUT_DIR="${caller_out_dir:-${RESULTS}/b200_${PROMPT// /_}}"

bash "$SLURM_SUBMIT_DIR/slurm/run_shard.sh"

# The answer, without waiting for run_metrics.py:
echo
echo "== one B200, ${MAX_SECONDS}s =="
python run_metrics.py "$OUT_DIR" || true
