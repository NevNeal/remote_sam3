#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Smoke test: one L4 GPU, 100 images, taxon 160559, prompt "flower".
#
# Run this FIRST, after build_sif.sh and setup_blue.sh.  It is the HPG
# equivalent of the CHTC 4-test harness: if this passes, the array will work.
#
#   ./hpg/submit.sh test
#
# --account/--qos/--output are supplied by submit.sh (SLURM does not expand
# shell variables inside #SBATCH lines, so they cannot be hardcoded here).
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=sam3_test
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48gb
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/config.sh"

TAXON_ID="${TAXON_ID:-160559}"
PROMPT="${PROMPT:-flower}"
OUTPUT_DIR="${OUTPUT_DIR:-${RESULTS_ROOT}/test_${TAXON_ID}}"

# 100 images is enough to prove container + GPU + parquet + S3 + SAM3 all work.
export LIMIT_FLAG="--limit ${LIMIT:-100}"
export WORKERS="${WORKERS:-8}"

date
"$HERE/run_shard.sh" "$TAXON_ID" "$OUTPUT_DIR" "$PROMPT" 0 1
date

echo
echo "── summary ──────────────────────────────────────────────────────────────"
module load apptainer
apptainer exec --bind "/blue/${HPG_GROUP}:/blue/${HPG_GROUP}" "$SIF" \
    python /app/merge_shards.py "$OUTPUT_DIR" --num-shards 1
