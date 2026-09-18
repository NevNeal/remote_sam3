#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# submit.sh — the one command you actually type.
#
# SLURM refuses to expand shell variables inside #SBATCH directives, so
# --account, --qos and the log paths (which all contain your group name) have to
# arrive as command-line flags.  This wrapper reads them from hpg/config.sh and
# passes them through, and exports the run settings the job scripts read.
#
# Usage:
#   ./hpg/submit.sh test                    # 1 L4, 100 images — do this first
#   ./hpg/submit.sh array <N> [maxconc]     # N single-L4 shards
#   ./hpg/submit.sh b200                    # 1 node, 8 B200s, 8 shards
#   ./hpg/submit.sh merge                    # merge shard CSVs (runs on login node)
#
# Run settings come from the environment:
#   TAXON_ID=62741 PROMPT=petal ./hpg/submit.sh array 40
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/config.sh"

[[ "$HPG_GROUP" == "CHANGEME" ]] && { echo "ERROR: set HPG_GROUP in hpg/config.sh"; exit 1; }

MODE="${1:?usage: submit.sh test|array <N> [maxconc]|b200|merge}"

TAXON_ID="${TAXON_ID:-160559}"
PROMPT="${PROMPT:-flower}"
OUTPUT_DIR="${OUTPUT_DIR:-${RESULTS_ROOT}/${TAXON_ID}_${PROMPT// /_}}"

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

# GPU partitions have no burst QOS on HiPerGator, so the investment QOS
# (= the plain group name) is the only option for these jobs.
COMMON=(
    --account="$HPG_GROUP"
    --qos="$HPG_GROUP"
    --mail-type=END,FAIL
)

export TAXON_ID PROMPT OUTPUT_DIR

case "$MODE" in
  test)
    sbatch "${COMMON[@]}" \
        --output="$LOG_DIR/sam3_test-%j.out" \
        --error="$LOG_DIR/sam3_test-%j.err" \
        --export=ALL \
        "$HERE/sbatch_l4_test.sh"
    ;;

  array)
    N="${2:?usage: submit.sh array <num_shards> [max_concurrent]}"
    MAXCONC="${3:-}"
    ARRAY_SPEC="0-$((N - 1))"
    [[ -n "$MAXCONC" ]] && ARRAY_SPEC="${ARRAY_SPEC}%${MAXCONC}"

    # Hard cap: SLURM on HiPerGator tops out at 3000 jobs per user and array
    # task IDs cannot exceed 3000.
    (( N > 3000 )) && { echo "ERROR: max 3000 array tasks on HiPerGator"; exit 1; }

    export NUM_SHARDS="$N"
    echo "submitting array $ARRAY_SPEC  taxon=$TAXON_ID prompt='$PROMPT'"
    echo "  output: $OUTPUT_DIR"
    sbatch "${COMMON[@]}" \
        --array="$ARRAY_SPEC" \
        --output="$LOG_DIR/sam3_l4-%A_%a.out" \
        --error="$LOG_DIR/sam3_l4-%A_%a.err" \
        --export=ALL \
        "$HERE/sbatch_l4_array.sh"
    ;;

  b200)
    export NUM_SHARDS="${NUM_SHARDS:-8}"
    export SHARD_BASE="${SHARD_BASE:-0}"
    sbatch "${COMMON[@]}" \
        --output="$LOG_DIR/sam3_b200-%j.out" \
        --error="$LOG_DIR/sam3_b200-%j.err" \
        --export=ALL \
        "$HERE/sbatch_b200_node.sh"
    ;;

  merge)
    module load apptainer
    apptainer exec --bind "/blue/${HPG_GROUP}:/blue/${HPG_GROUP}" "$SIF" \
        python /app/merge_shards.py "$OUTPUT_DIR" "${@:2}"
    ;;

  *)
    echo "unknown mode: $MODE"
    echo "usage: submit.sh test|array <N> [maxconc]|b200|merge"
    exit 1
    ;;
esac
