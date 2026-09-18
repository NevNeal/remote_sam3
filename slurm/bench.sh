#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Benchmark a whole taxon across N GPUs, keep only the timing analysis.
#
#     TAXON_ID=62741 bash slurm/bench.sh          # 10 GPUs (default)
#     TAXON_ID=62741 bash slurm/bench.sh 0-19     # 20 GPUs
#     CPUS=4 MEM=16gb TAXON_ID=62741 bash slurm/bench.sh   # fit a tight group CPU limit
#
# Queues two jobs:
#   1. slurm/bench.sbatch         the GPU array; outputs deleted as they're timed
#   2. slurm/bench_report.sbatch  starts when every array task has ENDED (even
#                                 failed ones): saves logs, sacct, per-photo
#                                 timing and GPU telemetry to benchmarks/<name>/,
#                                 writes report.md, then deletes the run's
#                                 results/ directory
#
# What survives: benchmarks/<taxon>_<prompt>_<stamp>/ (tens of MB). Nothing
# under results/.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
cd "$(dirname "$0")/.."
source slurm/settings.sh

ARRAY="${1:-0-9}"
NAME="${TAXON_ID}_${PROMPT// /_}_$(date +%Y%m%d_%H%M%S)"
export OUT_DIR="${RESULTS}/bench_${NAME}"
export BENCH_DIR="${PROJECT}/benchmarks/${NAME}"
mkdir -p logs "$BENCH_DIR"

# CPUS / MEM override bench.sbatch's per-task request. Shrink them when the
# group's CPU or memory allocation, not its GPUs, is what caps concurrency —
# squeue shows (QOSGrpCpuLimit) / (QOSGrpMemLimit) on the pending tasks.
# Downloads are I/O-bound threads, so WORKERS does not need to shrink with CPUS.
extra=()
[[ -n "${CPUS:-}" ]] && extra+=(--cpus-per-task="$CPUS")
[[ -n "${MEM:-}" ]] && extra+=(--mem="$MEM")

array_id=$(sbatch --parsable --array="$ARRAY" "${extra[@]}" slurm/bench.sbatch)
array_id="${array_id%%;*}"
report_id=$(sbatch --parsable --dependency="afterany:${array_id}" \
            --export="ALL,ARRAY_JOB_ID=${array_id}" slurm/bench_report.sbatch)
report_id="${report_id%%;*}"

cat <<EOF
taxon $TAXON_ID, prompt '$PROMPT', array $ARRAY, per task: ${CPUS:-12} CPUs, ${MEM:-48gb}
  GPU array  : job $array_id   logs/sam3_bench-${array_id}_<shard>.out/.err
  report     : job $report_id  (waits for the array; logs/bench_report-${report_id}.out)
  run dir    : $OUT_DIR   (deleted by the report job)
  analysis   : $BENCH_DIR/report.md

watch:  squeue -u \$USER
cancel: scancel $array_id $report_id
EOF
