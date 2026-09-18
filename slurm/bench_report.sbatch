#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Benchmark report + cleanup. Queued by slurm/bench.sh with
# --dependency=afterany:<array>, so it runs once every GPU task has ended,
# whether it succeeded or not.
#
#   1. save the evidence into BENCH_DIR: sacct accounting, every shard's log,
#      per-photo timing CSVs, shard timing JSONs, GPU telemetry
#   2. write BENCH_DIR/report.md (+ per_shard.csv, per_node.csv, summary.json)
#   3. delete the run directory under results/
#
# Step 3 always removes the heavy folders (images/masks/overlays/segments).
# The directory itself, with its small CSVs, is only removed once step 1 has
# verifiably copied them, so a failure here never loses the timing data.
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=bench_report
#SBATCH --partition=hpg-default
#SBATCH --cpus-per-task=2
#SBATCH --mem=8gb
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -uo pipefail                 # no -e: cleanup must run even if a step fails
cd "$SLURM_SUBMIT_DIR"
source slurm/settings.sh
: "${OUT_DIR:?}" "${BENCH_DIR:?}" "${ARRAY_JOB_ID:?}"

# Refuse to delete anything that isn't a benchmark run inside this project.
case "$OUT_DIR" in
    "$PROJECT"/results/bench_*) ;;
    *) echo "refusing: OUT_DIR=$OUT_DIR is not under $PROJECT/results/bench_*" >&2; exit 1 ;;
esac

mkdir -p "$BENCH_DIR/raw" "$BENCH_DIR/logs"

# ── 1. evidence ─────────────────────────────────────────────────────────────
sacct -j "$ARRAY_JOB_ID" -P --units=M \
    --format=JobID,JobName,Partition,NodeList,State,ExitCode,Submit,Start,End,ElapsedRaw,AllocCPUS,TotalCPU,MaxRSS,ReqMem,AllocTRES%80 \
    > "$BENCH_DIR/sacct.txt"
cp -p logs/sam3_bench-"${ARRAY_JOB_ID}"_* "$BENCH_DIR/logs/" 2>/dev/null
cp -p "$OUT_DIR"/results_shard_*.csv "$OUT_DIR"/timing_shard_*.json "$BENCH_DIR/raw/" 2>/dev/null
cp -p "$OUT_DIR"/errors_shard_*.txt "$BENCH_DIR/raw/" 2>/dev/null
cp -rp "$OUT_DIR/telemetry" "$BENCH_DIR/raw/" 2>/dev/null

copied_ok=1
for f in "$OUT_DIR"/results_shard_*.csv; do
    [[ -e "$f" ]] || continue
    cmp -s "$f" "$BENCH_DIR/raw/$(basename "$f")" || { copied_ok=0; echo "copy mismatch: $f" >&2; }
done

# ── 2. report ───────────────────────────────────────────────────────────────
module purge
module load conda
set +u; conda activate "$CONDA_ENV"; set -u    # conda's activate scripts trip over set -u
python bench_report.py "$BENCH_DIR" || echo "report failed — raw data is in $BENCH_DIR/raw" >&2

# ── 3. cleanup ──────────────────────────────────────────────────────────────
echo "run dir before cleanup: $(du -sh "$OUT_DIR" 2>/dev/null | cut -f1)"
rm -rf "$OUT_DIR/images" "$OUT_DIR/masks" "$OUT_DIR/overlays" "$OUT_DIR/segments"
if (( copied_ok )); then
    rm -rf "$OUT_DIR"
    echo "deleted $OUT_DIR"
else
    echo "kept $OUT_DIR (CSVs only) because the copy didn't verify" >&2
fi
echo "analysis: $BENCH_DIR  ($(du -sh "$BENCH_DIR" | cut -f1))"
