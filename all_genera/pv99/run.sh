#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# pv99 run: 1M flower-detected angiosperm photos phenovision scored >= 0.99,
# at least 50 per genus where it has them, SAM3 at min score 0.5, top 5 masks, .npz masks + metadata only. Run from the
# project root (the flower_sam3 checkout):
#
#     bash all_genera/pv99/run.sh            # index only, if missing; then stop
#     bash all_genera/pv99/run.sh go         # GPU array + export, chained
#
#   1. index.sbatch     CPU   data/pv99_index.parquet; count is in its log
#   2. segment.sbatch   B200  NUM_SHARDS array tasks (default 2), one GPU each
#   3. export.sbatch    CPU   export/pv99/pv99_segment_metadata.parquet
#
# The first call stops after the index so the photo count can be checked before
# GPU time is spent. TARGET=, PER_GENUS=, MIN_PV= change the index (with
# REBUILD_INDEX=1 if one exists); NUM_SHARDS=4 ... go splits the GPU work 4 ways.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
if [[ ! -f segment.py || ! -f all_genera/pv99/run.sh ]]; then
    echo "ERROR: run from the project root (the flower_sam3 checkout)" >&2
    exit 1
fi
mkdir -p logs
INDEX=data/pv99_index.parquet
NUM_SHARDS="${NUM_SHARDS:-2}"

if [[ "${1:-}" != "go" ]]; then
    if [[ -f "$INDEX" && -z "${REBUILD_INDEX:-}" ]]; then
        echo "index      : $INDEX exists ($(python -c "import pyarrow.parquet as p; print(f'{p.ParquetFile(\"$INDEX\").metadata.num_rows:,}')" 2>/dev/null || echo '?') photos)"
        echo "next       : bash all_genera/pv99/run.sh go"
    else
        index=$(sbatch --parsable all_genera/pv99/index.sbatch)
        echo "1. index   : job $index"
        echo "count      : tail -n 5 logs/sam3_pv99_index-$index.out   (when it finishes)"
        echo "next       : bash all_genera/pv99/run.sh go"
    fi
    exit 0
fi

if [[ ! -f "$INDEX" ]]; then
    echo "ERROR: no $INDEX yet - run bash all_genera/pv99/run.sh first" >&2
    exit 1
fi
gpu=$(sbatch --parsable --export=ALL,NUM_SHARDS="$NUM_SHARDS" \
      --array=0-$((NUM_SHARDS - 1)) all_genera/pv99/segment.sbatch)
echo "2. segment : job $gpu   (hpg-b200, $NUM_SHARDS tasks x 1 GPU)"
report=$(sbatch --parsable --dependency="afterok:$gpu" \
         --export=ALL,NUM_SHARDS="$NUM_SHARDS" all_genera/pv99/export.sbatch)
echo "3. export  : job $report"
echo
echo "watch      : squeue -u \$USER"
echo "metadata   : export/pv99/          (when job $report has finished)"
echo "masks      : results/pv99_flower/masks/"
