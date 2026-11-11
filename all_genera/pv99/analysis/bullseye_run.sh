#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Bullseye pattern calls over every masked pv99 photo, from the project root
# (the flower_sam3 checkout), after the pv99 export:
#
#     bash all_genera/pv99/analysis/bullseye_run.sh test   # 200 photos, one GPU, ~10 min
#     bash all_genera/pv99/analysis/bullseye_run.sh        # the whole run
#
# Login node: checks scipy and caches Qwen3-VL-8B-Instruct (~17 GB, once).
# Then NUM_SHARDS (default 4) B200 array tasks, and the report job after them
# (afterany, so it still reports on whatever finished).
# Outputs: export/pv99/analysis/bullseye/
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
A=all_genera/pv99/analysis
if [[ ! -f segment.py || ! -f $A/bullseye_run.sh ]]; then
    echo "ERROR: run from the project root (the flower_sam3 checkout)" >&2
    exit 1
fi
source slurm/settings.sh
mkdir -p logs export/pv99/analysis/bullseye
module purge
module load conda
set +u; conda activate "$CONDA_ENV"; set -u
python -c "import scipy" 2>/dev/null || pip install --quiet scipy
python -c "from huggingface_hub import snapshot_download as s; print('model      :', s('Qwen/Qwen3-VL-8B-Instruct'))"

if [[ "${1:-}" == "test" ]]; then
    job=$(sbatch --parsable --time=1:00:00 \
          --export=ALL,NUM_SHARDS=1,BULLSEYE_ARGS="--limit 200 --out export/pv99/analysis/bullseye/test_200.csv" \
          $A/bullseye.sbatch)
    echo "test       : job $job -> export/pv99/analysis/bullseye/test_200.csv"
    echo "log        : logs/pv99_bullseye-${job}_0.out  (photos/s is in it)"
    exit 0
fi

NUM_SHARDS="${NUM_SHARDS:-4}"
gpu=$(sbatch --parsable --export=ALL,NUM_SHARDS="$NUM_SHARDS" --array=0-$((NUM_SHARDS - 1)) \
      $A/bullseye.sbatch)
echo "1. classify: job $gpu   (hpg-b200, $NUM_SHARDS tasks)"
rep=$(sbatch --parsable --dependency=afterany:$gpu $A/bullseye_report.sbatch)
echo "2. report  : job $rep"
echo
echo "watch      : squeue -u \$USER;  tail -f logs/pv99_bullseye-${gpu}_0.out"
echo "outputs    : export/pv99/analysis/bullseye/"
