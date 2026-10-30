#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Flower morphology per genus, as two chained jobs. Run from the project root
# (the flower_sam3 checkout), after all-genera run 2:
#
#     bash morphology/run.sh
#
#   1. classify.sbatch   B200  select 10 photos per genus, Qwen3-VL answers the
#                              morphology questions, per-genus summary (<8 h)
#   2. sheets.sbatch     CPU   contact sheet per genus, copy to export/morphology/
#
# Anything classify.sbatch reads can be set here, e.g. a short test first:
#
#     LIMIT=2000 bash morphology/run.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
if [[ ! -f segment.py || ! -f morphology/run.sh ]]; then
    echo "ERROR: run from the project root (the flower_sam3 checkout)" >&2
    exit 1
fi
mkdir -p logs

gpu=$(sbatch --parsable morphology/classify.sbatch)
echo "1. classify : job $gpu   (hpg-b200, 1 GPU, < 8 h)"
sheets=$(sbatch --parsable --dependency="afterok:$gpu" morphology/sheets.sbatch)
echo "2. sheets   : job $sheets"
echo
echo "watch      : squeue -u \$USER ;  tail -f logs/morphology-$gpu.out"
echo "download   : export/morphology/   (when job $sheets has finished)"
