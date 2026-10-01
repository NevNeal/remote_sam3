#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Summarise a large CSV on /blue and push the summary to GitHub.
#
#     sbatch slurm/summarize_csv.sbatch                     # the default CSV below
#     CSV=/path/to/other.csv sbatch slurm/summarize_csv.sbatch
#
# Writes csv_summaries/<csv name without .csv>/ (summary.md, summary.json,
# columns.csv, head.csv, sample.csv — all small), then commits that folder on
# the current branch and pushes it, so the report can be read on GitHub.
#
# Set PUSH=0 to skip the commit/push. If the push fails (no credentials on the
# compute node, branch moved), the commit stays local and the log says how to
# push it from the login node.
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=summarize_csv
#SBATCH --partition=hpg-default
#SBATCH --cpus-per-task=2
#SBATCH --mem=64gb
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/settings.sh

CSV="${CSV:-/home/neal.nevyn/blue_guralnick/share/r.dinnage/Projects/phenovision/output/production_datasets/2026-03-15/annotations_ingest_all_repro-v1.1.0_leaves-v1.0.0_2026-03-15.csv}"
PUSH="${PUSH:-1}"
name="$(basename "$CSV" .csv)"
SUMMARY_DIR="${SUMMARY_DIR:-csv_summaries/$name}"

echo "csv:     $CSV ($(du -h "$CSV" | cut -f1))"
echo "summary: $SUMMARY_DIR"

module purge
module load conda
set +u; conda activate "$CONDA_ENV"; set -u    # conda's activate scripts trip over set -u
python summarize_csv.py "$CSV" "$SUMMARY_DIR" --image-root "$IMAGE_ROOT"
cp "logs/summarize_csv-${SLURM_JOB_ID}.out" "$SUMMARY_DIR/job.log" 2>/dev/null || true

(( PUSH )) || exit 0
branch="$(git rev-parse --abbrev-ref HEAD)"
git add "$SUMMARY_DIR"
git commit -m "CSV summary: $name (job $SLURM_JOB_ID)" -- "$SUMMARY_DIR" || { echo "nothing to commit"; exit 0; }
if git pull --rebase origin "$branch" && git push origin "$branch"; then
    echo "pushed $SUMMARY_DIR to origin/$branch"
else
    echo "push failed — the commit is local. From the login node run:" >&2
    echo "    cd $PWD && git pull --rebase origin $branch && git push origin $branch" >&2
    exit 1
fi
