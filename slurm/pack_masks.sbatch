#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Pack a run's .npy masks for export: one .tar.zst per batch_NNNNN folder, plus
# the results and metrics CSVs, under export/<run>/. CPU only.
#
#     sbatch slurm/pack_masks.sbatch                       # taxon 62741
#     RUN_DIR=results/160559_flower sbatch slurm/pack_masks.sbatch
#
# Why pack rather than point Globus at masks/: a taxon is a few hundred thousand
# files, and Globus pays a per-file cost on every one. And a binary mask stored as
# uint8 is almost all zero bytes, so zstd shrinks it by one to two orders of
# magnitude. Hundreds of medium archives transfer far faster than 300k files.
#
# Each archive is verified (zstd -t) before the next is written, and a
# MANIFEST.tsv lists every archive with its file count, raw bytes and packed
# bytes, so the export can be checked on the other side. The masks/ originals
# are left in place.
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=sam3_pack
#SBATCH --partition=hpg-default
#SBATCH --cpus-per-task=16
#SBATCH --mem=16gb
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
caller_taxon_id="${TAXON_ID:-}"
source slurm/settings.sh
TAXON_ID="${caller_taxon_id:-62741}"
RUN_DIR="${RUN_DIR:-${RESULTS}/${TAXON_ID}_${PROMPT// /_}}"
NAME="$(basename "$RUN_DIR")"
EXPORT="${EXPORT:-${PROJECT}/export/${NAME}}"
LEVEL="${ZSTD_LEVEL:-6}"
THREADS="${SLURM_CPUS_PER_TASK:-16}"

mkdir -p "$EXPORT"
printf 'archive\tfiles\traw_bytes\tpacked_bytes\n' > "$EXPORT/MANIFEST.tsv"

for dir in "$RUN_DIR"/masks/batch_*/; do
    batch="$(basename "$dir")"
    archive="$EXPORT/masks_${batch}.tar.zst"
    files=$(find "$dir" -name '*.npy' | wc -l)
    raw=$(du -sb --apparent-size "$dir" | cut -f1)
    tar -C "$RUN_DIR" -cf - "masks/$batch" | zstd -q -T"$THREADS" -"$LEVEL" -o "$archive" -f
    zstd -q -t "$archive"
    packed=$(stat -c %s "$archive")
    printf '%s\t%s\t%s\t%s\n' "$(basename "$archive")" "$files" "$raw" "$packed" \
        >> "$EXPORT/MANIFEST.tsv"
    echo "$batch: $files masks, $((raw / 1000000)) MB -> $((packed / 1000000)) MB"
done

for f in results.csv run_metrics.csv metrics.json per_shard.csv; do
    [[ -f "$RUN_DIR/$f" ]] && cp "$RUN_DIR/$f" "$EXPORT/"
done
(cd "$EXPORT" && sha256sum masks_*.tar.zst > SHA256SUMS)

awk -F'\t' 'NR > 1 {n += $2; r += $3; p += $4}
    END {printf "\ntotal: %d masks, %.2f GB raw -> %.2f GB packed (%.0fx)\n",
         n, r / 1e9, p / 1e9, (p ? r / p : 0)}' "$EXPORT/MANIFEST.tsv"
echo "export: $EXPORT"
