#!/bin/bash
# push_to_researchdrive.sh — copy a local directory to Research Drive via smbclient.
#
# IMPORTANT: this CANNOT run on a compute node. Research Drive is only
# reachable from CHTC submit/transfer nodes.
#
# Run from: transfer.chtc.wisc.edu (for /staging sources)
#       or: ap2001.chtc.wisc.edu / ap2002.chtc.wisc.edu (for /home sources)
#
# Usage:
#   ./chtc/push_to_researchdrive.sh <local_dir> <remote_subpath>
# Example:
#   ./chtc/push_to_researchdrive.sh test_output/ sam3/2026-05-14/591507
#
# Side effects:
#   - tars the local dir to a temp file
#   - smbclient puts the tar into //research.drive.wisc.edu/<PI_SHARE>/<remote_subpath>/
#   - writes a JSON receipt to push_receipt.json (read by analyze_results.py)

set -euo pipefail

LOCAL_DIR="${1:?usage: push_to_researchdrive.sh <local_dir> <remote_subpath>}"
REMOTE_SUBPATH="${2:?usage: push_to_researchdrive.sh <local_dir> <remote_subpath>}"

PI_SHARE="${PI_SHARE:-<PI_SHARE_NAME>}"
USE_KERBEROS="${USE_KERBEROS:-1}"

if [[ "$PI_SHARE" == "<PI_SHARE_NAME>" ]]; then
    echo "ERROR: set PI_SHARE env var, e.g. PI_SHARE=jadams"
    exit 1
fi

if [[ ! -d "$LOCAL_DIR" ]]; then
    echo "ERROR: local dir not found: $LOCAL_DIR"
    exit 1
fi

TAR_NAME="$(basename "$LOCAL_DIR")_$(date -u +%Y%m%dT%H%M%SZ).tar.gz"
TAR_PATH="/tmp/$TAR_NAME"

echo "[push] taring $LOCAL_DIR -> $TAR_PATH"
START_TAR=$(date +%s)
tar -czf "$TAR_PATH" -C "$(dirname "$LOCAL_DIR")" "$(basename "$LOCAL_DIR")"
END_TAR=$(date +%s)

TAR_BYTES=$(stat -c%s "$TAR_PATH")
echo "[push] tar size: $(numfmt --to=iec "$TAR_BYTES")"

SMBCLIENT_ARGS=(//research.drive.wisc.edu/"$PI_SHARE")
[[ "$USE_KERBEROS" == "1" ]] && SMBCLIENT_ARGS+=(-k)

echo "[push] creating remote dir $REMOTE_SUBPATH (errors ignored if it exists)"
echo "mkdir \"$REMOTE_SUBPATH\"; quit" | smbclient "${SMBCLIENT_ARGS[@]}" >/dev/null 2>&1 || true

echo "[push] uploading tar to Research Drive..."
START_PUT=$(date +%s)
smbclient "${SMBCLIENT_ARGS[@]}" <<EOF
cd "$REMOTE_SUBPATH"
put "$TAR_PATH" "$TAR_NAME"
quit
EOF
END_PUT=$(date +%s)

# Verify by listing the remote dir
LISTING=$(smbclient "${SMBCLIENT_ARGS[@]}" -c "cd \"$REMOTE_SUBPATH\"; ls" 2>&1 || true)

cat > push_receipt.json <<EOF
{
  "local_dir": "$LOCAL_DIR",
  "remote_share": "//research.drive.wisc.edu/$PI_SHARE",
  "remote_subpath": "$REMOTE_SUBPATH",
  "tar_path": "$TAR_PATH",
  "tar_bytes": $TAR_BYTES,
  "tar_duration_sec": $((END_TAR - START_TAR)),
  "put_duration_sec": $((END_PUT - START_PUT)),
  "remote_listing": $(printf '%s' "$LISTING" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')
}
EOF

rm -f "$TAR_PATH"
echo "[push] done. Receipt: push_receipt.json"
