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
#   ./chtc/push_to_researchdrive.sh test_output sam3/2026-05-14/593537
#
# Authentication (pick one):
#   1. Username + password via env vars (recommended for scripted use):
#        SMB_USER=dli55 SMB_PASS='your-password' ./chtc/push_to_researchdrive.sh ...
#   2. Username only — smbclient will prompt for the password:
#        SMB_USER=dli55 ./chtc/push_to_researchdrive.sh ...
#   3. Kerberos (only if your CHTC account is tied to your campus NetID):
#        USE_KERBEROS=1 ./chtc/push_to_researchdrive.sh ...
#
# Side effects:
#   - tars the local dir to a temp file
#   - smbclient puts the tar into //research.drive.wisc.edu/<PI_SHARE>/<remote_subpath>/
#   - writes a JSON receipt to push_receipt.json (read by analyze_results.py)

set -euo pipefail

LOCAL_DIR="${1:?usage: push_to_researchdrive.sh <local_dir> <remote_subpath>}"
REMOTE_SUBPATH="${2:?usage: push_to_researchdrive.sh <local_dir> <remote_subpath>}"

PI_SHARE="${PI_SHARE:-dli55}"
SMB_USER="${SMB_USER:-}"
SMB_PASS="${SMB_PASS:-}"
USE_KERBEROS="${USE_KERBEROS:-0}"

if [[ ! -d "$LOCAL_DIR" ]]; then
    echo "ERROR: local dir not found: $LOCAL_DIR"
    exit 1
fi

# Build smbclient args + authentication
SMBCLIENT_ARGS=(//research.drive.wisc.edu/"$PI_SHARE")
CREDS_FILE=""

if [[ "$USE_KERBEROS" == "1" ]]; then
    SMBCLIENT_ARGS+=(-k)
elif [[ -n "$SMB_USER" && -n "$SMB_PASS" ]]; then
    # Write a temp credentials file with 600 perms so the password never
    # appears on the command line or in /proc.
    CREDS_FILE=$(mktemp)
    chmod 600 "$CREDS_FILE"
    cat > "$CREDS_FILE" <<EOF
username=$SMB_USER
password=$SMB_PASS
EOF
    SMBCLIENT_ARGS+=(--authentication-file="$CREDS_FILE")
    trap 'rm -f "$CREDS_FILE"' EXIT
elif [[ -n "$SMB_USER" ]]; then
    # smbclient will prompt for the password interactively
    SMBCLIENT_ARGS+=(-U "$SMB_USER")
else
    echo "ERROR: no authentication configured."
    echo "  Set SMB_USER and SMB_PASS, or USE_KERBEROS=1."
    echo "  Example: SMB_USER=dli55 SMB_PASS='...' ./chtc/push_to_researchdrive.sh test_output sam3/test"
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
