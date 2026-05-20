#!/bin/bash
# watch_and_push.sh — forward incrementally-shipped batch tars to Research Drive.
#
# The A100 job ships each completed 1000-image batch off the compute node via
# `condor_chirp put`, which lands the tar in the job's submit directory (Iwd) on
# the access point. That directory is on shared /home, so it is also visible from
# transfer.chtc.wisc.edu — the only place that can reach Research Drive.
#
# This script polls that directory and pushes each new batch_*.tar.gz to RD,
# then deletes the local copy (RD is the store) so /home quota is not exhausted.
# It is idempotent: already-pushed basenames are recorded and skipped.
#
# Run it (on a node that can reach Research Drive, i.e. transfer.chtc.wisc.edu)
# BEFORE or just after submitting the job, and leave it running for the job's
# duration:
#
#   ssh <NETID>@transfer.chtc.wisc.edu
#   cd remote_sam3
#   SMB_USER=<NETID> SMB_PASS='...' \
#     ./chtc/watch_and_push.sh sam3/$(date +%F)/336671
#
# Args:
#   $1  remote_subpath   RD path under the share (e.g. sam3/2026-05-19/336671)  [required]
#   $2  incoming_dir     where chirp drops the tars (default: current directory)
#
# Env:
#   PI_SHARE     Research Drive share name           (default: dli55)
#   SMB_USER     SMB username \
#   SMB_PASS     SMB password  >  pick one auth method (see push_to_researchdrive.sh)
#   USE_KERBEROS USE_KERBEROS=1 to use Kerberos /
#   POLL_SEC     seconds between scans                (default: 30)
#   PUSH_GLOB    filename glob to match               (default: batch_*.tar.gz)
#   ONCE         ONCE=1 to do a single pass and exit  (default: loop forever)
#   KEEP_LOCAL   KEEP_LOCAL=1 to keep local tars after a successful push
set -euo pipefail

REMOTE_SUBPATH="${1:?usage: watch_and_push.sh <remote_subpath> [incoming_dir]}"
INCOMING_DIR="${2:-.}"

PI_SHARE="${PI_SHARE:-dli55}"
SMB_USER="${SMB_USER:-}"
SMB_PASS="${SMB_PASS:-}"
USE_KERBEROS="${USE_KERBEROS:-0}"
POLL_SEC="${POLL_SEC:-30}"
PUSH_GLOB="${PUSH_GLOB:-batch_*.tar.gz}"
ONCE="${ONCE:-0}"
KEEP_LOCAL="${KEEP_LOCAL:-0}"

if [[ ! -d "$INCOMING_DIR" ]]; then
    echo "ERROR: incoming dir not found: $INCOMING_DIR"
    exit 1
fi

STATE_FILE="$INCOMING_DIR/.rd_pushed.log"
touch "$STATE_FILE"

# ---- Build smbclient args + authentication (mirrors push_to_researchdrive.sh) ----
SMBCLIENT_ARGS=(//research.drive.wisc.edu/"$PI_SHARE")
CREDS_FILE=""

if [[ "$USE_KERBEROS" == "1" ]]; then
    SMBCLIENT_ARGS+=(-k)
elif [[ -n "$SMB_USER" && -n "$SMB_PASS" ]]; then
    CREDS_FILE=$(mktemp)
    chmod 600 "$CREDS_FILE"
    cat > "$CREDS_FILE" <<EOF
username=$SMB_USER
password=$SMB_PASS
EOF
    SMBCLIENT_ARGS+=(--authentication-file="$CREDS_FILE")
    trap 'rm -f "$CREDS_FILE"' EXIT
elif [[ -n "$SMB_USER" ]]; then
    SMBCLIENT_ARGS+=(-U "$SMB_USER")
else
    echo "ERROR: no authentication configured. Set SMB_USER and SMB_PASS, or USE_KERBEROS=1."
    exit 1
fi

# Create the remote dir once (ignore error if it already exists).
echo "[watch] ensuring remote dir $REMOTE_SUBPATH"
echo "mkdir \"$REMOTE_SUBPATH\"; quit" | smbclient "${SMBCLIENT_ARGS[@]}" >/dev/null 2>&1 || true

already_pushed() { grep -qxF "$1" "$STATE_FILE"; }

push_one() {
    local path="$1"
    local name
    name="$(basename "$path")"

    # Skip partial chirp writes.
    case "$name" in
        *.part|*.tmp) return 0 ;;
    esac

    if already_pushed "$name"; then
        return 0
    fi

    echo "[watch] pushing $name -> //research.drive.wisc.edu/$PI_SHARE/$REMOTE_SUBPATH/"
    if smbclient "${SMBCLIENT_ARGS[@]}" <<EOF
cd "$REMOTE_SUBPATH"
put "$path" "$name"
quit
EOF
    then
        echo "$name" >> "$STATE_FILE"
        echo "[watch] pushed $name"
        if [[ "$KEEP_LOCAL" != "1" ]]; then
            rm -f "$path"
        fi
    else
        echo "[watch] WARNING: push failed for $name (will retry next pass)"
    fi
}

scan_once() {
    shopt -s nullglob
    local f
    for f in "$INCOMING_DIR"/$PUSH_GLOB; do
        push_one "$f"
    done
    shopt -u nullglob
}

echo "[watch] watching $INCOMING_DIR for '$PUSH_GLOB' -> RD:$REMOTE_SUBPATH (poll ${POLL_SEC}s, once=$ONCE)"

if [[ "$ONCE" == "1" ]]; then
    scan_once
    echo "[watch] single pass complete."
    exit 0
fi

while true; do
    scan_once
    sleep "$POLL_SEC"
done
