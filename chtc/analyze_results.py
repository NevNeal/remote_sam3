"""
analyze_results.py — parse HTCondor logs + test_results.json + push_receipt.json
and produce a single human-readable summary.

Usage:
    python chtc/analyze_results.py logs/test.<cluster>.<process>
        (the prefix; .log .out .err siblings will be picked up automatically)

Prints:
  - HTCondor lifecycle: submit time, start time, end time, queue wait, wall time
  - Final job status + exit code
  - Per-test status, duration, and error
  - Where output files landed (counts of images/masks/overlays)
  - Research Drive push receipt (if push_receipt.json exists)
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path


def parse_condor_log(log_path: Path):
    """Parse the HTCondor user log (the .log file) for key timestamps + exit info."""
    if not log_path.exists():
        return {"present": False, "path": str(log_path)}

    text = log_path.read_text(errors="replace")
    events = []
    # Events look like:
    #   000 (12345.000.000) 2026-05-14 14:01:33 Job submitted from host: ...
    #   001 (12345.000.000) 2026-05-14 14:05:11 Job executing on host: ...
    #   005 (12345.000.000) 2026-05-14 14:18:02 Job terminated.
    event_re = re.compile(r"^(\d{3})\s+\((\d+\.\d+\.\d+)\)\s+(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\s+(.+)$", re.M)
    for m in event_re.finditer(text):
        code, jobid, ts, msg = m.groups()
        events.append({
            "code": code,
            "job_id": jobid,
            "timestamp": ts,
            "message": msg.strip(),
        })

    def ts_for(code):
        for ev in events:
            if ev["code"] == code:
                return ev["timestamp"]
        return None

    def parse_dt(ts):
        return datetime.strptime(ts, "%Y-%m-%d %H:%M:%S") if ts else None

    submit_ts = ts_for("000")
    start_ts = ts_for("001")
    end_ts = ts_for("005") or ts_for("009")  # 009 = aborted

    submit_dt = parse_dt(submit_ts)
    start_dt = parse_dt(start_ts)
    end_dt = parse_dt(end_ts)

    queue_wait_sec = (start_dt - submit_dt).total_seconds() if submit_dt and start_dt else None
    wall_clock_sec = (end_dt - start_dt).total_seconds() if start_dt and end_dt else None

    # Exit code line:
    exit_code = None
    m = re.search(r"\(return value (\d+)\)", text)
    if m:
        exit_code = int(m.group(1))

    # Held reason
    held_reason = None
    m = re.search(r"Job was held\.\s*\n\s*(.+)", text)
    if m:
        held_reason = m.group(1).strip()

    return {
        "present": True,
        "path": str(log_path),
        "submitted_at": submit_ts,
        "started_at": start_ts,
        "ended_at": end_ts,
        "queue_wait_sec": queue_wait_sec,
        "wall_clock_sec": wall_clock_sec,
        "exit_code": exit_code,
        "held_reason": held_reason,
        "event_count": len(events),
    }


def parse_test_json(json_path: Path):
    if not json_path.exists():
        return {"present": False, "path": str(json_path)}
    data = json.loads(json_path.read_text())
    data["present"] = True
    data["path"] = str(json_path)
    return data


def parse_push_receipt(json_path: Path):
    if not json_path.exists():
        return None
    return json.loads(json_path.read_text())


def tail(path: Path, n=20):
    if not path.exists():
        return None
    lines = path.read_text(errors="replace").splitlines()
    return "\n".join(lines[-n:])


def main():
    parser = argparse.ArgumentParser(description="Summarize a CHTC test job")
    parser.add_argument(
        "log_prefix",
        help="HTCondor log prefix (e.g. logs/test.12345.0). The .log, .out, .err will be found.",
    )
    parser.add_argument("--test-json", default="test_results.json")
    parser.add_argument("--push-json", default="push_receipt.json")
    args = parser.parse_args()

    prefix = Path(args.log_prefix)
    log_info = parse_condor_log(prefix.with_suffix(".log"))
    test_info = parse_test_json(Path(args.test_json))
    push_info = parse_push_receipt(Path(args.push_json))
    err_tail = tail(prefix.with_suffix(".err"), n=30)

    print("=" * 70)
    print("CHTC TEST RESULT SUMMARY")
    print("=" * 70)

    print("\n[HTCondor lifecycle]")
    if log_info["present"]:
        print(f"  log file       : {log_info['path']}")
        print(f"  submitted at   : {log_info['submitted_at']}")
        print(f"  started at     : {log_info['started_at']}")
        print(f"  ended at       : {log_info['ended_at']}")
        if log_info["queue_wait_sec"] is not None:
            print(f"  queue wait     : {log_info['queue_wait_sec']:.0f} s")
        if log_info["wall_clock_sec"] is not None:
            print(f"  wall clock     : {log_info['wall_clock_sec']:.0f} s")
        print(f"  exit code      : {log_info['exit_code']}")
        if log_info["held_reason"]:
            print(f"  HELD REASON    : {log_info['held_reason']}")
    else:
        print(f"  (no log file at {log_info['path']})")

    print("\n[test_on_chtc.py results]")
    if test_info["present"]:
        print(f"  results file   : {test_info['path']}")
        print(f"  total duration : {test_info['total_duration_sec']:.1f} s")
        print(f"  all passed     : {test_info['all_ok']}")
        for r in test_info["results"]:
            marker = {"ok": "PASS", "failed": "FAIL", "skipped": "SKIP"}[r["status"]]
            print(f"    [{marker}] {r['name']:14s} {r['duration_sec']:>7.2f} s  {r['error']}")
            # Surface a couple of useful detail fields per test
            d = r.get("detail", {})
            if r["name"] == "environment" and d.get("gpus"):
                for gpu in d["gpus"]:
                    print(f"             gpu[{gpu['index']}]: {gpu['name']} ({gpu['memory_gb']} GB, cc {gpu['capability']})")
            if r["name"] == "pipeline" and "files" in d:
                f = d["files"]
                print(f"             output: {f.get('images', 0)} images, {f.get('masks', 0)} masks, {f.get('overlays', 0)} overlays")
                print(f"             base  : {d.get('output_base')}")
    else:
        print(f"  (no results JSON at {test_info['path']})")

    print("\n[Research Drive push]")
    if push_info:
        print(f"  local dir      : {push_info['local_dir']}")
        print(f"  remote         : {push_info['remote_share']}/{push_info['remote_subpath']}")
        print(f"  tar bytes      : {push_info['tar_bytes']:,}")
        print(f"  tar duration   : {push_info['tar_duration_sec']} s")
        print(f"  put duration   : {push_info['put_duration_sec']} s")
    else:
        print("  (no push_receipt.json — run chtc/push_to_researchdrive.sh from a transfer node)")

    if err_tail:
        print("\n[stderr tail]")
        for line in err_tail.splitlines():
            print(f"  {line}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
