#!/usr/bin/env python3
"""job_checker.py — quick health snapshot of all your HTCondor jobs.

Run this on a CHTC access point (e.g. ap2001.chtc.wisc.edu). It shells out to
`condor_q` and `condor_history`, then writes a plain-text report covering, for
every job:

  * how long it has run (wall-clock)
  * which GPU type it landed on (A100 / H100 / H200 / ...)
  * any potential problems (held + hold reason, non-zero exit, signal kills,
    repeated restarts/evictions, long idle waits)

Usage:
    python chtc_tools/job_checker.py                 # writes job_status_<ts>.txt
    python chtc_tools/job_checker.py -o status.txt   # custom output file
    python chtc_tools/job_checker.py --history 100   # include last 100 finished
    python chtc_tools/job_checker.py --user someone  # check another user

It uses only the Python standard library and the condor CLI tools, so it needs
no extra packages or the HTCondor Python bindings.
"""

import argparse
import getpass
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime

# HTCondor JobStatus codes -> human label
JOB_STATUS = {
    1: "IDLE",
    2: "RUNNING",
    3: "REMOVED",
    4: "COMPLETED",
    5: "HELD",
    6: "TRANSFERRING_OUTPUT",
    7: "SUSPENDED",
}

# Keys that may carry the GPU model string, in order of preference.
_DEVICE_NAME_KEYS = (
    "GPUs_DeviceName",
    "MachineAttrGPUs_DeviceName0",
    "MachineAttrGPUs_DeviceName",
    "AssignedGPUs",
)


def run_condor(cmd):
    """Run a condor command and return parsed JSON (list of ad dicts).

    Returns [] on any failure (tool missing, no jobs, parse error) so the
    report still renders.
    """
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=60,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        print(f"WARNING: '{' '.join(cmd)}' failed: {exc}", file=sys.stderr)
        return []
    out = proc.stdout.strip()
    if not out:
        return []
    try:
        data = json.loads(out)
        return data if isinstance(data, list) else [data]
    except json.JSONDecodeError:
        print(f"WARNING: could not parse JSON from '{' '.join(cmd)}'", file=sys.stderr)
        return []


def fmt_duration(seconds):
    """Seconds -> compact 'Xd Yh Zm' string."""
    try:
        seconds = int(seconds)
    except (TypeError, ValueError):
        return "n/a"
    if seconds <= 0:
        return "0m"
    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    mins = rem // 60
    parts = []
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    parts.append(f"{mins}m")
    return " ".join(parts)


def job_id(ad):
    return f"{ad.get('ClusterId', '?')}.{ad.get('ProcId', 0)}"


def runtime_seconds(ad, now):
    """Best wall-clock estimate for a job ad.

    Running jobs: now - JobCurrentStartDate.
    Finished/history jobs: RemoteWallClockTime (HTCondor's accumulated wall time)
    falling back to CommittedTime.
    """
    status = ad.get("JobStatus")
    start = ad.get("JobCurrentStartDate") or ad.get("JobStartDate")
    if status == 2 and start:
        return max(0, int(now) - int(start))
    for key in ("RemoteWallClockTime", "CommittedTime"):
        val = ad.get(key)
        if val:
            return int(val)
    if start:
        return max(0, int(now) - int(start))
    return 0


def gpu_type(ad, host_cache):
    """Return a readable GPU model for the job, or 'n/a'.

    Tries device-name keys present on the ad; if absent and the job is running,
    queries the execute machine via condor_status (cached per host).
    """
    for key in _DEVICE_NAME_KEYS:
        val = ad.get(key)
        if val and isinstance(val, str) and val.strip() and not val.startswith("GPU-"):
            return val.strip()

    # Fallback: ask the machine the running job is on.
    host = ad.get("RemoteHost") or ad.get("LastRemoteHost")
    if not host:
        return "n/a"
    machine = host.split("@")[-1]  # strip slotN@
    if machine in host_cache:
        return host_cache[machine]
    name = "n/a"
    try:
        proc = subprocess.run(
            ["condor_status", machine, "-af", "GPUs_DeviceName"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30,
        )
        for line in proc.stdout.splitlines():
            line = line.strip()
            if line and line.lower() != "undefined":
                name = line
                break
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    host_cache[machine] = name
    return name


def find_problems(ad, now):
    """Return a list of human-readable problem strings for a job ad."""
    problems = []
    status = ad.get("JobStatus")

    if status == 5:  # HELD
        reason = ad.get("HoldReason", "(no reason recorded)")
        code = ad.get("HoldReasonCode")
        problems.append(f"HELD: {reason}" + (f" [code {code}]" if code else ""))

    exit_code = ad.get("ExitCode")
    if exit_code not in (None, 0) and status in (3, 4):
        problems.append(f"exited non-zero (ExitCode={exit_code})")

    if ad.get("ExitBySignal") in (True, 1):
        sig = ad.get("ExitSignal", "?")
        problems.append(f"killed by signal {sig}")

    starts = ad.get("NumJobStarts")
    if isinstance(starts, int) and starts > 1:
        problems.append(f"restarted {starts}x (evicted/preempted)")

    holds = ad.get("NumHolds")
    if isinstance(holds, int) and holds > 0 and status != 5:
        problems.append(f"held {holds}x previously")

    # Long idle wait
    if status == 1:
        qdate = ad.get("QDate")
        if qdate:
            waited = int(now) - int(qdate)
            if waited > 6 * 3600:
                problems.append(f"idle {fmt_duration(waited)} (still unmatched)")

    return problems


def format_job(ad, now, host_cache):
    status = JOB_STATUS.get(ad.get("JobStatus"), str(ad.get("JobStatus")))
    rid = job_id(ad)
    runtime = fmt_duration(runtime_seconds(ad, now))
    gpu = gpu_type(ad, host_cache)
    problems = find_problems(ad, now)

    lines = [f"  Job {rid:<12} {status:<20} runtime={runtime:<12} gpu={gpu}"]
    host = ad.get("RemoteHost") or ad.get("LastRemoteHost")
    if host:
        lines.append(f"      host: {host}")
    cmd = ad.get("Cmd", "")
    args = ad.get("Args") or ad.get("Arguments") or ""
    if cmd:
        lines.append(f"      cmd:  {os.path.basename(cmd)} {args}".rstrip())
    if problems:
        for p in problems:
            lines.append(f"      !! {p}")
    return "\n".join(lines), bool(problems)


def main():
    parser = argparse.ArgumentParser(description="HTCondor job health snapshot.")
    parser.add_argument("-o", "--out", default=None,
                        help="output text file (default: job_status_<timestamp>.txt)")
    parser.add_argument("--user", default=None,
                        help="username to query (default: current user)")
    parser.add_argument("--history", type=int, default=20,
                        help="how many recently finished jobs to include (default: 20; 0 to skip)")
    args = parser.parse_args()

    user = args.user or os.environ.get("USER") or getpass.getuser()
    out_path = args.out or f"job_status_{datetime.now():%Y%m%d_%H%M%S}.txt"
    now = time.time()
    host_cache = {}

    active = run_condor(["condor_q", user, "-json"])
    history = []
    if args.history > 0:
        history = run_condor(
            ["condor_history", user, "-json", "-limit", str(args.history)]
        )

    # Tally
    by_status = {}
    for ad in active:
        label = JOB_STATUS.get(ad.get("JobStatus"), "OTHER")
        by_status[label] = by_status.get(label, 0) + 1

    report = []
    report.append("=" * 72)
    report.append(f"HTCondor job report for '{user}'")
    report.append(f"generated: {datetime.now():%Y-%m-%d %H:%M:%S}")
    report.append("=" * 72)
    report.append("")
    summary = ", ".join(f"{k}={v}" for k, v in sorted(by_status.items())) or "none"
    report.append(f"Active jobs ({len(active)}): {summary}")
    report.append("")

    flagged = 0
    if active:
        report.append("-" * 72)
        report.append("ACTIVE / QUEUED JOBS")
        report.append("-" * 72)
        for ad in sorted(active, key=lambda a: (a.get("ClusterId", 0), a.get("ProcId", 0))):
            text, has_problem = format_job(ad, now, host_cache)
            report.append(text)
            report.append("")
            flagged += int(has_problem)
    else:
        report.append("(no active jobs in the queue)")
        report.append("")

    if history:
        report.append("-" * 72)
        report.append(f"RECENTLY FINISHED (last {len(history)})")
        report.append("-" * 72)
        for ad in history:
            text, has_problem = format_job(ad, now, host_cache)
            report.append(text)
            report.append("")
            flagged += int(has_problem)

    report.append("=" * 72)
    report.append(f"Jobs with potential problems: {flagged}")
    report.append("=" * 72)

    text = "\n".join(report) + "\n"
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(text)

    # Also echo to stdout so it is useful when run interactively.
    print(text)
    print(f"[job_checker] wrote {out_path}")


if __name__ == "__main__":
    main()
