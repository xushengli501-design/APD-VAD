#!/usr/bin/env python3
# =============================================================================
# Profile a command's Peak GPU Memory (GB) by polling nvidia-smi for the
# process tree (works with `bash script.sh` -> `python train.py` chains).
#
# Usage:
#   python scripts/profile_gpu.py <command...>
#
# Reports on stdout (when the command exits):
#   [prof] Peak GPU Memory: X.XX GB (YYYY MiB)
#   [prof] Wall time: N.N s
#
# Exit code matches the wrapped command.  Numerically inert — never touches
# the wrapped command's behavior or its outputs.
# =============================================================================
import os
import subprocess
import sys
import time


def _children(pid):
    out = subprocess.run(
        ["pgrep", "-P", str(pid)], capture_output=True, text=True
    ).stdout.split()
    kids = [int(x) for x in out]
    for k in kids:
        kids.extend(_children(k))
    return kids


def _tree(pid):
    return [pid] + _children(pid)


def _peak_gpu_mib(pids):
    if not pids:
        return 0.0
    pid_set = set(pids)
    out = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    ).stdout
    total = 0.0
    for line in out.strip().splitlines():
        parts = line.split(",")
        if len(parts) == 2 and parts[0].strip().isdigit():
            if int(parts[0]) in pid_set:
                total += float(parts[1].strip())
    return total


def main():
    cmd = sys.argv[1:]
    if not cmd:
        print("usage: python scripts/profile_gpu.py <command...>", file=sys.stderr)
        return 2
    start = time.time()
    proc = subprocess.Popen(cmd)
    peak = 0.0
    while proc.poll() is None:
        m = _peak_gpu_mib(_tree(proc.pid))
        if m > peak:
            peak = m
        time.sleep(0.5)
    elapsed = time.time() - start
    print(f"[prof] Peak GPU Memory: {peak / 1024:.2f} GB ({peak:.0f} MiB)", flush=True)
    print(f"[prof] Wall time: {elapsed:.1f} s", flush=True)
    return proc.returncode or 0


if __name__ == "__main__":
    sys.exit(main())
