#!/usr/bin/env python3
"""Phase timings of `cargo-zisk prove -vv` logs (native or bridged), plus the
bridge's per-instance totals from its `ZZ_LOG=2` lines: what each family's
proves cost on the client, how long instances waited for it, and what the fixed
sections cost inside the prove slot against what was read and uploaded ahead
of it. Usage: summarize.py <run.log>..."""
from __future__ import annotations

import collections
import pathlib
import re
import sys

PHASES = (
    "INITIALIZING_PROOFMAN",
    "CALCULATING_CONTRIBUTIONS",
    "GENERATING_INNER_PROOFS",
    "GENERATE_VADCOP_FINAL_PROOF",
)
INSTANCE = (
    r"\[zz \+\s*([\d.]+)\] instance (\d+) (\w+) \(\w+\): ([\d.]+) s,"
    r" of which ([\d.]+) s waiting"
)
STREAMS = (
    r"Using (\d+) streams per GPU for basic proofs"
    r" and (\d+) streams per GPU for recursive"
)
FIXED = (
    r"\[zz \+\s*[\d.]+\] fixed sections for (\w+): ([\d.]+) s under the slot,"
    r" ([\d.]+) s ahead of it"
)
# A read-ahead upload the bridge caught and sent to the slot instead. The line
# quotes PJRT's message verbatim, so it has to come out of the log before the
# abort search below — a rescued run is one that finished, and reporting it as
# ABORTED would name the success this fallback exists to produce as the
# failure it exists to prevent.
RESCUED = r"\[zz \+\s*[\d.]+\] (\w+): read-ahead upload gave way to the slot \([^\n]*\)"
ABORT = r"PJRT error in \w+: (Out of memory[^\n]*)"
VERIFIED = ("Proof verified successfully", "Vadcop Final proof was verified")


def summarize(path: pathlib.Path) -> None:
    log = path.read_text(errors="replace")
    phases = {
        p: int(m.group(1)) / 1000
        for p in PHASES
        if (m := re.search(rf"INFO: <<< {p} \((\d+)ms\)", log))
    }
    wall = re.search(r"Elapsed \(wall clock\).*?(\d+):([\d.]+)", log)
    wall_s = int(wall.group(1)) * 60 + float(wall.group(2)) if wall else "?"
    # Both phrases, because the two stacks this tool compares do not share
    # one: a native pil2 run ends "Proof verified successfully", a bridged run
    # "Vadcop Final proof was verified". Matching only the first reported
    # every bridged run as unverified.
    verified = any(p in log for p in VERIFIED)
    line = f"## {path}  wall {wall_s} s  verified={verified}"
    if streams := re.search(STREAMS, log):
        line += f"  pil2 streams basic/recursive {streams.group(1)}/{streams.group(2)}"
    print(line)
    print("   " + "  ".join(f"{p.lower()} {v:.2f} s" for p, v in phases.items()))
    inst = re.findall(INSTANCE, log)
    if inst:
        report_bridge(log, inst)
    # Outside the block above: a run can abort before its first instance
    # finishes, and that is when it most often does, so neither of these may
    # sit behind "no instances completed".
    if rescued := re.findall(RESCUED, log):
        by_air = collections.Counter(rescued)
        print(
            f"   read-ahead uploads sent to the slot: {len(rescued)}"
            f" ({', '.join(f'{a} x{n}' for a, n in sorted(by_air.items()))})"
            " — the card was full when they were tried, and the prove went on"
        )
    if oom := re.search(ABORT, re.sub(RESCUED, "", log)):
        print(f"   ABORTED: {oom.group(1)}")


def report_bridge(log: str, inst: list) -> None:
    """The bridge's own per-instance and fixed-section totals."""
    own = collections.defaultdict(list)
    waited = 0.0
    for _, _, air, total, wait in inst:
        own[air].append(float(total) - float(wait))
        waited += float(wait)
    done = [float(i[0]) for i in inst]
    total_own = sum(map(sum, own.values()))
    print(
        f"   bridge: {len(inst)} instances, own {total_own:.2f} s,"
        f" waiting for the client {waited:.2f} s summed,"
        f" done +{min(done):.1f}..+{max(done):.1f} s"
    )
    for air, v in sorted(own.items(), key=lambda kv: -sum(kv[1])):
        per = sum(v) / len(v)
        print(f"     {air:24s} x{len(v):<3d} own {sum(v):6.2f} s  ({per:.2f}/instance)")
    fixed = re.findall(FIXED, log)
    if fixed:
        slot = sum(float(s) for _, s, _ in fixed)
        ahead = sum(float(s) for _, _, s in fixed)
        print(
            f"   fixed sections: {len(fixed)} builds, {slot:.2f} s under the slot,"
            f" {ahead:.2f} s read and uploaded ahead of it"
        )


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        summarize(pathlib.Path(arg))
