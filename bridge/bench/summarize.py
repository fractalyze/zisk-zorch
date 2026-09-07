#!/usr/bin/env python3
"""Phase timings of `cargo-zisk prove -vv` logs (native or bridged), plus the
bridge's per-instance totals from its `ZZ_LOG=2` lines: what each family's
proves cost on the client, how long instances waited for it, and the fixed
sections rebuilt on family switches. Usage: summarize.py <run.log>..."""
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
FIXED = r"\[zz \+\s*[\d.]+\] fixed sections for (\w+) in ([\d.]+) s"


def summarize(path: pathlib.Path) -> None:
    log = path.read_text(errors="replace")
    phases = {
        p: int(m.group(1)) / 1000
        for p in PHASES
        if (m := re.search(rf"INFO: <<< {p} \((\d+)ms\)", log))
    }
    wall = re.search(r"Elapsed \(wall clock\).*?(\d+):([\d.]+)", log)
    wall_s = int(wall.group(1)) * 60 + float(wall.group(2)) if wall else "?"
    verified = "Proof verified successfully" in log
    line = f"## {path}  wall {wall_s} s  verified={verified}"
    if streams := re.search(STREAMS, log):
        line += f"  pil2 streams basic/recursive {streams.group(1)}/{streams.group(2)}"
    print(line)
    print("   " + "  ".join(f"{p.lower()} {v:.2f} s" for p, v in phases.items()))
    inst = re.findall(INSTANCE, log)
    if not inst:
        return
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
        secs = sum(float(s) for _, s in fixed)
        print(f"   fixed sections: {len(fixed)} builds, {secs:.2f} s")
    if oom := re.search(r"PJRT error in \w+: (Out of memory[^\n]*)", log):
        print(f"   ABORTED: {oom.group(1)}")


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        summarize(pathlib.Path(arg))
