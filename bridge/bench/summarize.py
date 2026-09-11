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

# Python puts this file's own directory on sys.path rather than the repo root,
# so the package import below cannot resolve on its own. Under bazel the module
# is imported as `bridge.bench.summarize` and __package__ is already set.
if not __package__:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from bridge.bench.run_log import instances  # noqa: E402

PHASES = (
    "INITIALIZING_PROOFMAN",
    "CALCULATING_CONTRIBUTIONS",
    "GENERATING_INNER_PROOFS",
    "GENERATE_VADCOP_FINAL_PROOF",
)
STREAMS = (
    r"Using (\d+) streams per GPU for basic proofs"
    r" and (\d+) streams per GPU for recursive"
)
FIXED = (
    r"\[zz \+\s*[\d.]+\] fixed sections for (\w+): ([\d.]+) s under the slot,"
    r" ([\d.]+) s ahead of it"
)
# A read-ahead upload the bridge caught and sent to the slot instead. Only
# logged at ZZ_LOG>=1, so its absence means nothing.
RESCUED = r"\[zz \+\s*[\d.]+\] (\w+): read-ahead upload gave way to the slot \([^\n]*\)"
# An out-of-memory message. NOT evidence of an abort on its own: the bridge
# catches one of these and carries on, and Rust's default panic hook prints
# the message before `catch_unwind` ever sees it, unconditionally and whatever
# ZZ_LOG says. A rescued run therefore holds text identical to an aborted
# one's, and no amount of stripping the bridge's own line changes that.
OOM = r"PJRT error in \w+: (Out of memory[^\n]*)"
# Whether the run finished, which is what actually tells the two apart.
# run.sh appends this after the prover exits.
EXIT = r"^exit=(\d+)$"
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
    # Both phrases, because the wording has changed across prover builds and
    # neither identifies the mode: a native and a bridged run of 2026-09-06
    # both end "Proof verified successfully", while a native and a bridged run
    # built on 09-08 both end "Vadcop Final proof was verified". Match both and
    # do not infer the stack from which one appears.
    verified = any(p in log for p in VERIFIED)
    line = f"## {path}  wall {wall_s} s  verified={verified}"
    if streams := re.search(STREAMS, log):
        line += f"  pil2 streams basic/recursive {streams.group(1)}/{streams.group(2)}"
    print(line)
    print("   " + "  ".join(f"{p.lower()} {v:.2f} s" for p, v in phases.items()))
    inst = instances(log)
    if inst:
        report_bridge(log, inst)
    # Outside the block above: a run can abort before its first instance
    # finishes, and that is when it most often does, so neither of these may
    # sit behind "no instances completed".
    report_outcome(log, verified)


def report_outcome(log: str, verified: bool) -> None:
    """Whether the run finished, and what an out-of-memory in it meant.

    Keyed on the run's own outcome rather than on any message in the log: a
    rescued out-of-memory and a fatal one leave the same text behind."""
    ooms = re.findall(OOM, log)
    exited = re.search(EXIT, log, re.M)
    if exited:
        completed = exited.group(1) == "0"
    else:
        # No exit line (a partial capture, or a log not written by run.sh).
        # Do not cry abort on a log that merely stops early — say so only
        # when it also carries a failure to point at.
        completed = verified or not ooms
    if not completed:
        print(
            f"   ABORTED: {ooms[0]}" if ooms else "   ABORTED: the run did not finish"
        )
        return
    if not ooms:
        return
    # Completed with an out-of-memory in it: the read-ahead caught it.
    rescued = re.findall(RESCUED, log)
    if rescued:
        by_air = collections.Counter(rescued)
        airs = ", ".join(f"{a} x{n}" for a, n in sorted(by_air.items()))
        detail = f"{len(rescued)} ({airs})"
    else:
        # With ZZ_LOG unset the bridge logs nothing, leaving only the panic
        # hook's output, which does not name the AIR. Count those rather than
        # guessing an attribution.
        detail = f"{len(ooms)}, air not recorded (ZZ_LOG>=1 names them)"
    print(
        f"   read-ahead uploads sent to the slot: {detail}"
        " — the card was full, and the run went on"
    )


def report_bridge(log: str, inst: list) -> None:
    """The bridge's own per-instance and fixed-section totals."""
    own = collections.defaultdict(list)
    waited = 0.0
    for one in inst:
        own[one.air].append(one.held)
        waited += one.waiting
    done = [one.done for one in inst]
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
