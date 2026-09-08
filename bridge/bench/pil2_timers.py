#!/usr/bin/env python3
"""pil2's own per-instance GPU timers out of a `cargo-zisk prove -vv` log:
what each basic instance spent committing and proving, over which kernel
categories, so the bridge's per-program table (`nvtx_programs.py`) has
something to be read against. See docs/bridge.md "Profiling".

pil2 prints these blocks from every stream, so run it at one stream
(`ZZ_GPU_HEADROOM_GB` large enough on the fork): with more than one the lines
interleave and neither the phases nor the categories can be attributed.

Usage: pil2_timers.py <run.log>... [--global-info pilout.globalInfo.json]"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import re
import sys

# `TIMERS FOR INSTANCE ID <instance> [<air group>:<air>]`, then one phase
# line, then the phase's kernel categories. Blocks from a basic proof and
# from the recursive proofs over it carry the same header, so `GEN_PROOF`
# (which proofman logs only for basic instances) is what tells them apart.
INSTANCE = re.compile(r"TIMERS FOR INSTANCE ID (\d+) \[(\d+):(\d+)\]")
PHASE = re.compile(r"<-- STARK_GPU_(COMMIT|PROOF) : ([\d.]+) s")
CATEGORY = re.compile(r"^\s*([A-Z_0-9]+)\s+:\s+([\d.]+)s \([\d.]+%\)")
GEN_PROOF = re.compile(r">>> GEN_PROOF_(\d+) \[(\d+):(\d+)\]")
STREAMS = re.compile(r"Using (\d+) streams per GPU for basic proofs")

Air = tuple[int, int]
Key = tuple[int, Air]


class Timers:
    """The seconds of one or more phases and the kernel categories under them."""

    def __init__(self) -> None:
        self.seconds: dict[str, float] = collections.defaultdict(float)
        self.categories: dict[str, float] = collections.defaultdict(float)

    def add(self, other: Timers) -> None:
        for phase, secs in other.seconds.items():
            self.seconds[phase] += secs
        for name, secs in other.categories.items():
            self.categories[name] += secs

    def total(self) -> float:
        return sum(self.seconds.values())

    def hottest(self, n: int) -> str:
        top = sorted(self.categories.items(), key=lambda kv: -kv[1])[:n]
        return ", ".join(f"{name} {secs:.3f}" for name, secs in top)


def air_names(global_info: pathlib.Path | None) -> dict[Air, str]:
    """`airs[air group][air].name` — the only place the ids in a timer block
    are spelled out."""
    if global_info is None:
        return {}
    airs = json.loads(global_info.read_text())["airs"]
    return {
        (g, a): air["name"]
        for g, group in enumerate(airs)
        for a, air in enumerate(group)
    }


def parse(log: str) -> tuple[dict[Key, Timers], Timers]:
    """Each basic instance's timers, and the recursive proofs over them
    summed. Keyed by instance and not by air: a workload runs several
    instances of the same air, and their timers are not one instance's."""
    instances: set[Key] = {
        (int(m.group(1)), (int(m.group(2)), int(m.group(3))))
        for m in GEN_PROOF.finditer(log)
    }
    basic: dict[Key, Timers] = collections.defaultdict(Timers)
    recursive = Timers()
    proved: set[Key] = set()
    key: Key | None = None
    timers: Timers | None = None
    for line in log.splitlines():
        if m := INSTANCE.search(line):
            key, timers = (int(m.group(1)), (int(m.group(2)), int(m.group(3)))), None
        elif (m := PHASE.search(line)) and key is not None:
            phase, seconds = m.group(1), float(m.group(2))
            # Only a basic instance commits, and only its first proof is the
            # basic one; the rest of its blocks are recursive proofs.
            if key in instances and (phase == "COMMIT" or key not in proved):
                timers = basic[key]
                if phase == "PROOF":
                    proved.add(key)
            else:
                timers = recursive
            timers.seconds[phase] += seconds
        elif timers is not None and (
            m := CATEGORY.match(line.partition("PilStark:")[2])
        ):
            timers.categories[m.group(1)] += float(m.group(2))
    return basic, recursive


def by_air(basic: dict[Key, Timers]) -> dict[Air, list[Timers]]:
    """The instances of each air. A report row is one air, so it has to carry
    how many instances it sums — the sha-hasher workload runs 13 Main, and a
    13x sum printed bare reads as what one Main instance cost."""
    airs: dict[Air, list[Timers]] = collections.defaultdict(list)
    for (_, air), timers in basic.items():
        airs[air].append(timers)
    return airs


def streams(log: str) -> int | None:
    """The basic-proof streams pil2 ran with, when it says so."""
    m = STREAMS.search(log)
    return int(m.group(1)) if m else None


def report(path: pathlib.Path, names: dict[Air, str], top: int) -> None:
    log = path.read_text(errors="replace")
    basic, recursive = parse(log)
    warning = ""
    if (n := streams(log)) and n > 1:
        warning = f"  WARNING: {n} streams — the blocks interleave, re-run on one"
    print(f"## {path}  {len(basic)} basic instances{warning}")
    whole = Timers()
    rows: list[tuple[Air, int, Timers]] = []
    for air, instances in by_air(basic).items():
        row = Timers()
        for inst in instances:
            row.add(inst)
        whole.add(row)
        rows.append((air, len(instances), row))
    for air, count, row in sorted(rows, key=lambda r: -r[2].total()):
        commit, proof = row.seconds["COMMIT"], row.seconds["PROOF"]
        print(
            f"   {names.get(air, 'air %d:%d' % air):20s} x{count:<3d}"
            f" commit {commit:.3f} s  proof {proof:.3f} s"
            f"  total {row.total():.3f} s ({row.total() / count:.3f}/instance)"
            f"  {row.hottest(top)}"
        )
    label = "all %d" % len(basic)
    print(f"   {label:25s} total {whole.total():.3f} s  {whole.hottest(top)}")
    if recursive.total():
        total, hot = recursive.total(), recursive.hottest(top)
        print(f"   {'recursive proofs':25s} total {total:.3f} s  {hot}")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("log", nargs="+", type=pathlib.Path)
    ap.add_argument(
        "--global-info",
        type=pathlib.Path,
        help="the proving key's pilout.globalInfo.json, to name the airs",
    )
    ap.add_argument(
        "--top", type=int, default=4, help="kernel categories listed per instance"
    )
    args = ap.parse_args(argv[1:])
    names = air_names(args.global_info)
    for path in args.log:
        report(path, names, args.top)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
