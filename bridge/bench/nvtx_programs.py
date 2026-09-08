#!/usr/bin/env python3
"""Per-program device time from an `nsys stats --report nvtx_kern_sum` CSV:
what each program of a prove costs on the device, over how many kernels and
which of them, plus the same total rolled up by kernel family. The ranges are
the ones the bridge's `nvtx` feature opens around every `Artifact::run`; see
docs/bridge.md "Profiling" for the capture recipe.

XLA opens ranges of its own (the `TSL` domain) around the same kernels; only
the bridge's, which carry no domain, are counted here.

Usage: nvtx_programs.py <nvtx_kern_sum.csv>... [--top N] [--proves N]"""
from __future__ import annotations

import argparse
import collections
import csv
import pathlib
import re
import statistics
import sys


def program(nvtx_range: str) -> str | None:
    """The range name if it is one of the bridge's, else None. `nsys` writes a
    range that has a domain as `<domain>:<name>`; the bridge's ranges are in
    the default domain, which has no name."""
    domain, sep, name = nvtx_range.partition(":")
    if not sep:
        return nvtx_range
    return name if not domain else None


def family(kernel: str) -> str:
    """A kernel name without the index XLA appends to each instance of a
    fusion: `sponge_hash_1` and `sponge_hash_4` are one family."""
    return re.sub(r"_\d+$", "", kernel)


class Program:
    """One NVTX range's kernels, summed over the capture."""

    def __init__(self) -> None:
        self.runs = 0
        self.kernels = 0
        self.by_kernel: dict[str, int] = collections.defaultdict(int)

    def add(self, row: dict[str, str]) -> None:
        # "NVTX Inst" counts the range instances that contained this kernel,
        # so the largest over a program's kernels is the times it ran.
        self.runs = max(self.runs, int(row["NVTX Inst"]))
        self.kernels += int(row["Kern Inst"])
        self.by_kernel[row["Kernel Name"]] += int(row["Total Time (ns)"])

    @property
    def total_ns(self) -> int:
        return sum(self.by_kernel.values())


def read(path: pathlib.Path) -> dict[str, Program]:
    progs: dict[str, Program] = collections.defaultdict(Program)
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if name := program(row["NVTX Range"]):
                progs[name].add(row)
    return progs


def proves(progs: dict[str, Program]) -> int:
    """How many proves the capture holds. Most programs run exactly once per
    prove, so their run count is the commonest one; `const_setup` runs once
    per family and the quotient runs once per chunk. Descending, so a tie
    between two counts resolves to the larger."""
    return statistics.mode(sorted((p.runs for p in progs.values()), reverse=True))


def per_prove(progs: dict[str, Program], n: int) -> list[tuple[str, Program, int]]:
    """Every program with the share of the capture to divide its totals by:
    the `n` proves, except for one that ran fewer times than that —
    `const_setup` is a per-family cost, so its own total is the honest
    number. A program that runs several times per prove, like the quotient
    over its chunks, keeps all of them in one row. Costliest first."""
    return sorted(
        ((name, p, min(p.runs, n)) for name, p in progs.items()),
        key=lambda row: -row[1].total_ns / row[2],
    )


def report(path: pathlib.Path, top: int, override: int | None) -> None:
    progs = read(path)
    if not progs:
        print(f"## {path}  no bridge ranges — was zz_prove built with --features nvtx?")
        return
    n = override or proves(progs)
    rows = per_prove(progs, n)
    print(
        f"## {path}  {n} proves, {len(rows)} programs,"
        f" {sum(p.total_ns / share for _, p, share in rows) / 1e9:.3f} s on the device"
        f" per prove over {sum(p.kernels / share for _, p, share in rows):.0f} kernels"
    )
    for name, p, share in rows:
        hot = sorted(p.by_kernel.items(), key=lambda kv: -kv[1])[:top]
        label = name if p.runs == n else f"{name} (x{p.runs})"
        kernels = ", ".join(f"{k} {ns / share / 1e9:.3f}" for k, ns in hot)
        print(
            f"   {label:24s} {p.total_ns / share / 1e9:7.3f} s"
            f" {p.kernels / share:5.0f} kernels  {kernels}"
        )
    families: dict[str, float] = collections.defaultdict(float)
    for _, p, share in rows:
        for kernel, ns in p.by_kernel.items():
            families[family(kernel)] += ns / share
    ranked = sorted(families.items(), key=lambda kv: -kv[1])[:top]
    print("   by family: " + ", ".join(f"{f} {ns / 1e9:.3f} s" for f, ns in ranked))


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv", nargs="+", type=pathlib.Path)
    ap.add_argument("--top", type=int, default=3, help="kernels listed per program")
    ap.add_argument(
        "--proves", type=int, help="proves in the capture, when the guess is wrong"
    )
    args = ap.parse_args(argv[1:])
    for path in args.csv:
        report(path, args.top, args.proves)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
