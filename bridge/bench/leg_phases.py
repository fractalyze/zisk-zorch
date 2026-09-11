#!/usr/bin/env python3
"""The inner-proof leg's structure out of a `cargo-zisk prove -vv` run log:
the wall the basic proofs occupied, the wall the recursion occupied, and how
much of the leg neither of them explains. This is where the phase walls in
docs/bridge.md "The gap is the basic phase's wall" come from, so a native and
a bridged run can be compared by the shape of their legs and not only by the
one number at the end.

Two things this reads that are easy to get wrong by hand.

**Where a run's basic proofs are.** proofman brackets what it schedules, so a
native run's basic proofs are its `GEN_PROOF_n` spans. Under the bridge they
are not: `gen_proof` returns as soon as the work reaches a worker, so those
spans are milliseconds against proves that take seconds. The bridge's own
`ZZ_LOG` line per instance is the prove, and it separates the time an instance
held the client from the time it spent waiting for one. A log carrying those
lines is read as a bridged run; one without them is read as pil2's own, and
if that reading leaves the basic proofs covering almost none of the leg the
report says so rather than publishing it -- a bridged run with `ZZ_LOG` off
looks exactly like a native run whose proofs were free.

**A wall is a union, not a sum.** Proofs on three streams overlap, so summing
their spans counts the same nanosecond several times; the wall is what at
least one of them covered. Both are printed, because their ratio is the
concurrency the stack achieved -- and on contended streams that ratio
overstates what the streams buy, since a proof sharing a card takes longer
than it would alone.

**`leg - basic phase` is a residual, not the recursion's cost.** The two
phases overlap, and how much of the recursion hides under the basic phase
moves with how long the basic phase is: the same stack at one stream instead
of three leaves a different remainder with the same recursion. So the report
prints the overlap it can prove -- `basic + recursion - leg`, which needs no
common clock and is a lower bound -- beside the remainder, and neither should
be quoted as what the recursion cost.

Usage: leg_phases.py <run.log>...
"""
from __future__ import annotations

import argparse
import pathlib
import re
import statistics
import sys
from datetime import datetime, timezone

# Python puts this file's own directory on sys.path rather than the repo root,
# so the package import below cannot resolve on its own. Under bazel the module
# is imported as `bridge.bench.leg_phases` and __package__ is already set.
if not __package__:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from bridge.bench.nsys_trace import Span, covered, merge  # noqa: E402

TS = r"(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d+)Z"
LEG = re.compile(TS + r".*<<< GENERATING_INNER_PROOFS \((\d+)ms\)")
BASIC = re.compile(TS + r".*<<< GEN_PROOF_\d+ \[\d+:\d+\] \((\d+)ms\)")
RECURSIVE = re.compile(TS + r".*<<< GEN_RECURSIVE_PROOF_\w+ \[\d+:\d+\] \((\d+)ms\)")
# `[zz +  6.917] instance 9 Air_n21 (worker): 1.228 s, of which 0.681 s waiting`
# -- a streamed instance says `(streamed)`, and a tally that matches only
# `(worker)` silently drops it.
HELD = re.compile(
    r"\[zz \+\s*([\d.]+)\] instance \d+ \S+ \((?:worker|streamed)\): "
    r"([\d.]+) s, of which ([\d.]+) s waiting"
)

NS = 1_000_000_000
# Below this share of the leg, proofman's basic spans are not prove time and
# the log is almost certainly a bridged run with ZZ_LOG off.
MIN_BASIC_SHARE = 0.25


def stamp_ns(stamp: str) -> int:
    when = datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%S.%f")
    return round(when.replace(tzinfo=timezone.utc).timestamp() * NS)


def ended_at(pattern: re.Pattern[str], log: str) -> list[Span]:
    """The spans proofman closed, each read back from its end and duration."""
    spans = []
    for m in pattern.finditer(log):
        end, duration = stamp_ns(m.group(1)), int(m.group(2)) * NS // 1000
        spans.append((end - duration, end))
    return spans


def held(log: str) -> list[Span]:
    """The intervals in which an instance held a client, on the bridge's own
    clock -- seconds since bridge-up, which is not proofman's clock. Only
    durations and unions are taken of these, never an intersection with a
    proofman span."""
    spans = []
    for m in HELD.finditer(log):
        end = round(float(m.group(1)) * NS)
        spans.append((end - round((float(m.group(2)) - float(m.group(3))) * NS), end))
    return spans


class Leg:
    """One run's leg, its two phases, and what they leave over."""

    def __init__(self, log: str) -> None:
        m = LEG.search(log)
        if m is None:
            raise ValueError("no closing GENERATING_INNER_PROOFS in the log")
        self.leg = int(m.group(2)) / 1000
        self.bridged = bool(HELD.search(log))
        basic = held(log) if self.bridged else ended_at(BASIC, log)
        recursive = ended_at(RECURSIVE, log)
        self.n_basic, self.n_recursive = len(basic), len(recursive)
        self.basic_wall = covered(merge(basic)) / NS
        self.basic_spans = covered(basic) / NS
        self.recursive_wall = covered(merge(recursive)) / NS
        self.recursive_spans = covered(recursive) / NS

    @property
    def overlap(self) -> float:
        """At least this much of the recursion ran inside the basic phase.
        Both phases sit inside the leg, so whatever they cover past its length
        they cover at once -- true of durations alone, which is what lets it
        hold when the two phases were read off different clocks."""
        return max(0.0, self.basic_wall + self.recursive_wall - self.leg)

    @property
    def remainder(self) -> float:
        """`leg - basic phase`. A residual: it holds whatever recursion the
        basic phase did not cover, so it moves when the basic phase does."""
        return self.leg - self.basic_wall

    @property
    def suspect(self) -> bool:
        """proofman's spans were used and they explain almost none of the leg."""
        return not self.bridged and self.basic_wall < MIN_BASIC_SHARE * self.leg


def concurrency(wall: float, spans: float) -> str:
    return f"{spans / wall:.2f}x" if wall else "n/a"


def report(path: pathlib.Path, leg: Leg) -> None:
    source = "the bridge's ZZ_LOG" if leg.bridged else "proofman's GEN_PROOF"
    print(
        f"## {path}  leg {leg.leg:.3f} s, {leg.n_basic} basic"
        f" + {leg.n_recursive} recursive proofs, basic from {source}"
    )
    if leg.suspect:
        print(
            f"   WARNING: the basic proofs cover only {leg.basic_wall:.3f} s of"
            " the leg. If this is a bridged run, re-run it with ZZ_LOG set --"
            " proofman's spans are not prove time there."
        )
    print(
        f"   basic phase  {leg.basic_wall:7.3f} s of wall, {leg.basic_spans:7.3f} s"
        f" of spans, {concurrency(leg.basic_wall, leg.basic_spans)}"
    )
    print(
        f"   recursion    {leg.recursive_wall:7.3f} s of wall,"
        f" {leg.recursive_spans:7.3f} s of spans,"
        f" {concurrency(leg.recursive_wall, leg.recursive_spans)}"
    )
    print(
        f"   at least     {leg.overlap:7.3f} s of the recursion ran inside"
        " the basic phase"
    )
    print(
        f"   leg - basic  {leg.remainder:7.3f} s, a residual --"
        " see the module docstring"
    )


def summarize(legs: list[Leg]) -> None:
    """Median and range per quantity. Every figure this page's docs quote
    carries a pass count and a spread, because on this rig the leg drifts by
    more than most of what gets measured on it is worth."""
    print(f"== {len(legs)} logs")
    for label, get in (
        ("leg", lambda x: x.leg),
        ("basic phase", lambda x: x.basic_wall),
        ("recursion", lambda x: x.recursive_wall),
        ("overlap, at least", lambda x: x.overlap),
        ("leg - basic", lambda x: x.remainder),
    ):
        vals = [get(x) for x in legs]
        print(
            f"   {label:18s} median {statistics.median(vals):7.3f} s"
            f"  [{min(vals):.3f}-{max(vals):.3f}]"
        )


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("log", nargs="+", type=pathlib.Path)
    args = ap.parse_args(argv[1:])
    legs = []
    for path in args.log:
        leg = Leg(path.read_text(errors="replace"))
        report(path, leg)
        legs.append(leg)
    if len(legs) > 1:
        summarize(legs)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
