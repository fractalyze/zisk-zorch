#!/usr/bin/env python3
"""What the host was doing while the bridge's device sat idle, from a capture
of a whole run: an `nsys stats --report cuda_gpu_trace` CSV for the kernels
and an `nsys stats --report nvtx_pushpop_trace` CSV for the host phases.

The bridge's leg is its first kernel to its last. Its own kernels are busy
for less than half of it (#193: 2.60 s of a 5.47 s leg) and the uploads
account for 0.30 s of the difference, so the rest is time in which the device
has nothing to run because the host has not given it anything. This report
says which host phase was running in each such nanosecond.

The phases are the `host/` NVTX ranges the bridge opens (`src/nvtx.rs`), one
per step of `Bridge::take`, `prove_owned` and the schedule in `AirDriver::
prove`. A phase's *self* time excludes the ranges nested inside it, so the
per-program ranges around each `Artifact::run` subtract themselves out and
what a phase keeps is the host work between two enqueues — the downloads, the
transcript, the query draw. The program ranges are reported too: idle inside
one is the device waiting on that program's own dispatch.

**Only the prove holding the client can explain the idle.** The bridge runs
one prove at a time per client but hands every instance to a thread of its
own, so a dozen threads are alive at once and all but one are queued. A
queued thread's `host/admit` and `host/slot_wait` cover almost every idle
nanosecond in the leg by construction — they are waits, not costs, and
charging idle to them says only that a prove was waiting, which is the
premise rather than the finding. So a prove's *turn* is the span from where
its thread leaves `host/slot_wait` holding the slot mutex to the end of its
`host/prove`; the mutex serialises turns, and inside one the holder's phases
tile the time exactly. That makes the split exact rather than overlapping:

    device idle = idle under the holder's phases + idle with no prove holding

What the other threads were doing in the same nanoseconds is reported after
that, marked as overlapping rather than additive — it is where an upload
waiting on the client (#193) and `take` unpacking the next instance on
proofman's proof worker show up.

Read a phase's share as "the device idled this long with this phase running",
which is an upper bound on what removing the phase would return: a phase that
runs while the device would have idled anyway costs the leg nothing.

Capture recipe in docs/bridge.md "Profiling". The prover must be built with
`--features nvtx`, and three flags are not optional: `--cuda-graph-trace=node`
(XLA runs the fusions as CUDA graphs and nsys sees no kernels through them),
`-t cuda,nvtx` (or there are no ranges), and `--sample=none --cpuctxsw=none`
(nsys 2026.1.3 deadlocks in report generation on a run this size).

Usage: host_idle.py <cuda_gpu_trace.csv> <nvtx_pushpop_trace.csv>
                    [cuda_api_trace.csv] [--top N]"""
from __future__ import annotations

import argparse
import collections
import csv
import pathlib
import re
import sys
import typing

Span = tuple[int, int]

# nsys writes the unit into the column header, and which one it picks depends
# on the capture's length.
UNITS_NS = {"ns": 1, "us": 1_000, "µs": 1_000, "ms": 1_000_000, "s": 1_000_000_000}

# The prefix `src/nvtx.rs` puts on a host phase, so a phase is told from the
# per-program ranges around each `Artifact::run`.
HOST = "host/"

# A prove's turn on the client runs from where its thread leaves this phase
# (it has the slot mutex) to the end of this one.
SLOT_WAIT, PROVE = f"{HOST}slot_wait", f"{HOST}prove"


def column(header: list[str], prefix: str) -> tuple[str, int]:
    """The named column and the multiplier from its unit to nanoseconds."""
    for name in header:
        if name.startswith(prefix):
            unit = re.search(r"\(([^)]*)\)", name)
            scale = UNITS_NS.get(unit.group(1) if unit else "ns")
            if scale is None:
                raise ValueError(f"{name}: unit is not a time")
            return name, scale
    raise ValueError(f"no {prefix!r} column in {header}")


def merge(spans: list[Span]) -> list[Span]:
    """The spans as a sorted, non-overlapping cover of the same time."""
    out: list[Span] = []
    for start, end in sorted(spans):
        if out and start <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], end))
        else:
            out.append((start, end))
    return out


def covered(spans: list[Span]) -> int:
    return sum(end - start for start, end in spans)


def intersect(a: list[Span], b: list[Span]) -> list[Span]:
    """The time both merged covers hold."""
    out: list[Span] = []
    i = j = 0
    while i < len(a) and j < len(b):
        lo, hi = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if lo < hi:
            out.append((lo, hi))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return out


def subtract(a: list[Span], b: list[Span]) -> list[Span]:
    """The time the merged cover `a` holds and the merged cover `b` does
    not."""
    out: list[Span] = []
    j = 0
    for lo, hi in a:
        # Both covers are sorted, so the cursor into `b` only ever moves
        # forward across the whole sweep.
        while j < len(b) and b[j][1] <= lo:
            j += 1
        cur, k = lo, j
        while k < len(b) and b[k][0] < hi:
            if b[k][0] > cur:
                out.append((cur, b[k][0]))
            cur = max(cur, b[k][1])
            k += 1
        if cur < hi:
            out.append((cur, hi))
    return out


def owner_is_bridge(kernel_name: str) -> bool:
    """Whether the bridge emitted a kernel. A bridged run has two provers on
    one card: XLA writes a fusion's name with no argument list
    (`loop_add_fusion`, `sponge_hash_1`) and pil2's kernels are C++
    signatures (`_add(Goldilocks::Element *, ...)`), so a `(` in the name is
    what tells them apart. Same rule as `h2d_overlap.py`, which #196 lands;
    the two want one module once it does."""
    return "(" not in kernel_name


class RangeRow(typing.NamedTuple):
    """One NVTX push/pop range instance."""

    name: str
    span: Span
    tid: str
    range_id: str
    parent_id: str


def phase_name(nvtx_range: str) -> str | None:
    """The range name if it is one of the bridge's, else None. `nsys` writes
    a range that has a domain as `<domain>:<name>`; the bridge's are in the
    default domain, which nsys writes as a bare leading colon, and XLA's own
    are in `TSL`."""
    domain, sep, name = nvtx_range.partition(":")
    if not sep:
        return nvtx_range
    return name if not domain else None


def read_kernels(path: pathlib.Path) -> list[Span]:
    """The bridge's kernel spans. Memory operations are left out: a copy is
    not the device computing, and the leg's idle is what this report is
    about."""
    spans: list[Span] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        start_col, start_scale = column(reader.fieldnames or [], "Start")
        dur_col, dur_scale = column(reader.fieldnames or [], "Duration")
        for row in reader:
            name = row["Name"]
            if "memcpy" in name or "memset" in name or not owner_is_bridge(name):
                continue
            start = round(float(row[start_col]) * start_scale)
            spans.append((start, start + round(float(row[dur_col]) * dur_scale)))
    return spans


def read_ranges(path: pathlib.Path) -> list[RangeRow]:
    """Every bridge NVTX range instance in the capture."""
    rows: list[RangeRow] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        start_col, start_scale = column(reader.fieldnames or [], "Start")
        end_col, end_scale = column(reader.fieldnames or [], "End")
        for row in reader:
            name = phase_name(row["Name"])
            if name is None:
                continue
            start = round(float(row[start_col]) * start_scale)
            end = round(float(row[end_col]) * end_scale)
            rows.append(
                RangeRow(
                    name, (start, end), row["TID"], row["RangeId"], row["ParentId"]
                )
            )
    return rows


def self_spans(rows: list[RangeRow]) -> list[tuple[RangeRow, list[Span]]]:
    """Each range instance with the time of the ranges nested in it removed.

    A phase that encloses an `Artifact::run` would otherwise be charged with
    the whole program's device time; what is wanted is the host work between
    two enqueues. Only the bridge's own ranges are children here — XLA's are
    filtered out before this — so a program range keeps the dispatch gaps
    inside it rather than losing them to a `TSL` range."""
    children: dict[str, list[Span]] = collections.defaultdict(list)
    for row in rows:
        if row.parent_id:
            children[row.parent_id].append(row.span)
    return [
        (row, subtract([row.span], merge(children.get(row.range_id, []))))
        for row in rows
    ]


def turns(rows: list[RangeRow]) -> dict[str, list[Span]]:
    """Each thread's turns holding the client, from where it leaves
    `host/slot_wait` to the end of its `host/prove`. A thread proves once in
    the usual arrangement (`prove_async` gives every instance a thread), so
    the two lists pair up in order."""
    ends: dict[str, dict[str, list[int]]] = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )
    for row in rows:
        if row.name in (SLOT_WAIT, PROVE):
            ends[row.tid][row.name].append(row.span[1])
    out: dict[str, list[Span]] = {}
    for tid, by_name in ends.items():
        starts, stops = sorted(by_name[SLOT_WAIT]), sorted(by_name[PROVE])
        out[tid] = merge(list(zip(starts, stops)))
    return out


class ApiRow(typing.NamedTuple):
    """One CUDA driver call, from `nsys stats --report cuda_api_trace`."""

    name: str
    span: Span
    tid: str


def read_api(path: pathlib.Path) -> list[ApiRow]:
    """Every CUDA driver call in the capture, with the thread that made it."""
    rows: list[ApiRow] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        start_col, start_scale = column(reader.fieldnames or [], "Start")
        dur_col, dur_scale = column(reader.fieldnames or [], "Duration")
        for row in reader:
            start = round(float(row[start_col]) * start_scale)
            end = start + round(float(row[dur_col]) * dur_scale)
            rows.append(ApiRow(row["Name"], (start, end), row["Tid"]))
    return rows


def api_idle(
    api: list[ApiRow], turns: dict[str, list[Span]], idle: list[Span]
) -> tuple[dict[str, int], collections.Counter]:
    """Idle time a holding thread spent inside each CUDA driver call, and how
    many such calls it made.

    This is a second cut of the *same* idle the phases account for, not more
    of it: a module load happens inside some program's range, so the two
    sections answer "which step of the prove" and "what the driver was doing"
    about one nanosecond. Only a holder's calls are candidates, for the reason
    the phase attribution turns on — which is also why the count is of the
    holders' calls alone, so it divides the idle beside it."""
    held = {tid: intersect(idle, spans) for tid, spans in turns.items() if spans}
    by_name: dict[str, dict[str, list[Span]]] = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )
    calls: collections.Counter = collections.Counter()
    for row in api:
        if row.tid in held:
            by_name[row.name][row.tid].append(row.span)
            calls[row.name] += 1
    return {
        name: sum(
            covered(intersect(merge(spans), held[tid]))
            for tid, spans in per_tid.items()
        )
        for name, per_tid in by_name.items()
    }, calls


class Shares(typing.NamedTuple):
    """Idle time under each phase name, split by whether the phase's own
    thread held the client at the time."""

    holding: dict[str, int]
    other: dict[str, int]
    counts: collections.Counter


def shares(rows: list[RangeRow], idle: list[Span]) -> Shares:
    by_tid = turns(rows)
    holding: dict[str, int] = collections.defaultdict(int)
    other: dict[str, int] = collections.defaultdict(int)
    for row, own in self_spans(rows):
        hit = intersect(merge(own), idle)
        held = intersect(hit, by_tid.get(row.tid, []))
        holding[row.name] += covered(held)
        other[row.name] += covered(hit) - covered(held)
    return Shares(
        dict(holding), dict(other), collections.Counter(row.name for row in rows)
    )


def rank(share: dict[str, int], top: int) -> list[tuple[str, int]]:
    return [kv for kv in sorted(share.items(), key=lambda kv: -kv[1])[:top] if kv[1]]


def report(
    trace: pathlib.Path,
    nvtx: pathlib.Path,
    api: pathlib.Path | None,
    top: int,
    proves: int | None,
) -> None:
    kernels = merge(read_kernels(trace))
    if not kernels:
        print(
            f"## {trace.name}  no bridge kernels"
            " — was the capture --cuda-graph-trace=node?"
        )
        return
    rows = read_ranges(nvtx)
    if not rows:
        print(
            f"## {nvtx.name}  no bridge ranges — was the prover built --features nvtx?"
        )
        return
    leg = [(kernels[0][0], kernels[-1][1])]
    idle = subtract(leg, kernels)
    s = shares(rows, idle)
    # `host/prove` wraps one instance's turn on the client, so its instances
    # are the proves the capture holds.
    n = proves or s.counts.get(PROVE, 0) or 1
    held = sum(s.holding.values())

    def line(name: str, ns: int, indent: str = "      ") -> str:
        return (
            f"{indent}{name:22s} {ns / 1e9:7.3f} s"
            f"  {ns / n / 1e6:7.1f} ms per prove"
        )

    print(
        f"## {trace.name}  leg {covered(leg) / 1e9:.2f} s,"
        f" bridge kernels busy {covered(kernels) / 1e9:.2f} s,"
        f" device idle {covered(idle) / 1e9:.2f} s over {n} proves"
    )
    print(line("idle, client held", held, "   "))
    for name, ns in rank(s.holding, top):
        print(line(name, ns) + f"  x{s.counts[name]}")
    print(line("idle, client free", covered(idle) - held, "   "))
    print(
        "   meanwhile on other threads (overlaps the above, not additive;"
        " admit/slot_wait are queue waits by construction)"
    )
    for name, ns in rank(s.other, top):
        print(line(name, ns) + f"  x{s.counts[name]}")
    if api is None:
        return
    driver, calls = api_idle(read_api(api), turns(rows), idle)
    print("   the same idle, by what the holder was inside (CUDA driver call)")
    for name, ns in rank(driver, top):
        print(line(name, ns) + f"  x{calls[name]}")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("trace", type=pathlib.Path, help="cuda_gpu_trace CSV")
    ap.add_argument("nvtx", type=pathlib.Path, help="nvtx_pushpop_trace CSV")
    ap.add_argument(
        "api",
        type=pathlib.Path,
        nargs="?",
        help="cuda_api_trace CSV, to also cut the idle by CUDA driver call",
    )
    ap.add_argument("--top", type=int, default=14, help="phases listed per section")
    ap.add_argument(
        "--proves", type=int, help="proves in the capture, when the guess is wrong"
    )
    args = ap.parse_args(argv[1:])
    report(args.trace, args.nvtx, args.api, args.top, args.proves)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
