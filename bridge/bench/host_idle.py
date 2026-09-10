#!/usr/bin/env python3
"""What the host was doing while the bridge's device sat idle, from a capture
of a whole run: an `nsys stats --report cuda_gpu_trace` CSV for the kernels
and an `nsys stats --report nvtx_pushpop_trace` CSV for the host phases.

The bridge's leg is its first kernel to its last. Its own kernels are busy
for less than half of it (#193: 2.60 s of a 5.47 s leg) and the uploads
account for 0.31 s of the difference, so the rest is time in which the device
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

That identity needs **one client**. `ZZ_CLIENTS` defaults to 3, and with
several, two threads hold different slot mutexes at once: their turns
overlap, one nanosecond is charged to two phases, and the shares stop being
a split. The report totals the turns as a union so "client free" cannot go
negative, and prints a `counted twice` line when it happens rather than
letting it pass silently.

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
                    [cuda_api_trace.csv] [--top N] [--minus-call NAME]"""
from __future__ import annotations

import argparse
import collections
import csv
import pathlib
import sys
import typing

from bridge.bench.nsys_trace import (
    BRIDGE,
    Span,
    column,
    covered,
    intersect,
    merge,
    owner,
    subtract,
)

# The prefix `src/nvtx.rs` puts on a host phase, so a phase is told from the
# per-program ranges around each `Artifact::run`.
HOST = "host/"

# A prove's turn on the client runs from where its thread leaves this phase
# (it has the slot mutex) to the end of this one.
SLOT_WAIT, PROVE = f"{HOST}slot_wait", f"{HOST}prove"


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
            if "memcpy" in name or "memset" in name or owner(name) != BRIDGE:
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
        # The count has to be of the same population the nanoseconds come
        # from, because the report divides one by the other. A call on a
        # holder's thread made *before* it took the slot — the event records
        # and copies of `host/upload_inputs` — contributes no idle, so
        # counting it would deflate the per-call cost.
        if row.tid in held and intersect([row.span], held[row.tid]):
            by_name[row.name][row.tid].append(row.span)
            calls[row.name] += 1
    return {
        name: sum(
            covered(intersect(merge(spans), held[tid]))
            for tid, spans in per_tid.items()
        )
        for name, per_tid in by_name.items()
    }, calls


def without_call(
    api: list[ApiRow],
    rows: list[RangeRow],
    turns: dict[str, list[Span]],
    idle: list[Span],
    call: str,
) -> dict[str, int]:
    """Idle under each phase that the holder did **not** spend inside `call`.

    The two cuts above answer "which step of the prove" and "what the driver
    was doing" about the same nanoseconds; this crosses them, which is what
    sizing a bridge-side lever needs. When a driver call leaves the prove path
    it takes its share of every phase with it, and what a phase keeps is the
    rest — so the phase column alone over-states a lever by whatever the
    driver was doing inside that phase.

    Holder-only, for the reason the phase cut turns on: a call on a queued
    thread explains none of the holder's idle."""
    held = {tid: intersect(idle, spans) for tid, spans in turns.items() if spans}
    calls: dict[str, list[Span]] = collections.defaultdict(list)
    for row in api:
        if row.name == call and row.tid in held:
            calls[row.tid].append(row.span)
    in_call = {tid: intersect(merge(spans), held[tid]) for tid, spans in calls.items()}
    rest: dict[str, int] = collections.defaultdict(int)
    for row, own in self_spans(rows):
        hit = intersect(intersect(merge(own), idle), turns.get(row.tid, []))
        rest[row.name] += covered(subtract(hit, in_call.get(row.tid, [])))
    return dict(rest)


class Shares(typing.NamedTuple):
    """Idle time under each phase name, split by whether the phase's own
    thread held the client at the time."""

    holding: dict[str, int]
    other: dict[str, int]
    counts: collections.Counter


def shares(
    rows: list[RangeRow], idle: list[Span], by_tid: dict[str, list[Span]] | None = None
) -> Shares:
    # `by_tid` is derivable from `rows`, so a caller with one already — the
    # report needs it for its own totals — passes it rather than paying for
    # the same sweep again.
    by_tid = turns(rows) if by_tid is None else by_tid
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
    minus_call: str | None = None,
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
    by_tid = turns(rows)
    s = shares(rows, idle, by_tid)
    # `host/prove` wraps one instance's turn on the client, so its instances
    # are the proves the capture holds.
    n = proves or s.counts.get(PROVE, 0) or 1
    # Time inside *some* prove's turn, counted once. With one client the slot
    # mutex serialises turns and this equals the sum of the phase shares; with
    # several, two threads hold different mutexes at once, their turns overlap
    # and the shares double-count. Taking the union keeps `client free`
    # non-negative and makes the excess visible instead of silent.
    inside = covered(
        intersect(idle, merge([sp for spans in by_tid.values() for sp in spans]))
    )
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
    print(line("idle, client held", inside, "   "))
    for name, ns in rank(s.holding, top):
        print(line(name, ns) + f"  x{s.counts[name]}")
    if held > inside:
        print(
            f"   {'-> counted twice':22s} {(held - inside) / 1e9:7.3f} s"
            "  (turns overlap — several clients; the shares are not a split)"
        )
    print(line("idle, client free", covered(idle) - inside, "   "))
    print(
        "   meanwhile on other threads (overlaps the above, not additive;"
        " admit/slot_wait are queue waits by construction)"
    )
    for name, ns in rank(s.other, top):
        print(line(name, ns) + f"  x{s.counts[name]}")
    if api is None:
        return
    api_rows = read_api(api)
    driver, calls = api_idle(api_rows, by_tid, idle)
    print(
        "   the same idle, by what the holder was inside (CUDA driver call;"
        " x<n> counts the calls that contributed, not every call made)"
    )
    for name, ns in rank(driver, top):
        print(line(name, ns) + f"  x{calls[name]}")
    if minus_call is None:
        return
    # A name that is in no call in the capture would subtract nothing and
    # print the phase column back unchanged, which reads as "this call is
    # free" rather than "you misspelled it" — and the table above is
    # `--top`-capped, so its absence there proves nothing either way.
    if not any(row.name == minus_call for row in api_rows):
        raise ValueError(
            f"{minus_call}: no such call in {api.name}."
            f" The capture holds {len({row.name for row in api_rows})} call names;"
            " run without --minus-call to see the ones that cost idle."
        )
    rest = without_call(api_rows, rows, by_tid, idle, minus_call)
    kept = sum(rest.values())
    # A call that is made but never while the device starves is a real
    # answer, and its table is the phase column exactly. Say which of the
    # two identical-looking tables this is.
    took = held - kept
    note = "" if took else " — it contributed no idle, so this is the column above"
    print(f"   what each phase keeps once {minus_call} leaves the prove path{note}")
    for name, ns in rank(rest, top):
        print(line(name, ns))
    print(line("all phases", kept, "   "))


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
    ap.add_argument(
        "--minus-call",
        metavar="NAME",
        help="also report each phase's idle outside this CUDA driver call,"
        " for sizing what survives the call leaving the prove path",
    )
    args = ap.parse_args(argv[1:])
    if args.minus_call and args.api is None:
        # `report` returns before the driver-call cut when there is no API
        # trace, so the flag would otherwise be dropped in silence.
        ap.error("--minus-call needs the cuda_api_trace CSV argument")
    report(args.trace, args.nvtx, args.api, args.top, args.proves, args.minus_call)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
