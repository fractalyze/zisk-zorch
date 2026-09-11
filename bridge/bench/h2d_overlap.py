#!/usr/bin/env python3
"""How much of a run's host-to-device time the device actually waited for,
from an `nsys stats --report cuda_gpu_trace` CSV.

The bridge uploads a prove's trace and its AIR's fixed sections before the
prove takes the client, so a transfer that runs while the client's kernels
are on the device costs the leg nothing. What costs the leg is transfer time
with no kernel of that side running:

    on the critical path   union(uploads) minus union(that side's kernels)
    overlapped             union(uploads) intersect the same

It is an upper bound, not an exact attribution: a transfer that runs in a gap
the client would have idled through anyway (a host-side unpack, an executable
load) counts against it. Read it as "uploads are worth at most this much".
Two lines say how loose the bound is on a given capture. The critical path is
split at the leg — an upload before the side's first kernel had no kernel to
hide behind and belongs to no leg — and `idle_gaps` reports how long the
side's device had already been idle when each copy started. A copy that
starts into an idle device overlaps nothing because there was nothing to
overlap with, which is a different finding from a copy the runtime refused to
overlap, and only the second is worth chasing on the upload path.

Each side's uploads are also intersected with the *other* prover's kernels.
That number is the control: it is what the card does when the ordering
constraint does not apply, so a zero beside a non-zero says the hardware was
willing and the runtime was not.

A bridged run has two provers on one card, so the two are reported apart.
`nsys_trace.owner` tells their kernels apart; a transfer goes to the side
whose kernels share its stream, or, on a stream that carries only transfers,
to the side owning the kernels that follow its copies — see
`Capture.upload_owners`.

`SrcMemKd` says whether a transfer came out of pageable or pinned host
memory, so the same report shows which path the uploads take and at what
bandwidth.

Capture recipe in docs/bridge.md "Profiling". Two flags are not optional:
`--cuda-graph-trace=node`, or the kernels inside XLA's CUDA graphs are
invisible and every transfer looks unoverlapped, and `--sample=none
--cpuctxsw=none`, or nsys 2026.1.3 deadlocks in report generation on a run
this size.

Usage: h2d_overlap.py <cuda_gpu_trace.csv>... [--top N]"""
from __future__ import annotations

import argparse
import bisect
import collections
import csv
import pathlib
import statistics
import sys
import typing

# Under the `bench/h2d_overlap.py ...` recipe in docs/bridge.md "Profiling"
# Python puts this file's own directory on sys.path rather than the repo root,
# so the package import below cannot resolve on its own. Under bazel the module
# is imported as `bridge.bench.h2d_overlap` and __package__ is already set.
if not __package__:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from bridge.bench.nsys_trace import (  # noqa: E402
    BRIDGE,
    PIL2,
    UNITS_BYTES,
    Span,
    column,
    covered,
    merge,
    overlap,
    owner,
)


class Upload(typing.NamedTuple):
    """One host-to-device copy: when it ran, how big it was, whether it came
    out of pageable or pinned host memory, and on which stream."""

    span: Span
    n_bytes: int
    kind: str
    stream: str


def idle_gaps(uploads: list[Upload], kernels: list[Span]) -> list[int]:
    """Per copy, how long this side's device had been idle when the copy
    started: zero if one of the side's kernels was running, otherwise the
    time since the last one ended. `kernels` must be a merged cover, so the
    one span that could still be running at time t is the last that started
    at or before it. Copies before the side's first kernel have no "since"
    and are left out."""
    starts = [start for start, _ in kernels]
    gaps: list[int] = []
    for up in uploads:
        began = up.span[0]
        i = bisect.bisect_right(starts, began)
        if i == 0:
            continue
        end = kernels[i - 1][1]
        gaps.append(0 if began < end else began - end)
    return gaps


class Capture:
    """A `cuda_gpu_trace` CSV as kernels per side plus every upload."""

    def __init__(self) -> None:
        self.kernels: dict[str, list[Span]] = {BRIDGE: [], PIL2: []}
        self.uploads: list[Upload] = []
        # Which side's kernels each stream carries, for the streams that
        # carry any.
        self.kernel_streams: dict[str, collections.Counter] = collections.defaultdict(
            collections.Counter
        )

    def add(
        self, row: dict[str, str], start_ns: int, dur_ns: int, n_bytes: int
    ) -> None:
        span = (start_ns, start_ns + dur_ns)
        name = row["Name"]
        stream = row.get("Strm", "")
        if "memcpy" not in name and "memset" not in name:
            side = owner(name)
            self.kernels[side].append(span)
            self.kernel_streams[stream][side] += 1
        elif "Host-to-Device" in name:
            kind = row.get("SrcMemKd") or "unknown"
            self.uploads.append(Upload(span, n_bytes, kind, stream))

    def upload_owners(self) -> dict[str, str]:
        """Which side each upload stream belongs to. A stream that also
        carries kernels belongs to whoever wrote them — CUDA orders a copy
        against the kernels of its own stream, so the two are one side's
        work. A stream that carries only copies is a dedicated transfer
        stream: it goes to whoever owns the kernel that starts next after a
        copy ends, by majority so one interleaved copy cannot flip it."""
        starts = sorted(
            (span[0], side) for side, spans in self.kernels.items() for span in spans
        )
        when = [start for start, _ in starts]
        votes: dict[str, collections.Counter] = collections.defaultdict(
            collections.Counter
        )
        for up in self.uploads:
            if up.stream in self.kernel_streams:
                votes[up.stream] = self.kernel_streams[up.stream]
                continue
            i = bisect.bisect_left(when, up.span[1])
            if i < len(starts):
                votes[up.stream][starts[i][1]] += 1
        return {s: c.most_common(1)[0][0] for s, c in votes.items()}


def read(path: pathlib.Path) -> Capture:
    """Every kernel and every host-to-device copy in the capture. The size
    column goes through the same unit rule as the times — nsys picks the unit
    from the capture and this report publishes GB and GB/s, so a header this
    module cannot scale has to raise rather than print zeros."""
    cap = Capture()
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        start_col, start_scale = column(header, "Start")
        dur_col, dur_scale = column(header, "Duration")
        bytes_col, bytes_scale = column(header, "Bytes", UNITS_BYTES)
        for row in reader:
            start = round(float(row[start_col]) * start_scale)
            # A kernel row leaves the size column empty.
            n_bytes = round(float(row[bytes_col] or 0) * bytes_scale)
            cap.add(row, start, round(float(row[dur_col]) * dur_scale), n_bytes)
    return cap


def bandwidth(n_bytes: int, ns: int) -> str:
    # Bytes per nanosecond is already GB/s.
    return f"{n_bytes / ns:5.1f} GB/s" if ns else "    -     "


def report_leg(merged: list[Span], kernels: list[Span]) -> None:
    """Where the critical path fell relative to the leg. The share needs the
    same window top and bottom: an upload before the side's first kernel is
    time no kernel of that side could have hidden, and belongs to no leg, so
    counting it against the leg's length overstates the share."""
    first, last = kernels[0][0], kernels[-1][1]
    in_leg = overlap(merged, [(first, last)]) - overlap(merged, kernels)
    early = covered([(s, min(e, first)) for s, e in merged if s < first])
    late = covered([(max(s, last), e) for s, e in merged if e > last])
    print(
        f"          {in_leg / 1e9:5.2f} s of that fell inside the leg"
        f" ({in_leg / (last - first) * 100:.0f} % of it)"
    )
    print(
        f"          {early / 1e9:5.2f} s ran before the leg's first kernel and"
        f" {late / 1e9:5.2f} s after its last"
    )


def report_idle(uploads: list[Upload], kernels: list[Span]) -> None:
    """How long the side's own device had been idle when each copy started.
    An upload that overlaps nothing is not the same claim as an upload that
    had nothing to overlap with, and this line is what tells them apart."""
    gaps = [g / 1e6 for g in idle_gaps(uploads, kernels)]
    if len(gaps) < 2:
        return
    waited = [g for g in gaps if g > 1]
    print(
        f"          {len(waited):5d} of {len(gaps)} copies"
        f" ({len(waited) / len(gaps) * 100:.0f} %) started more than 1 ms"
        f" after the last kernel ended"
    )
    print(
        f"          idle before a copy: median {statistics.median(gaps):6.2f} ms,"
        f" p90 {statistics.quantiles(gaps, n=10)[8]:6.2f} ms,"
        f" max {max(gaps):7.2f} ms"
    )


def report_side(
    side: str,
    kernels: list[Span],
    uploads: list[Upload],
    other_kernels: list[Span],
    top: int,
) -> None:
    if not kernels and not uploads:
        return
    busy = covered(kernels)
    leg = kernels[-1][1] - kernels[0][0] if kernels else 0
    print(
        f"   {side:6s} leg {leg / 1e9:5.2f} s, kernels busy {busy / 1e9:5.2f} s"
        f" over {len(kernels)} spans"
    )
    if not uploads:
        return
    merged = merge([u.span for u in uploads])
    spent = covered(merged)
    hidden = overlap(merged, kernels)
    n_bytes = sum(u.n_bytes for u in uploads)
    print(
        f"          uploads {len(uploads):5d} copies, {n_bytes / 1e9:6.2f} GB"
        f" in {spent / 1e9:5.2f} s at {bandwidth(n_bytes, spent)}"
    )
    print(
        f"          {(spent - hidden) / 1e9:5.2f} s on the critical path;"
        f" {hidden / 1e9:5.2f} s overlapped by its own kernels,"
        f" {overlap(merged, other_kernels) / 1e9:5.2f} s by the other prover's"
    )
    if kernels:
        report_leg(merged, kernels)
        report_idle(uploads, kernels)
    by_kind: dict[str, list[Upload]] = collections.defaultdict(list)
    for u in uploads:
        by_kind[u.kind].append(u)
    for kind, group in sorted(by_kind.items(), key=lambda kv: -len(kv[1])):
        ns = covered(merge([u.span for u in group]))
        total = sum(u.n_bytes for u in group)
        print(
            f"          from {kind:9s} {len(group):5d} copies,"
            f" {total / 1e9:6.2f} GB in {ns / 1e9:5.2f} s at {bandwidth(total, ns)}"
        )
    biggest = sorted(uploads, key=lambda u: -u.n_bytes)[:top]
    print(
        "          largest: "
        + ", ".join(
            f"{u.n_bytes / 1e9:.2f} GB in {(u.span[1] - u.span[0]) / 1e6:.0f} ms"
            f" ({u.kind.lower()})"
            for u in biggest
        )
    )


def report(path: pathlib.Path, top: int) -> None:
    cap = read(path)
    if not cap.uploads and not any(cap.kernels.values()):
        print(f"## {path}  no kernels and no transfers — was -t cuda on?")
        return
    owners = cap.upload_owners()
    every = [s for spans in cap.kernels.values() for s in spans] + [
        u.span for u in cap.uploads
    ]
    span = max(e for _, e in every) - min(s for s, _ in every)
    print(f"## {path}  {span / 1e9:.2f} s of timeline")
    merged_kernels = {side: merge(spans) for side, spans in cap.kernels.items()}
    for side, other in ((BRIDGE, PIL2), (PIL2, BRIDGE)):
        uploads = [u for u in cap.uploads if owners.get(u.stream) == side]
        report_side(side, merged_kernels[side], uploads, merged_kernels[other], top)
    # A stream resolves unless no kernel starts after any of its copies —
    # a capture cut short, or one with no kernels at all. Those bytes
    # belong to neither side above, so say so rather than dropping them
    # out of the totals.
    orphans = [u for u in cap.uploads if u.stream not in owners]
    if orphans:
        streams = sorted({u.stream for u in orphans})
        print(
            f"   unattributed {len(orphans)} copies,"
            f" {sum(u.n_bytes for u in orphans) / 1e9:.2f} GB on"
            f" stream(s) {', '.join(streams)}: no kernel runs after them"
        )


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv", nargs="+", type=pathlib.Path)
    ap.add_argument("--top", type=int, default=4, help="largest uploads listed")
    args = ap.parse_args(argv[1:])
    for path in args.csv:
        report(path, args.top)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
