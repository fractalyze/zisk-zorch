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

A bridged run has two provers on one card, so the two are reported apart.
XLA writes a fusion's name with no argument list (`loop_add_fusion`,
`sponge_hash_1`); pil2's kernels are C++ signatures (`_add(Goldilocks::
Element *, ...)`), so a `(` in the name is what tells them apart. A transfer
goes to the side whose kernels share its stream, or, on a stream that
carries only transfers, to the side owning the kernels that follow its
copies — see `Capture.upload_owners`.

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
import re
import sys
import typing

Span = tuple[int, int]

BRIDGE, PIL2 = "bridge", "pil2"

# nsys writes the unit into the column header, and which one it picks
# depends on the capture's length.
UNITS_NS = {"ns": 1, "us": 1_000, "µs": 1_000, "ms": 1_000_000, "s": 1_000_000_000}


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


def overlap(a: list[Span], b: list[Span]) -> int:
    """Time covered by both merged covers. What each cover has to itself is
    then its own total minus this, so no second sweep is needed."""
    total = 0
    i = j = 0
    while i < len(a) and j < len(b):
        total += max(0, min(a[i][1], b[j][1]) - max(a[i][0], b[j][0]))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total


def owner(kernel_name: str) -> str:
    """Which prover emitted a kernel. XLA's names carry no argument list;
    pil2's are C++ signatures."""
    return PIL2 if "(" in kernel_name else BRIDGE


class Upload(typing.NamedTuple):
    """One host-to-device copy: when it ran, how big it was, whether it came
    out of pageable or pinned host memory, and on which stream."""

    span: Span
    n_bytes: int
    kind: str
    stream: str


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

    def add(self, row: dict[str, str], start_ns: int, dur_ns: int) -> None:
        span = (start_ns, start_ns + dur_ns)
        name = row["Name"]
        stream = row.get("Strm", "")
        if "memcpy" not in name and "memset" not in name:
            side = owner(name)
            self.kernels[side].append(span)
            self.kernel_streams[stream][side] += 1
        elif "Host-to-Device" in name:
            kind = row.get("SrcMemKd") or "unknown"
            self.uploads.append(Upload(span, size_bytes(row), kind, stream))

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


def size_bytes(row: dict[str, str]) -> int:
    """The row's `Bytes (MB)` (nsys means 10^6) as bytes. A kernel row leaves
    the column empty."""
    return round(float(row.get("Bytes (MB)", "") or 0) * 1e6)


def read(path: pathlib.Path) -> Capture:
    cap = Capture()
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        start_col, start_scale = column(reader.fieldnames or [], "Start")
        dur_col, dur_scale = column(reader.fieldnames or [], "Duration")
        for row in reader:
            start = round(float(row[start_col]) * start_scale)
            cap.add(row, start, round(float(row[dur_col]) * dur_scale))
    return cap


def bandwidth(n_bytes: int, ns: int) -> str:
    # Bytes per nanosecond is already GB/s.
    return f"{n_bytes / ns:5.1f} GB/s" if ns else "    -     "


def report_side(
    side: str, kernels: list[Span], uploads: list[Upload], top: int
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
    share = f", {(spent - hidden) / leg * 100:.0f} % of the leg" if leg else ""
    print(
        f"          {(spent - hidden) / 1e9:5.2f} s on the critical path{share};"
        f" {hidden / 1e9:5.2f} s overlapped"
    )
    # The first instance's upload has no kernel of its own to hide behind,
    # so it is exposed by construction; the rest could have overlapped and
    # did not.
    if kernels:
        first = kernels[0][0]
        early = covered([(s, min(e, first)) for s, e in merged if s < first])
        if early / 1e9 >= 0.005:
            print(
                f"          {early / 1e9:5.2f} s of that ran before the"
                f" leg's first kernel"
            )
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
    for side in (BRIDGE, PIL2):
        uploads = [u for u in cap.uploads if owners.get(u.stream) == side]
        report_side(side, merge(cap.kernels[side]), uploads, top)


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
