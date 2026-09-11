#!/usr/bin/env python3
"""Which device buffers a bridged prove holds at each stage of its schedule,
read out of a `ZZ_MEM_STAGES=1` run.

This is the reader behind docs/bridge.md "Where a prove's device memory goes";
`mem_budget.py` is a different tool on a different question (what a whole run
asked the card for, and which allocation a dead run died on) and cannot
produce these tables.

The four things it does that are easy to get wrong by hand:

**An inventory from a run that did not finish is missing its tail, and looks
complete.** Every stage the prove reached prints a full-looking block; the
stages after the abort simply are not there, and nothing in a block says so.
So the run's exit status is read first and printed beside every table, and
`--strict` refuses a log that did not finish rather than let a truncated
inventory be quoted. The same hazard in its blunter form -- a peak truncated
at an abort -- is why `mem_budget.py` says what it says about `MaxInUse`.

**The registry and the allocator are two different quantities, and their
difference is a term rather than an error.** `live` is what the bridge's own
buffers add up to, exactly, from the manifest; `in_use` is what the client's
allocator says it is holding at the same instant. The allocator is the larger
of the two and what it holds beyond the registry is what XLA allocated inside
an execution -- a fusion's scratch, an extend's output beside its input. The
report names that difference rather than reconciling it away.

**The peak is not at a boundary.** A snapshot taken between two stages cannot
see the high-water inside one, so the peak stage is not the boundary with the
largest `live`. It is read from `peak`, which the allocator keeps as a
monotonic high-water: the stage during which it last rose is the stage the
peak is in, whatever the boundaries around it look like.

**Stage blocks from two proves interleave.** The lines carry no instance id --
they are written inside `driver::prove`, which does not know one -- so a block
is attributed to the next instance line that follows it. That is sound only
while proves are serial, which is `ZZ_CLIENTS=1`. Two clients interleave their
blocks and the attribution would be silently wrong, so a second `stage1`
arriving before an instance line is an error, not a warning.

Usage:
  mem_stages.py <run.log> [--air Main_n22] [--manifest <artifacts>/<air>] [--strict]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

# Python puts this file's own directory on sys.path rather than the repo root,
# so the package import below cannot resolve on its own. Under bazel the module
# is imported as `bridge.bench.mem_stages` and __package__ is already set.
if not __package__:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from bridge.bench import run_log  # noqa: E402

MIB = 1 << 20

# A prove whose boundaries all report the same high-water did not set it: the
# allocator's peak is the client's, over its whole life, so only the prove that
# raised it can have a stage named from it.
NO_PEAK = "not reached in this prove -- the client high-water was set earlier"

HEAD = re.compile(
    r"\[zz \+\s*[\d.]+\] mem stage (\S+): in_use (-|\d+), peak (-|\d+),"
    r" pool (-|\d+), live (\d+) in (\d+) buffers(, INVENTORY INCOMPLETE)?"
)
BUF = re.compile(r"\[zz \+\s*[\d.]+\] mem stage (\S+) buf (\S+) (\d+) (\d+)")
RUN = re.compile(r"\[zz \+\s*[\d.]+\]\s+run (\S+): enqueue")
PROG = re.compile(
    r"\[zz \+\s*[\d.]+\] mem prog (\S+): in_use (-|\d+), peak (-|\d+), live (\d+)"
)


def _opt(text: str) -> int | None:
    """A statistic the allocator did not keep, as None rather than 0."""
    return None if text == "-" else int(text)


class Stage:
    """One stage boundary of one prove: the totals and the live set."""

    def __init__(self, name, in_use, peak, pool, live, count, incomplete):
        self.name = name
        self.in_use = in_use
        self.peak = peak
        self.pool = pool
        self.live = live
        self.count = count
        self.incomplete = incomplete
        # (origin, count, bytes), largest first as the bridge wrote them.
        self.rows: list[tuple[str, int, int]] = []
        # The programs that ran while this stage was open, in order. Only at
        # `ZZ_MEM_STAGES=2`, which is what puts a mark after each program.
        self.programs: list[str] = []

    @property
    def unnamed(self) -> int | None:
        """What the allocator holds beyond the bridge's own buffers: XLA's
        allocations inside an execution, which the registry cannot see."""
        return None if self.in_use is None else self.in_use - self.live


class Mark:
    """One reading of the allocator, and what ran just before it: a program at
    `ZZ_MEM_STAGES=2`, or the tail of the stage that just closed."""

    def __init__(self, ran: str, in_use, peak, live):
        self.ran = ran
        self.in_use = in_use
        self.peak = peak
        self.live = live


class Prove:
    """One instance's prove: its stages in order, the programs it ran, and --
    at `ZZ_MEM_STAGES=2` -- a reading of the allocator after each of them."""

    def __init__(self, stages: list[Stage], programs: list[str], marks: list[Mark]):
        self.stages = stages
        self.marks = marks
        # At `ZZ_MEM_STAGES=2` the marks carry the program order themselves,
        # and they are the list to read: a mark is written where the reading
        # was taken, so the order here and the order the stages were credited
        # from cannot disagree. The `run` lines are the fallback for level 1,
        # where there are no marks -- and they only exist at `ZZ_LOG=2`.
        from_marks = [m.ran for m in marks if not m.ran.startswith("(")]
        self.programs = from_marks or programs
        self.air = "?"
        self.index = -1

    @property
    def peak_program(self) -> tuple[str, int] | None:
        """The program the allocator's peak last rose across, and by how many
        bytes, or `None` when no program can be named.

        This is the one the stage table cannot give: most of a wide AIR's
        high-water is inside a stage rather than at either end of it, so the
        stage that holds the peak names a span of a dozen programs while the
        rise itself belongs to one of them.

        `None` covers two cases, and neither is a program. Without
        `ZZ_MEM_STAGES=2` there are only boundaries. With it, the last rise
        can still fall between a stage's last program and the next boundary --
        a download, the transcript, the query draw -- and reporting the
        boundary's own label there would print `(end of stage1)` under "peak
        rose across" as if it were a program. `peak_stage` is what names that
        case."""
        rose = None
        marks = [m for m in self.marks if m.peak is not None]
        for before, after in zip(marks, marks[1:]):
            if after.peak > before.peak:
                rose = (after.ran, after.peak - before.peak)
        if rose is None or rose[0].startswith("("):
            return None
        return rose

    @property
    def peak_stage(self) -> str | None:
        """The stage the high-water is in: the one during which the
        allocator's monotonic peak last rose. `None` when the allocator keeps
        no peak, in which case no stage can be named from this log.

        A boundary snapshot reports the peak as it stood when the stage before
        it closed, so a rise between two boundaries belongs to the earlier of
        the two. The last stage has a boundary after it for exactly this
        reason -- `memlog::Stage` reports once more as it is dropped."""
        rose = None
        marks = [(s.name, s.peak) for s in self.stages if s.peak is not None]
        for (name, peak), (_, later) in zip(marks, marks[1:]):
            if later > peak:
                rose = name
        return rose

    def stage(self, name: str) -> Stage | None:
        return next((s for s in self.stages if s.name == name), None)


def proves(log: str) -> list[Prove]:
    """Every prove in the log, each with its stage blocks and program order.

    A block belongs to the next instance line after it; see the module
    docstring on why that is only sound for a serial log, and what is raised
    when it is not."""
    out: list[Prove] = []
    stages: list[Stage] = []
    programs: list[str] = []
    marks: list[Mark] = []
    by_name: dict[str, Stage] = {}
    instances = iter(run_log.instances(log))
    for line in log.splitlines():
        head = HEAD.search(line)
        if head:
            name, in_use, peak, pool, live, count, incomplete = head.groups()
            if name in by_name:
                raise ValueError(
                    f"stage {name!r} arrived twice in one prove: the log"
                    " interleaves two clients' proves and no block in it can"
                    " be attributed to an instance. Re-run with ZZ_CLIENTS=1."
                )
            stage = Stage(
                name,
                _opt(in_use),
                _opt(peak),
                _opt(pool),
                int(live),
                int(count),
                bool(incomplete),
            )
            stages.append(stage)
            by_name[name] = stage
            # A boundary is a reading too, and what ran before it is whatever
            # the closing stage did after its last program.
            ran = f"(end of {stages[-2].name})" if len(stages) > 1 else "(prove start)"
            marks.append(Mark(ran, stage.in_use, stage.peak, stage.live))
            continue
        buf = BUF.search(line)
        if buf:
            name, origin, count, size = buf.groups()
            by_name[name].rows.append((origin, int(count), int(size)))
            continue
        prog = PROG.search(line)
        if prog:
            name, in_use, peak, live = prog.groups()
            marks.append(Mark(name, _opt(in_use), _opt(peak), int(live)))
            if stages:
                stages[-1].programs.append(name)
            continue
        run = RUN.search(line)
        if run:
            programs.append(run.group(1))
            continue
        if run_log.INSTANCE.search(line):
            # Advanced on every instance line, not only the ones with blocks
            # behind them: skipping one would shift every later prove's name
            # onto the wrong stages, which is the misattribution this reader
            # exists to avoid.
            instance = next(instances, None)
            if not stages:
                continue
            prove = Prove(stages, programs, marks)
            if instance is not None:
                prove.air, prove.index = instance.air, instance.index
            out.append(prove)
            stages, programs, marks, by_name = [], [], [], {}
    if stages:
        # Blocks with no instance line after them: the prove died mid-flight,
        # which is exactly the log this reader exists to print a
        # DID-NOT-FINISH table for. Dropping them reported "no ZZ_MEM_STAGES
        # blocks" for the one run whose inventory anyone needed to look at.
        # The AIR stays `?` -- the line that would have named it never came.
        out.append(Prove(stages, programs, marks))
    return out


def readers_of(manifest: dict, origin: str, programs: list[str]) -> list[str]:
    """The programs this run ran that bind `origin`'s buffer as an input, in
    the order they ran. The last of them is the buffer's last reader.

    Matched on the manifest's own input names. An origin whose name is not an
    input of any program comes back empty: `driver::set_fixed` renames the
    setup trees' digest layers as it binds them (`const_setup_layers_k` ->
    `const_layers_k`), so those rows are resolved by hand in the write-up
    rather than by a rename rule copied out of the driver into here, where it
    would drift."""
    name = origin.split("/", 1)[1]
    binds = {
        program
        for program, info in manifest["programs"].items()
        if any(spec["name"] == name for spec in info["inputs"])
    }
    # Run order, de-duplicated: a program that ran twice (the quotient's
    # chunks) reads the section at its last run.
    seen, order = set(), []
    for program in programs:
        if program in binds and program not in seen:
            seen.add(program)
            order.append(program)
    return order


ELEM = {"goldilocks": 8, "uint64": 8, "goldilocksx3": 24, "uint32": 4, "int32": 4}


def declared_sizes(manifest: dict) -> dict[str, set[int]]:
    """name -> the byte sizes this AIR declares for it, over every program's
    inputs and outputs. Sections are sized dtype x dims, so one AIR's `trace`
    and another's share a name at different sizes."""
    out: dict[str, set[int]] = {}
    for info in manifest["programs"].values():
        for spec in list(info["inputs"]) + list(info["outputs"]):
            elems = 1
            for dim in spec["dims"]:
                elems *= dim
            out.setdefault(spec["name"], set()).add(elems * ELEM[spec["dtype"]])
    return out


def own_bytes(origin: str, count: int, size: int, manifest: dict) -> int:
    """How much of one aggregated row belongs to the prove being reported.

    The registry is per client, not per prove, so a row can hold the next
    instance's upload as well as this one's: under the default admission
    (`ZZ_PENDING=2`) the next instance's `trace` is on the device while this
    prove runs, and on this workload that is up to 1,248 MiB. Two rules
    separate them, both from the manifest:

    - a copy whose size this AIR never declares for that name is another
      instance's outright -- that is how the co-resident `trace` is told from
      ours, since the eleven hello-world AIRs carry eleven different trace
      sizes;
    - otherwise the prove binds one buffer per name, except the quotient's row
      windows, of which the manifest says `quotient_chunks` are uploaded and
      all are this prove's.

    Charging a co-resident row to this prove is the same invented-finding
    failure `readers_of` guards the renamed layers against, and it lands on
    the largest buffer in the run rather than the smallest."""
    name = origin.split("/", 1)[1]
    per = size // count
    if per not in declared_sizes(manifest).get(name, set()):
        return 0
    mine = min(count, len(manifest["quotient_chunks"]) if name == "rows" else 1)
    return per * mine


def held_past_last_reader(
    prove: Prove, stage: Stage, manifest: dict
) -> list[tuple[str, int, str]]:
    """The rows alive at `stage` whose last reader already ran, as
    `(origin, bytes, last reader)`.

    This is attribution category (a) computed rather than argued: a section
    still bound after the last program that reads it is holding device memory
    for nothing this prove will do. It needs `ZZ_MEM_STAGES=2`, because
    without a mark per program there is nothing to say which stage a reader
    ran in; at level 1 it returns nothing rather than guessing.

    A row whose last reader cannot be resolved is left out, not assumed dead:
    the setup trees' layers are bound under names the driver renames, and
    calling them unread would invent the largest finding on the page. Only the
    bytes `own_bytes` attributes to this prove are counted, for the same
    reason: a co-resident upload has not outlived its reader, it is waiting
    for one."""
    order = [s.name for s in prove.stages]
    if not any(s.programs for s in prove.stages):
        return []
    ran_in = {
        program: order.index(s.name) for s in prove.stages for program in s.programs
    }
    here = order.index(stage.name)
    out = []
    for origin, count, size in stage.rows:
        mine = own_bytes(origin, count, size, manifest)
        if not mine:
            continue
        readers = readers_of(manifest, origin, prove.programs)
        if not readers:
            continue
        last = readers[-1]
        if last in ran_in and ran_in[last] < here:
            out.append((origin, mine, last))
    return out


def report(prove: Prove, manifest: dict | None, finished: bool) -> str:
    """The per-stage table for one prove, as docs/bridge.md carries it."""
    status = (
        "finished"
        if finished
        else "DID NOT FINISH -- stages after the abort are missing"
    )
    lines = [
        f"{prove.air} instance {prove.index}  [run {status}]",
        "",
        f"{'stage':<10} {'live':>9} {'in_use':>9} {'unnamed':>9} {'pool':>9}"
        f"  {'bufs':>5}  peak",
    ]
    for stage in prove.stages:
        mib = lambda b: "-" if b is None else f"{b / MIB:,.0f}"  # noqa: E731
        lines.append(
            f"{stage.name:<10} {mib(stage.live):>9} {mib(stage.in_use):>9}"
            f" {mib(stage.unnamed):>9} {mib(stage.pool):>9}  {stage.count:>5}"
            f"  {mib(stage.peak)}"
            + ("  INVENTORY INCOMPLETE" if stage.incomplete else "")
        )
    peak = prove.peak_stage
    lines += [
        "",
        f"peak stage: {peak or NO_PEAK}",
    ]
    rose = prove.peak_program
    if rose:
        lines.append(f"peak rose across: {rose[0]}, by {rose[1] / MIB:,.0f} MiB")
    # Two different stages answer two different questions, and quoting one
    # figure from the other is how a lifetime total of 1,488 MiB reads as 0.
    # The peak stage is where the high-water is; the largest live set is where
    # the most sections are bound at once, and that is where a section that
    # has outlived its reader shows up.
    largest = max(prove.stages, key=lambda s: s.live, default=None)
    blocks = []
    if peak:
        blocks.append(("peak", prove.stage(peak)))
    if largest is not None and largest not in [s for _, s in blocks]:
        blocks.append(("largest live set", largest))
    for label, stage in blocks:
        lines += ["", f"live set entering {stage.name} ({label}, MiB):", ""]
        for origin, count, size in stage.rows:
            last = readers_of(manifest, origin, prove.programs) if manifest else []
            lines.append(
                f"  {origin:<34} {count:>3}  {size / MIB:>9,.0f}"
                f"  last reader: {last[-1] if last else '?'}"
            )
        if manifest:
            past = held_past_last_reader(prove, stage, manifest)
            total = sum(size for _, size, _ in past)
            lines += [
                "",
                f"of which held past their last reader: {total / MIB:,.0f} MiB"
                " (this prove's own buffers only)",
                "",
            ]
            for origin, size, last in past:
                lines.append(
                    f"  {origin:<34}      {size / MIB:>9,.0f}  last read by {last}"
                )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("log", type=pathlib.Path)
    ap.add_argument("--air", help="only proves of this AIR")
    ap.add_argument(
        "--manifest", type=pathlib.Path, help="the AIR's exported artifact directory"
    )
    ap.add_argument(
        "--strict",
        action="store_true",
        help="refuse a log whose run did not finish, rather than report a"
        " truncated inventory",
    )
    args = ap.parse_args(argv)

    log = args.log.read_text(errors="replace")
    finished = run_log.completed(log)
    if args.strict and not finished:
        print(
            f"{args.log}: the run did not finish; its inventory stops at the abort",
            file=sys.stderr,
        )
        return 1
    manifest = None
    if args.manifest:
        manifest = json.loads((args.manifest / "manifest.json").read_text())

    found = [p for p in proves(log) if not args.air or p.air == args.air]
    if not found:
        where = f" for {args.air}" if args.air else ""
        print(f"{args.log}: no ZZ_MEM_STAGES blocks{where}", file=sys.stderr)
        return 1
    print("\n\n".join(report(p, manifest, finished) for p in found))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
