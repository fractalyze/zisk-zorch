#!/usr/bin/env python3
"""What one exported program allocates *inside* an execution, from XLA's own
buffer assignment.

`mem_stages.py` reports the buffers a client holds at each stage boundary and,
at `ZZ_MEM_STAGES=2`, the allocator's totals after every program. What neither
it nor the allocator can name is the part of the peak that exists only while a
program runs -- a fusion's scratch, an extend's output beside its input. That
term is the difference between the registry and `in_use` in those reports, and
it belongs to no section (docs/bridge.md, "What the excess actually is",
category (c)).

XLA already itemises it. Compiling with `--xla_dump_to` leaves, per executable,
a `-buffer-assignment.txt` (every allocation, its size and the HLO values that
own it), a `-live-range.txt` (each value's live range as an interval over the
printed instruction sequence) and a `-memory-usage-report.txt` (XLA's own
totals). This reader turns those three into the table, and reconciles it
against a run.

**The dump is written while compiling, so a warm cache produces none.** The
executables are cached (`ZZ_COMPILE_CACHE`, `$ZZ_ARTIFACTS/.pjrt-cache`), and a
run that loads them never reaches the code that writes these files -- the
obvious mechanism, a prove with the dump flags set, silently dumps nothing.
Point `ZZ_COMPILE_CACHE` at an empty scratch directory and compile with
`zz_prove --warm <artifacts> <AIR>...`: no prove, no proofman, and the client
grows on demand rather than preallocating an arena, so it coexists with another
tenant on the card. Never point that scratch at the warmed cache a leg
measurement depends on.

**What a program allocates is a property of the executable; what it costs the
client's peak is not.** The allocation totals here come from the compile and do
not move with the admission policy, the residency policy or the prove order.
The *rise* in a run's high-water does move with all three: it is measured
against whatever the previous high-water was and on top of whatever else is
live, so one executable's rise differs between two arms by more than a
gigabyte while its allocations are the same bytes (docs/bridge.md). Quote this
table for the executable and the run for the peak; `reconcile()` below relates
them and states what is left over rather than reconciling it away.

**A re-derived total with no check is a guess with a table's authority** --
`pil2_layout.py`'s rule, and it applies to a parse as much as to arithmetic.
Every allocation of every memory space must sum to the `Total bytes` XLA prints
in its own report, and `parse_executable` refuses a dump where it does not.

Usage:
  buffer_assignment.py <dump-dir> --manifest <artifacts>/<AIR>/manifest.json
  buffer_assignment.py <dump-dir> [--module <substring>] [--top N] [--list]

`reconcile()` is the library half: it takes one program's allocator readings
from a `ZZ_MEM_STAGES=2` log and says what the executable does not account
for.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import re
import sys

# Python puts this file's own directory on sys.path rather than the repo root,
# so the package import below cannot resolve on its own. Under bazel the module
# is imported as `bridge.bench.buffer_assignment` and __package__ is already
# set.
if not __package__:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

MIB = 1 << 20

# `allocation 11: size 5242880, preallocated-temp:` -- the trailing text is the
# kind, which carries commas of its own ("parameter 0, shape |f32[8]| at
# ShapeIndex {}"), so it is taken whole rather than split.
ALLOCATION = re.compile(r"^allocation (\d+): size (\d+), (.*):$")
# ` value: <36 gemm_fusion_dot @0> (size=4194304,offset=0): f32[4,512,512]{2,1,0}`
VALUE = re.compile(
    r"^\s+value: <(\d+) (\S+) @(\d+)> \(size=(\d+),offset=(\d+)\): (.*)$"
)
# `    a.1{}:0-12` under `BufferLiveRange:`
LIVE_RANGE = re.compile(r"^\s+(\S+?)\{(\S*)\}:(\d+)-(\d+)$")
# `Total bytes: 6291496 (6.00MiB)`
TOTAL_BYTES = re.compile(r"^Total bytes: (\d+)")

# The dump names a file
# `module_<id>.<module>.<platform>_gpu_after_optimizations-<what>.txt`.
# The id is the key here, not the name: every exported program's MLIR module is
# called `jit_fn`, so all 34 of an AIR's executables dump under one name and
# only the id tells them apart (see `identify` for how a program gets its name
# back). The platform tag carries a dot of its own (`sm_12.0a`), so matching it
# explicitly is what keeps that tag out of the name.
DUMP_FILE = re.compile(
    r"^module_(\d+)\.(.+?)(?:\.sm_[0-9.]+[a-z]*)?_gpu_after_optimizations-(.+)\.txt$"
)


@dataclasses.dataclass(frozen=True)
class Value:
    """One HLO value placed in an allocation. Several share one allocation
    when their live ranges do not overlap, which is why an allocation's size
    is not the sum of its values'."""

    number: int
    name: str
    index: int
    size: int
    offset: int
    shape: str


@dataclasses.dataclass(frozen=True)
class Allocation:
    """One buffer XLA reserves for an execution, and what lives in it."""

    index: int
    size: int
    kind: str
    values: tuple[Value, ...]

    @property
    def is_parameter(self) -> bool:
        """An input the caller already holds. For a bridged prove these are
        the registry's own buffers, so they are live before the execution and
        are not part of what it adds."""
        return self.kind.startswith("parameter")

    @property
    def is_output(self) -> bool:
        """A buffer the execution returns. Allocated during it and still alive
        after, so it is part of the rise and not of the transient."""
        return "maybe-live-out" in self.kind

    @property
    def is_temp(self) -> bool:
        """Scratch, freed when the execution ends: the transient this reader
        exists to name."""
        return "preallocated-temp" in self.kind


@dataclasses.dataclass(frozen=True)
class Executable:
    """One compiled program: its allocations, its values' live ranges, and the
    instruction sequence those ranges index."""

    module: str
    module_id: int
    allocations: tuple[Allocation, ...]
    live_ranges: dict[str, tuple[int, int]]
    sequence: tuple[str, ...]
    reported_total: int | None

    def _sum(self, pick) -> int:
        return sum(a.size for a in self.allocations if pick(a))

    @property
    def temp_bytes(self) -> int:
        return self._sum(lambda a: a.is_temp)

    @property
    def output_bytes(self) -> int:
        return self._sum(lambda a: a.is_output)

    @property
    def parameter_bytes(self) -> int:
        return self._sum(lambda a: a.is_parameter)

    @property
    def other_bytes(self) -> int:
        """Thread-local scratch and constants: everything that is neither an
        input, an output nor the temp arena. Small, but it is in XLA's total
        and dropping it would break the check that reproduces it."""
        return self._sum(lambda a: not (a.is_parameter or a.is_output or a.is_temp))

    @property
    def added_bytes(self) -> int:
        """What the execution puts on the device beyond the inputs it was
        given: the temp arena plus the outputs plus the rest. This is the
        quantity an allocator reading taken *during* the execution sees above
        what was live when it started."""
        return self.temp_bytes + self.output_bytes + self.other_bytes

    def live_range(self, value: Value) -> tuple[int, int] | None:
        return self.live_ranges.get(value.name)

    def temp_values(self) -> list[tuple[Value, Allocation]]:
        """Every value in a temp allocation, largest first. The owners of the
        transient."""
        pairs = [(v, a) for a in self.allocations if a.is_temp for v in a.values]
        return sorted(pairs, key=lambda p: -p[0].size)


def _parse_allocations(text: str) -> list[Allocation]:
    allocations: list[Allocation] = []
    index = size = None
    kind = ""
    values: list[Value] = []

    def flush() -> None:
        if index is not None:
            allocations.append(Allocation(index, size, kind, tuple(values)))

    for line in text.splitlines():
        head = ALLOCATION.match(line)
        if head:
            flush()
            index, size, kind = int(head.group(1)), int(head.group(2)), head.group(3)
            values = []
            continue
        value = VALUE.match(line)
        if value and index is not None:
            values.append(
                Value(
                    number=int(value.group(1)),
                    name=value.group(2),
                    index=int(value.group(3)),
                    size=int(value.group(4)),
                    offset=int(value.group(5)),
                    shape=value.group(6),
                )
            )
    flush()
    return allocations


def _parse_live_ranges(text: str) -> tuple[dict[str, tuple[int, int]], tuple[str, ...]]:
    """`-live-range.txt` into name -> (start, end) plus the instruction
    sequence those indices refer to. A range is meaningless without the
    sequence: `3-4` is a position, not a duration."""
    ranges: dict[str, tuple[int, int]] = {}
    sequence: list[tuple[int, str]] = []
    for line in text.splitlines():
        entry = LIVE_RANGE.match(line)
        if entry:
            ranges[entry.group(1)] = (int(entry.group(3)), int(entry.group(4)))
            continue
        step = re.match(r"^\s+(\d+):(\S+)$", line)
        if step:
            sequence.append((int(step.group(1)), step.group(2)))
    return ranges, tuple(name for _, name in sorted(sequence))


def parse_executable(dump: pathlib.Path, module_id: int) -> Executable:
    """The three reports for one module id, as one object.

    Raises when the allocations do not sum to the total XLA printed for the
    same executable: a parse that silently drops an allocation kind would
    otherwise understate the transient and read as a measurement."""
    files = {}
    module = "?"
    for path in dump.iterdir():
        name = DUMP_FILE.match(path.name)
        if name and int(name.group(1)) == module_id:
            module = name.group(2)
            files[name.group(3)] = path
    if "buffer-assignment" not in files:
        raise FileNotFoundError(
            f"no buffer assignment for module {module_id} in {dump} -- "
            "a warm compile cache dumps nothing; compile into an empty "
            "ZZ_COMPILE_CACHE (see the module docstring)"
        )
    allocations = _parse_allocations(files["buffer-assignment"].read_text())
    ranges: dict[str, tuple[int, int]] = {}
    sequence: tuple[str, ...] = ()
    if "live-range" in files:
        ranges, sequence = _parse_live_ranges(files["live-range"].read_text())
    reported = None
    if "memory-usage-report" in files:
        totals = [
            int(m.group(1))
            for m in (
                TOTAL_BYTES.match(line)
                for line in files["memory-usage-report"].read_text().splitlines()
            )
            if m
        ]
        reported = sum(totals) if totals else None
    executable = Executable(
        module=module,
        module_id=module_id,
        allocations=tuple(allocations),
        live_ranges=ranges,
        sequence=sequence,
        reported_total=reported,
    )
    assigned = sum(a.size for a in allocations)
    if reported is not None and assigned != reported:
        raise ValueError(
            f"{module}: allocations sum to {assigned} B but XLA's own report "
            f"says {reported} B -- the parse is missing an allocation kind, "
            "and a table from it would understate the transient"
        )
    return executable


def modules(dump: pathlib.Path) -> list[int]:
    """Every module id the dump holds a buffer assignment for."""
    found = set()
    for path in dump.iterdir():
        name = DUMP_FILE.match(path.name)
        if name and name.group(3) == "buffer-assignment":
            found.add(int(name.group(1)))
    return sorted(found)


# Bytes per element of the dtypes a manifest declares. Most of an AIR's
# arrays are Goldilocks words, but the row windows and a handful of scalars
# are 32-bit, and sizing those at 8 bytes would double them -- enough to move
# a signature off the program it belongs to.
DTYPE_BYTES = {"uint64": 8, "int64": 8, "uint32": 4, "int32": 4}


def declared(manifest: dict) -> dict[str, tuple[tuple[int, ...], tuple[int, ...]]]:
    """Each program's declared input and output byte sizes, sorted.

    The manifest is the only thing that can give a dumped module its program
    name back, because the dump cannot: `jit_fn` is every program's module
    name. A program's shape declaration is its signature."""

    def sizes(specs) -> tuple[int, ...]:
        out = []
        for spec in specs:
            dtype = spec.get("dtype", "uint64")
            if dtype not in DTYPE_BYTES:
                raise KeyError(
                    f"unknown dtype {dtype!r} in the manifest -- add its width "
                    "to DTYPE_BYTES; sizing it by guess would move a signature "
                    "onto the wrong program"
                )
            n = DTYPE_BYTES[dtype]
            for dim in spec["dims"]:
                n *= dim
            out.append(n)
        return tuple(sorted(out))

    return {
        name: (sizes(info["inputs"]), sizes(info["outputs"]))
        for name, info in manifest["programs"].items()
    }


def identify(
    executables: dict[int, Executable], manifest: dict
) -> tuple[dict[int, str], dict[int, list[str]]]:
    """Module id -> program name, by matching allocation sizes against the
    manifest's declared shapes.

    Returns the confident matches and, separately, every id whose signature
    fits more than one program. An ambiguous id is reported rather than
    resolved: guessing between two programs would put one program's transient
    under the other's name, which is the whole quantity under study.

    The manifest must be the one for the AIR that was compiled. Signatures are
    unique *within* an AIR but coincide across them -- the small opening and
    grind programs declare the same shapes on every AIR -- so a dump holding
    two AIRs' executables has two candidates behind those signatures and this
    function cannot see it. Compile one AIR per dump directory."""
    want = declared(manifest)
    matched: dict[int, str] = {}
    ambiguous: dict[int, list[str]] = {}
    for module_id, executable in executables.items():
        params = tuple(sorted(a.size for a in executable.allocations if a.is_parameter))
        outs = tuple(sorted(a.size for a in executable.allocations if a.is_output))
        fits = [
            name
            for name, (dec_in, dec_out) in want.items()
            if dec_in == params and dec_out == outs
        ]
        if len(fits) == 1:
            matched[module_id] = fits[0]
        elif fits:
            ambiguous[module_id] = sorted(fits)
    return matched, ambiguous


@dataclasses.dataclass(frozen=True)
class Reconciliation:
    """What a run's allocator readings around one program leave unexplained
    once the executable's own allocations are subtracted."""

    program: str
    entry_in_use: int
    peak_during: int
    added_measured: int
    added_assigned: int

    @property
    def unexplained(self) -> int:
        """Measured minus assigned. Zero is the executable accounting for the
        whole rise; positive is something else allocating in the same window
        -- on a default-admission log, the next instance's uploads, which run
        concurrently with the prove. Negative means the program's peak was
        never reached while the reading was taken, so the window bounds it
        from below only."""
        return self.added_measured - self.added_assigned


def reconcile(
    executable: Executable, entry_in_use: int, peak_during: int
) -> Reconciliation:
    """Relate one program's compile-time allocations to a run's readings.

    `entry_in_use` is the allocator's `in_use` after the program before this
    one and `peak_during` its high-water while this one ran -- both from a
    `ZZ_MEM_STAGES=2` log. The identity is
    `peak_during = entry_in_use + added_bytes`, and it holds only where
    nothing else allocates in the window: `ZZ_PENDING=1`, where no next
    instance is admitted beside the running prove."""
    return Reconciliation(
        program=executable.module,
        entry_in_use=entry_in_use,
        peak_during=peak_during,
        added_measured=peak_during - entry_in_use,
        added_assigned=executable.added_bytes,
    )


def _load(dump: pathlib.Path) -> dict[int, Executable]:
    out = {}
    for module_id in modules(dump):
        try:
            out[module_id] = parse_executable(dump, module_id)
        except ValueError as err:
            print(f"  !! {err}", file=sys.stderr)
    return out


def _report(
    dump: pathlib.Path, manifest: dict | None, wanted: str | None, top: int
) -> None:
    executables = _load(dump)
    if not executables:
        print(f"no buffer assignments in {dump}", file=sys.stderr)
        return
    names, ambiguous = ({}, {}) if manifest is None else identify(executables, manifest)
    rows = [
        (names.get(i, f"module {i}"), e)
        for i, e in executables.items()
        if wanted is None or wanted in names.get(i, f"module {i}")
    ]
    rows.sort(key=lambda r: -r[1].added_bytes)
    print(
        f"{'program':<24} {'temp':>10} {'outputs':>10} {'other':>8} "
        f"{'added':>10} {'inputs':>10}"
    )
    for name, e in rows:
        print(
            f"{name:<24} {e.temp_bytes / MIB:10,.0f} "
            f"{e.output_bytes / MIB:10,.0f} {e.other_bytes / MIB:8,.1f} "
            f"{e.added_bytes / MIB:10,.0f} {e.parameter_bytes / MIB:10,.0f}"
        )
    print(
        "\n(MiB. `added` is temp + outputs + other: what the execution puts "
        "on the device\nbeyond the inputs it was handed.)"
    )
    if manifest is not None:
        unnamed = [i for i in executables if i not in names]
        if unnamed:
            print(f"\nunidentified module ids: {unnamed}")
        for module_id, fits in ambiguous.items():
            print(
                f"  module {module_id} fits {len(fits)} programs: " f"{', '.join(fits)}"
            )

    for name, e in rows[: max(1, top)]:
        owners = e.temp_values()
        if not owners:
            continue
        print(
            f"\n{name}: the temp arena, {e.temp_bytes / MIB:,.0f} MiB, "
            f"by owning value"
        )
        span = len(e.sequence) or None
        for value, _ in owners[:top]:
            live = e.live_range(value)
            where = f"{live[0]}-{live[1]}" if live else "?"
            if live and span:
                where += f" of {span}"
            print(
                f"  {value.name:<34} {value.size / MIB:8,.1f} MiB  "
                f"@{value.offset:<10} live {where:<12} {value.shape}"
            )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dump", type=pathlib.Path, help="--xla_dump_to directory")
    ap.add_argument(
        "--manifest",
        type=pathlib.Path,
        help="<artifacts>/<AIR>/manifest.json -- what gives a module its "
        "program name back; without it modules are listed by id",
    )
    ap.add_argument("--module", help="only programs whose name contains this")
    ap.add_argument("--top", type=int, default=12, help="rows of owners to show")
    ap.add_argument("--list", action="store_true", help="list module ids and exit")
    args = ap.parse_args(argv)
    if not args.dump.is_dir():
        print(f"{args.dump} is not a directory", file=sys.stderr)
        return 2
    if args.list:
        for module_id in modules(args.dump):
            print(module_id)
        return 0
    manifest = None
    if args.manifest:
        manifest = json.loads(args.manifest.read_text())
    _report(args.dump, manifest, args.module, args.top)
    return 0


if __name__ == "__main__":
    sys.exit(main())
