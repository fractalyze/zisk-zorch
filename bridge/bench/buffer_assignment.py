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
  buffer_assignment.py <dump>/<AIR> --tree --manifest <artifacts>/<AIR>/manifest.json \
      [--log <run.log> --air <AIR>]
  buffer_assignment.py <dump-dir> [--manifest M] [--module <substring>] [--list]

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
# `    a.1{}:0-12` and `    loop_slice_fusion.1{3}:3-87` under
# `BufferLiveRange:`. The brace is part of the key, not decoration: a value
# that is one element of a returned tuple is written `name{index}` in both
# files, while a plain value is `name` in the assignment and `name{}` here.
LIVE_RANGE = re.compile(r"^\s+(\S+\{\S*\}):(\d+)-(\d+)$")
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

    @property
    def is_tuple(self) -> bool:
        """A tuple-shaped value: the index table XLA allocates for a program
        that returns more than one array, holding a pointer per element."""
        return self.shape.startswith("(")


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

    @property
    def is_tuple_table(self) -> bool:
        """The output tuple's index table -- one pointer per returned array,
        so eight bytes times the program's output count.

        XLA allocates it beside the outputs themselves and it is live-out like
        them, but the manifest does not declare it: it is an artefact of
        returning a tuple, not a program output. Anything comparing dumped
        outputs against declared ones has to set it aside, and `verify`
        checks its size rather than merely tolerating it."""
        return self.is_output and any(v.is_tuple for v in self.values)


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
        """Constants and thread-local scratch: everything that is neither an
        input, an output nor the temp arena.

        Deliberately not part of `added_bytes`: these are not allocated per
        execution -- a module's constants are placed when the executable is
        loaded and stay for its lifetime. Counting them as what a run adds
        overstates it by their total, which is how this came to be separate:
        including them left the reconciliation short by exactly this sum.
        Small per executable, but a client holding an AIR's whole set pays
        each one, and it is in XLA's printed total, so dropping it would
        break the check that reproduces that."""
        return self._sum(lambda a: not (a.is_parameter or a.is_output or a.is_temp))

    @property
    def added_bytes(self) -> int:
        """What the execution puts on the device beyond the inputs it was
        given: the temp arena plus the outputs.

        This is the quantity an allocator reading taken *during* the execution
        sees above what was live when it started. Inputs are already live (the
        registry's own buffers) and constants were placed at load."""
        return self.temp_bytes + self.output_bytes

    def live_range(self, value: Value) -> tuple[int, int] | None:
        """This value's live range, as an interval over `sequence`.

        The two files spell one value two ways: the assignment writes a plain
        value as `name` and a tuple element as `name{index}`, while the live
        ranges always carry a brace. Looking up the assignment's spelling
        unchanged finds nothing for every plain value, and -- worse, because
        it is silent -- nothing for the tuple elements that make up the
        largest arenas here."""
        key = value.name if "{" in value.name else f"{value.name}{{}}"
        return self.live_ranges.get(key)

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
            ranges[entry.group(1)] = (int(entry.group(2)), int(entry.group(3)))
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
    """One program's temp arena as the run saw it, beside what the compile
    says it is."""

    program: str
    in_use_after: int
    peak_during: int
    temp_measured: int
    temp_assigned: int

    @property
    def unexplained(self) -> int:
        """Measured minus assigned, in bytes.

        A few hundred bytes is the allocator's alignment. Positive beyond that
        is something else allocating inside the same window -- on a
        default-admission log, the next instance's uploads, which run
        concurrently with the prove. Negative means the readings are not the
        ones this executable ran between: most often `peak_during` is an older
        high-water because this program never raised it, in which case the run
        cannot measure this program's arena at all."""
        return self.temp_measured - self.temp_assigned


def reconcile(
    executable: Executable, in_use_after: int, peak_during: int
) -> Reconciliation:
    """Check one program's temp arena against a run's allocator readings.

    While a program runs the client holds what it already held, plus the
    program's outputs, plus its temp arena; when the program ends the arena is
    freed and the outputs stay. So

        temp = peak_during - in_use_after

    with both readings from a `ZZ_MEM_STAGES=2` log -- the allocator's
    high-water while the program ran, and its `in_use` on the line written
    after it. Nothing else is needed: no reading from before the program, and
    no term for what the driver released, which is why this is the form to
    quote. An earlier one measured from the reading *before* the program and
    needed a released-buffer term to close; that term is a free parameter an
    analyst can tune until the residual vanishes, and it hid a real
    discrepancy once.

    Two conditions. The peak must have risen during this program -- otherwise
    `peak_during` belongs to an earlier prove and the difference means
    nothing; `mem_stages.py`'s "peak rose across" names the programs where it
    did. And nothing else may allocate in the window, which is `ZZ_PENDING=1`,
    where no next instance is admitted beside a running prove.

    The companion identity, for when a release is suspected rather than
    measured, is `in_use_after - in_use_before = outputs - freed`: it isolates
    what the program left behind, and solving it for `freed` is how a section
    dropped at its last reader gets pinned to a size."""
    return Reconciliation(
        program=executable.module,
        in_use_after=in_use_after,
        peak_during=peak_during,
        temp_measured=peak_during - in_use_after,
        temp_assigned=executable.temp_bytes,
    )


def per_program(root: pathlib.Path) -> dict[str, Executable]:
    """A `dump/<AIR>/<program>/` tree, keyed by the program each directory is
    named for.

    One program per directory is how the dump is taken (XLA numbers modules
    per process, so a directory per compile keeps the ids from colliding), and
    it makes the program name a fact about the filesystem rather than
    something to infer. `verify` is what checks the name is the right one."""
    out = {}
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        ids = modules(child)
        if not ids:
            continue
        if len(ids) > 1:
            raise ValueError(
                f"{child} holds {len(ids)} modules; a per-program directory "
                "must hold one, or the name it carries is not the program's"
            )
        out[child.name] = parse_executable(child, ids[0])
    return out


def verify(executables: dict[str, Executable], manifest: dict) -> list[str]:
    """Every program whose dumped allocations do not match what the AIR
    declares for the name its directory carries.

    The failure this catches: a dump taken with the wrong `--only` argument,
    or a directory renamed by hand, puts one program's arena under another's
    name -- and since the arena is the quantity under study, nothing later
    would reveal it. Parameters and outputs are declared in the manifest, so
    they are checkable; the temp arena is not, which is the point of measuring
    it."""
    want = declared(manifest)
    wrong = []
    for name, e in executables.items():
        if name not in want:
            wrong.append(f"{name}: the AIR declares no such program")
            continue
        dec_in, dec_out = want[name]
        params = tuple(sorted(a.size for a in e.allocations if a.is_parameter))
        outs = tuple(
            sorted(
                a.size for a in e.allocations if a.is_output and not a.is_tuple_table
            )
        )
        tables = [a for a in e.allocations if a.is_tuple_table]
        if params != dec_in:
            wrong.append(f"{name}: inputs {params} but the manifest declares {dec_in}")
        elif outs != dec_out:
            wrong.append(f"{name}: outputs {outs} but the manifest declares {dec_out}")
        elif len(tables) > 1:
            wrong.append(
                f"{name}: {len(tables)} output tuple tables, expected at most one"
            )
        elif tables and tables[0].size != 8 * len(dec_out):
            wrong.append(
                f"{name}: the output tuple table is {tables[0].size} B for "
                f"{len(dec_out)} outputs, expected {8 * len(dec_out)}"
            )
    return wrong


def run_readings(log: str, air: str) -> dict[str, tuple[int, int, bool]]:
    """Per program of one AIR's prove: the allocator's `in_use` on the line
    after it, the high-water then standing, and whether that high-water rose
    across this program.

    The last flag is what says whether the run can measure this program's
    arena at all: `peak` is a client-lifetime high-water, so for every prove
    after the binding one it is an older number and `peak - in_use` is not an
    arena. Read from a `ZZ_MEM_STAGES=2` log through `mem_stages`, which owns
    the rule that a stage block belongs to the instance line after it."""
    from bridge.bench import mem_stages

    out: dict[str, tuple[int, int, bool]] = {}
    for prove in mem_stages.proves(log):
        if prove.air != air:
            continue
        marks = [
            m for m in prove.marks if m.peak is not None and not m.ran.startswith("(")
        ]
        previous = None
        for mark in marks:
            rose = previous is not None and mark.peak > previous
            out[mark.ran] = (mark.in_use, mark.peak, rose)
            previous = mark.peak
    return out


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


def _tree_report(
    root: pathlib.Path,
    manifest: dict,
    log_path: pathlib.Path | None,
    air: str,
    top: int,
) -> int:
    executables = per_program(root)
    if not executables:
        print(f"no per-program dumps under {root}", file=sys.stderr)
        return 1
    wrong = verify(executables, manifest)
    for line in wrong:
        print(f"  !! {line}", file=sys.stderr)
    readings = {}
    if log_path is not None:
        readings = run_readings(log_path.read_text(errors="replace"), air)

    rows = sorted(executables.items(), key=lambda kv: -kv[1].temp_bytes)
    head = f"{'program':<16} {'temp':>9} {'outputs':>9} {'inputs':>9} {'values':>7}"
    if readings:
        head += f" {'run temp':>9} {'delta':>9}"
    print(head)
    for name, e in rows:
        line = (
            f"{name:<16} {e.temp_bytes / MIB:9,.0f} {e.output_bytes / MIB:9,.0f} "
            f"{e.parameter_bytes / MIB:9,.0f} {len(e.temp_values()):7}"
        )
        if readings:
            reading = readings.get(name)
            if reading is None:
                line += f" {'no line':>9} {'':>9}"
            elif not reading[2]:
                line += f" {'no rise':>9} {'':>9}"
            else:
                r = reconcile(e, reading[0], reading[1])
                line += f" {r.temp_measured / MIB:9,.0f} {r.unexplained:9,}"
        print(line)
    print(
        "\n(MiB. `temp` is the arena from the compile; `run temp` is "
        "peak - in_use after\nthe program, and `delta` their difference in "
        "BYTES -- a few hundred is alignment.\n`no rise` is a program the "
        "run's high-water did not rise across, where the run\ncannot "
        "measure an arena at all.)"
    )
    if wrong:
        print(f"\n{len(wrong)} program(s) do not match the manifest; see above.")

    for name, e in rows[: max(1, top)]:
        owners = e.temp_values()
        if not owners:
            continue
        print(f"\n{name}: the arena, {e.temp_bytes / MIB:,.0f} MiB, by region")
        span = len(e.sequence) or None
        seen: dict[int, int] = {}
        for value, _ in owners:
            seen[value.offset] = max(seen.get(value.offset, 0), value.size)
        for offset, size in sorted(seen.items(), key=lambda kv: -kv[1])[:top]:
            here = [v for v, _ in owners if v.offset == offset]
            biggest = max(here, key=lambda v: v.size)
            live = e.live_range(biggest)
            where = f"{live[0]}-{live[1]}" if live else "?"
            if live and span:
                where += f"/{span}"
            print(
                f"  @{offset:<12,} {size / MIB:7,.0f} MiB  x{len(here):<3} "
                f"live {where:<10} {biggest.name} {biggest.shape}"
            )
    return 1 if wrong else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dump", type=pathlib.Path, help="--xla_dump_to directory")
    ap.add_argument(
        "--manifest",
        type=pathlib.Path,
        help="<artifacts>/<AIR>/manifest.json -- what gives a module its "
        "program name back, and what a per-program tree is verified against",
    )
    ap.add_argument(
        "--tree",
        action="store_true",
        help="`dump` is a dump/<AIR>/ holding one directory per program",
    )
    ap.add_argument(
        "--log",
        type=pathlib.Path,
        help="a ZZ_MEM_STAGES=2 run log to reconcile against",
    )
    ap.add_argument("--air", help="the AIR whose prove to read from --log")
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
    manifest = json.loads(args.manifest.read_text()) if args.manifest else None
    if args.tree:
        if manifest is None:
            print(
                "--tree needs --manifest to verify the directory names", file=sys.stderr
            )
            return 2
        if args.log is not None and not args.air:
            print("--log needs --air to say whose prove to read", file=sys.stderr)
            return 2
        return _tree_report(args.dump, manifest, args.log, args.air or "", args.top)
    _report(args.dump, manifest, args.module, args.top)
    return 0


if __name__ == "__main__":
    sys.exit(main())
