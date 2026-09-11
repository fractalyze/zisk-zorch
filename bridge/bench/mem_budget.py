#!/usr/bin/env python3
"""What a `cargo-zisk prove -vv` run asked the card for: the arena each bridge
client claimed, what pil2 was left and what it needs, and -- when the run died
-- which allocation it died on. This is where the walk table in
docs/bridge.md "Memory budget" comes from, so a floor can be re-derived from
logs already on disk instead of from a scratch script.

Four things this reads that are easy to get wrong by hand, and are why the
numbers here are read out of the run rather than computed beside it.

**The fraction is not the card.** `ZZ_MEMORY_FRACTION` is divided by the
client count and then applied by XLA to its own base, which on this card is
0.48 GiB below the 31.84 GiB the card has. A table whose share column is
`fraction x card` is high in every cell by that ratio, and the error rides
into everything derived from it. The run prints the product it actually
allocated; that line is the share.

**pil2's requirement is not the space left to it.** `Need X GB but only Y GB
available` is pil2's own check, and Y is free memory as pil2 finds it -- after
the bridge's clients have claimed their arenas and their module loads have
begun. Whatever the module loads have taken by then is already inside Y, so a
budget that adds a module-load term on top of "what pil2 needs left on the
card" counts it twice.

**pil2 refuses in two different sentences.** Below its requirement it prints
`Insufficient memory. Need X GB but only Y GB available`; earlier, when what
it can see is small enough that its own stream sizing asks for a card nobody
has, it prints `Not enough GPU memory to run the proof` and no figures at
all. Both are one verdict -- pil2 never started -- and a reader that knows
only the first calls the second a run that died while proving. The `Need` of
the first is a requirement only while the stream count beside it is the one
the run would have used: at two clients holding their floor pil2 sized 20
basic streams and asked for 162 GB, which is that sizing and not a floor.

**Whether a run finished is `run_log`'s rule, not this reader's.** The verify
line has two wordings across prover builds and an out-of-memory message
survives a rescue, so a cell scored on either alone is wrong in a direction
that raises the floor. Both live in `run_log` because `summarize.py` scores on
them too.

**The allocator is not the arm that was set.** `ZZ_ALLOCATOR` asks for a kind;
the plugin names the one it built on the same line it prints the arena on, so
the arm in the table below is read from the run rather than from what the
sweep meant to set -- a spelling the plugin does not know falls back to its
default without the sweep noticing. The kind also changes what the arena
*means*: under BFC it is a ceiling as well as a claim, and every allocation is
placed inside it, while under `cuda_async` it is the release threshold of the
device's memory pool, which is claimed up front but grows past it while the
card has room. A pass at a smaller arena is therefore not the same statement
in the two columns, and the table keeps them apart for that reason rather than
for tidiness.

**A failing run's `MaxInUse` is truncated at the abort**, so it is a lower
bound on what that arena had to hold, never the working set; and
`MaxAllocSize` is the largest single allocation, which is not the floor
either -- removing a 5.50 GiB one moved the floor 0.6 GiB (#191). Both are
reported because they name what is in the arena, not because either sizes it.

Usage: mem_budget.py <run.log>...
"""
from __future__ import annotations

import argparse
import collections
import pathlib
import re
import sys

# Python puts this file's own directory on sys.path rather than the repo root,
# so the package import below cannot resolve on its own. Under bazel the module
# is imported as `bridge.bench.mem_budget` and __package__ is already set.
if not __package__:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from bridge.bench import run_log  # noqa: E402

GIB = 1 << 30

ARENA = re.compile(
    r"XLA backend allocating (\d+) bytes on device \d+ for (\w+)Allocator"
)
PIL2_SEES = re.compile(r"Using minimum memory across \d+ GPUs: ([\d.]+) GB")
PIL2_NEEDS = re.compile(
    r"Insufficient memory\. Need ([\d.]+) GB but only ([\d.]+) GB available"
)
PIL2_STREAMS = re.compile(
    r"Using (\d+) streams per GPU for basic proofs and (\d+) streams"
)
PIL2_CONFIG = re.compile(r"Not enough GPU memory to run the proof")
CLIENT_OOM = re.compile(
    r"bridge: instance (\d+): PJRT error in \w+: Out of memory"
    r" while trying to allocate ([\d.]+)([KMG]iB)"
)
STAT = re.compile(r"^(MaxAllocSize|MaxInUse|Limit): +(.+)$", re.MULTILINE)

UNIT = {"KiB": 1 / (1 << 20), "MiB": 1 / (1 << 10), "GiB": 1.0}


class Run:
    """One prove's memory story, read out of its log."""

    def __init__(self, log: str):
        claimed = ARENA.findall(log)
        self.arenas = [int(b) / GIB for b, _ in claimed]
        self.allocators = [kind for _, kind in claimed]
        sees = PIL2_SEES.search(log)
        self.pil2_sees = float(sees.group(1)) if sees else None
        needs = PIL2_NEEDS.search(log)
        # Both figures come from the refusal itself, so the shortfall below
        # reconciles within one sentence rather than against `pil2_sees`,
        # which pil2 prints on a different line and need not be there at all.
        self.pil2_needs = float(needs.group(1)) if needs else None
        self.pil2_available = float(needs.group(2)) if needs else None
        self.pil2_refused = bool(needs) or bool(PIL2_CONFIG.search(log))
        streams = PIL2_STREAMS.search(log)
        if streams:
            self.streams = (int(streams.group(1)), int(streams.group(2)))
        else:
            self.streams = None
        # `run_log.completed` falls back to the log's text when there is no
        # `exit=` line, and a pil2 refusal carries neither an exit code nor an
        # out-of-memory for that fallback to catch -- so it would read as a run
        # that finished. The refusal is decisive on its own: nothing proved.
        self.completed = run_log.completed(log) and not self.pil2_refused
        oom = CLIENT_OOM.search(log)
        self.oom_instance = int(oom.group(1)) if oom else None
        self.oom_gib = float(oom.group(2)) * UNIT[oom.group(3)] if oom else None
        self.stats = {k: v.strip() for k, v in STAT.findall(log)}

    @property
    def share(self) -> float | None:
        """The per-client arena. Every client of a run gets the same one, so a
        run whose clients differ is a reading error rather than a
        configuration -- say so instead of picking one."""
        if not self.arenas:
            return None
        if max(self.arenas) - min(self.arenas) > 0.01:
            raise ValueError(f"clients claimed different arenas: {self.arenas}")
        return self.arenas[0]

    @property
    def allocator(self) -> str | None:
        """Which allocator the plugin built, in its own words. A run whose
        clients differ is a reading error rather than a configuration, as with
        the arena."""
        if not self.allocators:
            return None
        if len(set(self.allocators)) > 1:
            raise ValueError(f"clients built different allocators: {self.allocators}")
        return self.allocators[0]

    @property
    def outcome(self) -> str:
        if self.completed:
            # An out-of-memory in a run that finished is one the bridge caught
            # and re-sent under the slot, which is a cost rather than a stop.
            if self.oom_instance is not None:
                return f"verified, {self.oom_gib:.2f} GiB rescued to the slot"
            return "verified"
        if self.pil2_refused:
            return "pil2 refused"
        if self.oom_instance is not None:
            return f"client OOM (instance {self.oom_instance}, {self.oom_gib:.2f} GiB)"
        return "failed"


def report(path: pathlib.Path, run: Run) -> None:
    clients = len(run.arenas)
    share = run.share
    print(f"== {path}")
    if share is None:
        print("   no client arena in this log (native run, or the fraction was unset)")
    else:
        print(
            f"   {clients} client(s) x {share:.2f} GiB arena, {run.allocator} allocator"
        )
    if run.pil2_sees is not None:
        line = f"   pil2 sees {run.pil2_sees:.3f} GB"
        if run.streams:
            line += f", {run.streams[0]} basic / {run.streams[1]} recursive streams"
        print(line)
    if run.pil2_needs is not None:
        short = run.pil2_needs - run.pil2_available
        print(f"   pil2 needs {run.pil2_needs:.3f} GB -- short by {short:.3f} GB")
    print(f"   {run.outcome}")
    if run.stats:
        print("   " + "  ".join(f"{k} {v}" for k, v in run.stats.items()))


def walk(runs: list[Run]) -> None:
    """The walk table: how many runs at each arena size finished. A cell at
    the boundary is a race rather than a threshold (#188), so the count is
    what is quoted and a cell with one run is not a floor."""
    cells: dict[tuple[int, str, float], list[bool]] = collections.defaultdict(list)
    for run in runs:
        if run.share is not None:
            key = (len(run.arenas), run.allocator, round(run.share, 2))
            cells[key].append(run.completed)
    if not cells:
        return
    print(f"== {len(runs)} logs")
    print(f"   {'clients':>7}  {'allocator':>12}  {'arena':>10}  passes")
    for (clients, allocator, share), outcomes in sorted(
        cells.items(), key=lambda kv: (kv[0][0], kv[0][1], -kv[0][2])
    ):
        print(
            f"   {clients:>7}  {allocator:>12}  {share:7.2f} GiB "
            f" {sum(outcomes)}/{len(outcomes)}"
        )


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("log", nargs="+", type=pathlib.Path)
    args = ap.parse_args(argv[1:])
    runs = []
    for path in args.log:
        run = Run(path.read_text(errors="replace"))
        report(path, run)
        runs.append(run)
    if len(runs) > 1:
        walk(runs)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
