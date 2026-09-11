#!/usr/bin/env python3
"""The bridge's own `ZZ_LOG` line per instance, which is the one line in a
`cargo-zisk prove -vv` log that more than one reader here needs.

It exists as a module for the reason `nsys_trace` does: two copies of a rule
drift, and a change that lands in one reader and not the other silently
rewrites half the numbers this repo publishes. `summarize.py` reports these
per air, and `leg_phases.py` turns them into the spans a bridged run's basic
phase is made of, because under the bridge proofman's own `GEN_PROOF` spans
are not prove time.

The kind in parentheses is matched loosely on purpose. A worker instance says
`(worker)` and a streamed one says `(streamed)`, and a reader that once
matched only the first under-counted by one and misaligned every prove-to-AIR
mapping built on the count -- silently, because the line is still there and
still well formed. A third kind should arrive in both readers as an instance
with a new name in that field, not vanish from one of them.

Whether a run *finished* is the module's other shared rule, for the same
reason: two readers score cells on it, and both ways of getting it wrong are
silent.

- **The verify line has two wordings.** A prover built before 2026-09-08 ends
  `Proof verified successfully` and a later one `Vadcop Final proof was
  verified`. Neither identifies the stack -- native and bridged runs of one
  vintage end the same way -- so match both and infer nothing from which
  appears. A reader matching one wording scores every log of the other
  vintage as a failure, which reads as a floor several cells too high.
- **An out-of-memory message is not an abort.** The bridge catches a
  read-ahead upload's OOM and uploads under the slot instead, and Rust's
  default panic hook prints the message before `catch_unwind` ever sees it,
  unconditionally and whatever `ZZ_LOG` says. A rescued run therefore holds
  text identical to an aborted one's. What tells them apart is `exit=`, which
  `run.sh` appends after the prover exits."""
from __future__ import annotations

import re
import typing

INSTANCE = re.compile(
    r"\[zz \+\s*([\d.]+)\] instance (\d+) (\S+) \((\w+)\): ([\d.]+) s,"
    r" of which ([\d.]+) s waiting"
)


class Instance(typing.NamedTuple):
    """One instance's prove, on the bridge's clock -- seconds since bridge-up,
    which is not proofman's clock."""

    done: float
    index: int
    air: str
    kind: str
    total: float
    waiting: float

    @property
    def held(self) -> float:
        """The prove itself: the time the instance held a client, which is its
        total less what it spent queued for one."""
        return self.total - self.waiting

    @property
    def span(self) -> tuple[float, float]:
        """The interval the instance held a client, read back from its end."""
        return (self.done - self.held, self.done)


def instances(log: str) -> list[Instance]:
    return [
        Instance(float(done), int(index), air, kind, float(total), float(waiting))
        for done, index, air, kind, total, waiting in INSTANCE.findall(log)
    ]


VERIFIED = ("Proof verified successfully", "Vadcop Final proof was verified")
OOM = re.compile(r"PJRT error in \w+: (Out of memory[^\n]*)")
EXIT = re.compile(r"^exit=(\d+)$", re.M)
# A read-ahead upload the bridge caught and sent to the slot instead. Only
# logged at ZZ_LOG>=1, so its absence means nothing.
RESCUED = re.compile(
    r"\[zz \+\s*[\d.]+\] (\w+): read-ahead upload gave way to the slot \([^\n]*\)"
)


def verified(log: str) -> bool:
    """Whether the run printed a verified final proof, in either wording."""
    return any(p in log for p in VERIFIED)


def completed(log: str) -> bool:
    """Whether the run finished, keyed on `exit=` rather than on any message.

    Without that line -- a partial capture, or a log not written by `run.sh` --
    fall back to the text, and do not call a log that merely stops early an
    abort: say so only when it also carries a failure to point at."""
    exited = EXIT.search(log)
    if exited:
        return exited.group(1) == "0"
    return verified(log) or not OOM.findall(log)
