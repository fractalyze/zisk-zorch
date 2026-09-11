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
with a new name in that field, not vanish from one of them."""
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
