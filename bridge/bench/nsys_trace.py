"""What every reader of an `nsys stats --format csv` export in this directory
needs: the column-unit rule, the span algebra the reports are built out of,
and which prover a kernel belongs to.

The last one is why this is a module rather than two copies. A bridged run has
both provers on one card, so every report here has to tell their kernels
apart, and the rule is a heuristic on the name — it will need changing, and a
change that lands in one reader and not the other silently reassigns work
between provers in half the numbers this repo publishes.

Spans are `(start, end)` in nanoseconds, half-open. Every function below that
takes a list of them takes a *merged cover* — sorted and non-overlapping, as
`merge` returns — and the sweeps rely on it; passing raw spans gives wrong
answers rather than an error.
"""

from __future__ import annotations

import re

Span = tuple[int, int]

BRIDGE, PIL2 = "bridge", "pil2"

# nsys writes the unit into the column header, and which one it picks depends
# on the capture's length and size.
UNITS_NS = {"ns": 1, "us": 1_000, "µs": 1_000, "ms": 1_000_000, "s": 1_000_000_000}
UNITS_BYTES = {"B": 1, "KB": 1_000, "MB": 1_000_000, "GB": 1_000_000_000}


def column(
    header: list[str], prefix: str, units: dict[str, int] = UNITS_NS
) -> tuple[str, int]:
    """The named column and the multiplier from its unit to the base of
    `units` — nanoseconds for a time, bytes for a size. Raises rather than
    guessing: a header whose unit is not in the table would otherwise scale
    every value in the report by the wrong power of ten, or by zero."""
    for name in header:
        if name.startswith(prefix):
            unit = re.search(r"\(([^)]*)\)", name)
            scale = units.get(unit.group(1) if unit else "ns")
            if scale is None:
                raise ValueError(f"{name}: unit is not one of {sorted(units)}")
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


def overlap(a: list[Span], b: list[Span]) -> int:
    """How long both merged covers hold. What each has to itself is then its
    own total minus this, so no second sweep is needed."""
    return covered(intersect(a, b))


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


def owner(kernel_name: str) -> str:
    """Which prover emitted a kernel. XLA writes a fusion's name with no
    argument list (`loop_add_fusion`, `sponge_hash_1`); pil2's kernels are
    C++ signatures (`_add(Goldilocks::Element *, ...)`), so a `(` in the name
    is what tells them apart."""
    return PIL2 if "(" in kernel_name else BRIDGE
