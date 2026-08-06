"""Main's continuation predicate — the per-segment direct bus updates.

Main's cross-segment state rides 96 DIRECT interactions (single-row bus
updates whose operands are stage-1 air values, never trace columns), absent
from the rw manifest; under the sp1-zorch pattern the consumer ports them,
as `public_values.py` does for SP1's public-value digests. Ported from
`main.pil` at the fork rev the fixture tool pins
(https://github.com/fractalyze/zisk/blob/9c4044b6/state-machines/main/pil/main.pil):

- busid 1000 (`MAIN_CONTINUATION_ID`): one assumes of the inherited state
  `[segment, 0, initial_pc, previous_c]` and one proves of the successor
  state `[next_segment, last_segment, next_pc, last_c]`, where
  `next_segment = (segment + 1) * (1 - last_segment)` closes the cycle.
- busid 10 (`MEMORY_ID`), per register 1..=31 (reg 0 is ZisK scratch): a
  proves of the segment's last access at `last_reg_mem_step[r]` and a
  `(1 - last_segment)`-gated assumes of the same value re-issued at the
  segment-boundary step — the access the next segment's reload consumes.
- busid 102 / 103 (std range checks): the 31 reload monotonicity windows
  `boundary_step - last_reg_mem_step[r] - 1` in `[0, 2^24)`, and
  `main_segment`'s own range.

Each interaction is also one `im_airval` hint in the proving key (a
stage-2 scalar `±mult / D`), which is what pins this port: the golden
fixture evaluates the key's own SSA per interaction on the real block's
12 segment vectors, and `continuation_test` matches tuple-for-tuple.

The cycle ends are vadcop-global, not per-segment: `global_updates`
carries the boot/end continuation tuples and the 31 register-init
assumes the global bus sum consumes once per proof.
"""

from __future__ import annotations

from dataclasses import dataclass

from frx import Array

from zisk_zorch.golden import embed

_P = 0xFFFFFFFF00000001

# main.pil / opids.pil constants at the pinned fork rev.
MAIN_CONTINUATION_ID = 1000
MEMORY_ID = 10
REG_RANGE_ID = 102
SEGMENT_RANGE_ID = 103
MEMORY_REG_OP = 3
REG_BYTES = 8
BOOT_ADDR = 0x1000
END_PC_ADDR = 0x1004
REGS_IN_MAIN_FROM = 1
REGS_IN_MAIN = 31
RC = 2
# mem.pil's step composition: slot MAX-1 of a main step is reserved for the
# segment-boundary register re-issue (mem_offset 3 is never used by a
# register write).
RESERVED_MEM_STEPS = 1
MAX_MEM_STEPS_PER_MAIN_STEP = 4

_SCALARS = (
    "Main.main_segment",
    "Main.main_last_segment",
    "Main.segment_initial_pc",
    "Main.segment_next_pc",
)


@dataclass(frozen=True)
class DirectInteraction:
    """One single-row bus update: `numerator / D(bus_id, values)` in the
    grand sum. `numerator` is the signed multiplicity — proves +1,
    assumes -1, scaled by the pil selector (so the last segment's
    boundary re-issue carries 0)."""

    bus_id: int
    values: tuple[int, ...]
    numerator: int


@dataclass(frozen=True)
class MainSegmentAirValues:
    """One Main segment's stage-1 air values, regs indexed 0..30 for
    registers 1..=31."""

    main_segment: int
    main_last_segment: int
    segment_initial_pc: int
    segment_next_pc: int
    segment_previous_c: tuple[int, int]
    segment_last_c: tuple[int, int]
    last_reg_value: tuple[tuple[int, int], ...]
    last_reg_mem_step: tuple[int, ...]

    @classmethod
    def from_named(cls, named: dict[str, list[int]]) -> MainSegmentAirValues:
        """Build from a name -> values mapping in `airValuesMap` order —
        the shape the oracle binds (repeated names append in map order:
        `last_reg_value` arrives as 62 words, reg-major limb-minor)."""
        for name in _SCALARS + ("Main.segment_previous_c", "Main.segment_last_c"):
            if name not in named:
                raise KeyError(f"air value {name!r} missing from the mapping")
        values = named["Main.last_reg_value"]
        steps = named["Main.last_reg_mem_step"]
        if len(values) != REGS_IN_MAIN * RC or len(steps) != REGS_IN_MAIN:
            raise ValueError(
                f"register arrays carry {len(values)}/{len(steps)} words, "
                f"want {REGS_IN_MAIN * RC}/{REGS_IN_MAIN}"
            )
        return cls(
            main_segment=named["Main.main_segment"][0],
            main_last_segment=named["Main.main_last_segment"][0],
            segment_initial_pc=named["Main.segment_initial_pc"][0],
            segment_next_pc=named["Main.segment_next_pc"][0],
            segment_previous_c=tuple(named["Main.segment_previous_c"]),
            segment_last_c=tuple(named["Main.segment_last_c"]),
            last_reg_value=tuple(
                (values[2 * r], values[2 * r + 1]) for r in range(REGS_IN_MAIN)
            ),
            last_reg_mem_step=tuple(steps),
        )


def main_next_segment(av: MainSegmentAirValues) -> int:
    """`(main_segment + 1) * (1 - main_last_segment)` — the successor id,
    wrapping the last segment back to the global end tuple's 0."""
    return (av.main_segment + 1) * (1 - av.main_last_segment)


def boundary_mem_step(av: MainSegmentAirValues, n_bits: int) -> int:
    """`main_step_to_special_mem_step((main_segment + 1) * N - 1)` — the
    reserved mem-step slot of the segment's final row, `4 * (seg + 1) * N`
    under the current step composition."""
    step = (av.main_segment + 1) * (1 << n_bits) - 1
    return (
        RESERVED_MEM_STEPS
        + MAX_MEM_STEPS_PER_MAIN_STEP * step
        + MAX_MEM_STEPS_PER_MAIN_STEP
        - 1
    )


def continuation_assumes(av: MainSegmentAirValues) -> DirectInteraction:
    return DirectInteraction(
        MAIN_CONTINUATION_ID,
        (av.main_segment, 0, av.segment_initial_pc, *av.segment_previous_c),
        -1,
    )


def continuation_proves(av: MainSegmentAirValues) -> DirectInteraction:
    return DirectInteraction(
        MAIN_CONTINUATION_ID,
        (
            main_next_segment(av),
            av.main_last_segment,
            av.segment_next_pc,
            *av.segment_last_c,
        ),
        1,
    )


def interactions(
    av: MainSegmentAirValues, n_bits: int
) -> tuple[DirectInteraction, ...]:
    """The segment's 96 direct updates, register-major like the key's
    `im_airval` hint order: per register the monotonicity range check,
    the last-access proves, then the boundary re-issue assumes; then the
    two continuation tuples and `main_segment`'s range."""
    boundary = boundary_mem_step(av, n_bits)
    out: list[DirectInteraction] = []
    for r in range(REGS_IN_MAIN):
        addr = REGS_IN_MAIN_FROM + r
        value = av.last_reg_value[r]
        out.append(
            DirectInteraction(
                REG_RANGE_ID, (boundary - av.last_reg_mem_step[r] - 1,), -1
            )
        )
        out.append(
            DirectInteraction(
                MEMORY_ID,
                (MEMORY_REG_OP, addr, av.last_reg_mem_step[r], REG_BYTES, *value),
                1,
            )
        )
        out.append(
            DirectInteraction(
                MEMORY_ID,
                (MEMORY_REG_OP, addr, boundary, REG_BYTES, *value),
                -(1 - av.main_last_segment),
            )
        )
    out.append(continuation_assumes(av))
    out.append(continuation_proves(av))
    out.append(DirectInteraction(SEGMENT_RANGE_ID, (av.main_segment,), -1))
    return tuple(out)


def fold_denominator(it: DirectInteraction, alpha: Array, gamma: Array) -> Array:
    """pil2's bus compression of one direct tuple: reverse-α Horner over
    the values (last slot at the highest power), `· α + bus_id` to append
    the bus id at the low end, `+ γ` — `LogUpBus._gsum_e`'s convention,
    which the golden pins 96/96 against the key's own `im_airval`
    denominators."""
    den = embed([it.values[-1]]).reshape(())
    for v in reversed(it.values[:-1]):
        den = den * alpha + embed([v]).reshape(())
    return den * alpha + embed([it.bus_id]).reshape(()) + gamma


def direct_term(
    av: MainSegmentAirValues, n_bits: int, alpha: Array, gamma: Array
) -> Array:
    """The segment's whole direct contribution to the grand sum:
    `Σ numerator / D` over the 96 updates — what pil2 adds to `gsum[-1]`
    when it exports the airgroup value."""
    total = embed([0]).reshape(())
    for it in interactions(av, n_bits):
        num = embed([it.numerator % _P]).reshape(())
        total = total + num / fold_denominator(it, alpha, gamma)
    return total


def global_updates() -> tuple[DirectInteraction, ...]:
    """The once-per-proof vadcop-global tuples that close every cycle the
    per-segment updates open: the boot proves / end assumes of the
    continuation chain, and the zero-init assumes of each register's
    access cycle (`global_init_mem`)."""
    zeros = (0,) * RC
    out = [
        DirectInteraction(MAIN_CONTINUATION_ID, (0, 0, BOOT_ADDR, *zeros), 1),
        DirectInteraction(MAIN_CONTINUATION_ID, (0, 1, END_PC_ADDR, *zeros), -1),
    ]
    for r in range(REGS_IN_MAIN):
        out.append(
            DirectInteraction(
                MEMORY_ID,
                (MEMORY_REG_OP, REGS_IN_MAIN_FROM + r, 0, REG_BYTES, *zeros),
                -1,
            )
        )
    return tuple(out)
