"""The recursion circuits' zkin builder — every witness input from OUR
artifacts.

A zkin is the flat u64 vector the circom witness calculator consumes.
Shapes (byte-pinned against the native tower on block 21740136):

    compressor / recursive1:  [prefix | flat child proof]
    recursive2:               [prefix+vk | 3 x (agg head | child proof)]
    vadcop_final:             [prefix    | 1 x (agg head | child proof)]

The prefix is [publics | proofValues (3 words per map entry) | global
challenge (3) | recursive2 verkey (4, where marked)]. A child's
aggregation head is

    [w0 w1 | 0 agv(3) | contribution lattice]

where `agv` is the limbwise mod-p sum of the child subtree's basic
airgroup values (the LogUp grand sums — the cross-instance carry of
zisk-zorch#109 is exactly this addition), the lattice is
`aggregate_contributions` over the subtree's `instance_contribution`s
(the same recipe the global challenge derives from), and the two lead
words are `[air_id + 2, 1]` for a recursive1/compressor child,
`[1, |subtree|]` for a recursive2 child, with the vadcop head counting
the planner's total instances (proved + planned) rather than the
aggregated set.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from zisk_zorch.harness.contributions import aggregate_contributions
from zisk_zorch.harness.pil2 import MODULUS, canon_u64


def _u64(a: np.ndarray) -> np.ndarray:
    """Every wire part as u64 BEFORE it reaches a concatenate: numpy
    promotes a mixed int64/uint64 concatenate to float64, which rounds
    away the low bits of any Goldilocks word above 2^53."""
    return np.asarray(a, dtype=np.uint64)


def _canon(a: np.ndarray) -> np.ndarray:
    """`canon_u64` over a part the caller may not have typed yet."""
    return canon_u64(_u64(a))


@dataclass(frozen=True)
class BasicArtifacts:
    """One basic instance's aggregation-relevant outputs: its airgroup
    value (the emitted proof's leading cubic) and its contribution
    lattice (`instance_contribution` of the stage-1 artifacts)."""

    airgroupvalue: np.ndarray  # (3,) canonical u64
    contribution: np.ndarray  # (lattice_size,) u64


def wire_prefix(
    publics: np.ndarray,
    proofvalues3: np.ndarray,
    global_challenge: np.ndarray,
    verkey: np.ndarray | None,
) -> np.ndarray:
    """The zkin prefix. `proofvalues3` is already wire-packed (3 words per
    ``proofValuesMap`` entry); `verkey` is the recursive2 verkey, which
    only recursive1 and recursive2 carry — the compressor and vadcop_final
    take `None` (`recursion.rs` sizes their prefix without the trailing 4
    and passes an empty verkey path)."""
    parts = [_u64(publics), _u64(proofvalues3), _u64(global_challenge)]
    if verkey is not None:
        parts.append(_u64(verkey))
    return np.concatenate(parts)


def subtree_head(
    basics: list[BasicArtifacts], lead: tuple[int, int]
) -> np.ndarray:
    """A child segment's aggregation head over its subtree's basics."""
    if not basics:
        raise ValueError("a subtree head needs at least one basic instance")
    agv = np.zeros(3, dtype=object)
    for b in basics:
        agv = (agv + b.airgroupvalue.astype(object)) % MODULUS
    lattice = aggregate_contributions([b.contribution for b in basics])
    return np.concatenate(
        [
            np.array(lead, dtype=np.uint64),
            np.zeros(1, dtype=np.uint64),
            agv.astype(np.uint64),
            _canon(lattice),
        ]
    )


def leaf_zkin(prefix: np.ndarray, proof: np.ndarray) -> np.ndarray:
    """compressor (no-verkey prefix) / recursive1-over-basic (verkey
    prefix): the prefix and the wrapped flat proof."""
    return np.concatenate([_u64(prefix), _canon(proof)])


def compressed_leaf_zkin(
    head: np.ndarray, prefix: np.ndarray, proof: np.ndarray
) -> np.ndarray:
    """recursive1 over a COMPRESSOR proof: the instance's aggregation head
    leads, then the verkey prefix, then the compressor's flat proof — the
    only zkin whose head precedes its prefix."""
    return np.concatenate([_u64(head), _u64(prefix), _canon(proof)])


def node_zkin(
    prefix: np.ndarray,
    segments: list[tuple[np.ndarray, np.ndarray] | None],
) -> np.ndarray:
    """recursive2 / vadcop_final: the prefix and fixed-width child
    segments — `(head, proof)` per child, `None` for an absent one
    (zero-filled, the circuit's null-segment convention)."""
    widths = {h.size + p.size for hp in segments if hp for h, p in [hp]}
    # Validation, not a sanity check: with `python -O` an assert here would
    # be stripped and a disagreeing set would pop an arbitrary width, so the
    # `None` slots would zero-fill to the wrong length — a silently
    # misaligned zkin.
    if not widths:
        raise ValueError("a node zkin needs at least one present segment")
    if len(widths) > 1:
        raise ValueError(f"segment widths disagree: {sorted(widths)}")
    width = widths.pop()
    out = [_u64(prefix)]
    for hp in segments:
        if hp is None:
            out.append(np.zeros(width, dtype=np.uint64))
        else:
            head, proof = hp
            out.append(np.concatenate([_u64(head), _canon(proof)]))
    return np.concatenate(out)
