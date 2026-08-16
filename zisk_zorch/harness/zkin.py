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

_P = (1 << 64) - (1 << 32) + 1


def _canon(a: np.ndarray) -> np.ndarray:
    return np.where(a >= np.uint64(_P), a - np.uint64(_P), a)


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
    ``proofValuesMap`` entry); `verkey` is the recursive2 verkey where the
    circuit carries one (recursive1/compressor/recursive2 — vadcop_final
    reads no verkey from its input)."""
    parts = [publics, proofvalues3, global_challenge]
    if verkey is not None:
        parts.append(verkey)
    return np.concatenate(parts).astype(np.uint64)


def subtree_head(
    basics: list[BasicArtifacts], lead: tuple[int, int]
) -> np.ndarray:
    """A child segment's aggregation head over its subtree's basics."""
    agv = np.zeros(3, dtype=object)
    for b in basics:
        agv = (agv + b.airgroupvalue.astype(object)) % _P
    lattice = aggregate_contributions([b.contribution for b in basics])
    return np.concatenate(
        [
            np.array(lead, dtype=np.uint64),
            np.zeros(1, dtype=np.uint64),
            agv.astype(np.uint64),
            _canon(np.asarray(lattice).astype(np.uint64)),
        ]
    )


def leaf_zkin(prefix: np.ndarray, proof: np.ndarray) -> np.ndarray:
    """compressor (no-verkey prefix) / recursive1-over-basic (verkey
    prefix): the prefix and the wrapped flat proof."""
    return np.concatenate([prefix, _canon(proof)]).astype(np.uint64)


def compressed_leaf_zkin(
    head: np.ndarray, prefix: np.ndarray, proof: np.ndarray
) -> np.ndarray:
    """recursive1 over a COMPRESSOR proof: the instance's aggregation head
    leads, then the verkey prefix, then the compressor's flat proof — the
    only zkin whose head precedes its prefix."""
    return np.concatenate([head, prefix, _canon(proof)]).astype(np.uint64)


def node_zkin(
    prefix: np.ndarray,
    segments: list[tuple[np.ndarray, np.ndarray] | None],
) -> np.ndarray:
    """recursive2 / vadcop_final: the prefix and fixed-width child
    segments — `(head, proof)` per child, `None` for an absent one
    (zero-filled, the circuit's null-segment convention)."""
    widths = {h.size + p.size for hp in segments if hp for h, p in [hp]}
    assert len(widths) == 1, f"segment widths disagree: {widths}"
    width = widths.pop()
    out = [prefix]
    for hp in segments:
        if hp is None:
            out.append(np.zeros(width, dtype=np.uint64))
        else:
            head, proof = hp
            out.append(np.concatenate([head, _canon(proof)]))
    return np.concatenate(out).astype(np.uint64)
