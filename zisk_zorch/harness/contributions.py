"""pil2's contributions phase — the global challenge from stage-1 artifacts.

The composed prover's transcript seeds with a global challenge every
instance's stage-1 commit feeds. proofman derives it in three steps
(`challenge_accumulation.rs`, `get_contribution_air`; ZisK's globalInfo:
``curve: None``, ``latticeSize: 368``):

1. Per instance: absorb ``[verkey(4) | root1(4) | stage-1 airvalues]`` into a
   fresh width-16 transcript of the key's hash family, take the flushed state,
   and expand it to the lattice width by chaining the width-16 permutation —
   block ``j+1`` is the permutation of block ``j``.
2. Aggregate all instances by elementwise Goldilocks addition. Addition is
   commutative, so distributed provers can sum partial contributions in any
   order.
3. Absorb the proof-level publics, the stage-1 proof values (when the key
   declares any), and the aggregate into a fresh transcript; the squeezed
   cubic challenge is the seed.
"""

from __future__ import annotations

from functools import cache

import frx
import frx.numpy as fnp
import numpy as np

from zisk_zorch.harness.pil2 import MODULUS, to_field, value_offsets
from zisk_zorch.transcript.transcript import _HASH_FAMILY_PERMS, Transcript

# The contribution hashes run the width-16 permutation for both families
# (`challenge_accumulation.rs` `const W`), independent of the prove
# transcript's own width.
W = 16


def stage1_values(packed: np.ndarray, values_map: list) -> np.ndarray:
    """The stage-1 entries of a packed value section, in map order (pil2
    packs stage-1 values as one word, stage>=2 as three)."""
    return np.array(
        [packed[off] for _, off, w in value_offsets(values_map) if w == 1],
        dtype=np.uint64,
    )


def instance_contribution(
    vk: np.ndarray,
    root1: np.ndarray,
    stage1_airvalues: np.ndarray,
    *,
    hash_family: str,
    lattice_size: int,
) -> np.ndarray:
    """Step 1: one instance's lattice contribution as canonical u64."""
    if lattice_size % W:
        raise ValueError(f"lattice_size must be a multiple of {W}, got {lattice_size}")
    values = np.concatenate(
        [
            np.asarray(vk, dtype=np.uint64).reshape(4),
            np.asarray(root1, dtype=np.uint64).reshape(4),
            np.asarray(stage1_airvalues, dtype=np.uint64).reshape(-1),
        ]
    )
    t = Transcript(W, hash_family)
    t.put(to_field(values))
    chained = _lattice_chain(hash_family, lattice_size)(t.get_state())
    return np.asarray(chained).reshape(-1).astype(np.uint64)


@cache
def _lattice_chain(hash_family: str, lattice_size: int):
    """The seed-state -> full-lattice permutation chain as ONE compiled
    kernel. The chain is sequential by construction, but running it eagerly
    costs a dispatch + device->host sync per block (~22 round-trips per
    instance at latticeSize 368) — a measured slice of the block composite's
    per-instance commit leg (#144). Shape is (family, size)-fixed, so one
    cache entry serves every instance of a key."""
    perm = _HASH_FAMILY_PERMS[hash_family](W)
    n_blocks = lattice_size // W - 1

    def chain(state):
        blocks = [state.reshape(W)]
        for _ in range(n_blocks):
            blocks.append(perm.permute(blocks[-1]))
        return fnp.concatenate(blocks)

    return frx.jit(chain)


def aggregate_contributions(contributions: list[np.ndarray]) -> np.ndarray:
    """Step 2: the elementwise Goldilocks sum of every instance's lattice."""
    acc = np.zeros_like(contributions[0], dtype=object)
    for c in contributions:
        acc = (acc + c) % MODULUS
    return acc.astype(np.uint64)


def global_challenge(
    publics: np.ndarray,
    stage1_proofvalues: np.ndarray,
    aggregate: np.ndarray,
    *,
    hash_family: str,
) -> np.ndarray:
    """Step 3: the cubic global challenge as its 3 Goldilocks words."""
    t = Transcript(W, hash_family)
    t.put(to_field(np.asarray(publics, dtype=np.uint64)))
    if len(stage1_proofvalues):
        t.put(to_field(np.asarray(stage1_proofvalues, dtype=np.uint64)))
    t.put(to_field(aggregate))
    return np.asarray(t.get_field())
