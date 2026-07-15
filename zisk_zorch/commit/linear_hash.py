"""pil2-stark's chained linear hash — the Merkle leaf hasher ZisK commits with.

A thin adapter over zorch's `Sponge.hash(..., SpongeType.MERKLE_DAMGARD)` (the
Merkle-Damgard construction): pil2's `linear_hash_seq` zero-pads a partial block
and chains by copying the previous output's first 4 lanes into the capacity slots
`[rate, rate+4)` before each block after the first. (v0.15.0 short-circuited
rows of <= 4 elements to the zero-padded row unhashed; v1.0.0-alpha removed that
shortcut — every row is permuted.) Reference:
https://github.com/0xPolygonHermez/pil2-proofman/blob/v1.0.0-alpha/fields/src/merkle.rs#L14-L39

Fixes rate/out to pil2's convention (rate = width - 4, out = 4, so
rate + out == width) and duck-types zorch's Merkle leaf-hasher surface (`hash`,
`out`, `has_dedicated_fusion`), so `MerkleTree(LinearHash(perm), compressor)`
builds exactly pil2's tree.
"""

from __future__ import annotations

from frx import Array
from zorch.hash.poseidon2.poseidon2 import Poseidon2
from zorch.hash.sponge import Sponge, SpongeParams, SpongeType

# pil2 digests are always 4 Goldilocks elements (HASH_SIZE == CAPACITY).
DIGEST_ELEMS = 4


class LinearHash:
    """pil2 `linear_hash_seq` over a width `>= 8` Poseidon2 permutation."""

    def __init__(self, permutation: Poseidon2) -> None:
        if permutation.width <= DIGEST_ELEMS:
            raise ValueError(
                f"linear hash needs width > {DIGEST_ELEMS}, got {permutation.width}"
            )
        self._permutation = permutation
        self.rate = permutation.width - DIGEST_ELEMS
        self.out = DIGEST_ELEMS
        # Cache the Sponge (value-equal for jit-zone keys) — pil2's rate/out
        # satisfy the chained precondition rate + out == width.
        self._sponge = Sponge(permutation, SpongeParams(rate=self.rate, out=self.out))

    @property
    def has_dedicated_fusion(self) -> bool:
        return self._permutation.has_dedicated_fusion

    def hash(self, input: Array) -> Array:
        """pil2 leaf digest of a row: (n,) over dtype -> (DIGEST_ELEMS,).

        Emits the fused `zorch.sponge_hash` (merkle_damgard) region — one
        register-resident kernel over all blocks — instead of a per-block
        permute+concatenate. `Sponge.hash(..., MERKLE_DAMGARD)` carries the same
        pil2 semantics (zero-pad partial tail; chain the prior digest through the
        capacity slots [rate, rate+DIGEST_ELEMS))."""
        if input.ndim != 1:
            raise ValueError(f"input must be 1-D, got ndim={input.ndim}")
        return self._sponge.hash(input, sponge_type=SpongeType.MERKLE_DAMGARD)
