"""pil2-stark's chained linear hash — the Merkle leaf hasher ZisK commits with.

NOT a zorch Sponge: pil2's `linear_hash_seq` zero-pads a partial block (zorch's
sponge is padding-free overwrite) and chains by copying the previous output's
first 4 lanes into the capacity slots `[rate, rate+4)` before each block after
the first. (v0.15.0 short-circuited rows of <= 4 elements to the zero-padded
row unhashed; v0.18.0 removed that shortcut — every row is permuted.)
Reference:
https://github.com/0xPolygonHermez/pil2-proofman/blob/v0.18.0/fields/src/poseidon2.rs#L135-L157

Duck-types zorch's Merkle leaf-hasher surface (`hash`, `out`,
`has_dedicated_fusion`), so `MerkleTree(LinearHash(perm), compressor)` builds
exactly pil2's tree.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from zorch.hash.poseidon2.poseidon2 import Poseidon2

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

    @property
    def has_dedicated_fusion(self) -> bool:
        return self._permutation.has_dedicated_fusion

    def hash(self, input: Array) -> Array:
        """pil2 leaf digest of a row: (n,) over dtype -> (DIGEST_ELEMS,)."""
        if input.ndim != 1:
            raise ValueError(f"input must be 1-D, got ndim={input.ndim}")
        n = input.shape[0]
        width = self._permutation.width
        state = jnp.zeros((width,), input.dtype)
        for start in range(0, n, self.rate):
            block = input[start : start + self.rate]
            tail = jnp.zeros((self.rate - block.shape[0],), input.dtype)
            if start == 0:
                capacity = jnp.zeros((DIGEST_ELEMS,), input.dtype)
            else:
                # Chain: the previous output's first 4 lanes ride in the
                # capacity slots [rate, rate+4).
                capacity = state[:DIGEST_ELEMS]
            state = self._permutation.permute(
                jnp.concatenate([block, tail, capacity])
            )
        return state[:DIGEST_ELEMS]
