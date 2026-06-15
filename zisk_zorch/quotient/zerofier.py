"""pil2-stark's inverse zerofier — the divisor stage-2's quotient multiplies by.

The quotient is `Q = C / Z_H`, where `C` is the composite constraint polynomial
and `Z_H(x) = x^N - 1` vanishes on the base trace domain `H` (size `N`). pil2
computes this on the blown-up coset by multiplying `C` pointwise with the
precomputed inverse zerofier `Zi = 1/Z_H` (pil2's `buildZHInv`,
setup_ctx.hpp).

On the coset `shift * <w(nBitsExt)>`, `x^N = shift^N * w(nBitsExt)^(jN)` and
`w(nBitsExt)^N = w(blowupBits)`, so `x^N = shift^N * w(blowupBits)^j` takes only
`2^blowupBits` distinct values as `j` runs the domain. The inverse zerofier is
therefore that period tiled across the extended domain (natural order) — never
zero, since a nonzero coset shift keeps `x^N != 1`.

https://github.com/0xPolygonHermez/pil2-proofman/blob/v0.18.0/pil2-stark/src/starkpil/setup_ctx.hpp#L127-L146
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jax import Array
from zk_dtypes import goldilocks_mont as F
from zk_dtypes import pfinfo

# The Goldilocks modulus, pil2's coset shift, and the 2^32-order generator
# `Goldilocks::W[32]` (cf. zisk_zorch.evals.lev / zisk_zorch.fri.fold, which
# share these — pfinfo carries the modulus but not the generator or the shift).
_MODULUS = int(pfinfo(F).modulus)
_COSET_SHIFT = 7
_TWO_ADIC_ROOT = 7277203076849721926


def inv_zerofier(n_bits: int, blowup_bits: int) -> Array:
    """The `(2^(n_bits+blowup_bits),)` inverse zerofier `1/(x^N - 1)` on the
    blown-up coset, base-field (`goldilocks_mont`), in natural domain order.

    `n_bits` is the base trace domain `N = 2^n_bits`; `blowup_bits` the LDE
    blow-up (must be >= 1 — the quotient needs an extended domain).
    """
    if blowup_bits < 1:
        raise ValueError(f"blowup_bits must be >= 1, got {blowup_bits}")
    if not 0 <= n_bits + blowup_bits <= 32:
        raise ValueError(f"n_bits + blowup_bits must be in [0, 32]")

    extend = 1 << blowup_bits
    n_ext = 1 << (n_bits + blowup_bits)
    sn = pow(_COSET_SHIFT, 1 << n_bits, _MODULUS)
    w_ext = pow(_TWO_ADIC_ROOT, 1 << (32 - blowup_bits), _MODULUS)

    # One value of `1/(shift^N * w(blowupBits)^i - 1)` per coset residue class,
    # then tiled — `x^N` repeats with period `extend` over the domain.
    period = []
    w = 1
    for _ in range(extend):
        period.append(pow((sn * w - 1) % _MODULUS, -1, _MODULUS))
        w = w * w_ext % _MODULUS

    tiled = np.array(period * (n_ext // extend), dtype=np.uint64)
    return jnp.array(tiled, dtype=F)
