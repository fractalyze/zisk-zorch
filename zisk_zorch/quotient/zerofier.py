"""pil2-stark's inverse zerofier — the divisor stage-2's quotient multiplies by.

The quotient is `Q = C / Z_H`, where `C` is the composite constraint polynomial
and `Z_H(x) = x^N - 1` vanishes on the base trace domain `H` (size `N`). pil2
computes this on the blown-up coset by multiplying `C` pointwise with the
precomputed inverse zerofier `Zi = 1/Z_H` (pil2's `buildZHInv`).

On the coset `shift * <w(nBitsExt)>`, `x^N = shift^N * w(nBitsExt)^(jN)` and
`w(nBitsExt)^N = w(blowupBits)`, so `x^N = shift^N * w(blowupBits)^j` takes only
`2^blowupBits` distinct values as `j` runs the domain. The inverse zerofier is
therefore that period tiled across the extended domain (natural order) — never
zero, since a nonzero coset shift keeps `x^N != 1`. The firstRow / lastRow
(`buildOneRowZerofierInv`) and everyFrame (`buildFrameZerofierInv`) boundary
divisors build on the same coset points.

All arithmetic stays in the `goldilocks` field — the dtype reduces mod p on
every op, so there is no manual modulus juggling; `jnp.power` is the field-native
exponentiation (`lax.pow` needs a float dtype).

buildZHInv:   https://github.com/0xPolygonHermez/pil2-proofman/blob/v1.0.0-alpha/pil2-stark/src/starkpil/setup_ctx.hpp#L127-L146
buildOneRowZerofierInv: https://github.com/0xPolygonHermez/pil2-proofman/blob/v1.0.0-alpha/pil2-stark/src/starkpil/setup_ctx.hpp#L148-L161
buildFrameZerofierInv:  https://github.com/0xPolygonHermez/pil2-proofman/blob/v1.0.0-alpha/pil2-stark/src/starkpil/setup_ctx.hpp#L163-L191
"""

from __future__ import annotations

import frx.numpy as jnp
import numpy as np
from frx import Array
from zk_dtypes import goldilocks as F

from zorch.poly.univariate import powers

# pil2's coset shift and the 2^32-order generator `Goldilocks::W[32]`, as field
# scalars (cf. zisk_zorch.evals.lev / zisk_zorch.fri.fold, which share them —
# the field dtype carries the modulus but not the generator or the shift).
_SHIFT = jnp.array(np.array(7, dtype=np.uint64), dtype=F)
_TWO_ADIC_ROOT = jnp.array(np.array(7277203076849721926, dtype=np.uint64), dtype=F)
_ONE = jnp.ones((), F)

# `frx.lax.ntt` (and zorch's `ReedSolomon` / `eval_domain`) take the generator
# whose powers walk the subgroup backwards, i.e. `W[32]^-1`. Kept here with the
# root it inverts so the LDE, the FRI fold and the zerofier cannot drift apart on
# which root pil2 is on.
_PIL2_GENERATOR = int(jnp.power(_TWO_ADIC_ROOT, -1))


def _root(bits: int) -> Array:
    """The order-`2^bits` root of unity `Goldilocks::W[bits]`, a field scalar.

    pil2 fixes the two-adic generator at `W[32]`, so the subgroup it can reach
    tops out at order 2^32 — the guard names that ceiling, which the bare shift
    below would otherwise report as `negative shift count` from inside here."""
    if not 0 <= bits <= 32:
        raise ValueError(
            f"bits must be in [0, 32] — pil2's two-adic generator is W[32], so "
            f"there is no 2^{bits}-th root of unity; got {bits}"
        )
    return jnp.power(_TWO_ADIC_ROOT, 1 << (32 - bits))


def _check(n_bits: int, blowup_bits: int) -> None:
    if n_bits < 0:
        raise ValueError(f"n_bits must be non-negative, got {n_bits}")
    if blowup_bits < 1:
        raise ValueError(f"blowup_bits must be >= 1, got {blowup_bits}")
    if not 0 <= n_bits + blowup_bits <= 32:
        raise ValueError("n_bits + blowup_bits must be in [0, 32]")


def _coset_points(n_bits: int, blowup_bits: int) -> Array:
    """`x[i] = shift * w(nBitsExt)^i` on the extended coset — pil2 `computeX`."""
    n_ext = 1 << (n_bits + blowup_bits)
    return _SHIFT * powers(_root(n_bits + blowup_bits), n_ext)


def inv_zerofier(n_bits: int, blowup_bits: int) -> Array:
    """The `(2^(n_bits+blowup_bits),)` inverse zerofier `1/(x^N - 1)` on the
    blown-up coset, base-field (`goldilocks`), in natural domain order.

    `n_bits` is the base trace domain `N = 2^n_bits`; `blowup_bits` the LDE
    blow-up (must be >= 1 — the quotient needs an extended domain). This is the
    `everyRow` divisor (transition constraints hold on all of `H`); only the
    `2^blowup_bits` distinct values are computed, then tiled.
    """
    _check(n_bits, blowup_bits)
    extend = 1 << blowup_bits
    n_ext = 1 << (n_bits + blowup_bits)
    sn = jnp.power(_SHIFT, 1 << n_bits)  # shift^N
    period = _ONE / (sn * powers(_root(blowup_bits), extend) - _ONE)
    return jnp.tile(period, n_ext // extend)


def inv_one_row_zerofier(n_bits: int, blowup_bits: int, row_index: int) -> Array:
    """pil2 `buildOneRowZerofierInv`: the firstRow (`row_index=0`) / lastRow
    (`row_index=N`) boundary divisor `1/((x - w(nBits)^row_index) * Zi_everyRow)`
    over the extended coset. The everyRow inverse divides out `x^N - 1`, leaving
    the single excluded root in the denominator.
    """
    _check(n_bits, blowup_bits)
    x = _coset_points(n_bits, blowup_bits)
    zi_h = inv_zerofier(n_bits, blowup_bits)
    root = jnp.power(_root(n_bits), row_index)
    return _ONE / ((x - root) * zi_h)


def inv_frame_zerofier(
    n_bits: int, blowup_bits: int, offset_min: int, offset_max: int
) -> Array:
    """pil2 `buildFrameZerofierInv`: the everyFrame divisor — the product
    `prod_j (x - root_j)` over the first `offset_min` and last `offset_max` row
    roots (`w(nBits)^i` and `w(nBits)^(N-i-1)`). Despite pil2's name it stores
    the product, not its inverse — mirrored here for the byte-match.
    """
    _check(n_bits, blowup_bits)
    n = 1 << n_bits
    w_n = _root(n_bits)
    x = _coset_points(n_bits, blowup_bits)
    roots = [jnp.power(w_n, i) for i in range(offset_min)]
    roots += [jnp.power(w_n, n - i - 1) for i in range(offset_max)]

    acc = jnp.tile(_ONE, x.shape[0])
    for r in roots:
        acc = acc * (x - r)
    return acc
