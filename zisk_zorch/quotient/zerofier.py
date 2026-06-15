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


def _w(bits: int) -> int:
    """The order-`2^bits` root of unity, `Goldilocks::W[bits]`."""
    return pow(_TWO_ADIC_ROOT, 1 << (32 - bits), _MODULUS)


def _to_f(values: list[int]) -> Array:
    """Canonical ints -> a 1-D `goldilocks_mont` array (numpy-first, x64-safe)."""
    return jnp.array(np.array([v % _MODULUS for v in values], dtype=np.uint64), dtype=F)


def _check(n_bits: int, blowup_bits: int) -> None:
    if blowup_bits < 1:
        raise ValueError(f"blowup_bits must be >= 1, got {blowup_bits}")
    if not 0 <= n_bits + blowup_bits <= 32:
        raise ValueError("n_bits + blowup_bits must be in [0, 32]")


def _every_row_ints(n_bits: int, blowup_bits: int) -> list[int]:
    """Canonical `1/(x^N - 1)` over the coset — pil2 `buildZHInv`."""
    extend = 1 << blowup_bits
    n_ext = 1 << (n_bits + blowup_bits)
    sn = pow(_COSET_SHIFT, 1 << n_bits, _MODULUS)
    w_ext = _w(blowup_bits)

    # One value of `1/(shift^N * w(blowupBits)^i - 1)` per coset residue class,
    # then tiled — `x^N` repeats with period `extend` over the domain.
    period = []
    w = 1
    for _ in range(extend):
        period.append(pow((sn * w - 1) % _MODULUS, -1, _MODULUS))
        w = w * w_ext % _MODULUS
    return period * (n_ext // extend)


def _coset_points(n_bits: int, blowup_bits: int) -> list[int]:
    """`x[i] = shift * w(nBitsExt)^i` on the extended coset — pil2 `computeX`."""
    n_ext = 1 << (n_bits + blowup_bits)
    w_ext = _w(n_bits + blowup_bits)
    pts = [0] * n_ext
    x = _COSET_SHIFT % _MODULUS
    for i in range(n_ext):
        pts[i] = x
        x = x * w_ext % _MODULUS
    return pts


def inv_zerofier(n_bits: int, blowup_bits: int) -> Array:
    """The `(2^(n_bits+blowup_bits),)` inverse zerofier `1/(x^N - 1)` on the
    blown-up coset, base-field (`goldilocks_mont`), in natural domain order.

    `n_bits` is the base trace domain `N = 2^n_bits`; `blowup_bits` the LDE
    blow-up (must be >= 1 — the quotient needs an extended domain). This is the
    `everyRow` divisor (transition constraints hold on all of `H`).
    """
    _check(n_bits, blowup_bits)
    return _to_f(_every_row_ints(n_bits, blowup_bits))


def inv_one_row_zerofier(n_bits: int, blowup_bits: int, row_index: int) -> Array:
    """pil2 `buildOneRowZerofierInv`: the firstRow (`row_index=0`) / lastRow
    (`row_index=N`) boundary divisor `1/((x - w(nBits)^row_index) * Zi_everyRow)`
    over the extended coset. The everyRow inverse divides out `x^N - 1`, leaving
    the single excluded root in the denominator.
    """
    _check(n_bits, blowup_bits)
    x = _coset_points(n_bits, blowup_bits)
    zi_h = _every_row_ints(n_bits, blowup_bits)
    root = pow(_w(n_bits), row_index, _MODULUS)
    vals = [pow((x[i] - root) * zi_h[i] % _MODULUS, -1, _MODULUS) for i in range(len(x))]
    return _to_f(vals)


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
    w_n = _w(n_bits)
    x = _coset_points(n_bits, blowup_bits)
    roots = [pow(w_n, i, _MODULUS) for i in range(offset_min)]
    roots += [pow(w_n, n - i - 1, _MODULUS) for i in range(offset_max)]

    vals = []
    for xi in x:
        acc = 1
        for r in roots:
            acc = acc * ((xi - r) % _MODULUS) % _MODULUS
        vals.append(acc)
    return _to_f(vals)
