"""computeLEv — pil2-stark's Lagrange-evaluation vector for the opening phase.

To open the committed polynomials at the evaluation challenge `xi` (and its row
shifts), pil2 precomputes, per opening offset `p`, the coefficient vector of the
degree-`N` polynomial whose evaluations on the base subgroup are the geometric
series `g^k` (k in [0, N)), where `g = xi * w(nBits)^p * shift^-1` is the
shifted opening point (negative `p` inverts the root power). `evmap` then dots
that vector against a committed column to read off `P(xi * w^p)`.

pil2 builds the series and runs an INTT (`NTT_Goldilocks::INTT` over the base
domain N). The INTT of a geometric series has a closed form — the IDFT is

    c_j = N^-1 * (g^N - 1) * (g * w^-j - 1)^-1,

since `(g*w^-j)^N = g^N` collapses the sum — so the coefficients come out
directly from the cubic arithmetic, byte-identical to the INTT by IDFT
uniqueness and free of any NTT root-order convention (the same trick `fri.fold`
uses to dodge the root reindex). `g` carries a nonzero extension part for a real
extension challenge, so `g * w^-j != 1` and the denominator never vanishes.

Output is `(N, n_open)` cubic, entry `[k][i]` the k-th coefficient for opening
point `i` — pil2's row-major `LEv[k*nOpen + i]` layout. Host-driven and
un-jitted like the rest of the proof orchestration.

https://github.com/0xPolygonHermez/pil2-proofman/blob/v0.18.0/pil2-stark/src/starkpil/starks.hpp#L243-L279
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jax import Array
from zk_dtypes import goldilocks_mont as F
from zk_dtypes import goldilocksx3_mont as F3
from zk_dtypes import pfinfo

# The Goldilocks modulus, the LDE coset generator, and pil2's 2^32-order
# generator `Goldilocks::W[32]` (cf. zisk_zorch.fri.fold, which folds on the
# same root — pfinfo carries the modulus but not the generator or the shift).
_MODULUS = int(pfinfo(F).modulus)
_COSET_SHIFT = 7
_TWO_ADIC_ROOT = 7277203076849721926


def _base(value: int) -> Array:
    """A canonical Goldilocks int -> a montgomery scalar (value-converted)."""
    return jnp.array(np.array(value % _MODULUS, dtype=np.uint64), dtype=F)


def _base_powers(base: int, count: int) -> Array:
    """`[base^0, ..., base^(count-1)] mod p` as a 1-D montgomery array."""
    out = [1] * count
    for k in range(1, count):
        out[k] = out[k - 1] * base % _MODULUS
    return jnp.array(np.array(out, dtype=np.uint64), dtype=F)


def compute_lev(xi_challenge: Array, opening_points: list[int], n_bits: int) -> Array:
    """The `(N, len(opening_points))` cubic LEv coefficient matrix for opening at
    the cubic `xi_challenge`, `N = 2^n_bits` the base-domain size."""
    if not 0 <= n_bits <= 32:
        raise ValueError(f"n_bits must be in [0, 32], got {n_bits}")
    if not opening_points:
        raise ValueError("opening_points must be non-empty")

    n = 1 << n_bits
    one = jnp.ones((), F3)
    inv_n = _base(pow(n, -1, _MODULUS))
    w = pow(_TWO_ADIC_ROOT, 1 << (32 - n_bits), _MODULUS)
    shift_inv = pow(_COSET_SHIFT, -1, _MODULUS)
    # w^-j over the base domain — the per-coefficient evaluation points.
    wj_inv = _base_powers(pow(w, -1, _MODULUS), n)

    cols = []
    for p in opening_points:
        w_p = pow(w, abs(p), _MODULUS)
        if p < 0:
            w_p = pow(w_p, -1, _MODULUS)
        g = xi_challenge * _base(w_p * shift_inv % _MODULUS)
        num = jnp.power(g, n) - one
        # c_j = N^-1 * (g^N - 1) / (g * w^-j - 1), vectorized over j.
        cols.append(inv_n * num * (one / (g * wj_inv - one)))
    return jnp.stack(cols, axis=1)
