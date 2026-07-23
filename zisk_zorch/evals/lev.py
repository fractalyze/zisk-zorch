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

https://github.com/0xPolygonHermez/pil2-proofman/blob/v1.0.0-alpha/pil2-stark/src/starkpil/starks.hpp#L243-L279
"""

from __future__ import annotations

import frx.numpy as fnp
import numpy as np
from frx import Array
from zk_dtypes import goldilocks as F
from zk_dtypes import goldilocksx3 as F3

from zorch.poly.univariate import powers

# The LDE coset generator and pil2's 2^32-order generator `Goldilocks::W[32]`
# (cf. zisk_zorch.fri.fold, which folds on the same root).
_COSET_SHIFT = 7
_TWO_ADIC_ROOT = 7277203076849721926

# Host Goldilocks scalars for the compile-time-constant LEv arithmetic: the
# field type carries its own modulus, so the roots and inverses below are field
# ops on numpy scalars (host, hence constant-folded into the graph) rather than
# `pow(x, k, modulus)` restating the modulus.
_ONE = np.array(1, dtype=F)


def _fpow(base: Array, exp: int) -> Array:
    """`base ** exp` in Goldilocks on the host, `exp` possibly negative — the
    field type has no negative integer power, so `exp < 0` is `1 / base**|exp|`
    (a field inverse). Positive powers use its fast (square-and-multiply) `**`."""
    return base**exp if exp >= 0 else _ONE / base ** (-exp)


# Cubic one from explicit limbs: `fnp.ones` on an extension dtype lowers to an
# i64 constant the field type rejects under jit (same reason `fnp.power`'s
# integer exponent cannot trace — see compute_lev's squaring loop).
_CUBIC_ONE = fnp.array(
    np.array([1, 0, 0], dtype=np.uint64).astype(F).view(F3).reshape(())
)


def compute_lev(xi_challenge: Array, opening_points: list[int], n_bits: int) -> Array:
    """The `(N, len(opening_points))` cubic LEv coefficient matrix for opening at
    the cubic `xi_challenge`, `N = 2^n_bits` the base-domain size."""
    if not 0 <= n_bits <= 32:
        raise ValueError(f"n_bits must be in [0, 32], got {n_bits}")
    if not opening_points:
        raise ValueError("opening_points must be non-empty")

    n = 1 << n_bits
    one = _CUBIC_ONE
    inv_n = _ONE / np.array(n, dtype=F)
    w = _fpow(np.array(_TWO_ADIC_ROOT, dtype=F), 1 << (32 - n_bits))
    shift_inv = _ONE / np.array(_COSET_SHIFT, dtype=F)
    # w^-j over the base domain — the per-coefficient evaluation points.
    wj_inv = powers(fnp.array(_fpow(w, -1)), n)

    cols = []
    for p in opening_points:
        g = xi_challenge * fnp.array(_fpow(w, p) * shift_inv)
        # g^N by n_bits squarings: `fnp.power`'s integer exponent does not
        # lower for extension dtypes under jit, and N is a power of two.
        g_n = g
        for _ in range(n_bits):
            g_n = g_n * g_n
        num = g_n - one
        # c_j = N^-1 * (g^N - 1) / (g * w^-j - 1), vectorized over j.
        cols.append(inv_n * num * (one / (g * wj_inv - one)))
    return fnp.stack(cols, axis=1)
