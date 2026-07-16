"""Out-of-domain opening — pil2-stark's `computeLEv` + `evmap`.

The DEEP step evaluates every committed polynomial at the out-of-domain (OOD)
challenge point `z` (and its shifts `z·g^p` for the wrapped opening points `p`),
given only the polynomial's evaluations on the extended coset domain. pil2 does
this with a Lagrange-evaluation vector:

- `computeLEv(z)` builds, per opening point, the geometric series
  `pre[k] = (z·g^p·shift⁻¹)^k` over the *base* domain (`N = 2^nBits`) and INTTs
  it at the base root `W[nBits]`. The result `LEv[k]` are the barycentric weights
  such that `Σ_k LEv[k]·p(shift·g^k) = p(z·g^p)` for any `deg < N` polynomial —
  the identity `Σ_k LEv[k] g^{dk} = (z·g^p)^d` after the INTT (verified in the
  round-trip test).
- `evmap` subsamples each committed column's extended evals to the base coset
  (stride `2^extendBits`, `extendBits = nBitsExt − nBits`) and dots it with the
  opening's `LEv` column, yielding the OOD opening value.

Reference (v1.0.0-alpha, starks.hpp `computeLEv` / `evmap`):
https://github.com/0xPolygonHermez/pil2-proofman/blob/v1.0.0-alpha/pil2-stark/src/starkpil/starks.hpp
"""

from __future__ import annotations

from collections.abc import Sequence

import frx
import frx.numpy as jnp
from frx import Array
from zk_dtypes import goldilocks as F

from zorch.poly.univariate import powers

from zisk_zorch.fri.fold import intt
from zisk_zorch.fri.seam import _base_to_cubic, _cubic_to_base
from zisk_zorch.quotient.zerofier import _ONE, _SHIFT, _root

# Cubic multiplicative identity (limb0 = 1), for the geometric-series seed —
# never assumes a `jnp.ones` cubic-storage layout.
_CUBIC_ONE = _base_to_cubic(jnp.array([1, 0, 0], F)).reshape(())


def compute_lev(
    z: Array, opening_points: Sequence[int], n_bits: int
) -> Array:
    """pil2 `computeLEv`: the `(N, len(opening_points))` cubic Lagrange-evaluation
    matrix for OOD point `z` (its 3 Goldilocks limbs). Column `o` opens at
    `z·g^opening_points[o]` (negative openings invert `g`), `g = W[nBits]`."""
    if z.shape != (3,):
        raise ValueError(f"z must be a cubic challenge (3 limbs), got {z.shape}")
    n = 1 << n_bits
    zc = _base_to_cubic(z).reshape(())
    g = _root(n_bits)
    shift_inv = _ONE / _SHIFT
    columns = []
    for p in opening_points:
        w = jnp.power(g, abs(p))
        if p < 0:
            w = _ONE / w
        # pre[k] = (z·g^p·shift⁻¹)^k; INTT turns the series into the eval weights.
        columns.append(powers(zc * w * shift_inv, n))
    pre = jnp.stack(columns, axis=1)  # (N, n_openings) cubic
    coeffs = intt(_cubic_to_base(pre), n_bits)  # INTT each limb-column at W[nBits]
    return _base_to_cubic(coeffs)  # (N, n_openings) cubic


def open_columns(
    columns_ext: Array,
    lev: Array,
    opening_pos: Sequence[int],
    *,
    n_bits: int,
    blowup_bits: int,
) -> Array:
    """pil2 `evmap`: evaluate each cubic column of `columns_ext`
    (`(2^nBitsExt, M)`) at its opening point. `opening_pos[m]` selects column `m`'s
    `lev` column; the extended evals subsample to the base coset at stride
    `2^blowup_bits` before the dot. Returns the `(M,)` cubic OOD openings."""
    n = 1 << n_bits
    stride = 1 << blowup_bits
    if columns_ext.shape[0] != n << blowup_bits:
        raise ValueError(
            f"columns_ext must have 2^{n_bits + blowup_bits} rows, "
            f"got {columns_ext.shape[0]}"
        )
    if len(opening_pos) != columns_ext.shape[1]:
        raise ValueError(
            f"opening_pos length {len(opening_pos)} != column count "
            f"{columns_ext.shape[1]}"
        )
    base_coset = columns_ext[::stride]  # (N, M) — evals on shift·g^k
    lev_per_col = lev[:, jnp.array(opening_pos)]  # (N, M) cubic
    return jnp.sum(lev_per_col * base_coset, axis=0)  # (M,) cubic
