"""Out-of-domain opening — pil2-stark's `evmap`.

The DEEP step evaluates every committed polynomial at the out-of-domain (OOD)
challenge point `z` (and its shifts `z·g^p` for the wrapped opening points `p`),
given only the polynomial's evaluations on the extended coset domain. pil2 does
this with a Lagrange-evaluation vector: `computeLEv(z)` builds the barycentric
weights `LEv[k]` such that `Σ_k LEv[k]·p(shift·g^k) = p(z·g^p)` for any
`deg < N` polynomial ([`../evals/lev.py`](../evals/lev.py), pinned by a pil2
golden), and `evmap` — this module — subsamples each committed column's extended
evals to the base coset (stride `2^extendBits`, `extendBits = nBitsExt − nBits`)
and dots it with the opening's `LEv` column, yielding the OOD opening value.

Reference (v1.0.0-alpha, starks.hpp `evmap`):
https://github.com/0xPolygonHermez/pil2-proofman/blob/v1.0.0-alpha/pil2-stark/src/starkpil/starks.hpp
"""

from __future__ import annotations

from collections.abc import Sequence

import frx
import frx.numpy as jnp
from frx import Array

from zisk_zorch.fri.seam import _base_to_cubic, _cubic_to_base


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
