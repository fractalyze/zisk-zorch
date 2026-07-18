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
    base_cols: Array,
    cubic_cols: Array,
    lev: Array,
    opening_pos: Sequence[int],
    *,
    n_bits: int,
    blowup_bits: int,
) -> Array:
    """pil2 `evmap`: evaluate each committed column at its opening point. Columns
    arrive split by field — `base_cols` (`(N_ext, B)` base) then `cubic_cols`
    (`(N_ext, C)` cubic), matching the DEEP batching order (#69) — and the base
    dot keeps its 8-byte reads (`lev·base` is scalar×cubic, exact). `opening_pos[m]`
    selects column `m`'s `lev` column; the extended evals subsample to the base
    coset at stride `2^blowup_bits` before the dot. Returns the `(M,)` cubic OOD
    openings, base columns first."""
    stride = 1 << blowup_bits
    b, c = base_cols.shape[1], cubic_cols.shape[1]
    if base_cols.shape[0] != 1 << (n_bits + blowup_bits):
        raise ValueError(
            f"columns must have 2^{n_bits + blowup_bits} rows, "
            f"got {base_cols.shape[0]}"
        )
    if len(opening_pos) != b + c:
        raise ValueError(
            f"opening_pos length {len(opening_pos)} != column count {b + c} "
            f"({b} base + {c} cubic)"
        )
    lev_per_col = lev[:, jnp.array(opening_pos)]  # (N, M) cubic
    base_ev = jnp.sum(lev_per_col[:, :b] * base_cols[::stride], axis=0)  # (B,) cubic
    cubic_ev = jnp.sum(lev_per_col[:, b:] * cubic_cols[::stride], axis=0)  # (C,) cubic
    return jnp.concatenate([base_ev, cubic_ev])  # (M,) cubic
