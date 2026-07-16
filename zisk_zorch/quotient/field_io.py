"""Shared field-array helpers for the stage-2 quotient modules.

The base→cubic embed and the cyclic rotation are byte-match-load-bearing, so they
live in one place rather than being copied between `cexp_ref` and `reauthor`.
Cubic loading from decimal limbs is `zisk_zorch.golden.u64x3`; use that directly.
"""

from __future__ import annotations

import frx.numpy as jnp
import numpy as np
from frx import Array
from zk_dtypes import goldilocks as F
from zk_dtypes import goldilocksx3 as F3


def embed(values: list[str]) -> Array:
    """Base canonical-u64 decimals -> `F3` array of `(b, 0, 0)` embeddings.

    The decimals are already canonical (`< p`, the golden's `as_canonical_u64` /
    pil2's field literals), so `astype(F)` value-converts each straight into the
    plain field, then `astype(F3)` is the dtype's own base→cubic embedding."""
    base = jnp.array(np.array([int(v) for v in values], dtype=np.uint64), dtype=F)
    return base.astype(F3)


def embed_base(base: Array) -> Array:
    """An `F` base array -> `F3` `(b, 0, 0)` — the dtype's own value conversion."""
    return base.astype(F3)


def base_trace(case: dict, n_cols: int) -> Array:
    """The stage-1 base trace `(N, n_cols)` from a golden case's dim-1 `cm` columns
    (column id == index), as an `F` array — the input rw's `eval_constraints` and
    the interaction `VirtualPairCol`s index into."""
    cols = {c["id"]: c["values"] for c in case["cm"] if c["dim"] == 1}
    trace = np.stack(
        [np.array([int(v) for v in cols[j]], dtype=np.uint64) for j in range(n_cols)],
        axis=1,
    )
    return jnp.array(trace, dtype=F)


def rotate(col: Array, shift: int) -> Array:
    """`out[i] = col[(i + shift) mod n]` — the extended-domain image of a
    next/previous-row opening."""
    n = col.shape[0]
    s = shift % n
    return col if s == 0 else jnp.concatenate([col[s:], col[:s]])
