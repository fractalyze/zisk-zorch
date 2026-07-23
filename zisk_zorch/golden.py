"""Loaders for the golden/ JSON fixtures (canonical-u64 decimal strings)."""

from __future__ import annotations

import json
import pathlib

import frx.numpy as fnp
import numpy as np
from frx import Array
from zk_dtypes import goldilocks as F
from zk_dtypes import goldilocksx3 as F3


def load(path: pathlib.Path) -> dict:
    return json.loads(path.read_text())


def u64(values: list[str]) -> Array:
    """Decimal-string canonical u64s -> a 1-D goldilocks array."""
    return fnp.array(np.array([int(v) for v in values], dtype=np.uint64), dtype=F)


def u64x3(values: list[str]) -> Array:
    """Flat decimal-string canonical limbs (3 per cubic element) -> a 1-D
    goldilocksx3 array. Each element's limbs value-convert to plain
    in the base field, then the contiguous limb triples view as one cubic."""
    flat = np.array([int(v) for v in values], dtype=np.uint64).reshape(-1, 3)
    base = flat.astype(F)
    return fnp.array(base.view(F3).reshape(flat.shape[0]))


def embed(values: list[str]) -> Array:
    """Base canonical-u64 decimals -> `F3` array of `(b, 0, 0)` embeddings — the
    base-decimal sibling of `u64x3`. `astype(F3)` is the dtype's own base->cubic
    embedding, so a base operand stays exact scalar multiplication in `F3`."""
    return u64(values).astype(F3)


def base_trace(case: dict, n_cols: int) -> Array:
    """The stage-1 base trace `(N, n_cols)` from a golden case's dim-1 `cm` columns
    (column id == index), as an `F` array — the input rw's `eval_constraints` and
    the interaction `VirtualPairCol`s index into."""
    cols = {c["id"]: c["values"] for c in case["cm"] if c["dim"] == 1}
    trace = np.stack(
        [np.array([int(v) for v in cols[j]], dtype=np.uint64) for j in range(n_cols)],
        axis=1,
    )
    return fnp.array(trace, dtype=F)
