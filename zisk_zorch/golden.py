"""Loaders for the golden/ JSON fixtures (canonical-u64 decimal strings)."""

from __future__ import annotations

import json
import pathlib

import jax.numpy as jnp
import numpy as np
from jax import Array
from zk_dtypes import goldilocks_mont as F
from zk_dtypes import goldilocksx3_mont as F3


def load(path: pathlib.Path) -> dict:
    return json.loads(path.read_text())


def u64(values: list[str]) -> Array:
    """Decimal-string canonical u64s -> a 1-D goldilocks_mont array."""
    return jnp.array(np.array([int(v) for v in values], dtype=np.uint64), dtype=F)


def u64x3(values: list[str]) -> Array:
    """Flat decimal-string canonical limbs (3 per cubic element) -> a 1-D
    goldilocksx3_mont array. Each element's limbs value-convert to montgomery
    in the base field, then the contiguous limb triples view as one cubic."""
    flat = np.array([int(v) for v in values], dtype=np.uint64).reshape(-1, 3)
    base = flat.astype(F)
    return jnp.array(base.view(F3).reshape(flat.shape[0]))
