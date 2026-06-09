"""Loaders for the golden/ JSON fixtures (canonical-u64 decimal strings)."""

from __future__ import annotations

import json
import pathlib

import jax.numpy as jnp
import numpy as np
from jax import Array
from zk_dtypes import goldilocks_mont as F


def load(path: pathlib.Path) -> dict:
    return json.loads(path.read_text())


def u64(values: list[str]) -> Array:
    """Decimal-string canonical u64s -> a 1-D goldilocks_mont array."""
    return jnp.array(np.array([int(v) for v in values], dtype=np.uint64), dtype=F)
