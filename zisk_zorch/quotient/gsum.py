"""pil2-stark's LogUp grand-sum (std_sum) witness — the stage-2 bus argument.

ZisK proves its lookup/permutation buses with a log-derivative (LogUp) argument:
each row contributes `sum_i mult_i / den_i`, where `den_i` is the bus tuple
combined into a single cubic value, and the committed `gsum` column is the
running prefix sum of those local terms (pil2's `calculateWitnessSTD(prod=false)`
→ hint `gsum_col`). The boundary closes via the `__L1__'` constraint tying the
airgroup grand-sum result to the last row.

This module builds the two primitives the witness needs: the per-tuple
denominator (Horner in `std_alpha`, `+ std_gamma`) and the prefix-sum grand-sum.
The committed `gsum` column then feeds both the stage-2 commitment and the
quotient composite's bus / running-sum constraints (see
docs/stage2-constraint-ingest.md, quotient.py). Host-driven and un-jitted like
the rest of the proof orchestration.

std_sum driver: https://github.com/0xPolygonHermez/pil2-proofman/blob/v0.18.0/pil2-stark/src/starkpil/gen_proof.hpp#L23-L64
gsum/im hints:  https://github.com/0xPolygonHermez/pil2-proofman/blob/v0.18.0/pil2-stark/src/starkpil/hints.cpp
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array
from zk_dtypes import goldilocksx3_mont as F3


def bus_denominator(tuple_: Array, alpha: Array, gamma: Array) -> Array:
    """The LogUp bus denominator for one interaction: Horner in `alpha` over the
    tuple components, then `+ gamma`.

    `tuple_` is `[..., T]` cubic (the bus-tuple columns, `tuple_[..., 0]` the
    highest `alpha` power — pil2's order); `alpha`, `gamma` are cubic scalars.
    Returns the `[...]` cubic denominator.
    """
    den = tuple_[..., 0]
    for k in range(1, tuple_.shape[-1]):
        den = den * alpha + tuple_[..., k]
    return den + gamma


def grand_sum(numerators: Array, denominators: Array) -> Array:
    """The committed `gsum` column: the running prefix sum of each row's local
    term `sum_i numerator_i * denominator_i^-1`.

    `numerators`, `denominators` are `[N, I]` cubic (N rows, I interactions);
    returns the `[N]` cubic grand-sum. Row 0 is the raw local term (pil2's
    `gsum[0]`); the last entry is the airgroup `gsum_result` (modulo pil2's
    single-row direct update, handled by the caller).
    """
    # The zkx CPU emitter only handles elementwise cubic ops — `jnp.sum`/
    # `jnp.cumsum` (bitcast-expand), `lax.associative_scan` (interior padding),
    # and cubic matmul all crash. So fold the (small, static) interaction axis by
    # hand and take the prefix sum with a Hillis-Steele scan built from the ops
    # that do survive: slice, concatenate, add.
    ratio = numerators / denominators
    local = ratio[:, 0]
    for i in range(1, ratio.shape[1]):
        local = local + ratio[:, i]
    return _prefix_sum(local)


def _prefix_sum(x: Array) -> Array:
    """Inclusive prefix sum over axis 0 via a Hillis-Steele scan (log-depth, only
    slice/concat/add — the cubic ops the zkx CPU emitter supports)."""
    n = x.shape[0]
    acc = x
    shift = 1
    while shift < n:
        pad = jnp.zeros(shift, dtype=F3)
        acc = acc + jnp.concatenate([pad, acc[:-shift]])
        shift *= 2
    return acc
