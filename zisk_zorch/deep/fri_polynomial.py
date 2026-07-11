"""DEEP FRI polynomial — pil2-stark's `calculateFRIPolynomial`.

pil2 squeezes the OOD point `z`, opens every committed polynomial there
(`opening.py`), absorbs the openings, squeezes a batching challenge, then
evaluates the compiled `friExp` expression over the extended domain to get the
codeword FRI folds. `friExp` is the standard DEEP-ALI batched quotient

    f(x) = Σ_m  vf^m · (p_m(x) − p_m(ξ_{o(m)})) / (x − ξ_{o(m)})

each summand a genuine polynomial (the numerator vanishes at the opening point,
so the division is exact), so `f` has degree `< N` and FRI can fold it. `x` is
the extended coset (`zerofier._coset_points`), `ξ_o = z·g^{opening}` the OOD
points (no coset shift, unlike LEv), and `vf` the squeezed batching challenge.

**Byte-match boundary.** `friExp` in a real proving key bakes in *which* columns
are batched, their order, and each one's challenge power (from `expressions.bin`);
`deep_composition` implements the generic formula over the columns it is handed.
Matching a specific AIR's `friExp` byte-for-byte needs that compiled op list (the
same machinery `cexp_ref` interprets for the quotient) and a pil2 golden — a
later slice. This module is verified by the FRI low-degree property instead: a
correctly-opened `f` folds to a low-degree final polynomial; a wrong opening does
not.

https://github.com/0xPolygonHermez/pil2-proofman/blob/v1.0.0-alpha/pil2-stark/src/starkpil/starks.hpp
"""

from __future__ import annotations

import functools
from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from zisk_zorch.deep.opening import _cubic_powers, compute_lev, open_columns
from zisk_zorch.fri.seam import _base_to_cubic, _cubic_to_base
from zisk_zorch.quotient.zerofier import _ONE, _root

_GOLDILOCKS_P = 0xFFFFFFFF00000001
_COSET_SHIFT = 7  # Goldilocks::SHIFT, == zerofier._SHIFT
_TWO_ADIC_ROOT = 7277203076849721926  # Goldilocks::W[32], == zerofier._TWO_ADIC_ROOT


def _embed_base(col: Array) -> Array:
    """A base-field column as cubic elements `(b, 0, 0)` — the embedding under
    which base×cubic is exact scalar multiplication (cf. `cexp_ref`)."""
    zero = jnp.zeros_like(col)
    return _base_to_cubic(jnp.stack([col, zero, zero], axis=-1).reshape(-1))


@functools.lru_cache(maxsize=None)
def _coset_cubic(n_bits: int, blowup_bits: int) -> Array:
    """The extended coset `x[i] = SHIFT · w(nBitsExt)^i`, embedded to cubic —
    byte-identical to `_embed_base(zerofier._coset_points(...))`, but built on the
    host so it does NOT lower to `zerofier._powers`' `2^nBitsExt`-deep unrolled
    `jnp` running product. Under jit that unroll (plus the downstream cubic
    reciprocal in `deep_composition`) overwhelms the XLA NVPTX layout pass and
    aborts compilation; a host-materialised coset compiles in ~0.1s. Cached since
    it is fixed by `(n_bits, blowup_bits)`."""
    n_ext = 1 << (n_bits + blowup_bits)
    w = pow(_TWO_ADIC_ROOT, 1 << (32 - (n_bits + blowup_bits)), _GOLDILOCKS_P)
    out = [1] * n_ext
    out[0] = _COSET_SHIFT % _GOLDILOCKS_P
    for i in range(1, n_ext):
        out[i] = out[i - 1] * w % _GOLDILOCKS_P
    base = jnp.array(np.array(out, dtype=object).astype(np.uint64), dtype=_ONE.dtype)
    return _embed_base(base)


def _ood_points(z: Array, opening_points: Sequence[int], n_bits: int) -> Array:
    """`ξ_o = z·g^{opening}` (`g = W[nBits]`, negative openings invert) — the OOD
    points the composition divides by. No coset shift (that is LEv's, not this)."""
    zc = _base_to_cubic(z).reshape(())
    g = _root(n_bits)
    xis = []
    for p in opening_points:
        w = jnp.power(g, abs(p))
        if p < 0:
            w = _ONE / w
        xis.append(zc * w)
    return jnp.stack(xis)


def deep_composition(
    columns_ext: Array,
    evals: Array,
    xis: Array,
    opening_pos: Sequence[int],
    vf: Array,
    *,
    n_bits: int,
    blowup_bits: int,
) -> Array:
    """`f(x) = Σ_m vf^m·(col_m(x) − eval_m)/(x − ξ_{opening_pos[m]})` on the
    extended coset. `columns_ext` is `(2^nBitsExt, M)` cubic, `evals`/`xis` cubic,
    `vf` a cubic scalar. Returns the `(2^nBitsExt,)` cubic FRI codeword."""
    m = columns_ext.shape[1]
    if evals.shape[0] != m or len(opening_pos) != m:
        raise ValueError(
            f"evals ({evals.shape[0]}) and opening_pos ({len(opening_pos)}) must "
            f"match the {m} columns"
        )
    # Host-built coset, then an optimization barrier so XLA treats it as a
    # runtime value rather than a compile-time constant: a *constant* cubic array
    # feeding the reciprocal below crashes the NVPTX layout pass (both an unrolled
    # `jnp` coset and a materialised literal do), while a barriered value compiles.
    x = jax.lax.optimization_barrier(_coset_cubic(n_bits, blowup_bits))  # (N_ext,) cubic
    xis_per_col = xis[jnp.array(opening_pos)]  # (M,) cubic
    denom = x[:, None] - xis_per_col[None, :]  # (N_ext, M) cubic
    numer = columns_ext - evals[None, :]  # (N_ext, M) cubic
    vf_powers = _cubic_powers(vf, m)  # (M,) cubic
    return jnp.sum((numer / denom) * vf_powers[None, :], axis=1)


def _committed_columns(trace_ext: Array, quotient: Array) -> Array:
    """The committed cubic columns the DEEP opens: each extended trace column
    embedded as cubic, then the cubic quotient. `(2^nBitsExt, n_cols + 1)`."""
    cols = [_embed_base(trace_ext[:, c]) for c in range(trace_ext.shape[1])]
    cols.append(quotient)
    return jnp.stack(cols, axis=1)


def make_deep_combiner(opening_points: Sequence[int] = (0,)):
    """A `prove_inner` `fri_polynomial_fn` that runs the real DEEP flow, threading
    the transcript exactly as pil2's `genProof`: squeeze the OOD `z`, open the
    committed columns, absorb the openings, squeeze the batching challenge `vf`,
    and build `f`. `opening_points` are the AIR's wrapped opening shifts (default
    `(0,)` = open at `z` only)."""

    def combine(ctx) -> Array:
        columns = _committed_columns(ctx.trace.extended, ctx.quotient)
        opening_pos = [0] * columns.shape[1]  # all at z; wrapped openings are AIR-specific
        z = ctx.transcript.get_field()  # OOD point (pil2 stage nStages+2, stageId 0)
        lev = compute_lev(z, opening_points, ctx.n_bits)
        evals = open_columns(
            columns, lev, opening_pos, n_bits=ctx.n_bits, blowup_bits=ctx.blowup_bits
        )
        ctx.transcript.put(_cubic_to_base(evals))  # absorb openings
        vf = _base_to_cubic(ctx.transcript.get_field()).reshape(())  # batching challenge
        xis = _ood_points(z, opening_points, ctx.n_bits)
        return deep_composition(
            columns, evals, xis, opening_pos, vf,
            n_bits=ctx.n_bits, blowup_bits=ctx.blowup_bits,
        )

    return combine


# Default DEEP combiner: single opening at z.
deep_fri_polynomial = make_deep_combiner()
