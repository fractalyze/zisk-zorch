"""FRI fold — pil2-stark's `FRI::fold` on zorch's univariate machinery.

One fold step collapses a cubic-extension codeword from the previous coset
domain (size `2^prev_bits`) to the next layer (size `2^current_bits`) at a
cubic challenge. Output group `g` reads the `n_x = 2^(prev_bits - current_bits)`
codeword entries strided by `pol2n = 2^current_bits` — pol2's `ppar[j] =
pol[j*pol2n + g]` — which are the values of the codeword's restriction to the
coset `shift_eff * w(prev_bits)^(g + j*pol2n)`. pil2 recovers that degree-`n_x`
restriction by an INTT, undoes the coset with a per-coefficient rescale, then
evaluates at the challenge (`FRI::fold` for the prover, `verify_fold` for one
queried index — same arithmetic).

The interpolant through `n_x` distinct coset points is unique, so its value at
the challenge is one field element however it is computed: here it is a plain
Lagrange interpolation over the explicit coset evaluated at the challenge
(zorch's `fri_fold_k_values` k-ary fold kernel), byte-identical to pil2's
INTT-then-rescale by
construction and free of any NTT root-order convention. The coset points are
fixed by the static `(n_bits_ext, prev_bits, current_bits)`, so they are built
once on the host as a base-field constant.

https://github.com/0xPolygonHermez/pil2-proofman/blob/v0.18.0/pil2-stark/src/starkpil/fri/fri.hpp#L36-L113
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from zk_dtypes import goldilocks_mont as F

from zorch.coding.reed_solomon import fri_fold_k_values

# Goldilocks field modulus and the LDE coset generator (`Goldilocks::SHIFT`).
_GOLDILOCKS_P = 0xFFFFFFFF00000001
_COSET_SHIFT = 7

# pil2-stark's two-adic generator `Goldilocks::W[32]` (order 2^32); `W[bits]`,
# the 2^bits-th root, is `W[32]^(2^(32 - bits))`. Same element pil2 folds on,
# so the coset points match without any zk/pil2 root reindex (cf.
# zisk_zorch.commit.trace_commit, which bridges the native NTT's other root).
_TWO_ADIC_ROOT = 7277203076849721926


def _powers(base: int, count: int) -> np.ndarray:
    """`[base^0, base^1, ..., base^(count-1)] mod p` as an object-dtype array."""
    out = [1] * count
    for k in range(1, count):
        out[k] = out[k - 1] * base % _GOLDILOCKS_P
    return np.array(out, dtype=object)


def _coset_domain(n_bits_ext: int, prev_bits: int, current_bits: int) -> Array:
    """The per-group coset points as a `(2^current_bits, n_x)` base-field array:
    row `g` holds `shift_eff * w(prev_bits)^(g + j*cur_n)` for `j` in `[0, n_x)`,
    where `shift_eff = SHIFT^(2^(n_bits_ext - prev_bits))` is the previous
    layer's coset shift and `cur_n = 2^current_bits`."""
    cur_n = 1 << current_bits
    n_x = 1 << (prev_bits - current_bits)

    shift_eff = pow(_COSET_SHIFT, 1 << (n_bits_ext - prev_bits), _GOLDILOCKS_P)
    w = pow(_TWO_ADIC_ROOT, 1 << (32 - prev_bits), _GOLDILOCKS_P)

    # w^(g + j*cur_n) = w^g * (w^cur_n)^j, so the full 2^prev_bits power table is
    # the outer product of two short runs of lengths cur_n and n_x.
    col = _powers(w, cur_n)
    row = _powers(pow(w, cur_n, _GOLDILOCKS_P), n_x)
    canonical = (shift_eff * col[:, None] * row[None, :]) % _GOLDILOCKS_P
    return jnp.array(canonical.astype(np.uint64), dtype=F)


def fold(
    pol: Array, challenge: Array, n_bits_ext: int, prev_bits: int, current_bits: int
) -> Array:
    """Fold the cubic codeword `pol` (length `2^prev_bits`) to length
    `2^current_bits` at the cubic `challenge`, matching pil2's `FRI::fold`.

    A composer over the jitted Lagrange-basis kernel, so itself un-jitted (the
    coset domain is a static constant). `n_bits_ext` is the full extended
    domain's log size, fixing the previous layer's coset shift."""
    if not 0 <= current_bits < prev_bits <= n_bits_ext <= 32:
        raise ValueError(
            "need 0 <= current_bits < prev_bits <= n_bits_ext <= 32, got "
            f"current_bits={current_bits}, prev_bits={prev_bits}, "
            f"n_bits_ext={n_bits_ext}"
        )
    if pol.shape != (1 << prev_bits,):
        raise ValueError(f"pol must have shape {(1 << prev_bits,)}, got {pol.shape}")
    if challenge.shape != ():
        raise ValueError(f"challenge must be a scalar, got shape {challenge.shape}")

    cur_n = 1 << current_bits
    n_x = 1 << (prev_bits - current_bits)

    domain = _coset_domain(n_bits_ext, prev_bits, current_bits)
    # group[g, j] = pol[j*cur_n + g] — the n_x entries fold reads for group g.
    group = pol.reshape(n_x, cur_n).T

    def fold_group(values: Array, points: Array) -> Array:
        return fri_fold_k_values(values, challenge, points)

    return jax.vmap(fold_group)(group, domain)


def intt(evals: Array, n_bits: int) -> Array:
    """Inverse NTT of a `(2^n_bits, n_cols)` base-field evaluation matrix over the
    order-`2^n_bits` subgroup at pil2's root `W[n_bits]`, returning coefficients
    in natural order (`coeff[k]` is the `x^k` coefficient) — pil2's
    `NTT_Goldilocks::INTT` applied per column.

    An explicit Vandermonde-inverse matmul (`coeff[k] = N^-1 * sum_j evals[j] *
    W^-jk`): the final FRI polynomial the low-degree test runs on is tiny
    (`n_bits` is the last fold step), so the O(n^2) matrix form is fine and stays
    free of the zk<->pil2 root reindex the LDE's native NTT needs (cf.
    `zisk_zorch.commit.trace_commit`, which folds the same root mismatch into a
    gather). The subgroup-only INTT — no coset rescale — mirrors pil2, which
    INTTs the in-clear final pol on the plain subgroup; a coset only rescales
    coefficients, so it leaves the low-degree test's vanishing set unchanged."""
    n = 1 << n_bits
    if evals.ndim != 2 or evals.shape[0] != n:
        raise ValueError(f"evals must be (2^{n_bits}, n_cols) = ({n}, *), got {evals.shape}")

    w = pow(_TWO_ADIC_ROOT, 1 << (32 - n_bits), _GOLDILOCKS_P)
    w_inv = pow(w, -1, _GOLDILOCKS_P)
    n_inv = pow(n, -1, _GOLDILOCKS_P)

    # mat[k, j] = N^-1 * W^-(j*k mod N); the j*k table indexes one short run of
    # inverse-root powers, so the (n, n) matrix is a single gather.
    exps = (np.arange(n)[:, None] * np.arange(n)[None, :]) % n
    powers = _powers(w_inv, n)
    mat = (n_inv * powers[exps]) % _GOLDILOCKS_P
    m = jnp.array(mat.astype(np.uint64), dtype=F)
    # coeff[k, c] = sum_j mat[k, j] * evals[j, c].
    return jnp.sum(m[:, :, None] * evals[None, :, :], axis=1)


def verify_fold(
    values: Array,
    challenge: Array,
    n_bits_ext: int,
    prev_bits: int,
    current_bits: int,
    idx: int,
) -> Array:
    """pil2's `FRI::verify_fold` — fold one queried group's `n_x` cubic `values`
    to a single cubic value at `challenge`. The verifier's per-query counterpart
    to `fold`: the same coset Lagrange interpolation, for output group `idx`
    (so `verify_fold(pol's group idx, ...) == fold(pol, ...)[idx]`)."""
    if not 0 <= current_bits < prev_bits <= n_bits_ext <= 32:
        raise ValueError(
            "need 0 <= current_bits < prev_bits <= n_bits_ext <= 32, got "
            f"current_bits={current_bits}, prev_bits={prev_bits}, "
            f"n_bits_ext={n_bits_ext}"
        )
    n_x = 1 << (prev_bits - current_bits)
    if values.shape != (n_x,):
        raise ValueError(f"values must have shape {(n_x,)}, got {values.shape}")
    if challenge.shape != ():
        raise ValueError(f"challenge must be a scalar, got shape {challenge.shape}")

    points = _coset_domain(n_bits_ext, prev_bits, current_bits)[idx]
    return fri_fold_k_values(values, challenge, points)
