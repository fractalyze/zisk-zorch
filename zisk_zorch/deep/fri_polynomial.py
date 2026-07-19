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

`deep_composition` implements the generic formula over the columns it is handed;
a real proving key's `friExp` also fixes *which* columns are batched, their order,
and each one's challenge power (`docs/architecture.md`).

https://github.com/0xPolygonHermez/pil2-proofman/blob/v1.0.0-alpha/pil2-stark/src/starkpil/starks.hpp
"""

from __future__ import annotations

from collections.abc import Sequence

import frx.numpy as jnp
from frx import Array

from zorch.poly.univariate import powers

from zisk_zorch.deep.opening import open_columns
from zisk_zorch.evals.lev import compute_lev
from zisk_zorch.fri.seam import _base_to_cubic, _cubic_to_base
from zisk_zorch.quotient.zerofier import _coset_points, _root


def _ood_points(z: Array, opening_points: Sequence[int], n_bits: int) -> Array:
    """`ξ_o = z·g^{opening}` (`g = W[nBits]`, negative openings invert) — the OOD
    points the composition divides by. No coset shift (that is LEv's, not this)."""
    zc = _base_to_cubic(z).reshape(())
    g = _root(n_bits)
    return jnp.stack([zc * jnp.power(g, p) for p in opening_points])


def deep_composition(
    base_cols: Array,
    cubic_cols: Array,
    evals: Array,
    xis: Array,
    opening_pos: Sequence[int],
    vf: Array,
    *,
    n_bits: int,
    blowup_bits: int,
) -> Array:
    """`f(x) = Σ_m vf^m·(col_m(x) − eval_m)/(x − ξ_{opening_pos[m]})` on the
    extended coset. The committed columns arrive split by field: `base_cols`
    (`(N_ext, B)` base) then `cubic_cols` (`(N_ext, C)` cubic), in that batching
    order, so `vf^m`/`evals[m]` index base columns for `m < B` and cubic after.
    Keeping base columns base is the point (#69): pil2 reads dim-1 columns as 8 B,
    where embedding them to cubic up front would read (and materialize) 3×.

    Two structural choices keep it off HBM: the summand is accumulated
    column-by-column, so no `(N_ext, M)` cubic intermediate is ever built; and
    `vf^m·(col_m − eval_m)` is grouped by opening point, so each distinct `ξ`
    costs one cubic reciprocal (one in the wired flow — every column opens at
    `z`). Field arithmetic is exact, so this is byte-identical to the per-column
    `Σ (col−eval)/(x−ξ)` form. Returns the `(N_ext,)` cubic FRI codeword."""
    b, c = base_cols.shape[1], cubic_cols.shape[1]
    m = b + c
    if evals.shape[0] != m or len(opening_pos) != m:
        raise ValueError(
            f"evals ({evals.shape[0]}) and opening_pos ({len(opening_pos)}) must "
            f"match the {m} committed columns ({b} base + {c} cubic)"
        )
    x = _coset_points(n_bits, blowup_bits)  # (N_ext,) base — promotes below
    vfp = powers(vf, m)  # (M,) cubic

    numer_by_opening: dict[int, Array] = {}
    for col_m in range(m):
        col = base_cols[:, col_m] if col_m < b else cubic_cols[:, col_m - b]
        term = vfp[col_m] * (col - evals[col_m])  # (N_ext,) cubic; base−cubic ok
        o = opening_pos[col_m]
        numer_by_opening[o] = (
            term if o not in numer_by_opening else numer_by_opening[o] + term
        )
    f: Array | None = None
    for o, numer in numer_by_opening.items():
        term = numer / (x - xis[o])
        f = term if f is None else f + term
    return f


def _committed_columns(trace_ext: Array, quotient: Array) -> tuple[Array, Array]:
    """The committed columns the DEEP opens, split by field so DEEP reads base
    columns as 8 B not 24 B (#69): the extended trace kept **base**
    (`(N_ext, n_cols)`), and the cubic quotient (`(N_ext, 1)`). Batching order is
    base-then-cubic — `deep_composition` and `open_columns` follow it."""
    return trace_ext, quotient[:, None]


def deep_fri_polynomial(
    trace_ext: Array,
    quotient: Array,
    transcript,
    *,
    n_bits: int,
    blowup_bits: int,
    opening_points: Sequence[int] = (0,),
) -> tuple[Array, Array]:
    """The real DEEP flow, threading the transcript exactly as pil2's `genProof`:
    squeeze the OOD `z`, open the committed columns, absorb the openings, squeeze
    the batching challenge `vf`, and build `f`. `trace_ext` is the extended trace
    and `quotient` the cubic quotient codeword (the committed columns);
    `opening_points` are the AIR's wrapped opening shifts (default `(0,)` = open
    at `z` only).

    Returns `(f, evals)`. The openings travel in the proof (pil2's `evals`
    section) because the transcript absorbs them before squeezing `vf`: a
    verifier replaying the transcript cannot recompute them — it has no trace —
    so without them every later challenge diverges. `z` is not returned; it is
    squeezed, so the verifier re-derives it."""
    base_cols, cubic_cols = _committed_columns(trace_ext, quotient)
    m = base_cols.shape[1] + cubic_cols.shape[1]
    opening_pos = [0] * m  # all at z; wrapped openings are AIR-specific
    z = transcript.get_field()  # OOD point (pil2 stage nStages+2, stageId 0)
    lev = compute_lev(_base_to_cubic(z).reshape(()), list(opening_points), n_bits)
    evals = open_columns(
        base_cols, cubic_cols, lev, opening_pos, n_bits=n_bits, blowup_bits=blowup_bits
    )
    transcript.put(_cubic_to_base(evals))  # absorb openings
    vf = _base_to_cubic(transcript.get_field()).reshape(())  # batching challenge
    xis = _ood_points(z, opening_points, n_bits)
    f = deep_composition(
        base_cols, cubic_cols, evals, xis, opening_pos, vf,
        n_bits=n_bits, blowup_bits=blowup_bits,
    )
    return f, evals
