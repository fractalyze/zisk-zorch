"""DEEP FRI polynomial — pil2-stark's `calculateFRIPolynomial`.

pil2 squeezes the OOD point `z`, opens every committed polynomial there
(`zorch.pcs.deep.open_columns` over pil2's LEv weights), absorbs the openings,
squeezes a batching challenge, then evaluates the compiled `friExp` expression
over the extended domain to get the codeword FRI folds. `friExp` is the standard
DEEP-ALI batched quotient

    f(x) = Σ_m  vf^m · (p_m(x) − p_m(ξ_{o(m)})) / (x − ξ_{o(m)})

each summand a genuine polynomial (the numerator vanishes at the opening point,
so the division is exact), so `f` has degree `< N` and FRI can fold it. `x` is
the extended coset (`zerofier._coset_points`), `ξ_o = z·g^{opening}` the OOD
points (no coset shift, unlike LEv), and `vf` the squeezed batching challenge.

The batched quotient and the opening are `zorch.pcs.deep`'s scheme-neutral
`deep_composition` / `open_columns`; this module supplies pil2's coset, root, LEv
(`compute_lev`), and transcript sequencing. A real proving key's `friExp` also
fixes *which* columns are batched, their order, and each one's challenge power
(`docs/architecture.md`).

https://github.com/0xPolygonHermez/pil2-proofman/blob/v1.0.0-alpha/pil2-stark/src/starkpil/starks.hpp
"""

from __future__ import annotations

from collections.abc import Sequence

import frx
import frx.numpy as fnp
from frx import Array
from zk_dtypes import goldilocksx3 as F3

from zorch.pcs.deep import deep_composition, open_columns
from zorch.utils.field import join_coeffs, split_coeffs

from zisk_zorch.evals.lev import compute_lev
from zisk_zorch.quotient.zerofier import _coset_points, _root

# Jitted here, not run eager: per-column dispatch is ~85% of both stages' wall
# (opening 28.2 → 3.6 ms, composition 58.0 → 9.2 ms at the wired 2^22 shape).
# This does not re-trip #67 — its trigger is a coset *built inside* the trace,
# and both cosets here (`lev`, `domain`) enter as inputs. `opening_pos` must be
# a tuple: static args are hashed.
_open_columns = frx.jit(open_columns, static_argnames=("opening_pos", "stride"))
_deep_composition = frx.jit(deep_composition, static_argnames=("opening_pos",))


def _ood_points(z: Array, opening_points: Sequence[int], n_bits: int) -> Array:
    """`ξ_o = z·g^{opening}` (`g = W[nBits]`, negative openings invert) — the OOD
    points the composition divides by. No coset shift (that is LEv's, not this)."""
    zc = join_coeffs(z.reshape(-1, 3), F3).reshape(())
    g = _root(n_bits)
    return fnp.stack([zc * fnp.power(g, p) for p in opening_points])


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
    opening_pos = (0,) * m  # all at z; wrapped openings are AIR-specific
    z = transcript.get_field()  # OOD point (pil2 stage nStages+2, stageId 0)
    zc = join_coeffs(z.reshape(-1, 3), F3).reshape(())
    lev = compute_lev(zc, list(opening_points), n_bits)
    evals = _open_columns(
        base_cols, cubic_cols, lev, opening_pos, stride=1 << blowup_bits
    )
    transcript.put(split_coeffs(evals).reshape(-1))  # absorb openings
    # batching challenge
    vf = join_coeffs(transcript.get_field().reshape(-1, 3), F3).reshape(())
    xis = _ood_points(z, opening_points, n_bits)
    domain = _coset_points(n_bits, blowup_bits)  # (N_ext,) base — DEEP divides on it
    f = _deep_composition(base_cols, cubic_cols, evals, xis, opening_pos, vf, domain)
    return f, evals
