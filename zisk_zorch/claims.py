"""The claims crossing the inner proof's stage seams, plus the top witness.

A claim holds only what both roles can derive — the verifier from the wire,
the prover from its own run — never prover-only data; each stage's roles
(`quotient/`, `opening/`) consume and produce these, and the composites in
`prover.py` / `verifier.py` construct the bound forms after the shared
transcript operations. They live here, below every stage, so a stage never
imports the composite that runs it.
"""

from __future__ import annotations

from dataclasses import dataclass

from frx import Array


@dataclass(frozen=True, kw_only=True)
class InnerClaim:
    """Some trace of this shape satisfies the AIR.

    Spelled out: there exists a ``(2**n_bits, n_cols)`` base-field trace on
    which every one of the AIR's `n_constraints` constraints vanishes on every
    row. Nothing here names that trace — it is existentially quantified, and
    the prover exhibits one by committing to it, which is why the trace root
    is proof data rather than a field of the statement. The AIR's circuits
    (`eval_fn`) stay static configuration on the roles — both are built
    against the same AIR — while the statement instance's shape lives here,
    where both read it: the prover cross-checks its witness against it, the
    verifier sizes the alpha fold and the openings by it.
    """

    n_bits: int
    n_cols: int
    n_constraints: int


@dataclass(frozen=True)
class InnerWitness:
    """The trace that makes an `InnerClaim` true: the ``(2**n_bits, n_cols)``
    base-field evaluation matrix."""

    trace: Array


@dataclass(frozen=True, kw_only=True)
class TraceBoundClaim:
    """The trace committed under `trace_root` satisfies `inner`'s AIR.

    What `InnerClaim` becomes once the opening scheme's commit half runs and
    the composite binds the root: the existential is discharged — one concrete
    trace is now named by its commitment. Both roles derive it, the prover
    from committing, the verifier from the root on the wire, which is why the
    composite (not a stage) constructs it after `bind_trace_commitment`.
    """

    inner: InnerClaim
    trace_root: Array


@dataclass(frozen=True, kw_only=True)
class QuotientBoundClaim:
    """The codeword committed under `quotient_root` is the `alpha`-fold of
    `inner`'s AIR constraints on the trace committed under `trace_root`,
    divided by the zerofier.

    What the quotient stage reduces the AIR statement to: the division is
    exact only when the alpha-folded constraints vanish on the whole base
    domain, so a single violated row makes the true quotient a non-polynomial
    no committed codeword can equal at the opening's out-of-domain point.
    `inner` is the source statement the reduction conditions on — the opening
    still needs its shape to size what it opens. Both roles hold the roots —
    the prover from committing, the verifier off the wire — so they are claim
    data: they name what the opening is checked against. `alpha` (the folding
    challenge's power vector) is likewise derived by both roles from the
    transcript; the opening's verifier folds the out-of-domain constraint
    check with it. The fields are keyword-only so the two same-shaped roots
    cannot be passed to each other's slot.
    """

    inner: InnerClaim
    trace_root: Array
    quotient_root: Array
    alpha: Array
