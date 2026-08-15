"""The inner proof's shared data types: the statement, its witnesses, the
claims crossing the stage seams, and the commit-data handoffs.

Everything here is data more than one package reads — a stage's roles
(`quotient/`, `opening/`) consume and produce these, and the composites in
`prover.py` / `verifier.py` construct the bound forms after the shared
transcript operations — so it sits below every stage and a stage never
imports the composite that runs it — the proof types included, since a
proof one role produces is what the other role and the wire consume.

Two lines the docstrings below keep drawing:

- **Claim data vs configuration.** A claim holds only what both roles can
  derive — the verifier from the wire, the prover from its own run. What is
  fixed for every instance (the AIR's circuits, arity, the fold schedule) is
  role configuration instead.
- **Claim data vs prover data.** Commitments' wire projection (a root) is
  claim data; the matrices and digest layers behind it are prover data,
  riding commitment/witness types the composite hands between stages.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
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


@dataclass(frozen=True)
class TraceCommitment:
    """What the opening scheme's commit half hands forward: the 4-element
    root (the wire projection the composite binds), the digest layers the
    query phase opens with, and the extended matrix every later stage
    evaluates or opens against — the latter two prover data."""

    root: Array
    digest_layers: list[Array]
    extended: Array


@dataclass(frozen=True)
class QuotientCommitment:
    """The quotient stage's reduction proof, the quotient analogue of
    ``TraceCommitment``. Its wire projection is `root`; the cubic codeword the
    DEEP batch opens and the base-limb `matrix`/`layers` the query phase opens
    the committed tree with ride along as prover data — the quotient commits
    mid-transcript (alpha precedes it), so unlike a PCS trace commit this
    cannot be split into a pre-transcript commit half."""

    codeword: Array
    root: Array
    matrix: Array
    layers: list[Array]


@dataclass(frozen=True)
class OpeningWitness:
    """What discharging a `QuotientBoundClaim` takes: both committed trees'
    prover data — the extended trace and quotient codeword the DEEP batch
    reads, and the digest layers the query phase opens."""

    trace_commit: TraceCommitment
    quotient: QuotientCommitment


@dataclass(frozen=True)
class FriProof:
    """The fold loop's wire data: the per-layer roots (transcript order) and
    the final polynomial sent in clear."""

    roots: list[Array]
    final_pol: Array


@dataclass(frozen=True)
class OpeningProof:
    """Discharges a `QuotientBoundClaim`, leaving nothing to prove: the
    out-of-domain column openings (`evals`; None under `EchoOpening`, which
    opens nothing), the FRI fold proof, the grinding nonce, the squeezed
    query positions, and every committed tree's per-query openings."""

    evals: Array | None
    fri: FriProof
    nonce: int
    positions: np.ndarray
    trace_openings: list[list[np.ndarray]]
    quotient_openings: list[list[np.ndarray]]
    fri_openings: list[list[np.ndarray]]


@dataclass(frozen=True)
class InnerProof:
    """What a verifier needs to check an `InnerClaim` without the trace: the
    per-stage roots the transcript absorbed, then the opening stage's
    discharge as its own section. The quotient stage's reduction proof
    carries prover data, so only its root goes on the wire."""

    trace_root: Array
    quotient_root: Array
    opening: OpeningProof
