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
    trace_openings: list[list[Array]]
    quotient_openings: list[list[Array]]
    fri_openings: list[list[Array]]


@dataclass(frozen=True)
class InnerProof:
    """What a verifier needs to check an `InnerClaim` without the trace: the
    per-stage roots the transcript absorbed, then the opening stage's
    discharge as its own section. The quotient stage's reduction proof
    carries prover data, so only its root goes on the wire."""

    trace_root: Array
    quotient_root: Array
    opening: OpeningProof


@dataclass(frozen=True, kw_only=True)
class Pil2Claim:
    """A pil2 genProof instance's statement: some trace of this shape,
    together with these scalar sections, satisfies the proving key's AIR.

    The pil2-mode claims mirror the wired family above but condition on the
    proving key (the roles' `Pil2Key` configuration) instead of an `eval_fn`.
    The scalar sections ride as the dumped u64 word layouts (`pil2.values_env`
    owns the packing) because both roles read them that way: the prover packs
    them into interpreter environments and transcript absorbs, the verifier
    reads the same words off the wire. `global_challenge` is the
    contributions-phase seed — the transcript arrives already bound to every
    instance's stage-1 root through it, which is why no pil2 claim carries a
    bound trace root the way `TraceBoundClaim` does: the composite still
    commits the trace, but the schedule never absorbs `root1`."""

    n_bits: int
    n_cols: int
    publics: np.ndarray
    airvalues: np.ndarray
    proofvalues: np.ndarray
    global_challenge: np.ndarray


@dataclass(frozen=True, kw_only=True)
class Pil2TraceBoundClaim:
    """`Pil2Claim` after the commit half runs: one concrete trace is named by
    `trace_root`. The root is claim data for the wire even though the
    non-recursive schedule never absorbs it (see `Pil2Claim`)."""

    pil2: Pil2Claim
    trace_root: Array


@dataclass(frozen=True, kw_only=True)
class Stage2BoundClaim:
    """The stage-2 witness-STD columns committed under `root2` chain from the
    trace committed under `trace_root` via the key's hint expressions.

    `challenges` carries every challenge squeezed so far, keyed by its
    ``challengesMap`` index — the id the later stages' SSA operands reference.
    Both roles derive them from the transcript, so they are claim data, like
    `alpha` on `QuotientBoundClaim`. `airgroupvalues` is the stage's other
    product (the grand-sum result the last gsum row settles), transmitted on
    the wire and read back by the quotient's SSA."""

    pil2: Pil2Claim
    trace_root: Array
    root2: Array
    challenges: dict[int, Array]
    airgroupvalues: dict[int, Array]


@dataclass(frozen=True, kw_only=True)
class Pil2QuotientBoundClaim:
    """The codeword committed under `quotient_root` is the key's composite
    cExp over the committed sections, divided by the zerofier — the pil2
    analogue of `QuotientBoundClaim`, with the fold already baked into the
    key's pre-folded expression instead of an `alpha` power vector."""

    pil2: Pil2Claim
    trace_root: Array
    root2: Array
    quotient_root: Array
    challenges: dict[int, Array]
    airgroupvalues: dict[int, Array]


@dataclass(frozen=True)
class Stage2Commitment:
    """The stage-2 role's reduction proof: the base-domain witness-STD
    columns (the byte-gates compare them against pil2's ``cm2`` section) and
    their extend-and-merkelize commitment."""

    matrix: Array
    commitment: TraceCommitment


@dataclass(frozen=True)
class Pil2QuotientWitness:
    """What the pil2 quotient stage's SSA reads: both committed sections so
    far — the extended trace and the extended stage-2 columns."""

    trace_commit: TraceCommitment
    stage2: Stage2Commitment


@dataclass(frozen=True)
class Pil2OpeningWitness:
    """What discharging a `Pil2QuotientBoundClaim` takes: every committed
    tree's prover data — the extended sections the evals and DEEP batch read,
    and the digest layers the query phase opens."""

    trace_commit: TraceCommitment
    stage2: Stage2Commitment
    quotient: QuotientCommitment


@dataclass(frozen=True, kw_only=True)
class Pil2InnerProof:
    """The pil2-mode wire: the three stage roots, the airgroup values the
    stage-2 witness settles, and the opening discharge."""

    trace_root: Array
    root2: Array
    quotient_root: Array
    airgroupvalues: dict[int, Array]
    opening: OpeningProof
