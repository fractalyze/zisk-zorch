"""End-to-end inner-proof prover — the inner proof as composite Stage roles.

`InnerProver` reduces the inner statement to the trivial claim over one duplex
`Transcript`, the byte stream pil2-proofman's `genProof` drives: the opening
scheme's commit half binds the witness, then two `zorch.stage.ProverStage`
roles — quotient, opening — each discharge the claim the one before produced,
so what crosses a seam is a claim both roles derive rather than a shared
mutable carry.

- **Claim** — the statement a stage consumes and reduces (`InnerClaim`,
  `QuotientBoundClaim`, `zorch.stage.TrivialClaim`). Claims hold only what
  both roles can derive — the verifier from the wire, the prover from its own
  run — never prover-only data.
- **Witness** — what makes a claim true, plus the prover data discharging it
  takes (`InnerWitness`, `OpeningWitness`). Prover data produced by one part
  and consumed by another — the trace commitment, the quotient codeword —
  rides witness wrappers the composite assembles; it belongs to no claim.
- Static configuration (the AIR's `eval_fn`, arity, the fold schedule) lives
  on the role instances, the statement on the claim, the trace on the witness.

pil2's DEEP, FRI, and query phases are ONE terminal stage here
(`OpeningProver`): each of those sub-steps consumes a challenge the previous
one squeezed and produces the next one's prover input, so their seams are
prover-data seams, not claim boundaries a `ProverStage` pair could meet at.

The quotient-commit leaf layout mirrors the FRI seam's cubic convention (each
cubic row -> its 3 contiguous Goldilocks limbs, cf. `zorch.utils.field`), which
is pil2's `FIELD_EXTENSION`-contiguous memory order.

See `docs/architecture.md` for the DEEP seam and its byte-match boundary.

https://github.com/0xPolygonHermez/pil2-proofman/blob/v1.0.0-alpha/pil2-stark/src/starkpil/gen_proof.hpp
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial

import frx
import frx.numpy as fnp
import numpy as np
from frx import Array
from zk_dtypes import goldilocksx3 as F3
from zorch.poly.univariate import powers
from zorch.stage import ProveResult, ProverStage, TrivialClaim
from zorch.utils.field import join_coeffs, split_coeffs

from zisk_zorch.commit.openings import group_proof
from zisk_zorch.commit.trace_commit import TraceCommitment, commit_trace, merkle_tree
from zisk_zorch.deep.fri_polynomial import deep_fri_polynomial
from zisk_zorch.evals.lev import LevConstants, lev_constants
from zisk_zorch.fri.prover import FriProof, prove, prove_queries
from zisk_zorch.fri.queries import sample_query_positions
from zisk_zorch.quotient.quotient import quotient_from_constraints
from zisk_zorch.quotient.zerofier import _coset_points
from zisk_zorch.transcript.transcript import Transcript


@partial(frx.jit, static_argnames=("n_bits", "blowup_bits", "opening_points"))
def _opening_deep_jit(
    trace_ext: Array,
    quotient: Array,
    domain: Array,
    lev_consts: LevConstants,
    transcript: Transcript,
    *,
    n_bits: int,
    blowup_bits: int,
    opening_points: tuple[int, ...],
) -> tuple[Transcript, Array, Array]:
    """The DEEP leg — z squeeze, column openings, evals absorb, vf squeeze,
    composition — as one shape-invariant ``@jit`` zone (the sp1-zorch
    jagged-PCS pattern): the transcript rides through as a traced pytree, so
    the whole leg is one compiled executable instead of eager transcript hops
    between the inner `open_columns`/`deep_composition` zones. The compile
    keys on the committed shapes and the static schedule alone.

    `domain` and `lev_consts` arrive from the eager prologue: an in-trace
    coset feeding the composition's cubic reciprocal is #67's NVPTX crash
    trigger, and field constants enter as arguments per `LevConstants`. The
    zone RETURNS its transcript — the caller's object is not advanced in
    place across the boundary."""
    fri_pol, evals = deep_fri_polynomial(
        trace_ext,
        quotient,
        transcript,
        n_bits=n_bits,
        blowup_bits=blowup_bits,
        opening_points=opening_points,
        domain=domain,
        lev_consts=lev_consts,
    )
    return transcript, fri_pol, evals


def _fold_steps(n_bits_ext: int, fold_bits: int, final_bits: int) -> list[int]:
    """Strictly-decreasing FRI layer schedule `n_bits_ext -> ... -> final_bits`,
    folding by `fold_bits` per layer (the tail folds the remainder). Same schedule
    `bench_inner_proof._fold_steps` builds; kept here so the prover owns its FRI
    shape without importing a benchmark private.

    The two degenerate schedules are rejected here rather than downstream:
    `Pil2FriCode` accepts a one-step schedule, so a `final_bits` at or above the
    extended domain would fold zero layers and prove nothing, and a non-positive
    `fold_bits` reaches `range` as a zero or inverted step. Everything else the
    schedule needs is checked where it is used — `arity` by `merkle_tree`,
    `blowup_bits` by the zerofier, and the step sequence by `Pil2FriCode`."""
    if fold_bits < 1:
        raise ValueError(f"fold_bits must be >= 1, got {fold_bits}")
    if final_bits >= n_bits_ext:
        raise ValueError(
            f"final_bits must be below the extended domain size {n_bits_ext} "
            f"or FRI folds no layers, got {final_bits}"
        )
    steps = list(range(n_bits_ext, final_bits, -fold_bits))
    if not steps or steps[-1] != final_bits:
        steps.append(final_bits)
    return steps


@dataclass(frozen=True, kw_only=True)
class InnerClaim:
    """Some trace of this shape satisfies the AIR.

    Spelled out: there exists a ``(2**n_bits, n_cols)`` base-field trace on
    which every one of the AIR's `n_constraints` constraints vanishes on every
    row. Nothing here names that trace — it is existentially quantified, and
    the prover exhibits one by committing to it, which is why the trace root
    is proof data rather than a field of the statement. The AIR's circuits
    (`eval_fn`) stay static configuration on `QuotientProver` — both roles are
    built against the same AIR — while the statement instance's shape lives
    here, where both read it: the prover cross-checks its witness against it,
    the verifier sizes the alpha fold and the openings by it.
    """

    n_bits: int
    n_cols: int
    n_constraints: int


@dataclass(frozen=True, kw_only=True)
class QuotientBoundClaim:
    """The codeword committed under `quotient_root` is the alpha-fold of
    `inner`'s AIR constraints on the trace committed under `trace_root`,
    divided by the zerofier.

    What the quotient stage reduces the AIR statement to: the division is
    exact only when the alpha-folded constraints vanish on the whole base
    domain, so a single violated row makes the true quotient a non-polynomial
    no committed codeword can equal at the opening's out-of-domain point.
    `inner` is the source statement the reduction conditions on — the opening
    still needs its shape to size what it opens. Both roles hold the roots —
    the prover from committing, the verifier off the wire — so they are claim
    data: they name what the opening is checked against. The fields are
    keyword-only so the two same-shaped roots cannot be passed to each
    other's slot.
    """

    inner: InnerClaim
    trace_root: Array
    quotient_root: Array


@dataclass(frozen=True)
class InnerWitness:
    """The trace that makes an `InnerClaim` true: the ``(2**n_bits, n_cols)``
    base-field evaluation matrix."""

    trace: Array


@dataclass(frozen=True)
class QuotientCommitment:
    """The quotient stage's reduction proof, the quotient analogue of
    ``TraceCommitment``. Its wire projection is `root`; the cubic codeword the
    DEEP batch opens and the base-limb `matrix`/`layers` the query phase opens
    the committed tree with ride along as prover data — pil2 commits the
    quotient mid-transcript (alpha precedes it), so unlike a PCS trace commit
    this cannot be split into a pre-transcript commit half."""

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
class OpeningProof:
    """Discharges a `QuotientBoundClaim`, leaving nothing to prove: pil2's
    `evals` section (the out-of-domain column openings; None under
    `EchoOpening`, which opens nothing), the FRI fold proof, the grinding
    nonce, the squeezed query positions, and every committed tree's per-query
    openings."""

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
    per-stage roots the transcript absorbed and the opening stage's discharge,
    assembled flat for the wire."""

    trace_root: Array
    quotient_root: Array
    # pil2's `evals` section: the committed columns opened at the OOD point.
    # None when the opening is EchoOpening (no openings to send).
    evals: Array | None
    fri: FriProof
    final_pol: Array
    nonce: int
    query_positions: np.ndarray
    trace_openings: list[list[Array]]
    quotient_openings: list[list[Array]]
    fri_openings: list[list[Array]]


def bind_trace_commitment(transcript: Transcript, root: Array) -> Transcript:
    """Bind the committed trace into the stream — pil2's stage-1 root absorb,
    what sits between the opening scheme's commit half and the first squeeze.

    A transcript-only schedule operation, so it is one shared function both
    roles call rather than a stage: the prover and the verifier dual reach the
    post-commit transcript state through this single definition, and an
    ordering edit cannot land in one Fiat-Shamir stream and not the other.
    """
    transcript.put(root)
    return transcript


class QuotientProver(
    ProverStage[InnerClaim, TraceCommitment, QuotientBoundClaim, QuotientCommitment]
):
    """pil2 `calculateQuotientPolynomial` as one claim reduction: squeeze
    alpha, fold the constraints by its powers, divide by the zerofier, commit
    `Q`, absorb its root.

    What it proves: every constraint evaluates to zero on every trace row —
    conditionally on the reduced claim, which the opening stage discharges.
    The witness is the trace commitment (constraints must be evaluated on the
    coset, where the zerofier is invertible). The statement's shape — domain
    size, constraint count — is read off the claim; only the AIR's circuits
    and the protocol parameters are configuration. Cubic rows commit as 3
    contiguous base limbs (pil2 `FIELD_EXTENSION` layout), so the leaf hash
    matches the FRI seam."""

    def __init__(
        self,
        eval_fn: Callable[[Array], Array],
        *,
        blowup_bits: int,
        arity: int,
    ) -> None:
        self._eval_fn = eval_fn
        self._blowup_bits = blowup_bits
        self._arity = arity

    def prove(
        self,
        claim: InnerClaim,
        witness: TraceCommitment,
        transcript: Transcript,
    ) -> ProveResult[QuotientBoundClaim, QuotientCommitment]:
        # pil2 folds the K constraints by powers of the stage-`nStages+1`
        # challenge — exactly the coefficient vector `zorch.constraint_eval` takes.
        alpha = powers(
            join_coeffs(transcript.get_field().reshape(-1, 3), F3).reshape(()),
            claim.n_constraints,
        )
        quotient = quotient_from_constraints(
            self._eval_fn,
            witness.extended,
            alpha,
            claim.n_bits,
            self._blowup_bits,
        )
        matrix = split_coeffs(quotient)
        root, layers = merkle_tree(self._arity).commit(matrix)
        transcript.put(root)
        commitment = QuotientCommitment(
            codeword=quotient, root=root, matrix=matrix, layers=layers
        )
        return ProveResult(
            QuotientBoundClaim(
                inner=claim, trace_root=witness.root, quotient_root=root
            ),
            commitment,
            transcript,
        )


class OpeningProver(
    ProverStage[QuotientBoundClaim, OpeningWitness, TrivialClaim, OpeningProof]
):
    """pil2's whole opening — `calculateFRIPolynomial`, `FRI::fold`,
    `proveQueries` — as the terminal stage discharging a `QuotientBoundClaim`.

    The two halves bracket the inner proof: `commit` binds the trace before
    any challenge exists, `prove` is the open. Only the open reduces a claim,
    so only the open is the Stage role; the commit is the scheme's other half.

    What it proves, in three internal steps whose seams are prover-data
    seams (each consumes a challenge the previous step squeezed and produces
    the next step's input, which is why they share one stage):

    - DEEP: the transmitted OOD openings are the committed columns' true
      values. Each term `(f(x) - f(z)) / (x - z)` is a polynomial only if the
      claimed `f(z)` is genuine, so batching the terms reduces every opening's
      honesty to one low-degree claim.
    - FRI: that DEEP codeword is (close to) low-degree. Each beta-fold cuts
      the degree by `2**fold_bits` while preserving low-degreeness iff it held
      before, until the final polynomial is small enough to transmit and check
      directly.
    - Queries: the fold chain was computed honestly. Everything so far
      exchanged Merkle roots, not data; the query openings let the verifier
      re-derive each fold at `n_queries` random positions and check it against
      the committed layers, back to the trace and quotient leaves themselves.
      Grinding (`pow_bits`) makes retrying for favorable positions costly.
    """

    def __init__(
        self,
        *,
        blowup_bits: int,
        steps: list[int],
        arity: int,
        pow_bits: int,
        n_queries: int,
        opening_points: Sequence[int] = (0,),
        jit: bool = True,
    ) -> None:
        self._blowup_bits = blowup_bits
        self._steps = steps
        self._arity = arity
        self._pow_bits = pow_bits
        self._n_queries = n_queries
        self._opening_points = opening_points
        self._jit = jit

    def commit(self, witness: InnerWitness) -> TraceCommitment:
        """Commit the trace — pil2 `extendAndMerkelize`: LDE onto the coset,
        Merkle-commit the rows. The commit half of the scheme whose open half
        is `prove`; only the open reduces a claim, so only the open is the
        Stage role, and `TraceCommitment` is what commit hands forward.

        No transcript argument: committing is not a transcript operation. The
        composite absorbs the returned root through `bind_trace_commitment`,
        so the Fiat-Shamir binding has one visible home rather than hiding
        inside the scheme."""
        return commit_trace(
            witness.trace, blowup=1 << self._blowup_bits, arity=self._arity
        )

    def _fri_input(
        self,
        claim: QuotientBoundClaim,
        witness: OpeningWitness,
        transcript: Transcript,
    ) -> tuple[Array, Array | None, Transcript]:
        """The codeword FRI folds, the OOD openings the proof transmits, and
        the advanced transcript — pil2 `calculateFRIPolynomial`. `EchoOpening`
        overrides this hook.

        Under the `jit` knob the leg runs as the `_opening_deep_jit` zone,
        with the domain coset materialized here in the eager prologue (#67);
        eagerly it is the same flow with transcript hops between the inner
        zones. Byte-identical either way — exact field ops, same schedule."""
        if self._jit:
            domain = _coset_points(claim.inner.n_bits, self._blowup_bits)
            consts = lev_constants(list(self._opening_points), claim.inner.n_bits)
            transcript, fri_pol, evals = _opening_deep_jit(
                witness.trace_commit.extended,
                witness.quotient.codeword,
                domain,
                consts,
                transcript,
                n_bits=claim.inner.n_bits,
                blowup_bits=self._blowup_bits,
                opening_points=tuple(self._opening_points),
            )
            return fri_pol, evals, transcript
        fri_pol, evals = deep_fri_polynomial(
            witness.trace_commit.extended,
            witness.quotient.codeword,
            transcript,
            n_bits=claim.inner.n_bits,
            blowup_bits=self._blowup_bits,
            opening_points=self._opening_points,
        )
        return fri_pol, evals, transcript

    def prove(
        self,
        claim: QuotientBoundClaim,
        witness: OpeningWitness,
        transcript: Transcript,
    ) -> ProveResult[TrivialClaim, OpeningProof]:
        fri_pol, evals, transcript = self._fri_input(claim, witness, transcript)
        fri = prove(fri_pol, self._steps, arity=self._arity, transcript=transcript)
        n_bits_ext = claim.inner.n_bits + self._blowup_bits
        positions, nonce = sample_query_positions(
            transcript,
            fri.final_pol,
            pow_bits=self._pow_bits,
            n_queries=self._n_queries,
            n_bits_ext=n_bits_ext,
        )
        # Every challenge is squeezed by now and openings never re-enter the
        # transcript, so the per-query Merkle walks may batch freely: one
        # vmapped kernel per tree instead of a dispatch per query, with no
        # effect on the byte stream.
        ext_mask = (1 << n_bits_ext) - 1
        idx_ext = fnp.asarray(np.asarray(positions)) & ext_mask
        trace_batched = frx.vmap(
            partial(
                group_proof,
                merkle_tree(self._arity),
                witness.trace_commit.extended,
                witness.trace_commit.digest_layers,
            )
        )(idx_ext)
        quotient_batched = frx.vmap(
            partial(
                group_proof,
                merkle_tree(self._arity),
                witness.quotient.matrix,
                witness.quotient.layers,
            )
        )(idx_ext)
        trace_openings = [[trace_batched[q]] for q in range(len(positions))]
        quotient_openings = [[quotient_batched[q]] for q in range(len(positions))]
        fri_openings = prove_queries(fri, positions)
        return ProveResult(
            TrivialClaim(),
            OpeningProof(
                evals=evals,
                fri=fri,
                nonce=nonce,
                positions=positions,
                trace_openings=trace_openings,
                quotient_openings=quotient_openings,
                fri_openings=fri_openings,
            ),
            transcript,
        )


class EchoOpening(OpeningProver):
    """Placeholder opening: fold FRI over the quotient codeword itself,
    skipping the out-of-domain opening (`evals` stays None). The quotient is a
    valid cubic FRI input, so this drives the spine end to end for
    wiring/shape tests — but with no OOD openings the quotient's consistency
    with the committed trace is never tested, so it proves nothing about the
    trace and a proof built with it does not byte-match pil2. Not for
    conformance."""

    def _fri_input(
        self,
        claim: QuotientBoundClaim,
        witness: OpeningWitness,
        transcript: Transcript,
    ) -> tuple[Array, Array | None, Transcript]:
        return witness.quotient.codeword, None, transcript


class InnerProver(ProverStage[InnerClaim, InnerWitness, TrivialClaim, InnerProof]):
    """The ZisK inner prover: the trace commit, then two Stages.

    A composite role, so the wiring has one definition and the benchmark, the
    byte-match runnables, and proof assembly cannot drift on it. The quotient
    and opening Stages reduce the statement to the trivial claim, each one's
    reduced claim the next one's source claim. They are bracketed by the
    opening scheme's two halves: `opening.commit` binds the trace up front —
    before any challenge exists — and `opening.prove` discharges the reduced
    claim at the end, with the `TraceCommitment` held here in between because
    it belongs to no claim.

    `echo_deep` swaps the opening for `EchoOpening`, the trivial fallback that
    skips the OOD opening."""

    def __init__(
        self,
        eval_fn: Callable[[Array], Array],
        *,
        n_bits: int,
        blowup_bits: int = 1,
        arity: int = 2,
        fold_bits: int = 3,
        final_bits: int = 5,
        pow_bits: int = 16,
        n_queries: int = 64,
        echo_deep: bool = False,
        jit: bool = True,
    ) -> None:
        self.quotient = QuotientProver(eval_fn, blowup_bits=blowup_bits, arity=arity)
        opening_cls = EchoOpening if echo_deep else OpeningProver
        self.opening = opening_cls(
            blowup_bits=blowup_bits,
            steps=_fold_steps(n_bits + blowup_bits, fold_bits, final_bits),
            arity=arity,
            pow_bits=pow_bits,
            n_queries=n_queries,
            jit=jit,
        )

    def prove(
        self,
        claim: InnerClaim,
        witness: InnerWitness,
        transcript: Transcript,
    ) -> ProveResult[TrivialClaim, InnerProof]:
        # The verifier dual reads the statement's shape off the claim while
        # the prover has it in the witness's trace; a pair that disagrees
        # would otherwise only surface later, as a transcript divergence.
        assert witness.trace.shape == (
            1 << claim.n_bits,
            claim.n_cols,
        ), "claim's trace shape does not match the witness"
        commitment = self.opening.commit(witness)
        transcript = bind_trace_commitment(transcript, commitment.root)
        quotient = self.quotient.prove(claim, commitment, transcript)
        opening = self.opening.prove(
            quotient.reduced_claim,
            OpeningWitness(commitment, quotient.reduction_proof),
            quotient.transcript,
        )
        proof = opening.reduction_proof
        return ProveResult(
            TrivialClaim(),
            InnerProof(
                trace_root=commitment.root,
                quotient_root=quotient.reduction_proof.root,
                evals=proof.evals,
                fri=proof.fri,
                final_pol=proof.fri.final_pol,
                nonce=proof.nonce,
                query_positions=proof.positions,
                trace_openings=proof.trace_openings,
                quotient_openings=proof.quotient_openings,
                fri_openings=proof.fri_openings,
            ),
            opening.transcript,
        )


def prove_inner(
    trace: Array,
    eval_fn: Callable[[Array], Array],
    *,
    n_constraints: int,
    blowup_bits: int = 1,
    arity: int = 2,
    fold_bits: int = 3,
    final_bits: int = 5,
    pow_bits: int = 16,
    n_queries: int = 64,
    echo_deep: bool = False,
    jit: bool = True,
    transcript: Transcript | None = None,
) -> InnerProof:
    """Run `InnerProver` over one shared `Transcript` and return the proof.

    `trace` is the `(2^n_bits, n_cols)` base-field evaluation matrix; `eval_fn`
    produces the `n_constraints` constraints in its trailing axis (pil2's cExp
    order). `echo_deep` swaps the opening for the trivial `EchoOpening`
    fallback that skips the OOD opening."""
    if trace.ndim != 2:
        raise ValueError(f"trace must be 2-D (rows, cols), got ndim={trace.ndim}")
    n = trace.shape[0]
    if n & (n - 1):
        raise ValueError(f"trace height must be a power of two, got {n}")
    n_bits = n.bit_length() - 1

    prover = InnerProver(
        eval_fn,
        n_bits=n_bits,
        blowup_bits=blowup_bits,
        arity=arity,
        fold_bits=fold_bits,
        final_bits=final_bits,
        pow_bits=pow_bits,
        n_queries=n_queries,
        echo_deep=echo_deep,
        jit=jit,
    )
    claim = InnerClaim(
        n_bits=n_bits, n_cols=int(trace.shape[1]), n_constraints=n_constraints
    )
    result = prover.prove(claim, InnerWitness(trace), transcript or Transcript())
    return result.reduction_proof
