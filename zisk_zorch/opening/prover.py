"""The opening stage's prover role — the DEEP composition, FRI fold, and
query openings as the terminal stage discharging a `QuotientBoundClaim`,
plus the scheme's commit half."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial

import frx
import frx.numpy as fnp
import numpy as np
from frx import Array
from zorch.stage import ProveResult, ProverStage, TrivialClaim

from zisk_zorch.commit.openings import group_proof
from zisk_zorch.commit.trace_commit import commit_trace, merkle_tree
from zisk_zorch.deep.fri_polynomial import deep_fri_polynomial
from zisk_zorch.evals.lev import LevConstants, lev_constants
from zisk_zorch.fri.prover import FriProof, prove, prove_queries
from zisk_zorch.fri.queries import sample_query_positions
from zisk_zorch.quotient.zerofier import _coset_points
from zisk_zorch.transcript.transcript import Transcript
from zisk_zorch.types import (
    InnerWitness,
    OpeningWitness,
    QuotientBoundClaim,
    TraceCommitment,
)


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
    composition — as one shape-invariant ``@jit`` zone: the transcript rides
    through as a traced pytree, so the whole leg is one compiled executable
    instead of eager transcript hops between the inner
    `open_columns`/`deep_composition` zones. The compile keys on the committed
    shapes and the static schedule alone.

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


@dataclass(frozen=True)
class OpeningProof:
    """Discharges a `QuotientBoundClaim`, leaving nothing to prove: the
    out-of-domain column openings (`evals`; None under `EchoOpening`, which
    opens nothing), the FRI fold proof, the grinding
    nonce, the squeezed query positions, and every committed tree's per-query
    openings."""

    evals: Array | None
    fri: FriProof
    nonce: int
    positions: np.ndarray
    trace_openings: list[list[Array]]
    quotient_openings: list[list[Array]]
    fri_openings: list[list[Array]]


class OpeningProver(
    ProverStage[QuotientBoundClaim, OpeningWitness, TrivialClaim, OpeningProof]
):
    """The whole opening — the DEEP composition, the FRI fold, and the
    query openings — as the terminal stage discharging a `QuotientBoundClaim`.

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
        """Commit the trace: LDE onto the coset, Merkle-commit the rows —
        the commit half of the scheme whose open half is `prove`; only the
        open reduces a claim, so only the open is the Stage role, and
        `TraceCommitment` is what commit hands forward.

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
        the advanced transcript — the DEEP composition. `EchoOpening`
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
        fri, fri_layers = prove(
            fri_pol, self._steps, arity=self._arity, transcript=transcript
        )
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
        fri_openings = prove_queries(fri_layers, positions)
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
    trace. Not for conformance."""

    def _fri_input(
        self,
        claim: QuotientBoundClaim,
        witness: OpeningWitness,
        transcript: Transcript,
    ) -> tuple[Array, Array | None, Transcript]:
        return witness.quotient.codeword, None, transcript
