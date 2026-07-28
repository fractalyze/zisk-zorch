"""End-to-end inner-proof prover — the inner proof as composite Stage roles.

`InnerProver` reduces the inner statement to the trivial claim over one duplex
`Transcript`: the opening scheme's commit half binds the witness, then two
`zorch.stage.ProverStage` roles — quotient, opening — each discharge the
claim the one before produced,
so what crosses a seam is a claim both roles derive rather than a shared
mutable carry.

Each stage's roles live in its own package — `quotient/prover.py`,
`opening/prover.py` (with their verifier duals alongside) — and the claims
they exchange in `types.py`:

- **Claim** — the statement a stage consumes and reduces (`InnerClaim`,
  `TraceBoundClaim`, `QuotientBoundClaim`, `zorch.stage.TrivialClaim`).
  Claims hold only what both roles can derive — the verifier from the wire,
  the prover from its own run — never prover-only data.
- **Witness** — what makes a claim true, plus the prover data discharging it
  takes (`InnerWitness`, `OpeningWitness`). Prover data produced by one part
  and consumed by another — the trace commitment, the quotient codeword —
  rides witness wrappers the composite assembles; it belongs to no claim.
- Static configuration (the AIR's `eval_fn`, arity, the fold schedule) lives
  on the role instances, the statement on the claim, the trace on the witness.

The DEEP, FRI, and query phases are ONE terminal stage
(`opening.OpeningProver`): each of those sub-steps consumes a challenge the
previous one squeezed and produces the next one's prover input, so their
seams are prover-data seams, not claim boundaries a `ProverStage` pair could
meet at.

See `docs/architecture.md` for the DEEP seam and its byte-match boundary.
"""

from __future__ import annotations

from collections.abc import Callable

from frx import Array
from zorch.stage import ProveResult, ProverStage, TrivialClaim

from zisk_zorch.opening.prover import EchoOpening, OpeningProver
from zisk_zorch.quotient.prover import QuotientProver
from zisk_zorch.transcript.transcript import Transcript
from zisk_zorch.types import (
    InnerClaim,
    InnerProof,
    InnerWitness,
    OpeningWitness,
    TraceBoundClaim,
)


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


def bind_trace_commitment(transcript: Transcript, root: Array) -> Transcript:
    """Bind the committed trace into the stream — the root absorb sitting
    between the opening scheme's commit half and the first squeeze.

    A transcript-only schedule operation, so it is one shared function both
    roles call rather than a stage: the prover and the verifier dual reach the
    post-commit transcript state through this single definition, and an
    ordering edit cannot land in one Fiat-Shamir stream and not the other.
    """
    transcript.put(root)
    return transcript


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
        bound = TraceBoundClaim(inner=claim, trace_root=commitment.root)
        quotient = self.quotient.prove(bound, commitment, transcript)
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
    produces the `n_constraints` constraints in its trailing axis.
    `echo_deep` swaps the opening for the trivial `EchoOpening` fallback that
    skips the OOD opening."""
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
