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

from zisk_zorch.opening.prover import EchoOpening, OpeningProver, Pil2Opening
from zisk_zorch.pil2 import Pil2Key, absorb_words
from zisk_zorch.quotient.prover import Pil2QuotientProver, QuotientProver
from zisk_zorch.quotient.witness_std import WitnessStdProver
from zisk_zorch.transcript.transcript import Transcript
from zisk_zorch.types import (
    InnerClaim,
    InnerProof,
    InnerWitness,
    OpeningWitness,
    Pil2Claim,
    Pil2InnerProof,
    Pil2OpeningWitness,
    Pil2QuotientWitness,
    Pil2TraceBoundClaim,
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
        return ProveResult(
            TrivialClaim(),
            InnerProof(
                trace_root=commitment.root,
                quotient_root=quotient.reduction_proof.root,
                opening=opening.reduction_proof,
            ),
            opening.transcript,
        )


class Pil2InnerProver:
    """The pil2-mode composite: genProof's non-recursive schedule over the
    three pil2 roles — witness-STD, cExp quotient, pil2 opening.

    Same composite shape as `InnerProver` with the two schedule differences
    the protocol dictates: the transcript seeds with the claim's
    contributions-phase global challenge instead of absorbing the trace root
    (`root1` never enters the non-recursive schedule — it rides inside the
    seed), and the stage-2 witness role runs between the commit and the
    quotient. The roles configure themselves from one `Pil2Key`, so a
    prover exists per proving key, claims per instance.

    `prove_stages` is the byte-gate seam: it yields each stage's result as
    the stage finishes, so `verify_inner_proof`'s composed gate can check
    every reduction the moment it lands and stop before the later stages
    pay their compute on a mismatch — the same fail-fast shape as
    sp1-zorch's `verify_prove_shard` — while `prove` consumes the same
    generator, so the two paths cannot drift on the wiring."""

    def __init__(self, key: Pil2Key) -> None:
        self.stage2 = WitnessStdProver(key)
        self.quotient = Pil2QuotientProver(key)
        self.opening = Pil2Opening(key)

    def prove_stages(
        self,
        claim: Pil2Claim,
        witness: InnerWitness,
        transcript: Transcript,
    ):
        """Run the schedule, yielding ``(stage_name, result)`` as each stage
        finishes — ``commit`` yields the `TraceCommitment`, the Stage roles
        their `ProveResult`s, prover data included."""
        assert witness.trace.shape == (
            1 << claim.n_bits,
            claim.n_cols,
        ), "claim's trace shape does not match the witness"
        commitment = self.opening.commit(witness)
        yield "commit", commitment
        absorb_words(transcript, claim.global_challenge)
        bound = Pil2TraceBoundClaim(pil2=claim, trace_root=commitment.root)
        stage2 = self.stage2.prove(bound, witness, transcript)
        yield "stage2", stage2
        quotient = self.quotient.prove(
            stage2.reduced_claim,
            Pil2QuotientWitness(commitment, stage2.reduction_proof),
            stage2.transcript,
        )
        yield "quotient", quotient
        yield (
            "opening",
            self.opening.prove(
                quotient.reduced_claim,
                Pil2OpeningWitness(
                    commitment, stage2.reduction_proof, quotient.reduction_proof
                ),
                quotient.transcript,
            ),
        )

    def prove(
        self,
        claim: Pil2Claim,
        witness: InnerWitness,
        transcript: Transcript,
    ) -> ProveResult[TrivialClaim, Pil2InnerProof]:
        stages = dict(self.prove_stages(claim, witness, transcript))
        stage2, opening = stages["stage2"], stages["opening"]
        return ProveResult(
            TrivialClaim(),
            Pil2InnerProof(
                trace_root=stages["commit"].root,
                root2=stage2.reduced_claim.root2,
                quotient_root=stages["quotient"].reduction_proof.root,
                airgroupvalues=stage2.reduced_claim.airgroupvalues,
                opening=opening.reduction_proof,
            ),
            opening.transcript,
        )
