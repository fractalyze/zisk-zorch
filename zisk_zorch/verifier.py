"""End-to-end inner-proof verifier — the composite dual of `prover.InnerProver`.

Each stage's verifier role lives in its stage's package —
`quotient/verifier.py`, `opening/verifier.py` — consuming the same claims as
its prover twin. `InnerVerifier` walks the same chain over the same transcript
schedule, so the two composites cannot drift on the Fiat-Shamir stream (they
share `bind_trace_commitment` and the claim constructors).

The verifier replays the prover's transcript from a clean seed and re-derives
every challenge the prover claimed to have been bound by: the constraint-fold
`alpha`, the out-of-domain point `z`, the DEEP batching challenge `vf`, the
FRI fold betas, and the query positions. Nothing challenge-shaped is read from
the proof — only the committed roots, the openings, and the grinding nonce,
each of which the replay then checks. The four independent checks (Merkle,
constraint at `z`, DEEP, FRI) live on `opening.OpeningVerifier`, which
discharges the `QuotientBoundClaim`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from frx import Array
from zorch.stage import TrivialClaim, VerifierStage, VerifyResult

from zisk_zorch.opening.verifier import OpeningVerifier
from zisk_zorch.prover import _fold_steps, bind_trace_commitment
from zisk_zorch.quotient.verifier import QuotientVerifier
from zisk_zorch.transcript.transcript import Transcript
from zisk_zorch.types import InnerClaim, InnerProof, TraceBoundClaim


class InnerVerifier(VerifierStage[InnerClaim, TrivialClaim, InnerProof]):
    """The ZisK inner verifier: the trace-root bind, then two Stage duals.

    The mirror of `InnerProver.prove`, step for step: bind the trace root off
    the wire, derive `TraceBoundClaim`, run `QuotientVerifier`, and discharge
    its reduced claim with `OpeningVerifier` — the same claims, the same
    transcript schedule, no witness anywhere. The verdict is the AND of the
    stage verdicts."""

    def __init__(
        self,
        eval_fn: Callable[[Array], Array],
        *,
        n_bits: int,
        blowup_bits: int = 1,
        arity: int,
        fold_bits: int = 3,
        final_bits: int = 5,
        pow_bits: int = 16,
        opening_points: Sequence[int] = (0,),
    ) -> None:
        self.quotient = QuotientVerifier()
        self.opening = OpeningVerifier(
            eval_fn,
            blowup_bits=blowup_bits,
            steps=_fold_steps(n_bits + blowup_bits, fold_bits, final_bits),
            arity=arity,
            pow_bits=pow_bits,
            opening_points=opening_points,
        )

    def verify(
        self,
        claim: InnerClaim,
        reduction_proof: InnerProof,
        transcript: Transcript,
    ) -> VerifyResult[TrivialClaim]:
        proof = reduction_proof
        transcript = bind_trace_commitment(transcript, proof.trace_root)
        bound = TraceBoundClaim(inner=claim, trace_root=proof.trace_root)
        quotient = self.quotient.verify(bound, proof.quotient_root, transcript)
        opening = self.opening.verify(
            quotient.reduced_claim, proof.opening, quotient.transcript
        )
        return VerifyResult(
            TrivialClaim(), opening.transcript, quotient.ok & opening.ok
        )
