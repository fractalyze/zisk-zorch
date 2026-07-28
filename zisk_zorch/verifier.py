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
from zisk_zorch.types import InnerClaim, InnerProof, OpeningProof, TraceBoundClaim


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
        arity: int = 2,
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
            quotient.reduced_claim,
            OpeningProof(
                evals=proof.evals,
                fri=proof.fri,
                nonce=proof.nonce,
                positions=proof.query_positions,
                trace_openings=proof.trace_openings,
                quotient_openings=proof.quotient_openings,
                fri_openings=proof.fri_openings,
            ),
            quotient.transcript,
        )
        return VerifyResult(
            TrivialClaim(), opening.transcript, quotient.ok & opening.ok
        )


def verify_inner(
    proof: InnerProof,
    eval_fn: Callable[[Array], Array],
    *,
    n_constraints: int,
    n_bits: int,
    blowup_bits: int = 1,
    arity: int = 2,
    fold_bits: int = 3,
    final_bits: int = 5,
    pow_bits: int = 16,
    opening_points: Sequence[int] = (0,),
    transcript: Transcript | None = None,
) -> bool:
    """Whether `proof` is a valid inner proof of the AIR `eval_fn` over a
    `2^n_bits`-row trace — `prove_inner`'s dual, run through `InnerVerifier`.

    `transcript` must be seeded exactly as the prover's was. The keyword
    parameters are the prover's, and are the verifier's half of the statement:
    they fix the fold schedule and the domain, so passing different ones
    checks a different claim."""
    if proof.evals is None:
        raise ValueError(
            "proof carries no out-of-domain openings — it was built with "
            "EchoOpening, which opens nothing and so cannot be verified"
        )
    # The committed columns are the trace's base columns then the one cubic
    # quotient column (`deep._committed_columns`), so the openings pin the
    # claim's width.
    n_cols = proof.evals.shape[0] - 1
    verifier = InnerVerifier(
        eval_fn,
        n_bits=n_bits,
        blowup_bits=blowup_bits,
        arity=arity,
        fold_bits=fold_bits,
        final_bits=final_bits,
        pow_bits=pow_bits,
        opening_points=opening_points,
    )
    claim = InnerClaim(n_bits=n_bits, n_cols=n_cols, n_constraints=n_constraints)
    result = verifier.verify(claim, proof, transcript or Transcript())
    return bool(result.ok)
