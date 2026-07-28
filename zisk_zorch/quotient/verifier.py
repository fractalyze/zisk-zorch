"""The quotient stage's verifier role — `QuotientProver`'s dual over
`zorch.stage.VerifierStage`."""

from __future__ import annotations

import frx.numpy as fnp
from frx import Array
from zk_dtypes import goldilocksx3 as F3
from zorch.poly.univariate import powers
from zorch.stage import VerifierStage, VerifyResult
from zorch.utils.field import join_coeffs

from zisk_zorch.claims import QuotientBoundClaim, TraceBoundClaim
from zisk_zorch.transcript.transcript import Transcript


class QuotientVerifier(VerifierStage[TraceBoundClaim, QuotientBoundClaim, Array]):
    """`QuotientProver`'s dual: replay the alpha squeeze, absorb the committed
    quotient root off the wire, and derive the same `QuotientBoundClaim`.

    The verdict is structurally true: this stage's reduction has no content a
    verifier can check on its own — the claim it produces is conditional, and
    `OpeningVerifier` discharges it (the constraint check at `z` is exactly
    the check of this reduction, runnable only once `z` and the openings
    exist). What this stage contributes is the Fiat-Shamir binding: alpha is
    squeezed before the quotient root is absorbed, mirroring the prover."""

    def verify(
        self,
        claim: TraceBoundClaim,
        reduction_proof: Array,
        transcript: Transcript,
    ) -> VerifyResult[QuotientBoundClaim]:
        alpha = powers(
            join_coeffs(transcript.get_field().reshape(-1, 3), F3).reshape(()),
            claim.inner.n_constraints,
        )
        transcript.put(reduction_proof)
        return VerifyResult(
            QuotientBoundClaim(
                inner=claim.inner,
                trace_root=claim.trace_root,
                quotient_root=reduction_proof,
                alpha=alpha,
            ),
            transcript,
            fnp.asarray(True),
        )
