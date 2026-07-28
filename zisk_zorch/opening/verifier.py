"""The opening stage's verifier role — `OpeningProver`'s dual over
`zorch.stage.VerifierStage`."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import frx.numpy as fnp
import numpy as np
from frx import Array
from zk_dtypes import goldilocksx3 as F3
from zorch.poly.univariate import powers
from zorch.stage import TrivialClaim, VerifierStage, VerifyResult
from zorch.utils.field import join_coeffs, split_coeffs

from zisk_zorch.commit.openings import verify_group_proof
from zisk_zorch.commit.trace_commit import merkle_tree
from zisk_zorch.deep.fri_polynomial import _ood_points
from zisk_zorch.fri.queries import (
    grind_is_valid,
    grinding_seed_challenge,
    query_positions_for,
)
from zisk_zorch.fri.seam import Pil2SeamTranscript
from zisk_zorch.fri.verifier import verify_query_layers
from zisk_zorch.quotient.zerofier import _coset_points
from zisk_zorch.transcript.transcript import Transcript
from zisk_zorch.types import OpeningProof, QuotientBoundClaim


class OpeningVerifier(VerifierStage[QuotientBoundClaim, TrivialClaim, OpeningProof]):
    """`OpeningProver`'s dual: discharge a `QuotientBoundClaim` from the proof
    alone — the constraint check at `z`, then the Merkle, DEEP, and FRI checks
    at the re-derived query positions.

    `eval_fn` is this role's configuration and not the prover role's: the
    opening's prover never evaluates constraints, but discharging the claim
    means checking the committed pair actually encodes the AIR, so the
    verifier needs the circuits. The query positions are NOT trusted from the
    proof; they are re-derived from the transcript (final-pol absorb +
    grinding-seed squeeze + reseed with challenge++nonce), with the prover's
    nonce bound by the O(1) grind check first."""

    def __init__(
        self,
        eval_fn: Callable[[Array], Array],
        *,
        blowup_bits: int,
        steps: list[int],
        arity: int,
        pow_bits: int,
        opening_points: Sequence[int] = (0,),
    ) -> None:
        self._eval_fn = eval_fn
        self._blowup_bits = blowup_bits
        self._steps = steps
        self._arity = arity
        self._pow_bits = pow_bits
        self._opening_points = opening_points

    def verify(
        self,
        claim: QuotientBoundClaim,
        reduction_proof: OpeningProof,
        transcript: Transcript,
    ) -> VerifyResult[TrivialClaim]:
        proof = reduction_proof
        if proof.evals is None:
            raise ValueError(
                "proof carries no out-of-domain openings — it was built with "
                "EchoOpening, which opens nothing and so cannot be verified"
            )
        n_bits = claim.inner.n_bits
        n_cols = claim.inner.n_cols
        if proof.evals.shape[0] != n_cols + 1:
            raise ValueError(
                f"proof opens {proof.evals.shape[0]} columns, the claim's "
                f"trace shape implies {n_cols + 1} (trace columns + quotient)"
            )

        # Replay the DEEP head exactly as the prover: squeeze z, absorb the
        # openings, squeeze the batching challenge.
        t = transcript
        z = t.get_field()
        t.put(split_coeffs(proof.evals).reshape(-1))
        vf = join_coeffs(t.get_field().reshape(-1, 3), F3).reshape(())

        # FRI betas chain off the same transcript, one per committed layer.
        num_rounds = len(self._steps) - 1
        seam = Pil2SeamTranscript(t)
        betas = []
        for layer in range(num_rounds):
            seam, beta = seam.observe(proof.fri.roots[layer]).sample()
            betas.append(beta)

        # The grind binds the nonce before any position is read, so a prover
        # cannot search for query positions that miss its lie.
        challenge = grinding_seed_challenge(t, proof.fri.final_pol)
        if not grind_is_valid(challenge, proof.nonce, self._pow_bits):
            return VerifyResult(TrivialClaim(), t, fnp.asarray(False))
        positions = query_positions_for(
            challenge,
            t.width,
            proof.nonce,
            n_queries=len(proof.fri_openings),
            n_bits_ext=self._steps[0],
        )

        evals = proof.evals
        ok = self._check_constraint_at_z(evals, claim.alpha, z, n_bits)
        ok = ok and self._check_queries(claim, proof, evals, z, vf, positions)
        ok = ok and self._check_fri_chain(proof, betas, positions, n_bits)
        return VerifyResult(TrivialClaim(), t, fnp.asarray(ok))

    def _check_constraint_at_z(
        self, evals: Array, alpha: Array, z: Array, n_bits: int
    ) -> bool:
        """The AIR identity at the out-of-domain point: `Σ_k alpha^k c_k(z)`
        must equal `Q(z)·(z^N − 1)`.

        `QuotientProver` built `Q = C·Zi` with `Zi = 1/(x^N − 1)`, so
        multiplying the claimed `Q(z)` back by the zerofier recovers the
        composite the prover claims. An AIR's constraint expression is
        row-wise, so `eval_fn` takes the opened row directly — a `(1, n_cols)`
        cubic matrix."""
        n_cols = evals.shape[0] - 1
        trace_at_z = evals[:n_cols].reshape(1, n_cols)
        constraints = self._eval_fn(trace_at_z)
        if constraints.shape[-1] != alpha.shape[0]:
            raise ValueError(
                f"eval_fn produced {constraints.shape[-1]} constraints, "
                f"n_constraints={alpha.shape[0]}"
            )
        composite = fnp.sum(constraints.reshape(-1) * alpha)
        zc = join_coeffs(z.reshape(-1, 3), F3).reshape(())
        zerofier = zc ** (1 << n_bits) - fnp.ones((), zc.dtype)
        return bool(composite == evals[n_cols] * zerofier)

    def _check_queries(
        self,
        claim: QuotientBoundClaim,
        proof: OpeningProof,
        evals: Array,
        z: Array,
        vf: Array,
        positions: np.ndarray,
    ) -> bool:
        """Per query: the trace and quotient openings hash to the claim's
        roots, and the DEEP composition rebuilt from them equals the value FRI
        folded — what binds the FRI instance to the committed columns."""
        n_bits = claim.inner.n_bits
        n_cols = claim.inner.n_cols
        steps = self._steps
        ext_mask = (1 << steps[0]) - 1
        trace_tree = merkle_tree(self._arity)
        quotient_tree = merkle_tree(self._arity)
        x = _coset_points(n_bits, self._blowup_bits)
        xis = _ood_points(z, self._opening_points, n_bits)
        # Every column opens at `z`; wrapped openings are AIR-specific,
        # matching `deep_fri_polynomial`'s `opening_pos`.
        xi = xis[0]
        vf_pows = powers(vf, n_cols + 1)

        for q, idx in enumerate(positions):
            row = int(idx) & ext_mask
            trace_open = proof.trace_openings[q][0]
            quotient_open = proof.quotient_openings[q][0]
            if not verify_group_proof(
                trace_tree, claim.trace_root, row, trace_open, n_cols
            ):
                return False
            if not verify_group_proof(
                quotient_tree, claim.quotient_root, row, quotient_open, 3
            ):
                return False
            # Rebuild f(x) at this row. Layer 0's opening is the whole k-group
            # whose fold lands at `row mod 2^steps[1]`; this query's own
            # element sits at `row >> steps[1]` within it — the strided order
            # `Pil2FriCode.group_indices` lays down.
            numer = fnp.zeros((), evals.dtype)
            for m in range(n_cols):
                # base − cubic promotes
                numer = numer + vf_pows[m] * (trace_open[m] - evals[m])
            q_value = join_coeffs(quotient_open[:3].reshape(-1, 3), F3).reshape(())
            numer = numer + vf_pows[-1] * (q_value - evals[-1])
            expected = numer / (x[row] - xi)
            group_width = (1 << (steps[0] - steps[1])) * 3
            group = join_coeffs(
                proof.fri_openings[q][0][:group_width].reshape(-1, 3), F3
            )
            if not bool(expected == group[row >> steps[1]]):
                return False
        return True

    def _check_fri_chain(
        self,
        proof: OpeningProof,
        betas: list[Array],
        positions: np.ndarray,
        n_bits: int,
    ) -> bool:
        """The FRI half — `fri.verifier.verify_query_layers`, the one
        definition this and the standalone FRI verifier share."""
        return verify_query_layers(
            proof.fri.roots,
            proof.fri.final_pol,
            proof.fri_openings,
            positions,
            betas,
            steps=self._steps,
            arity=self._arity,
            n_bits=n_bits,
        )
