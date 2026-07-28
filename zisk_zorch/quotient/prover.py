"""The quotient stage's prover role — pil2 `calculateQuotientPolynomial` as
one claim reduction over `zorch.stage.ProverStage`."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from frx import Array
from zk_dtypes import goldilocksx3 as F3
from zorch.poly.univariate import powers
from zorch.stage import ProveResult, ProverStage
from zorch.utils.field import join_coeffs, split_coeffs

from zisk_zorch.claims import QuotientBoundClaim, TraceBoundClaim
from zisk_zorch.commit.trace_commit import TraceCommitment, merkle_tree
from zisk_zorch.quotient.quotient import quotient_from_constraints
from zisk_zorch.transcript.transcript import Transcript


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


class QuotientProver(
    ProverStage[
        TraceBoundClaim, TraceCommitment, QuotientBoundClaim, QuotientCommitment
    ]
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
        claim: TraceBoundClaim,
        witness: TraceCommitment,
        transcript: Transcript,
    ) -> ProveResult[QuotientBoundClaim, QuotientCommitment]:
        # pil2 folds the K constraints by powers of the stage-`nStages+1`
        # challenge — exactly the coefficient vector `zorch.constraint_eval` takes.
        alpha = powers(
            join_coeffs(transcript.get_field().reshape(-1, 3), F3).reshape(()),
            claim.inner.n_constraints,
        )
        quotient = quotient_from_constraints(
            self._eval_fn,
            witness.extended,
            alpha,
            claim.inner.n_bits,
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
                inner=claim.inner,
                trace_root=claim.trace_root,
                quotient_root=root,
                alpha=alpha,
            ),
            commitment,
            transcript,
        )
