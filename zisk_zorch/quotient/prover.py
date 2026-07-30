"""The quotient stage's prover roles — one claim reduction over
`zorch.stage.ProverStage`, in two forms: `QuotientProver` alpha-folds an
`eval_fn`'s constraints, `Pil2QuotientProver` interprets the proving key's
pre-folded composite cExp."""

from __future__ import annotations

from collections.abc import Callable

import frx
from frx import Array
from zk_dtypes import goldilocksx3 as F3
from zorch.poly.univariate import powers
from zorch.stage import ProveResult, ProverStage
from zorch.utils.field import join_coeffs, split_coeffs

from zisk_zorch.commit.trace_commit import merkle_tree
from zisk_zorch.pil2 import (
    Pil2Key,
    cm_env,
    const_env,
    custom_env,
    publics_env,
    squeeze_stage_challenges,
    values_env,
)
from zisk_zorch.quotient.cexp_ref import _run_block
from zisk_zorch.quotient.quotient import quotient_from_constraints
from zisk_zorch.quotient.zerofier import inv_zerofier
from zisk_zorch.transcript.transcript import Transcript
from zisk_zorch.types import (
    Pil2QuotientBoundClaim,
    Pil2QuotientWitness,
    QuotientBoundClaim,
    QuotientCommitment,
    Stage2BoundClaim,
    TraceBoundClaim,
    TraceCommitment,
)


class QuotientProver(
    ProverStage[
        TraceBoundClaim, TraceCommitment, QuotientBoundClaim, QuotientCommitment
    ]
):
    """One claim reduction: squeeze alpha, fold the constraints by its
    powers, divide by the zerofier, commit `Q`, absorb its root.

    What it proves: every constraint evaluates to zero on every trace row —
    conditionally on the reduced claim, which the opening stage discharges.
    The witness is the trace commitment (constraints must be evaluated on the
    coset, where the zerofier is invertible). The statement's shape — domain
    size, constraint count — is read off the claim; only the AIR's circuits
    and the protocol parameters are configuration. Cubic rows commit as 3
    contiguous base limbs, so the leaf hash matches the FRI seam."""

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
        # The K constraints fold by ascending powers of the squeezed
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


class Pil2QuotientProver(
    ProverStage[
        Stage2BoundClaim,
        Pil2QuotientWitness,
        Pil2QuotientBoundClaim,
        QuotientCommitment,
    ]
):
    """pil2's CALCULATE_QUOTIENT as a claim reduction: squeeze the stage
    ``nStages + 1`` challenges, interpret the key's composite cExp over the
    committed extended sections, and commit ``q = cExp / Z_H``.

    No alpha fold happens here — the key's composite expression arrives
    pre-folded (its own quotient challenge is one of the squeezed stage
    challenges the SSA reads by id), so the reduction is a straight
    interpreter run. The environment mixes the witness's committed sections
    (extended trace, extended stage-2 columns) with the key's extended
    constant/custom sections and the claim's scalar sections; the inverse
    zerofier rides as the ``Zi`` operand. Cubic rows commit as 3 contiguous
    base limbs, exactly as `QuotientProver` does."""

    def __init__(self, key: Pil2Key) -> None:
        si, ss = key.starkinfo, key.starkinfo["starkStruct"]
        self._key = key
        self._si = si
        self._nb = ss["nBits"]
        self._blowup_bits = ss["nBitsExt"] - ss["nBits"]
        self._arity = ss["merkleTreeArity"]
        self._code = next(
            e
            for e in key.expressionsinfo["expressionsCode"]
            if e["expId"] == si["cExpId"]
        )["code"]

    def prove(
        self,
        claim: Stage2BoundClaim,
        witness: Pil2QuotientWitness,
        transcript: Transcript,
    ) -> ProveResult[Pil2QuotientBoundClaim, QuotientCommitment]:
        si = self._si
        challenges = dict(claim.challenges)
        challenges.update(
            squeeze_stage_challenges(transcript, si["challengesMap"], si["nStages"] + 1)
        )
        bufs = {
            ("cm", 1): witness.trace_commit.extended,
            ("cm", 2): witness.stage2.commitment.extended,
            ("const", 0): self._key.const_ext,
        }
        for ci, buf in self._key.custom_ext.items():
            bufs[("custom", ci)] = buf
        env = {
            "cm": cm_env(si["cmPolsMap"], bufs),
            "const": const_env(bufs, si["nConstants"]),
            "custom": custom_env(bufs, si.get("customCommits", [])),
            "challenges": challenges,
            "publics": publics_env(claim.pil2.publics),
            "airvalues": values_env(claim.pil2.airvalues, si["airValuesMap"]),
            "airgroupvalues": claim.airgroupvalues,
            "proofvalues": values_env(claim.pil2.proofvalues, si["proofValuesMap"]),
            "zi": {0: inv_zerofier(self._nb, self._blowup_bits)},
        }
        # The environment enters the jit zone as an argument: closure-captured
        # arrays lower as in-graph constants, which crashes the compiler on
        # the zerofier coset (#67).
        quotient = frx.jit(
            lambda env: _run_block(self._code, env, 1 << self._blowup_bits)
        )(env)
        matrix = split_coeffs(quotient)
        root, layers = merkle_tree(self._arity).commit(matrix)
        transcript.put(root)
        return ProveResult(
            Pil2QuotientBoundClaim(
                pil2=claim.pil2,
                trace_root=claim.trace_root,
                root2=claim.root2,
                quotient_root=root,
                challenges=challenges,
                airgroupvalues=claim.airgroupvalues,
            ),
            QuotientCommitment(
                codeword=quotient, root=root, matrix=matrix, layers=layers
            ),
            transcript,
        )
