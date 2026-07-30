"""pil2's CALCULATE_WITNESS_STD + IM_POLS — the stage-2 witness role.

`WitnessStdProver` reduces a `Pil2TraceBoundClaim` to a `Stage2BoundClaim`:
squeeze the stage-2 challenges, chain the proving key's hint expressions into
the committed stage-2 columns, extend-and-merkelize them, and absorb the root
plus the stage-2 air values — pil2 genProof's STEP_2 on our transcript.

The column chain follows the key's hints, not per-AIR wiring:

- every ``im_col`` hint's reference column is its numerator over its
  denominator (both SSA expressions over the base domain);
- the ``gsum_col`` hint's reference is the running prefix sum of its
  ``numerator_air / denominator_air`` local terms (`grand_sum`), and its
  ``result`` names the airgroup value the last row settles;
- every ``imPol``-flagged ``cmPolsMap`` entry evaluates its own expression.

gsum runs before the imPol expressions: an ImPol may read the gsum column
(SpecifiedRanges does), while gsum's inputs are im references the im_col
hints already produced. Verified byte-identical to pil2's ``cm2`` section on
SpecifiedRanges, FibonacciSquare, and Module captures.

std driver: https://github.com/0xPolygonHermez/pil2-proofman/blob/v1.0.0-alpha/pil2-stark/src/starkpil/gen_proof.hpp#L24-L65
"""

from __future__ import annotations

import frx
import frx.numpy as fnp
from frx import Array
from zorch.stage import ProveResult, ProverStage
from zorch.utils.field import split_coeffs

from zisk_zorch.commit.trace_commit import commit_trace
from zisk_zorch.golden import embed
from zisk_zorch.pil2 import (
    Pil2Key,
    absorb_words,
    const_env,
    publics_env,
    squeeze_stage_challenges,
    values_env,
)
from zisk_zorch.quotient.cexp_ref import _run_block
from zisk_zorch.quotient.gsum import grand_sum
from zisk_zorch.transcript.transcript import Transcript
from zisk_zorch.types import (
    InnerWitness,
    Pil2TraceBoundClaim,
    Stage2BoundClaim,
    Stage2Commitment,
)


def _hint_value(v: dict, env: dict, exps: dict, n: int) -> Array:
    """A hint-field value as an `(n,)` column over `env` — a committed
    column, an SSA expression, or a literal broadcast."""
    if v.get("rowOffset", 0):
        raise NotImplementedError(f"hint value with a row offset: {v}")
    if v["op"] == "cm":
        return env["cm"][v["id"]]
    if v["op"] == "tmp":
        return _run_block(exps[v["id"]], env, 1)
    if v["op"] == "number":
        return fnp.broadcast_to(embed([v["value"]]).reshape(()), (n,))
    raise NotImplementedError(f"unhandled hint value op {v['op']!r}")


class WitnessStdProver(
    ProverStage[Pil2TraceBoundClaim, InnerWitness, Stage2BoundClaim, Stage2Commitment]
):
    """One claim reduction: squeeze the stage-2 challenges, materialize the
    witness-STD columns from the key's hints, commit them, absorb the root
    and the stage-2 air values.

    The proving key (`Pil2Key`) is configuration — fixed for every instance
    of the AIR — while the scalar sections the hint expressions read arrive
    on the claim. The witness is the settled stage-1 trace: the hints
    evaluate over the base domain, unlike the quotient's coset run."""

    def __init__(self, key: Pil2Key) -> None:
        si, ss = key.starkinfo, key.starkinfo["starkStruct"]
        self._key = key
        self._n = 1 << ss["nBits"]
        self._blowup = 1 << (ss["nBitsExt"] - ss["nBits"])
        self._arity = ss["merkleTreeArity"]
        ei = key.expressionsinfo
        self._exps = {e["expId"]: e["code"] for e in ei["expressionsCode"]}
        self._im_hints = [h for h in ei["hintsInfo"] if h["name"] == "im_col"]
        gsums = [h for h in ei["hintsInfo"] if h["name"] == "gsum_col"]
        assert len(gsums) == 1, f"expected one gsum_col hint, got {len(gsums)}"
        self._gsum_hint = gsums[0]
        self._stage2_cols = sorted(
            ((i, p) for i, p in enumerate(si["cmPolsMap"]) if p["stage"] == 2),
            key=lambda ip: ip[1]["stagePos"],
        )

    @staticmethod
    def _field(hint: dict, name: str) -> dict:
        return next(f for f in hint["fields"] if f["name"] == name)["values"][0]

    def prove(
        self,
        claim: Pil2TraceBoundClaim,
        witness: InnerWitness,
        transcript: Transcript,
    ) -> ProveResult[Stage2BoundClaim, Stage2Commitment]:
        si, n = self._key.starkinfo, self._n
        challenges = squeeze_stage_challenges(transcript, si["challengesMap"], 2)
        # The interpreter environment over the base domain. Absent classes
        # (custom, zi, later-stage sections) stay empty so a stage-2 hint
        # expression reaching for one fails loudly instead of silently
        # reading the wrong domain.
        env = {
            "cm": {i: witness.trace[:, i] for i in range(claim.pil2.n_cols)},
            "const": const_env({("const", 0): self._key.const_base}, si["nConstants"]),
            "custom": {},
            "challenges": challenges,
            "publics": publics_env(claim.pil2.publics),
            "airvalues": values_env(claim.pil2.airvalues, si["airValuesMap"]),
            "airgroupvalues": {},
            "proofvalues": values_env(claim.pil2.proofvalues, si["proofValuesMap"]),
            "zi": {},
        }
        if self._field(self._gsum_hint, "numerator_direct")["value"] != "0":
            raise NotImplementedError("gsum_col with a direct term")

        # One jitted zone with the environment as its argument: a
        # closure-captured array lowers as an in-graph constant, which
        # crashes the compiler on large cases (#67).
        def stage2_columns(env):
            for hint in self._im_hints:
                num = _hint_value(self._field(hint, "numerator"), env, self._exps, n)
                den = _hint_value(self._field(hint, "denominator"), env, self._exps, n)
                env["cm"][self._field(hint, "reference")["id"]] = num / den
            gsum_num = _hint_value(
                self._field(self._gsum_hint, "numerator_air"), env, self._exps, n
            )
            gsum_den = _hint_value(
                self._field(self._gsum_hint, "denominator_air"), env, self._exps, n
            )
            gsum = grand_sum(gsum_num[:, None], gsum_den[:, None])
            env["cm"][self._field(self._gsum_hint, "reference")["id"]] = gsum
            for i, p in self._stage2_cols:
                if p.get("imPol"):
                    env["cm"][i] = _run_block(self._exps[p["expId"]], env, 1)
            return (
                fnp.concatenate(
                    [split_coeffs(env["cm"][i]) for i, _ in self._stage2_cols], axis=1
                ),
                gsum[-1],
            )

        matrix, gsum_result = frx.jit(stage2_columns)(env)
        airgroupvalues = {self._field(self._gsum_hint, "result")["id"]: gsum_result}
        assert matrix.shape[1] == si["mapSectionsN"]["cm2"], "cm2 width mismatch"
        commitment = commit_trace(matrix, blowup=self._blowup, arity=self._arity)
        transcript.put(commitment.root)
        words, off = claim.pil2.airvalues, 0
        for v in si["airValuesMap"]:
            if v["stage"] == 1:
                off += 1
            elif v["stage"] == 2:
                absorb_words(transcript, words[off : off + 3])
                off += 3
        return ProveResult(
            Stage2BoundClaim(
                pil2=claim.pil2,
                trace_root=claim.trace_root,
                root2=commitment.root,
                challenges=challenges,
                airgroupvalues=airgroupvalues,
            ),
            Stage2Commitment(matrix=matrix, commitment=commitment),
            transcript,
        )
