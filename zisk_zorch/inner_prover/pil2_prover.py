"""The pil2-mode prover — the byte-match harness's composite, not a
production role.

Everything here exists to reproduce a real pil2-proofman ``genProof``
byte-for-byte so `verify_inner_proof` can seal the composition against a
capture; the production protocol is the wired `InnerProver` family, which
is why these roles and their claim types live beside the byte-match tools
rather than in the stage packages and `types.py`. The split keeps the
production wire surface free of pil2-conformance shapes — nothing outside
this harness may import them.

Same composite discipline as the production side: three `ProverStage`
roles — witness-STD, cExp quotient, pil2 opening — exchange claims both
sides could derive, configured once per proving key (`pil2.Pil2Key`),
with the statement on `Pil2Claim` and the trace on `InnerWitness`. The
schedule differences pil2 dictates: the transcript seeds with the
contributions-phase global challenge (``root1`` is never absorbed — it
rides inside the seed), the stage-2 witness role runs between the commit
and the quotient, and section absorbs are ``calculateHash`` digests under
a ``hashCommits`` stark struct.

Schedule source: ``gen_proof.hpp`` on the pinned pil2-proofman fork
(https://github.com/fractalyze/pil2-proofman/blob/11999a69/pil2-stark/src/starkpil/gen_proof.hpp).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import frx
import frx.numpy as fnp
import numpy as np
from frx import Array
from zk_dtypes import goldilocks as F
from zk_dtypes import goldilocksx3 as F3
from zk_dtypes import pfinfo
from zorch.pcs.deep import open_columns
from zorch.stage import ProveResult, ProverStage, TrivialClaim
from zorch.utils.field import split_coeffs

from zisk_zorch.commit.openings import group_proof
from zisk_zorch.commit.trace_commit import commit_trace, merkle_tree
from zisk_zorch.evals.lev import compute_lev
from zisk_zorch.fri.prover import prove, prove_queries
from zisk_zorch.fri.queries import sample_query_positions
from zisk_zorch.golden import embed
from zisk_zorch.inner_prover.pil2 import (
    Pil2Key,
    absorb_words,
    cm_env,
    committed_column,
    const_env,
    custom_env,
    named_challenge,
    publics_env,
    squeeze_stage_challenges,
    values_env,
)
from zisk_zorch.quotient.cexp_ref import _run_block
from zisk_zorch.quotient.gsum import grand_sum
from zisk_zorch.quotient.zerofier import _coset_points, _root, inv_zerofier
from zisk_zorch.transcript.transcript import Transcript, transcript_hash
from zisk_zorch.types import (
    InnerWitness,
    OpeningProof,
    QuotientCommitment,
    TraceCommitment,
)

# -- the harness's claim / witness / proof types ------------------------------


@dataclass(frozen=True, kw_only=True)
class Pil2Claim:
    """A pil2 genProof instance's statement: some trace of this shape,
    together with these scalar sections, satisfies the proving key's AIR.

    The pil2-mode claims mirror the wired family above but condition on the
    proving key (the roles' `Pil2Key` configuration) instead of an `eval_fn`.
    The scalar sections ride as the dumped u64 word layouts (`pil2.values_env`
    owns the packing) because both roles read them that way: the prover packs
    them into interpreter environments and transcript absorbs, the verifier
    reads the same words off the wire. `global_challenge` is the
    contributions-phase seed — the transcript arrives already bound to every
    instance's stage-1 root through it, which is why no pil2 claim carries a
    bound trace root the way `TraceBoundClaim` does: the composite still
    commits the trace, but the schedule never absorbs `root1`."""

    n_bits: int
    n_cols: int
    publics: np.ndarray
    airvalues: np.ndarray
    proofvalues: np.ndarray
    global_challenge: np.ndarray


@dataclass(frozen=True, kw_only=True)
class Pil2TraceBoundClaim:
    """`Pil2Claim` after the commit half runs: one concrete trace is named by
    `trace_root`. The root is claim data for the wire even though the
    non-recursive schedule never absorbs it (see `Pil2Claim`)."""

    pil2: Pil2Claim
    trace_root: Array


@dataclass(frozen=True, kw_only=True)
class Stage2BoundClaim:
    """The stage-2 witness-STD columns committed under `root2` chain from the
    trace committed under `trace_root` via the key's hint expressions.

    `challenges` carries every challenge squeezed so far, keyed by its
    ``challengesMap`` index — the id the later stages' SSA operands reference.
    Both roles derive them from the transcript, so they are claim data, like
    `alpha` on `QuotientBoundClaim`. `airgroupvalues` is the stage's other
    product (the grand-sum result the last gsum row settles), transmitted on
    the wire and read back by the quotient's SSA."""

    pil2: Pil2Claim
    trace_root: Array
    root2: Array
    challenges: dict[int, Array]
    airgroupvalues: dict[int, Array]


@dataclass(frozen=True, kw_only=True)
class Pil2QuotientBoundClaim:
    """The codeword committed under `quotient_root` is the key's composite
    cExp over the committed sections, divided by the zerofier — the pil2
    analogue of `QuotientBoundClaim`, with the fold already baked into the
    key's pre-folded expression instead of an `alpha` power vector."""

    pil2: Pil2Claim
    trace_root: Array
    root2: Array
    quotient_root: Array
    challenges: dict[int, Array]
    airgroupvalues: dict[int, Array]


@dataclass(frozen=True)
class Stage2Commitment:
    """The stage-2 role's reduction proof: the base-domain witness-STD
    columns (the byte-gates compare them against pil2's ``cm2`` section) and
    their extend-and-merkelize commitment."""

    matrix: Array
    commitment: TraceCommitment


@dataclass(frozen=True)
class Pil2QuotientWitness:
    """What the pil2 quotient stage's SSA reads: both committed sections so
    far — the extended trace and the extended stage-2 columns."""

    trace_commit: TraceCommitment
    stage2: Stage2Commitment


@dataclass(frozen=True)
class Pil2OpeningWitness:
    """What discharging a `Pil2QuotientBoundClaim` takes: every committed
    tree's prover data — the extended sections the evals and DEEP batch read,
    and the digest layers the query phase opens."""

    trace_commit: TraceCommitment
    stage2: Stage2Commitment
    quotient: QuotientCommitment


@dataclass(frozen=True, kw_only=True)
class Pil2InnerProof:
    """The pil2-mode wire: the three stage roots, the airgroup values the
    stage-2 witness settles, and the opening discharge."""

    trace_root: Array
    root2: Array
    quotient_root: Array
    airgroupvalues: dict[int, Array]
    opening: OpeningProof


# -- the three pil2 stage roles ----------------------------------------------


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


class Pil2Opening(
    ProverStage[Pil2QuotientBoundClaim, Pil2OpeningWitness, TrivialClaim, OpeningProof]
):
    """pil2's opening phase — STEP_EVALS through the query draw — as the
    terminal stage discharging a `Pil2QuotientBoundClaim`.

    Same three-step shape as `OpeningProver` (DEEP, FRI, queries), with the
    pil2 protocol differences the key dictates:

    - the out-of-domain openings run over the key's ``evMap`` (every
      committed, constant, and custom column it names, at every
      ``openingPoints`` shift) via `compute_lev`'s weight matrix;
    - the evals absorb is `calculateHash` of the section under a
      ``hashCommits`` stark struct, as is the final polynomial's;
    - the DEEP batch is pil2's two-challenge form — vf2-Horner within an
      opening group, one reciprocal per group, vf1-Horner across groups —
      which zorch's one-challenge `deep_composition` cannot express.

    The query phase opens the trace, quotient, and FRI-layer trees. pil2
    additionally opens the stage-2, constant, and custom trees per query;
    those openings are not produced here (the byte-gates compare the
    positions and nonce, which pin the whole draw) — the verifier dual is
    deferred with them."""

    def __init__(self, key: Pil2Key) -> None:
        si, ss = key.starkinfo, key.starkinfo["starkStruct"]
        self._key = key
        self._si = si
        self._nb, self._nbe = ss["nBits"], ss["nBitsExt"]
        self._steps = [s["nBits"] for s in ss["steps"]]
        self._arity = ss["merkleTreeArity"]
        self._pow_bits = ss["powBits"]
        self._n_queries = ss["nQueries"]
        self._hash_commits = ss.get("hashCommits", False)

    def commit(self, witness: InnerWitness) -> TraceCommitment:
        """The scheme's commit half — identical to `OpeningProver.commit`;
        the pil2 composite absorbs no root for it (the global challenge
        arrives already bound to it)."""
        return commit_trace(
            witness.trace, blowup=1 << (self._nbe - self._nb), arity=self._arity
        )

    def _absorb_section(self, transcript: Transcript, limbs: Array) -> None:
        if self._hash_commits:
            transcript.put(transcript_hash(limbs, transcript.width))
        else:
            transcript.put(limbs)

    def prove(
        self,
        claim: Pil2QuotientBoundClaim,
        witness: Pil2OpeningWitness,
        transcript: Transcript,
    ) -> ProveResult[TrivialClaim, OpeningProof]:
        si, n = self._si, 1 << self._nb
        ev_map, opening_points = si["evMap"], si["openingPoints"]
        challenges = dict(claim.challenges)
        challenges.update(
            squeeze_stage_challenges(transcript, si["challengesMap"], si["nStages"] + 2)
        )
        xi = challenges[named_challenge(si["challengesMap"], "std_xi")]

        bufs = {
            ("cm", 1): witness.trace_commit.extended,
            ("cm", 2): witness.stage2.commitment.extended,
            ("cm", si["nStages"] + 1): witness.quotient.matrix,
            ("const", 0): self._key.const_ext,
        }
        for ci, buf in self._key.custom_ext.items():
            bufs[("custom", ci)] = buf
        columns = [committed_column(e, si["cmPolsMap"], bufs) for e in ev_map]

        # open_columns wants the base and extension columns as two blocks;
        # the split re-permutes to evMap order afterwards so the absorbed
        # section matches pil2's layout.
        lev = compute_lev(xi, list(opening_points), self._nb)
        is_base = [col.dtype == F for col in columns]
        order = [i for i, b in enumerate(is_base) if b] + [
            i for i, b in enumerate(is_base) if not b
        ]
        base = [columns[i] for i in order if is_base[i]]
        ext = [columns[i] for i in order[len(base) :]]
        stride = 1 << (self._nbe - self._nb)
        ne = 1 << self._nbe
        split = open_columns(
            fnp.stack(base, axis=1) if base else fnp.zeros((ne, 0), F),
            fnp.stack(ext, axis=1) if ext else fnp.zeros((ne, 0), F3),
            lev,
            [ev_map[i]["openingPos"] for i in order],
            stride=stride,
        )
        evals = fnp.zeros_like(split).at[np.array(order)].set(split)
        self._absorb_section(transcript, split_coeffs(evals).reshape(-1))

        challenges.update(
            squeeze_stage_challenges(transcript, si["challengesMap"], si["nStages"] + 3)
        )
        vf1 = challenges[named_challenge(si["challengesMap"], "std_vf1")]
        vf2 = challenges[named_challenge(si["challengesMap"], "std_vf2")]

        # pil2's computeFRIExpression: vf2-Horner within an opening group
        # (evMap order), one reciprocal per group, vf1-Horner across groups.
        # The coset enters the jit zone as an argument — an in-trace coset
        # feeding the cubic reciprocal is #67's compiler-crash trigger.
        g = int(np.asarray(_root(self._nb)))
        modulus = int(pfinfo(F).modulus)
        domain = _coset_points(self._nb, self._nbe - self._nb)

        def deep(columns, evals, domain, xi, vf1, vf2):
            fri_pol = None
            for k, prime in enumerate(opening_points):
                acc = None
                for i, e in enumerate(ev_map):
                    if e["openingPos"] != k:
                        continue
                    term = columns[i] - evals[i]
                    acc = term if acc is None else acc * vf2 + term
                shift = fnp.array(np.uint64(pow(g, prime % n, modulus)).astype(F))
                group = acc / (domain - xi * shift)
                fri_pol = group if fri_pol is None else fri_pol * vf1 + group
            return fri_pol

        fri_pol = frx.jit(deep)(columns, evals, domain, xi, vf1, vf2)

        fri, fri_layers = prove(
            fri_pol, self._steps, arity=self._arity, transcript=transcript
        )
        positions, nonce = sample_query_positions(
            transcript,
            fri.final_pol,
            pow_bits=self._pow_bits,
            n_queries=self._n_queries,
            n_bits_ext=self._steps[0],
            hash_commits=self._hash_commits,
        )
        idx_ext = fnp.asarray(np.asarray(positions))
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
        return ProveResult(
            TrivialClaim(),
            OpeningProof(
                evals=evals,
                fri=fri,
                nonce=nonce,
                positions=positions,
                trace_openings=[[trace_batched[q]] for q in range(len(positions))],
                quotient_openings=[
                    [quotient_batched[q]] for q in range(len(positions))
                ],
                fri_openings=prove_queries(fri_layers, positions),
            ),
            transcript,
        )


# -- the composite -----------------------------------------------------------


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
