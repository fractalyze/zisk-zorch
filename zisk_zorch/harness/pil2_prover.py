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
from zorch.stage import ProveResult, ProverStage, TrivialClaim
from zorch.utils.field import split_coeffs

from zisk_zorch.commit.openings import group_proof
from zisk_zorch.commit.trace_commit import commit_trace, merkle_tree
from zisk_zorch.evals.lev import compute_lev
from zisk_zorch.fri.prover import prove, prove_queries
from zisk_zorch.fri.queries import sample_query_positions
from zisk_zorch.golden import embed
from zisk_zorch.harness.pil2 import (
    Pil2Key,
    absorb_stage2_airvalues,
    absorb_words,
    challenge_id,
    cm_env,
    committed_column,
    const_env,
    custom_env,
    deep_two_challenge,
    hint_value,
    open_evmap_columns,
    scalar_env,
    squeeze_stage_challenges,
)
from zisk_zorch.logup.bus import LogUpBus
from zisk_zorch.quotient.cexp_ref import run_block
from zisk_zorch.quotient.compute_q import compute_q
from zisk_zorch.quotient.zerofier import _coset_points, inv_zerofier
from zisk_zorch.transcript.transcript import Transcript, absorb_section
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
class LogUpBoundClaim:
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
class LogUpCommitment:
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
    logup: LogUpCommitment


@dataclass(frozen=True)
class Pil2OpeningWitness:
    """What discharging a `Pil2QuotientBoundClaim` takes: every committed
    tree's prover data — the extended sections the evals and DEEP batch read,
    and the digest layers the query phase opens."""

    trace_commit: TraceCommitment
    logup: LogUpCommitment
    quotient: QuotientCommitment


# -- the three pil2 stage roles ----------------------------------------------


def _hint_column(v: dict, env: dict, exps: dict, n: int) -> Array:
    """A hint-field value as an `(n,)` column over `env` — a committed
    column, an SSA expression, or a literal broadcast."""
    if v.get("rowOffset", 0):
        raise NotImplementedError(f"hint value with a row offset: {v}")
    if v["op"] == "cm":
        return env["cm"][v["id"]]
    if v["op"] == "tmp":
        return run_block(exps[v["id"]], env, 1)
    if v["op"] == "number":
        return fnp.broadcast_to(embed([v["value"]]).reshape(()), (n,))
    raise NotImplementedError(f"unhandled hint value op {v['op']!r}")


class LogUpWitnessProver(
    ProverStage[Pil2TraceBoundClaim, InnerWitness, LogUpBoundClaim, LogUpCommitment]
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
        self._family = key.hash_family
        self._n = 1 << ss["nBits"]
        self._blowup = 1 << (ss["nBitsExt"] - ss["nBits"])
        self._arity = ss["merkleTreeArity"]
        ei = key.expressionsinfo
        self._exps = {e["expId"]: e["code"] for e in ei["expressionsCode"]}
        self._im_hints = [h for h in ei["hintsInfo"] if h["name"] == "im_col"]
        gsums = [h for h in ei["hintsInfo"] if h["name"] == "gsum_col"]
        assert len(gsums) == 1, f"expected one gsum_col hint, got {len(gsums)}"
        self._gsum_hint = gsums[0]
        self._logup_cols = sorted(
            ((i, p) for i, p in enumerate(si["cmPolsMap"]) if p["stage"] == 2),
            key=lambda ip: ip[1]["stagePos"],
        )
        # Jitted once per role: the SSA blocks are key-fixed configuration,
        # and a per-prove `frx.jit` wrapper would retrace on every instance
        # proved with the same key.
        self._logup_jit = frx.jit(self._logup_columns)
        # Jitted for the same reasons as the opening role's commit: a dedicated
        # hash fusion only exists inside a jit region, and the eager LDE emits a
        # standalone `lax.ntt` pass fusion that can overrun ptxas's 48 KB static
        # shared-memory cap at ZisK Main width.
        self._commit_jit = frx.jit(self._commit_fn)

    def _commit_fn(self, matrix):
        # Components, not the dataclass: `TraceCommitment` is not a pytree.
        c = commit_trace(
            matrix,
            blowup=self._blowup,
            arity=self._arity,
            hash_family=self._family,
        )
        return c.root, c.digest_layers, c.extended

    def _logup_columns(self, env):
        # The environment enters as an argument: a closure-captured array
        # lowers as an in-graph constant, which crashes the compiler on
        # large cases (#67).
        n = self._n
        for hint in self._im_hints:
            num = _hint_column(hint_value(hint, "numerator"), env, self._exps, n)
            den = _hint_column(hint_value(hint, "denominator"), env, self._exps, n)
            env["cm"][hint_value(hint, "reference")["id"]] = num / den
        gsum_num = _hint_column(
            hint_value(self._gsum_hint, "numerator_air"), env, self._exps, n
        )
        gsum_den = _hint_column(
            hint_value(self._gsum_hint, "denominator_air"), env, self._exps, n
        )
        gsum = LogUpBus.grand_sum(gsum_num[None, :], gsum_den[None, :])
        env["cm"][hint_value(self._gsum_hint, "reference")["id"]] = gsum
        # pil2's single-row direct update: bus terms that never materialize
        # an im column enter the airgroup export as one scalar added to the
        # running sum's last entry — they are NOT part of the committed
        # column (std_sum.pil; the dumped cm2 pins this).
        direct = self._direct_term(env)
        result = LogUpBus.gsum_result(gsum)
        if direct is not None:
            result = result + direct
        for i, p in self._logup_cols:
            if p.get("imPol"):
                env["cm"][i] = run_block(self._exps[p["expId"]], env, 1)
        # A stage-2 column may be base-field (dim 1, e.g. Keccakf.ImPol):
        # it contributes one limb column where a cubic contributes three.
        return (
            fnp.concatenate(
                [
                    split_coeffs(env["cm"][i])
                    if p["dim"] == 3
                    else env["cm"][i].reshape(n, 1)
                    for i, p in self._logup_cols
                ],
                axis=1,
            ),
            result,
        )

    def _direct_term(self, env):
        """The gsum hint's direct contribution — a scalar expression (or the
        literal 0), divided by its denominator unless that is the literal 1
        (a dead cubic reciprocal otherwise)."""
        num_v = hint_value(self._gsum_hint, "numerator_direct")
        if num_v["op"] == "number" and int(num_v["value"]) == 0:
            return None
        num = run_block(self._exps[num_v["id"]], env, 1)
        den_v = hint_value(self._gsum_hint, "denominator_direct")
        if not (den_v["op"] == "number" and int(den_v["value"]) == 1):
            num = num / run_block(self._exps[den_v["id"]], env, 1)
        return num

    def prove(
        self,
        claim: Pil2TraceBoundClaim,
        witness: InnerWitness,
        transcript: Transcript,
    ) -> ProveResult[LogUpBoundClaim, LogUpCommitment]:
        si = self._key.starkinfo
        challenges = squeeze_stage_challenges(transcript, si["challengesMap"], 2)
        # The interpreter environment over the base domain. Absent classes
        # (custom, zi, later-stage sections) stay empty so a stage-2 hint
        # expression reaching for one fails loudly instead of silently
        # reading the wrong domain.
        env = {
            "cm": {i: witness.trace[:, i] for i in range(claim.pil2.n_cols)},
            "const": const_env({("const", 0): self._key.const_base}, si["nConstants"]),
            "custom": {},
            **scalar_env(
                si,
                publics=claim.pil2.publics,
                airvalues=claim.pil2.airvalues,
                proofvalues=claim.pil2.proofvalues,
                challenges=challenges,
                airgroupvalues={},
            ),
            "zi": {},
        }

        matrix, gsum_result = self._logup_jit(env)
        airgroupvalues = {hint_value(self._gsum_hint, "result")["id"]: gsum_result}
        assert matrix.shape[1] == si["mapSectionsN"]["cm2"], "cm2 width mismatch"
        root, digest_layers, extended = self._commit_jit(matrix)
        commitment = TraceCommitment(
            root=root, digest_layers=digest_layers, extended=extended
        )
        transcript.put(commitment.root)
        absorb_stage2_airvalues(transcript, claim.pil2.airvalues, si["airValuesMap"])
        return ProveResult(
            LogUpBoundClaim(
                pil2=claim.pil2,
                trace_root=claim.trace_root,
                root2=commitment.root,
                challenges=challenges,
                airgroupvalues=airgroupvalues,
            ),
            LogUpCommitment(matrix=matrix, commitment=commitment),
            transcript,
        )


class Pil2QuotientProver(
    ProverStage[
        LogUpBoundClaim,
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
    zerofier rides as the ``Zi`` operand. The committed section is pil2's
    ``computeQ`` of the evaluations — the identity (3 contiguous base limbs,
    `QuotientProver`'s shape) only when ``qDeg == 1``; a real AIR like ZisK's
    Main has qDeg 2, committing ``3·qDeg`` lanes per row."""

    def __init__(self, key: Pil2Key) -> None:
        si, ss = key.starkinfo, key.starkinfo["starkStruct"]
        self._key = key
        self._family = key.hash_family
        self._si = si
        self._nb = ss["nBits"]
        self._blowup_bits = ss["nBitsExt"] - ss["nBits"]
        self._arity = ss["merkleTreeArity"]
        self._code = next(
            e
            for e in key.expressionsinfo["expressionsCode"]
            if e["expId"] == si["cExpId"]
        )["code"]
        # Jitted once per role (key-fixed SSA); the environment enters as an
        # argument — closure-captured arrays lower as in-graph constants,
        # which crashes the compiler on the zerofier coset (#67).
        self._q_jit = frx.jit(
            lambda env: run_block(self._code, env, 1 << self._blowup_bits)
        )
        q_deg = si["mapSectionsN"]["cm" + str(si["nStages"] + 1)] // 3
        self._qsec_jit = frx.jit(
            split_coeffs
            if q_deg == 1
            else lambda q: compute_q(
                split_coeffs(q), ss["nBits"], ss["nBitsExt"], q_deg
            )
        )

    def prove(
        self,
        claim: LogUpBoundClaim,
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
            ("cm", 2): witness.logup.commitment.extended,
            ("const", 0): self._key.const_ext,
        }
        for ci, buf in self._key.custom_ext.items():
            bufs[("custom", ci)] = buf
        env = {
            "cm": cm_env(si["cmPolsMap"], bufs),
            "const": const_env(bufs, si["nConstants"]),
            "custom": custom_env(bufs, si.get("customCommits", [])),
            **scalar_env(
                si,
                publics=claim.pil2.publics,
                airvalues=claim.pil2.airvalues,
                proofvalues=claim.pil2.proofvalues,
                challenges=challenges,
                airgroupvalues=claim.airgroupvalues,
            ),
            "zi": {0: inv_zerofier(self._nb, self._blowup_bits)},
        }
        quotient = self._q_jit(env)
        matrix = self._qsec_jit(quotient)
        root, layers = merkle_tree(self._arity, self._family).commit(matrix)
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


class Pil2OpeningProver(
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
        self._family = key.hash_family
        self._si = si
        self._nb, self._nbe = ss["nBits"], ss["nBitsExt"]
        self._steps = [s["nBits"] for s in ss["steps"]]
        self._arity = ss["merkleTreeArity"]
        self._pow_bits = ss["powBits"]
        self._n_queries = ss["nQueries"]
        self._hash_commits = ss.get("hashCommits", False)
        # Jitted once per role; evals/coset enter as arguments (the in-trace
        # coset feeding the cubic reciprocal is #67's crash trigger). The
        # evMap COLUMNS are built inside each trace from the section
        # buffers: an eager per-entry `committed_column` list materializes
        # |evMap| full cubic columns on device, which exceeds device memory
        # at ZisK Main width (183 entries x 2^23) — inside the jit they are
        # fused slices that never exist whole.
        self._evals_jit = frx.jit(self._evals_fn)
        self._deep_jit = frx.jit(self._deep_fn)
        # The commit is jitted for the same reason, plus two of its own. A
        # dedicated hash fusion only exists INSIDE a jit region, so an eager
        # commit silently gives up the fused Poseidon kernel; and the eager
        # LDE dispatches `lax.ntt` as a standalone pass fusion, whose config
        # is assembled outside the NTT rewriter and can request more STATIC
        # shared memory than ptxas allows (48 KB) even though it fits the
        # device's dynamic budget -- a hard `uses too much shared data` build
        # error at ZisK Main width on sm_120.
        self._commit_jit = frx.jit(self._commit_fn)

    def _columns(self, bufs: dict) -> list:
        return [
            committed_column(e, self._si["cmPolsMap"], bufs) for e in self._si["evMap"]
        ]

    def _evals_fn(self, bufs: dict, lev):
        return open_evmap_columns(
            self._columns(bufs),
            self._si["evMap"],
            lev,
            stride=1 << (self._nbe - self._nb),
        )

    def _deep_fn(self, bufs: dict, evals, domain, xi, vf1, vf2):
        return deep_two_challenge(
            self._columns(bufs),
            evals,
            domain,
            xi,
            vf1,
            vf2,
            ev_map=self._si["evMap"],
            openings=self._si["openingPoints"],
            n_bits=self._nb,
        )

    def _commit_fn(self, trace):
        # `TraceCommitment` is a plain dataclass, not a registered pytree, so
        # the traced function hands back its components and `commit` rebuilds
        # it outside the trace.
        c = commit_trace(
            trace,
            blowup=1 << (self._nbe - self._nb),
            arity=self._arity,
            hash_family=self._family,
        )
        return c.root, c.digest_layers, c.extended

    def commit(self, witness: InnerWitness) -> TraceCommitment:
        """The scheme's commit half — identical to `OpeningProver.commit`;
        the pil2 composite absorbs no root for it (the global challenge
        arrives already bound to it)."""
        root, digest_layers, extended = self._commit_jit(witness.trace)
        return TraceCommitment(
            root=root, digest_layers=digest_layers, extended=extended
        )

    def prove(
        self,
        claim: Pil2QuotientBoundClaim,
        witness: Pil2OpeningWitness,
        transcript: Transcript,
    ) -> ProveResult[TrivialClaim, OpeningProof]:
        si = self._si
        opening_points = si["openingPoints"]
        challenges = dict(claim.challenges)
        challenges.update(
            squeeze_stage_challenges(transcript, si["challengesMap"], si["nStages"] + 2)
        )
        xi = challenges[challenge_id(si["challengesMap"], "std_xi")]

        bufs = {
            ("cm", 1): witness.trace_commit.extended,
            ("cm", 2): witness.logup.commitment.extended,
            ("cm", si["nStages"] + 1): witness.quotient.matrix,
            ("const", 0): self._key.const_ext,
        }
        for ci, buf in self._key.custom_ext.items():
            bufs[("custom", ci)] = buf

        # lev materialized before the openings (its trace inside the same
        # zone regresses the openings' fusion — see
        # zisk-zorch@lev-must-be-materialized).
        lev = compute_lev(xi, list(opening_points), self._nb)
        evals = self._evals_jit(bufs, lev)
        absorb_section(
            transcript, split_coeffs(evals).reshape(-1), hashed=self._hash_commits
        )

        challenges.update(
            squeeze_stage_challenges(transcript, si["challengesMap"], si["nStages"] + 3)
        )
        vf1 = challenges[challenge_id(si["challengesMap"], "std_vf1")]
        vf2 = challenges[challenge_id(si["challengesMap"], "std_vf2")]

        domain = _coset_points(self._nb, self._nbe - self._nb)
        fri_pol = self._deep_jit(bufs, evals, domain, xi, vf1, vf2)

        fri, fri_layers = prove(
            fri_pol,
            self._steps,
            arity=self._arity,
            transcript=transcript,
            hash_family=self._family,
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
                merkle_tree(self._arity, self._family),
                witness.trace_commit.extended,
                witness.trace_commit.digest_layers,
            )
        )(idx_ext)
        quotient_batched = frx.vmap(
            partial(
                group_proof,
                merkle_tree(self._arity, self._family),
                witness.quotient.matrix,
                witness.quotient.layers,
            )
        )(idx_ext)
        # One device→host copy per tree, then host row views — a per-query
        # device slice here is thousands of eager dispatches (see
        # `prove_queries`).
        trace_rows, quotient_rows = np.asarray(trace_batched), np.asarray(
            quotient_batched
        )
        return ProveResult(
            TrivialClaim(),
            OpeningProof(
                evals=evals,
                fri=fri,
                nonce=nonce,
                positions=positions,
                trace_openings=[[row] for row in trace_rows],
                quotient_openings=[[row] for row in quotient_rows],
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

    `prove_stages` is the whole API and the byte-gate seam: it yields each
    stage's result as the stage finishes, so `verify_inner_proof`'s composed
    gate can check every reduction the moment it lands and stop before the
    later stages pay their compute on a mismatch. A wire-proof assembler
    can return when something consumes one (the deferred serialize/FFI
    work)."""

    def __init__(self, key: Pil2Key) -> None:
        self.logup = LogUpWitnessProver(key)
        self.quotient = Pil2QuotientProver(key)
        self.opening = Pil2OpeningProver(key)

    def prove_stages(
        self,
        claim: Pil2Claim,
        witness: InnerWitness,
        transcript: Transcript,
    ):
        """Run the schedule, yielding ``(stage_name, result)`` as each stage
        finishes — ``trace_commit`` yields the `TraceCommitment`; the Stage
        roles (``logup_witness``, ``quotient``, ``opening``) their
        `ProveResult`s, prover data included."""
        assert witness.trace.shape == (
            1 << claim.n_bits,
            claim.n_cols,
        ), "claim's trace shape does not match the witness"
        commitment = self.opening.commit(witness)
        yield "trace_commit", commitment
        absorb_words(transcript, claim.global_challenge)
        bound = Pil2TraceBoundClaim(pil2=claim, trace_root=commitment.root)
        logup = self.logup.prove(bound, witness, transcript)
        yield "logup_witness", logup
        quotient = self.quotient.prove(
            logup.reduced_claim,
            Pil2QuotientWitness(commitment, logup.reduction_proof),
            logup.transcript,
        )
        yield "quotient", quotient
        yield (
            "opening",
            self.opening.prove(
                quotient.reduced_claim,
                Pil2OpeningWitness(
                    commitment, logup.reduction_proof, quotient.reduction_proof
                ),
                quotient.transcript,
            ),
        )
