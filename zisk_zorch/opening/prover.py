"""The opening stage's prover role — the DEEP composition, FRI fold, and
query openings as the terminal stage discharging a `QuotientBoundClaim`,
plus the scheme's commit half."""

from __future__ import annotations

from collections.abc import Sequence
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
from zisk_zorch.deep.fri_polynomial import deep_fri_polynomial
from zisk_zorch.evals.lev import LevConstants, compute_lev, lev_constants
from zisk_zorch.fri.prover import prove, prove_queries
from zisk_zorch.fri.queries import sample_query_positions
from zisk_zorch.pil2 import (
    Pil2Key,
    committed_column,
    named_challenge,
    squeeze_stage_challenges,
)
from zisk_zorch.quotient.zerofier import _coset_points, _root
from zisk_zorch.transcript.transcript import Transcript, transcript_hash
from zisk_zorch.types import (
    InnerWitness,
    OpeningProof,
    OpeningWitness,
    Pil2OpeningWitness,
    Pil2QuotientBoundClaim,
    QuotientBoundClaim,
    TraceCommitment,
)


@partial(frx.jit, static_argnames=("n_bits", "blowup_bits", "opening_points"))
def _opening_deep_jit(
    trace_ext: Array,
    quotient: Array,
    domain: Array,
    lev_consts: LevConstants,
    transcript: Transcript,
    *,
    n_bits: int,
    blowup_bits: int,
    opening_points: tuple[int, ...],
) -> tuple[Transcript, Array, Array]:
    """The DEEP leg — z squeeze, column openings, evals absorb, vf squeeze,
    composition — as one shape-invariant ``@jit`` zone: the transcript rides
    through as a traced pytree, so the whole leg is one compiled executable
    instead of eager transcript hops between the inner
    `open_columns`/`deep_composition` zones. The compile keys on the committed
    shapes and the static schedule alone.

    `domain` and `lev_consts` arrive from the eager prologue: an in-trace
    coset feeding the composition's cubic reciprocal is #67's NVPTX crash
    trigger, and field constants enter as arguments per `LevConstants`. The
    zone RETURNS its transcript — the caller's object is not advanced in
    place across the boundary."""
    fri_pol, evals = deep_fri_polynomial(
        trace_ext,
        quotient,
        transcript,
        n_bits=n_bits,
        blowup_bits=blowup_bits,
        opening_points=opening_points,
        domain=domain,
        lev_consts=lev_consts,
    )
    return transcript, fri_pol, evals


class OpeningProver(
    ProverStage[QuotientBoundClaim, OpeningWitness, TrivialClaim, OpeningProof]
):
    """The whole opening — the DEEP composition, the FRI fold, and the
    query openings — as the terminal stage discharging a `QuotientBoundClaim`.

    The two halves bracket the inner proof: `commit` binds the trace before
    any challenge exists, `prove` is the open. Only the open reduces a claim,
    so only the open is the Stage role; the commit is the scheme's other half.

    What it proves, in three internal steps whose seams are prover-data
    seams (each consumes a challenge the previous step squeezed and produces
    the next step's input, which is why they share one stage):

    - DEEP: the transmitted OOD openings are the committed columns' true
      values. Each term `(f(x) - f(z)) / (x - z)` is a polynomial only if the
      claimed `f(z)` is genuine, so batching the terms reduces every opening's
      honesty to one low-degree claim.
    - FRI: that DEEP codeword is (close to) low-degree. Each beta-fold cuts
      the degree by `2**fold_bits` while preserving low-degreeness iff it held
      before, until the final polynomial is small enough to transmit and check
      directly.
    - Queries: the fold chain was computed honestly. Everything so far
      exchanged Merkle roots, not data; the query openings let the verifier
      re-derive each fold at `n_queries` random positions and check it against
      the committed layers, back to the trace and quotient leaves themselves.
      Grinding (`pow_bits`) makes retrying for favorable positions costly.
    """

    def __init__(
        self,
        *,
        blowup_bits: int,
        steps: list[int],
        arity: int,
        pow_bits: int,
        n_queries: int,
        opening_points: Sequence[int] = (0,),
        jit: bool = True,
    ) -> None:
        self._blowup_bits = blowup_bits
        self._steps = steps
        self._arity = arity
        self._pow_bits = pow_bits
        self._n_queries = n_queries
        self._opening_points = opening_points
        self._jit = jit

    def commit(self, witness: InnerWitness) -> TraceCommitment:
        """Commit the trace: LDE onto the coset, Merkle-commit the rows —
        the commit half of the scheme whose open half is `prove`; only the
        open reduces a claim, so only the open is the Stage role, and
        `TraceCommitment` is what commit hands forward.

        No transcript argument: committing is not a transcript operation. The
        composite absorbs the returned root through `bind_trace_commitment`,
        so the Fiat-Shamir binding has one visible home rather than hiding
        inside the scheme."""
        return commit_trace(
            witness.trace, blowup=1 << self._blowup_bits, arity=self._arity
        )

    def _fri_input(
        self,
        claim: QuotientBoundClaim,
        witness: OpeningWitness,
        transcript: Transcript,
    ) -> tuple[Array, Array | None, Transcript]:
        """The codeword FRI folds, the OOD openings the proof transmits, and
        the advanced transcript — the DEEP composition. `EchoOpening`
        overrides this hook.

        Under the `jit` knob the leg runs as the `_opening_deep_jit` zone,
        with the domain coset materialized here in the eager prologue (#67);
        eagerly it is the same flow with transcript hops between the inner
        zones. Byte-identical either way — exact field ops, same schedule."""
        if self._jit:
            domain = _coset_points(claim.inner.n_bits, self._blowup_bits)
            consts = lev_constants(list(self._opening_points), claim.inner.n_bits)
            transcript, fri_pol, evals = _opening_deep_jit(
                witness.trace_commit.extended,
                witness.quotient.codeword,
                domain,
                consts,
                transcript,
                n_bits=claim.inner.n_bits,
                blowup_bits=self._blowup_bits,
                opening_points=tuple(self._opening_points),
            )
            return fri_pol, evals, transcript
        fri_pol, evals = deep_fri_polynomial(
            witness.trace_commit.extended,
            witness.quotient.codeword,
            transcript,
            n_bits=claim.inner.n_bits,
            blowup_bits=self._blowup_bits,
            opening_points=self._opening_points,
        )
        return fri_pol, evals, transcript

    def prove(
        self,
        claim: QuotientBoundClaim,
        witness: OpeningWitness,
        transcript: Transcript,
    ) -> ProveResult[TrivialClaim, OpeningProof]:
        fri_pol, evals, transcript = self._fri_input(claim, witness, transcript)
        fri, fri_layers = prove(
            fri_pol, self._steps, arity=self._arity, transcript=transcript
        )
        n_bits_ext = claim.inner.n_bits + self._blowup_bits
        positions, nonce = sample_query_positions(
            transcript,
            fri.final_pol,
            pow_bits=self._pow_bits,
            n_queries=self._n_queries,
            n_bits_ext=n_bits_ext,
        )
        # Every challenge is squeezed by now and openings never re-enter the
        # transcript, so the per-query Merkle walks may batch freely: one
        # vmapped kernel per tree instead of a dispatch per query, with no
        # effect on the byte stream.
        ext_mask = (1 << n_bits_ext) - 1
        idx_ext = fnp.asarray(np.asarray(positions)) & ext_mask
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
        trace_openings = [[trace_batched[q]] for q in range(len(positions))]
        quotient_openings = [[quotient_batched[q]] for q in range(len(positions))]
        fri_openings = prove_queries(fri_layers, positions)
        return ProveResult(
            TrivialClaim(),
            OpeningProof(
                evals=evals,
                fri=fri,
                nonce=nonce,
                positions=positions,
                trace_openings=trace_openings,
                quotient_openings=quotient_openings,
                fri_openings=fri_openings,
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


class EchoOpening(OpeningProver):
    """Placeholder opening: fold FRI over the quotient codeword itself,
    skipping the out-of-domain opening (`evals` stays None). The quotient is a
    valid cubic FRI input, so this drives the spine end to end for
    wiring/shape tests — but with no OOD openings the quotient's consistency
    with the committed trace is never tested, so it proves nothing about the
    trace. Not for conformance."""

    def _fri_input(
        self,
        claim: QuotientBoundClaim,
        witness: OpeningWitness,
        transcript: Transcript,
    ) -> tuple[Array, Array | None, Transcript]:
        return witness.quotient.codeword, None, transcript
