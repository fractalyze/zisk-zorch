"""End-to-end inner-proof prover — a `ProveChain` of Stages over one transcript.

`prove_inner_chain` is a `zorch.round.ProveChain` of five **Stages** — trace
commit, quotient, DEEP, FRI, queries — threading one duplex `Transcript` and a
single **Bridge** (`InnerBridge`). Each stage's Merkle root is absorbed before
the next stage's challenges (`alpha`, the FRI fold betas, the query positions)
are squeezed from that running state — the byte stream pil2-proofman's `genProof`
drives.

- **Stage** — one step of the inner proof's heterogeneous sequence, a `Round`
  subclass named `*Stage`. Each runs its own inner rounds (FRI's layer chain).
- **Bridge** — the state a Stage hands the next (`InnerBridge`). It holds only
  what a later Stage reads from an earlier one; a Stage writes its own fields via
  `replace` and passes the rest through. Static config (arity, the fold schedule,
  `eval_fn`) lives on the Stage, not the Bridge.

The quotient-commit leaf layout mirrors the FRI seam's cubic convention (each
cubic row -> its 3 contiguous Goldilocks limbs, cf. `seam._cubic_to_base`), which
is pil2's `FIELD_EXTENSION`-contiguous memory order.

See `docs/architecture.md` for the DEEP seam and its byte-match boundary.

https://github.com/0xPolygonHermez/pil2-proofman/blob/v1.0.0-alpha/pil2-stark/src/starkpil/gen_proof.hpp
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

import numpy as np
from frx import Array

from zorch.poly.univariate import powers
from zorch.round import ProveChain, Round

from zisk_zorch.commit.openings import group_proof
from zisk_zorch.commit.trace_commit import TraceCommitment, commit_trace, merkle_tree
from zisk_zorch.deep.fri_polynomial import deep_fri_polynomial
from zisk_zorch.fri.prover import FriProof, prove, prove_queries
from zisk_zorch.fri.queries import sample_query_positions
from zisk_zorch.fri.seam import _base_to_cubic, _cubic_to_base
from zisk_zorch.quotient.quotient import quotient_from_constraints
from zisk_zorch.transcript.transcript import Transcript


def _fold_steps(n_bits_ext: int, fold_bits: int, final_bits: int) -> list[int]:
    """Strictly-decreasing FRI layer schedule `n_bits_ext -> ... -> final_bits`,
    folding by `fold_bits` per layer (the tail folds the remainder). Same schedule
    `bench_inner_proof._fold_steps` builds; kept here so the prover owns its FRI
    shape without importing a benchmark private."""
    steps = list(range(n_bits_ext, final_bits, -fold_bits))
    if not steps or steps[-1] != final_bits:
        steps.append(final_bits)
    return steps


@dataclass(frozen=True)
class InnerBridge:
    """What flows between stages: the trace plus each stage's outputs the next
    one consumes. Stages return it via ``replace`` — a stage writes its own
    fields and passes the rest through untouched."""

    trace: Array
    # Written by TraceCommitStage; read by QuotientStage (the extended trace it
    # folds constraints over), DeepStage (the committed columns it opens), and
    # QueryStage (the tree it opens per query position).
    trace_commit: TraceCommitment | None = None
    # Written by QuotientStage; read by DeepStage (a committed column) and, as
    # the base-limb matrix plus its digest layers, by QueryStage.
    quotient: Array | None = None
    quotient_root: Array | None = None
    quotient_matrix: Array | None = None
    quotient_layers: list[Array] | None = None
    # Written by DeepStage; read by FriStage as its codeword.
    fri_pol: Array | None = None
    # Written by FriStage; read by QueryStage (the layer trees it opens).
    fri: FriProof | None = None
    # Written by QueryStage; read by proof assembly.
    nonce: int | None = None
    query_positions: np.ndarray | None = None
    trace_openings: list[list[Array]] | None = None
    quotient_openings: list[list[Array]] | None = None
    fri_openings: list[list[Array]] | None = None


class TraceCommitStage(Round):
    """pil2 `extendAndMerkelize`: LDE the trace onto the coset and Merkle-commit
    it. Fiat-Shamir requires the root be absorbed here, before `QuotientStage`
    squeezes alpha off the same transcript. The message is the trace root."""

    def __init__(self, *, blowup: int, arity: int) -> None:
        self._blowup = blowup
        self._arity = arity

    def __call__(
        self, bridge: InnerBridge, transcript: Transcript
    ) -> tuple[InnerBridge, Transcript, Array]:
        commitment = commit_trace(bridge.trace, blowup=self._blowup, arity=self._arity)
        transcript.put(commitment.root)
        return replace(bridge, trace_commit=commitment), transcript, commitment.root


class QuotientStage(Round):
    """pil2 `calculateQuotientPolynomial`: squeeze alpha, fold the constraints by
    its powers, divide by the zerofier, commit `Q`. Cubic rows commit as 3
    contiguous base limbs (pil2 `FIELD_EXTENSION` layout), so the leaf hash
    matches the FRI seam. The message is the quotient root."""

    def __init__(
        self,
        eval_fn: Callable[[Array], Array],
        *,
        n_constraints: int,
        n_bits: int,
        blowup_bits: int,
        arity: int,
    ) -> None:
        self._eval_fn = eval_fn
        self._n_constraints = n_constraints
        self._n_bits = n_bits
        self._blowup_bits = blowup_bits
        self._arity = arity

    def __call__(
        self, bridge: InnerBridge, transcript: Transcript
    ) -> tuple[InnerBridge, Transcript, Array]:
        # pil2 folds the K constraints by powers of the stage-`nStages+1`
        # challenge — exactly the coefficient vector `zorch.constraint_eval` takes.
        alpha = powers(
            _base_to_cubic(transcript.get_field()).reshape(()), self._n_constraints
        )
        quotient = quotient_from_constraints(
            self._eval_fn,
            bridge.trace_commit.extended,
            alpha,
            self._n_bits,
            self._blowup_bits,
        )
        matrix = _cubic_to_base(quotient).reshape(quotient.shape[0], 3)
        root, layers = merkle_tree(self._arity).commit(matrix)
        transcript.put(root)
        return (
            replace(
                bridge,
                quotient=quotient,
                quotient_root=root,
                quotient_matrix=matrix,
                quotient_layers=layers,
            ),
            transcript,
            root,
        )


class DeepStage(Round):
    """pil2 `calculateFRIPolynomial`: build the codeword FRI folds. Owns its
    out-of-domain squeeze, so it sits between the quotient's root and the FRI
    betas on this transcript, reading the committed trace and quotient off the
    bridge. The message is the codeword."""

    def __init__(
        self,
        *,
        n_bits: int,
        blowup_bits: int,
        opening_points: Sequence[int] = (0,),
    ) -> None:
        self._n_bits = n_bits
        self._blowup_bits = blowup_bits
        self._opening_points = opening_points

    def __call__(
        self, bridge: InnerBridge, transcript: Transcript
    ) -> tuple[InnerBridge, Transcript, Array]:
        fri_pol = deep_fri_polynomial(
            bridge.trace_commit.extended,
            bridge.quotient,
            transcript,
            n_bits=self._n_bits,
            blowup_bits=self._blowup_bits,
            opening_points=self._opening_points,
        )
        return replace(bridge, fri_pol=fri_pol), transcript, fri_pol


class QuotientEchoStage(Round):
    """Placeholder DEEP: fold FRI over the quotient codeword itself, skipping the
    out-of-domain opening. The quotient is a valid cubic FRI input, so this drives
    the spine end to end for wiring/shape tests — but it is NOT pil2's DEEP
    batching (no trace openings), so a proof built with it does not byte-match
    pil2. Not for conformance."""

    def __call__(
        self, bridge: InnerBridge, transcript: Transcript
    ) -> tuple[InnerBridge, Transcript, Array]:
        return replace(bridge, fri_pol=bridge.quotient), transcript, bridge.quotient


class FriStage(Round):
    """pil2 `FRI::fold`: fold the codeword down the layer chain, committing each
    layer. Its inner rounds are the layer folds, whose betas chain off the same
    transcript — so its state carries the absorbed trace and quotient roots into
    every layer challenge. The message is the fold output."""

    def __init__(self, *, steps: list[int], arity: int) -> None:
        self._steps = steps
        self._arity = arity

    def __call__(
        self, bridge: InnerBridge, transcript: Transcript
    ) -> tuple[InnerBridge, Transcript, FriProof]:
        fri = prove(bridge.fri_pol, self._steps, arity=self._arity, transcript=transcript)
        return replace(bridge, fri=fri), transcript, fri


class QueryStage(Round):
    """pil2 `proveQueries`: absorb the final polynomial, grind, squeeze the query
    positions, and open every committed tree at each. The message is the
    per-tree openings."""

    def __init__(
        self, *, n_bits_ext: int, arity: int, pow_bits: int, n_queries: int
    ) -> None:
        self._n_bits_ext = n_bits_ext
        self._arity = arity
        self._pow_bits = pow_bits
        self._n_queries = n_queries

    def __call__(
        self, bridge: InnerBridge, transcript: Transcript
    ) -> tuple[InnerBridge, Transcript, tuple[list, list, list]]:
        positions, nonce = sample_query_positions(
            transcript,
            bridge.fri.final_pol,
            pow_bits=self._pow_bits,
            n_queries=self._n_queries,
            n_bits_ext=self._n_bits_ext,
        )
        ext_mask = (1 << self._n_bits_ext) - 1
        trace_tree = merkle_tree(self._arity)
        quotient_tree = merkle_tree(self._arity)
        trace_openings = [
            [
                group_proof(
                    trace_tree,
                    bridge.trace_commit.extended,
                    bridge.trace_commit.digest_layers,
                    int(idx) & ext_mask,
                )
            ]
            for idx in positions
        ]
        quotient_openings = [
            [
                group_proof(
                    quotient_tree,
                    bridge.quotient_matrix,
                    bridge.quotient_layers,
                    int(idx) & ext_mask,
                )
            ]
            for idx in positions
        ]
        fri_openings = prove_queries(bridge.fri, positions)
        return (
            replace(
                bridge,
                nonce=nonce,
                query_positions=positions,
                trace_openings=trace_openings,
                quotient_openings=quotient_openings,
                fri_openings=fri_openings,
            ),
            transcript,
            (trace_openings, quotient_openings, fri_openings),
        )


@dataclass(frozen=True)
class InnerProof:
    """The wired inner proof: the per-stage roots the transcript absorbed, the
    FRI fold output, and the query-phase openings of every committed tree."""

    trace_root: Array
    quotient_root: Array
    fri: FriProof
    final_pol: Array
    nonce: int
    query_positions: np.ndarray
    trace_openings: list[list[Array]]
    quotient_openings: list[list[Array]]
    fri_openings: list[list[Array]]


def prove_inner_chain(
    eval_fn: Callable[[Array], Array],
    *,
    n_constraints: int,
    n_bits: int,
    blowup_bits: int = 1,
    arity: int = 2,
    fold_bits: int = 3,
    final_bits: int = 5,
    pow_bits: int = 16,
    n_queries: int = 64,
    deep_stage: Round | None = None,
) -> ProveChain:
    """The ZisK inner-proof chain. One definition for the stage wiring so the
    benchmark, the byte-match runnables, and proof assembly cannot drift on it.

    `n_bits` sizes the base trace domain; the trace itself rides the bridge.
    `deep_stage` fills the DEEP slot; it defaults to the real `DeepStage` — pass
    `QuotientEchoStage()` for the trivial fallback that skips the OOD opening."""
    n_bits_ext = n_bits + blowup_bits
    return ProveChain(
        [
            TraceCommitStage(blowup=1 << blowup_bits, arity=arity),
            QuotientStage(
                eval_fn,
                n_constraints=n_constraints,
                n_bits=n_bits,
                blowup_bits=blowup_bits,
                arity=arity,
            ),
            deep_stage or DeepStage(n_bits=n_bits, blowup_bits=blowup_bits),
            FriStage(
                steps=_fold_steps(n_bits_ext, fold_bits, final_bits), arity=arity
            ),
            QueryStage(
                n_bits_ext=n_bits_ext,
                arity=arity,
                pow_bits=pow_bits,
                n_queries=n_queries,
            ),
        ]
    )


def prove_inner(
    trace: Array,
    eval_fn: Callable[[Array], Array],
    *,
    n_constraints: int,
    blowup_bits: int = 1,
    arity: int = 2,
    fold_bits: int = 3,
    final_bits: int = 5,
    pow_bits: int = 16,
    n_queries: int = 64,
    deep_stage: Round | None = None,
    transcript: Transcript | None = None,
) -> InnerProof:
    """Run `prove_inner_chain` over one shared `Transcript` and assemble the proof.

    `trace` is the `(2^n_bits, n_cols)` base-field evaluation matrix; `eval_fn`
    produces the `n_constraints` constraints in its trailing axis (pil2's cExp
    order). `deep_stage` fills the DEEP slot — default real `DeepStage`, or
    `QuotientEchoStage()` for the trivial fallback."""
    if trace.ndim != 2:
        raise ValueError(f"trace must be 2-D (rows, cols), got ndim={trace.ndim}")
    n = trace.shape[0]
    if n & (n - 1):
        raise ValueError(f"trace height must be a power of two, got {n}")

    chain = prove_inner_chain(
        eval_fn,
        n_constraints=n_constraints,
        n_bits=n.bit_length() - 1,
        blowup_bits=blowup_bits,
        arity=arity,
        fold_bits=fold_bits,
        final_bits=final_bits,
        pow_bits=pow_bits,
        n_queries=n_queries,
        deep_stage=deep_stage,
    )
    bridge, _, _ = chain(InnerBridge(trace=trace), transcript or Transcript())

    return InnerProof(
        trace_root=bridge.trace_commit.root,
        quotient_root=bridge.quotient_root,
        fri=bridge.fri,
        final_pol=bridge.fri.final_pol,
        nonce=bridge.nonce,
        query_positions=bridge.query_positions,
        trace_openings=bridge.trace_openings,
        quotient_openings=bridge.quotient_openings,
        fri_openings=bridge.fri_openings,
    )
