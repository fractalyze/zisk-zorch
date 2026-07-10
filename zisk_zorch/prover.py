"""End-to-end inner-proof prover — the Fiat-Shamir spine threading every stage.

Each stage of the inner proof already exists as an isolated primitive
(`commit_trace`, `quotient_from_constraints`, `fri.prove`, `sample_query_positions`,
`prove_queries`); `bench_inner_proof.py` times them separately on *random* inputs.
This module is the missing wiring: it runs them in pil2's order and threads one
`Transcript` so each stage's Merkle root is absorbed and the next stage's
challenges (`alpha`, the FRI fold betas, the query positions) are squeezed from
that running state — the Fiat-Shamir byte stream pil2-proofman's `genProof`
drives (`gen_proof.hpp`).

Two honest boundaries, both load-bearing given this repo's byte-match contract:

- **The DEEP / FRI-polynomial construction** (pil2's `calculateFRIPolynomial`)
  now lives in `zisk_zorch/deep/`: squeeze the out-of-domain point, open the
  committed polynomials there (`deep.opening`), absorb the openings, squeeze the
  batching challenge, and build the codeword (`deep.fri_polynomial`). It is the
  default `fri_polynomial_fn`. Its generic DEEP-ALI formula is not yet pinned to
  a specific AIR's compiled `friExp` (that needs a proving-key op list + golden,
  a later slice) — `quotient_as_fri_polynomial` remains as a trivial fallback
  that folds FRI over the quotient itself.

- **The quotient-commit leaf layout** mirrors the FRI seam's cubic convention
  (each cubic row -> its 3 contiguous Goldilocks limbs, cf. `seam._cubic_to_base`),
  which is pil2's `FIELD_EXTENSION`-contiguous memory order.

https://github.com/0xPolygonHermez/pil2-proofman/blob/v1.0.0-alpha/pil2-stark/src/starkpil/gen_proof.hpp
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from jax import Array
from zk_dtypes import goldilocks as F

from zisk_zorch.commit.openings import group_proof
from zisk_zorch.commit.trace_commit import TraceCommitment, commit_trace, merkle_tree
from zisk_zorch.deep.fri_polynomial import deep_fri_polynomial
from zisk_zorch.fri.prover import FriProof, prove, prove_queries
from zisk_zorch.fri.queries import sample_query_positions
from zisk_zorch.fri.seam import _base_to_cubic, _cubic_to_base
from zisk_zorch.quotient.quotient import quotient_from_constraints
from zisk_zorch.transcript.transcript import Transcript

# Cubic multiplicative identity, built from limbs so it never depends on a
# `jnp.ones` cubic-storage assumption (limb0 = 1 is the field one).
_CUBIC_ONE = _base_to_cubic(jnp.array([1, 0, 0], F)).reshape(())


def _fold_steps(n_bits_ext: int, fold_bits: int, final_bits: int) -> list[int]:
    """Strictly-decreasing FRI layer schedule `n_bits_ext -> ... -> final_bits`,
    folding by `fold_bits` per layer (the tail folds the remainder). Same schedule
    `bench_inner_proof._fold_steps` builds; kept here so the prover owns its FRI
    shape without importing a benchmark private."""
    steps = list(range(n_bits_ext, final_bits, -fold_bits))
    if not steps or steps[-1] != final_bits:
        steps.append(final_bits)
    return steps


def _alpha_powers(challenge: Array, n_constraints: int) -> Array:
    """`[a^0, a^1, ..., a^(K-1)]` for the cubic challenge `a` (its 3 Goldilocks
    limbs) — pil2 folds the `K` constraints by powers of the stage-`nStages+1`
    challenge, which is exactly the per-constraint coefficient vector
    `zorch.constraint_eval` consumes as `alpha`."""
    a = _base_to_cubic(challenge).reshape(())
    powers = []
    cur = _CUBIC_ONE
    for _ in range(n_constraints):
        powers.append(cur)
        cur = cur * a
    return jnp.stack(powers)


@dataclass(frozen=True)
class FriPolynomialContext:
    """Everything the DEEP / FRI-polynomial seam may combine, plus the live
    transcript it squeezes its out-of-domain challenge from. A byte-match
    implementation reads the committed evaluations here and absorbs/squeezes on
    `transcript`; the placeholder ignores all but `quotient`."""

    trace: TraceCommitment
    quotient: Array  # cubic, length 2^n_bits_ext (the extended domain)
    quotient_root: Array
    n_bits: int
    blowup_bits: int
    transcript: Transcript


def quotient_as_fri_polynomial(ctx: "FriPolynomialContext") -> Array:
    """Runnable placeholder combiner: fold FRI over the quotient codeword itself.

    The quotient is a cubic codeword of length `2^n_bits_ext`, so it is a valid
    FRI input and drives the whole prover spine end to end. It is NOT pil2's DEEP
    batching (no out-of-domain openings of the trace), so a proof built with it
    does not byte-match pil2 — use it for wiring/shape tests, not conformance."""
    return ctx.quotient


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
    fri_polynomial_fn: Callable[["FriPolynomialContext"], Array] | None = None,
    transcript: Transcript | None = None,
) -> InnerProof:
    """Run commit -> quotient -> FRI -> queries on one shared `Transcript`.

    `trace` is the `(2^n_bits, n_cols)` base-field evaluation matrix; `eval_fn`
    produces the `n_constraints` constraints in its trailing axis (pil2's cExp
    order). `fri_polynomial_fn` supplies the FRI codeword from the committed
    stages (the DEEP stage); it defaults to the real `deep_fri_polynomial`
    combiner — pass `quotient_as_fri_polynomial` for the trivial fallback."""
    if trace.ndim != 2:
        raise ValueError(f"trace must be 2-D (rows, cols), got ndim={trace.ndim}")
    n = trace.shape[0]
    if n & (n - 1):
        raise ValueError(f"trace height must be a power of two, got {n}")
    fri_polynomial_fn = fri_polynomial_fn or deep_fri_polynomial
    n_bits = n.bit_length() - 1
    n_bits_ext = n_bits + blowup_bits
    blowup = 1 << blowup_bits
    transcript = transcript or Transcript()

    # Fiat-Shamir requires the trace root be absorbed before alpha is squeezed.
    trace_commit = commit_trace(trace, blowup=blowup, arity=arity)
    transcript.put(trace_commit.root)
    alpha = _alpha_powers(transcript.get_field(), n_constraints)

    # Q = C/Z on the extended domain; its cubic rows commit as 3 contiguous base
    # limbs (pil2 FIELD_EXTENSION layout) so the leaf hash matches the FRI seam.
    quotient = quotient_from_constraints(
        eval_fn, trace_commit.extended, alpha, n_bits, blowup_bits
    )
    quotient_matrix = _cubic_to_base(quotient).reshape(quotient.shape[0], 3)
    quotient_root, quotient_layers = merkle_tree(arity).commit(quotient_matrix)
    transcript.put(quotient_root)

    # The DEEP seam owns its out-of-domain squeeze; see module docstring for why
    # the FRI codeword is injected rather than built here.
    fri_pol = fri_polynomial_fn(
        FriPolynomialContext(
            trace=trace_commit,
            quotient=quotient,
            quotient_root=quotient_root,
            n_bits=n_bits,
            blowup_bits=blowup_bits,
            transcript=transcript,
        )
    )

    # FRI fold betas chain off the same transcript, so its state carries the
    # absorbed trace and quotient roots into every layer challenge.
    steps = _fold_steps(n_bits_ext, fold_bits, final_bits)
    fri = prove(fri_pol, steps, arity=arity, transcript=transcript)

    # Grind + derive positions, then open every committed tree at each position.
    positions, nonce = sample_query_positions(
        transcript,
        fri.final_pol,
        pow_bits=pow_bits,
        n_queries=n_queries,
        n_bits_ext=n_bits_ext,
    )
    ext_mask = (1 << n_bits_ext) - 1
    trace_tree = merkle_tree(arity)
    quotient_tree = merkle_tree(arity)
    trace_openings = [
        [group_proof(trace_tree, trace_commit.extended, trace_commit.digest_layers,
                     int(idx) & ext_mask)]
        for idx in positions
    ]
    quotient_openings = [
        [group_proof(quotient_tree, quotient_matrix, quotient_layers,
                     int(idx) & ext_mask)]
        for idx in positions
    ]
    fri_openings = prove_queries(fri, positions)

    return InnerProof(
        trace_root=trace_commit.root,
        quotient_root=quotient_root,
        fri=fri,
        final_pol=fri.final_pol,
        nonce=nonce,
        query_positions=positions,
        trace_openings=trace_openings,
        quotient_openings=quotient_openings,
        fri_openings=fri_openings,
    )
