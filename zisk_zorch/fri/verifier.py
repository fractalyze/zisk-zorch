"""FRI verifier — pil2-stark's FRI query-phase checks (`stark_verify.hpp`).

Given the proof a prover transmits (the per-layer roots, the in-clear final
polynomial, and one Merkle group-proof per query per layer), the verifier:

  1. re-derives the fold challenges by replaying the transcript over the roots
     and the final polynomial — the prover's exact absorb/squeeze discipline;
  2. for each query and each fold step, verifies the layer's k-ary Merkle
     opening against its root, folds the opened group with `verify_fold`, and
     checks the result equals the next layer's opening (or the final polynomial)
     at the position this fold maps to.

The final-polynomial low-degree test (`INTT` then assert the coefficients above
the degree bound vanish) is out of scope here: it needs a genuine low-degree
FRI polynomial and the base trace size `nBits`, neither of which exists until
the upstream STARK (stage-2 / Q / evals) produces a real `f`. The fold
consistency + Merkle checks are exactly the part a randomly-built codeword can
exercise, and what pins the prover's output as self-consistent and openable.

Host-driven, mirroring the prover's transcript discipline.

https://github.com/0xPolygonHermez/pil2-proofman/blob/v0.18.0/pil2-stark/src/starkpil/stark_verify.hpp#L564-L670
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jax import Array
from zk_dtypes import goldilocks_mont as F
from zk_dtypes import goldilocksx3_mont as F3

from zisk_zorch.commit.openings import verify_group_proof
from zisk_zorch.commit.trace_commit import merkle_tree
from zisk_zorch.fri.fold import verify_fold
from zisk_zorch.transcript.transcript import Transcript

# Cubic <-> base limb views (host round-trip; see docs/conventions.md). Mirror
# the prover's conversions — the verifier absorbs the final pol as base limbs,
# reads each squeezed challenge as a cubic, and reads opened rows back to cubic.
def _cubic_to_base(values: Array) -> Array:
    return jnp.array(np.ascontiguousarray(np.asarray(values)).view(F))


def _base_to_cubic(values: Array) -> Array:
    return jnp.array(np.ascontiguousarray(np.asarray(values)).view(F3))


def _replay_challenges(
    roots: list[Array], final_pol: Array, steps: list[int], transcript: Transcript
) -> list[Array]:
    """Re-derive each fold challenge as the prover did: absorb each layer root
    (the final polynomial in clear at the last step), squeeze a cubic each step."""
    challenges = []
    for step in range(len(steps)):
        if step < len(steps) - 1:
            transcript.put(roots[step])
        else:
            transcript.put(_cubic_to_base(final_pol))
        challenges.append(_base_to_cubic(transcript.get_field()).reshape(()))
    return challenges


def verify(
    roots: list[Array],
    final_pol: Array,
    query_openings: list[list[Array]],
    query_indices: np.ndarray | list[int],
    *,
    steps: list[int],
    arity: int,
    transcript: Transcript,
) -> bool:
    """Check the FRI fold consistency and Merkle openings for every query.

    `query_openings[q][layer]` is the flat `getGroupProof` array for layer
    `layer` at query `q` (as produced by `prover.prove_queries`). Returns whether
    every Merkle opening verifies and every fold lands on the next layer's
    opening (or the final polynomial). `transcript` must be seeded exactly as the
    prover's was."""
    n_bits_ext = steps[0]
    challenges = _replay_challenges(roots, final_pol, steps, transcript)
    tree = merkle_tree(arity)

    for q, query in enumerate(query_indices):
        query = int(query)
        for step in range(1, len(steps)):
            prev_bits, current_bits = steps[step - 1], steps[step]
            layer = step - 1
            open_idx = query % (1 << current_bits)
            n_cols = (1 << (prev_bits - current_bits)) * 3

            proof = query_openings[q][layer]
            if not verify_group_proof(tree, roots[layer], open_idx, proof, n_cols):
                return False

            row = _base_to_cubic(proof[:n_cols])  # the n_x opened cubic values
            folded = verify_fold(
                row, challenges[layer], n_bits_ext, prev_bits, current_bits, open_idx
            )

            if step < len(steps) - 1:
                # The fold for group `open_idx` lands at the next layer's opening,
                # position `open_idx / 2^steps[step+1]` within its regrouped row.
                next_bits = steps[step + 1]
                group_idx = open_idx // (1 << next_bits)
                next_cols = (1 << (current_bits - next_bits)) * 3
                next_row = _base_to_cubic(query_openings[q][step][:next_cols])
                expected = next_row[group_idx]
            else:
                expected = final_pol[open_idx]

            if not bool(jnp.all(folded == expected)):
                return False
    return True
