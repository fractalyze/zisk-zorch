"""FRI verifier — pil2-stark's FRI query-phase checks, on zorch's k-ary fold seam.

Given the proof a prover transmits (the per-layer roots, the in-clear final
polynomial, and one Merkle group-proof per query per layer), the verifier:

  1. re-derives the fold challenges by replaying the transcript over the roots
     (`Pil2SeamTranscript` observe+sample per layer) — the prover's exact
     absorb/squeeze discipline;
  2. checks every query's k-ary Merkle opening against its layer root
     (`verify_group_proof`, pil2's `getGroupProof` deserialization), and
  3. checks the fold chain — each layer's opened cubic group folds to the next
     layer's opening (or the final polynomial) — via
     `zorch.pcs.fold.verify_group_fold_chain` over the `Pil2FriCode` seam.

The fold consistency + Merkle checks are exactly the part a randomly-built
codeword can exercise. The final-polynomial low-degree test (`INTT` then assert
the high coefficients vanish) is out of scope here: it needs a genuine low-degree
FRI polynomial and the base trace size `nBits`, neither of which exists until the
upstream STARK (stage-2 / Q / evals) produces a real `f`.

Host-driven, mirroring the prover's transcript discipline. The query positions
are NOT trusted from the prover: the verifier re-derives them from the transcript
(`sample_query_positions` — finalPol absorb + grinding-seed squeeze + reseed with
challenge++nonce), so pil2's trailing finalPol absorb IS replayed here. The
grinding/PoW check that binds `nonce` is deferred with the search (see queries.py).

https://github.com/0xPolygonHermez/pil2-proofman/blob/v0.18.0/pil2-stark/src/starkpil/stark_verify.hpp#L564-L670
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jax import Array

from zisk_zorch.commit.openings import verify_group_proof
from zisk_zorch.commit.trace_commit import merkle_tree
from zisk_zorch.fri.queries import sample_query_positions
from zisk_zorch.fri.seam import Pil2FriCode, Pil2SeamTranscript, _base_to_cubic
from zisk_zorch.transcript.transcript import Transcript
from zorch.commit.merkle import Opening
from zorch.pcs.fold import verify_group_fold_chain


def verify(
    roots: list[Array],
    final_pol: Array,
    query_openings: list[list[Array]],
    *,
    steps: list[int],
    arity: int,
    transcript: Transcript,
    nonce: int,
) -> bool:
    """Check the FRI fold consistency and Merkle openings for every query.

    `query_openings[q][layer]` is the flat `getGroupProof` array for layer
    `layer` at query `q` (as produced by `prover.prove_queries`). The query
    positions are re-derived from the transcript (not trusted from the prover):
    `nonce` is the PoW witness the prover used. Returns whether every Merkle
    opening verifies and every fold lands on the next layer's opening (or the
    final polynomial). `transcript` must be seeded exactly as the prover's was."""
    code = Pil2FriCode(tuple(steps))
    tree = merkle_tree(arity)
    num_rounds = len(steps) - 1

    # Replay the fold challenges: observe each layer root, squeeze a cubic — the
    # prover's per-round discipline.
    t = Pil2SeamTranscript(transcript)
    betas = []
    for layer in range(num_rounds):
        t, beta = t.observe(roots[layer]).sample()
        betas.append(beta)

    # Re-derive the query positions from the transcript exactly as the prover did
    # (finalPol absorb + grinding-seed squeeze + reseed with challenge++nonce) —
    # the verifier trusts the transcript, not the prover-supplied indices. One
    # position per opened group.
    query_indices = sample_query_positions(
        transcript, final_pol, nonce, n_queries=len(query_openings), n_bits_ext=steps[0]
    )

    # Query indices stay on the host for the Merkle loop — indexing a JAX array
    # per (query, layer) would force a device→host sync each time. They cross to
    # JAX only for the fold-chain check below.
    positions = query_indices.astype(np.int64)  # (Q,)
    leaf_indices = code.group_layer_positions(positions, num_rounds)  # per layer (Q,)

    # Merkle: each query opens layer `layer` at `query mod 2^steps[layer+1]`; the
    # flat group-proof must rebuild that layer's root. Width = the cubic group's
    # base-limb count (n_x * 3).
    openings_seam: list[Opening] = []
    for layer in range(num_rounds):
        n_cols = (1 << (steps[layer] - steps[layer + 1])) * 3
        rows = []
        for q in range(len(positions)):
            proof = query_openings[q][layer]
            open_idx = int(leaf_indices[layer][q])
            if not verify_group_proof(tree, roots[layer], open_idx, proof, n_cols):
                return False
            rows.append(_base_to_cubic(proof[:n_cols]))  # (n_x,) cubic
        # The seam folds the opened cubic group, so feed it cubic-viewed rows
        # (the linear-hash leaf stays base-limb above, in verify_group_proof).
        openings_seam.append(Opening(row=jnp.stack(rows), path=[]))  # (Q, n_x)

    # Each layer's opened group folds to the next layer's opening, or the final
    # polynomial at the last layer — the shared k-ary fold-chain check. The leaf
    # indices cross to JAX only here, for the device-side fold arithmetic.
    leaf_indices_jax = [jnp.asarray(idx) for idx in leaf_indices]
    ok = verify_group_fold_chain(code, openings_seam, betas, leaf_indices_jax, final_pol)
    return bool(ok)
