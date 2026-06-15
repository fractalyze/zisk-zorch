"""Full FRI prover — pil2-stark's `FRI` fold/merkelize/query loop, driven on
zorch's k-ary fold seam.

The per-layer commit→squeeze→fold chain is `zorch.pcs.fold.PreFoldKGroupCommitRound`
run by `zorch.prove.fold_rounds`: each round regroups the codeword into the next
fold's coset leaves (`getTransposed`), commits them, observes the root, squeezes
the cubic challenge, and folds — exactly pil2's `gen_proof.hpp`
STARK_FRI_FOLDING loop. The pil2-specific grouping, coset Lagrange fold, and
transcript ride the `Pil2FriCode` / `Pil2SeamTranscript` adapters (`seam.py`), so
this module is the thin driver: seed → fold rounds → final polynomial → queries.

The codeword that enters the chain (pil2's FRI polynomial `f`) is the round-0
carry; step 0's no-op fold is implicit (the seam folds *after* committing, so the
first committed layer is `f` itself). The last round folds to the final
polynomial, which pil2 sends uncompressed — observed outside the rounds. `prove`
itself stops at the fold loop: pil2's trailing finalPol absorb + grinding-seed
squeeze and the query-index derivation live in `queries.sample_query_positions`,
which the caller runs on the post-fold transcript to obtain the indices it then
feeds to `prove_queries`.

prove_queries mirrors `FRI::proveFRIQueries`: layer `s` opens at
`query_index mod 2^(steps[s+1])`, reusing the `getGroupProof` serialization.

https://github.com/0xPolygonHermez/pil2-proofman/blob/v0.18.0/pil2-stark/src/starkpil/gen_proof.hpp#L235-L282
https://github.com/0xPolygonHermez/pil2-proofman/blob/v0.18.0/pil2-stark/src/starkpil/fri/fri.hpp#L36-L143
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from jax import Array

from zisk_zorch.commit.openings import group_proof
from zisk_zorch.commit.trace_commit import merkle_tree
from zisk_zorch.fri.seam import Pil2FriCode, Pil2SeamTranscript
from zisk_zorch.transcript.transcript import Transcript
from zorch.commit.merkle import MerkleTree
from zorch.pcs.fold import PreFoldKGroupCommitRound
from zorch.prove import fold_rounds


@dataclass(frozen=True)
class _Layer:
    """A committed FRI layer kept for the query phase."""

    tree: MerkleTree
    matrix: Array
    digest_layers: list[Array]
    leaf_bits: int  # log2 of the layer height = the openings' index modulus


@dataclass(frozen=True)
class FriProof:
    """Output of the fold loop: the per-layer roots (transcript order), the
    final polynomial sent in clear, and the layers retained for opening."""

    roots: list[Array]
    final_pol: Array
    layers: list[_Layer]


def prove(
    fri_pol: Array,
    steps: list[int],
    *,
    arity: int,
    transcript: Transcript,
) -> FriProof:
    """Fold `fri_pol` (cubic, length `2^steps[0]`) down the layer chain,
    committing each intermediate layer and squeezing challenges from
    `transcript`. `steps[0]` is the extended-domain log size (`nBitsExt`)."""
    code = Pil2FriCode(tuple(steps))  # validates the step schedule
    tree = merkle_tree(arity)  # validates arity; fixed across layers, build once.
    n_bits_ext = steps[0]
    if fri_pol.shape != (1 << n_bits_ext,):
        raise ValueError(
            f"fri_pol must have shape {(1 << n_bits_ext,)}, got {fri_pol.shape}"
        )
    num_rounds = len(steps) - 1

    # Each round commits the current layer's next-fold cosets, squeezes the cubic
    # challenge, and folds — the carry is the running codeword (pil2's `f`), and
    # the last round's carry is the final polynomial.
    final_pol, _, layers_msg = fold_rounds(
        PreFoldKGroupCommitRound(code, tree),
        fri_pol,
        Pil2SeamTranscript(transcript),
        num_rounds,
    )

    roots = [msg.root for msg in layers_msg]
    layers = [
        _Layer(tree, msg.leaves, msg.digest_layers, steps[i + 1])
        for i, msg in enumerate(layers_msg)
    ]
    return FriProof(roots=roots, final_pol=final_pol, layers=layers)


def prove_queries(proof: FriProof, query_indices: np.ndarray | list[int]) -> list[list[Array]]:
    """`FRI::proveFRIQueries`: open every committed layer at each query. Layer
    `s` opens at `query_index mod 2^(leaf_bits)`, the regrouped height."""
    out: list[list[Array]] = []
    for idx in query_indices:
        per_layer = [
            group_proof(
                layer.tree,
                layer.matrix,
                layer.digest_layers,
                int(idx) % (1 << layer.leaf_bits),
            )
            for layer in proof.layers
        ]
        out.append(per_layer)
    return out
