"""Full FRI prover — pil2-stark's `FRI` fold/merkelize/query loop on zorch.

Drives the single-step `fold` over the layer chain exactly as pil2's STARK
driver does (`gen_proof.hpp`'s STARK_FRI_FOLDING / STARK_FRI_QUERIES loops):

  for step in steps:                       # steps[0].nBits == nBitsExt
      fold(prev -> current) at the challenge     # step 0 is a no-op fold
      if not last: merkelize the codeword, absorb the root
      else:        absorb the final polynomial in clear
      squeeze the next cubic challenge

The codeword that enters the chain (pil2's FRI polynomial `f`) is committed
first, at step 0, where `prevBits == currentBits == nBitsExt` makes the fold a
no-op (`FRI::fold`'s `if (step != 0)` guard). Each later step folds the
previous layer to the next, commits a tree over the regrouped codeword, and the
last step sends the final polynomial uncompressed. Challenges are squeezed from
the pil2 transcript after each commit — the consumer owns this @jit-free zone
(the per-layer coset domain is host-side, so `fold` stays un-jitted; jit lives
at the zorch leaf kernels it composes).

merkelize regroups the layer with pil2's `getTransposed`: a degree-`2^current`
codeword becomes a `(2^next, 2^(current-next))` matrix, row `i` holding the
strided entries `pol[j*2^next + i]` — the cosets the next fold reads. The k-ary
tree over that matrix is the same `MerkleTreeGL` the stage-1 commit builds.

prove_queries mirrors `FRI::proveFRIQueries`: layer `s` (committed at step `s`)
opens at `query_index mod 2^(steps[s+1])`, reusing the `getGroupProof`
serialization from the query phase.

https://github.com/0xPolygonHermez/pil2-proofman/blob/v0.18.0/pil2-stark/src/starkpil/gen_proof.hpp#L235-L282
https://github.com/0xPolygonHermez/pil2-proofman/blob/v0.18.0/pil2-stark/src/starkpil/fri/fri.hpp#L36-L143
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from jax import Array
from zk_dtypes import goldilocks_mont as F
from zk_dtypes import goldilocksx3_mont as F3

from zisk_zorch.commit.openings import group_proof
from zisk_zorch.commit.trace_commit import merkle_tree
from zisk_zorch.fri.fold import fold
from zisk_zorch.transcript.transcript import Transcript
from zorch.commit.merkle import MerkleTree


def _cubic_to_base(values: Array) -> np.ndarray:
    """View a cubic array as its Goldilocks limbs, the cubic axis expanded
    in place (element `[c0, c1, c2]` -> three contiguous base lanes), matching
    pil2's `FIELD_EXTENSION`-contiguous memory layout."""
    return np.ascontiguousarray(np.asarray(values)).view(F)


def _limbs_to_cubic(limbs: Array) -> Array:
    """The transcript's 3 squeezed Goldilocks limbs -> one cubic challenge
    (the limbs are already that element's coefficients, cf. golden.u64x3)."""
    return jnp.array(np.ascontiguousarray(np.asarray(limbs)).view(F3)).reshape(())


def _regroup(pol: Array, current_bits: int, next_bits: int) -> Array:
    """pil2's `getTransposed`: codeword (length `2^current_bits`) -> a
    `(2^next_bits, 2^(current_bits - next_bits))` base matrix, row `i` holding
    `pol[j*2^next_bits + i]` for `j` in `[0, 2^(current_bits - next_bits))`."""
    w = 1 << next_bits
    h = 1 << (current_bits - next_bits)
    # reshape(h, w).T[i, j] = pol[j*w + i] — the strided coset the next fold reads;
    # the cubic view then expands each entry to its 3 limbs, giving (w, 3*h).
    rows = pol.reshape(h, w).T
    return jnp.array(_cubic_to_base(rows))


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
    if len(steps) < 1:
        raise ValueError("steps must list at least the extended-domain size")
    if any(not 0 <= s <= 32 for s in steps):
        raise ValueError(f"each step must be a domain log size in [0, 32], got {steps}")
    if steps != sorted(steps, reverse=True) or len(set(steps)) != len(steps):
        raise ValueError(f"steps must strictly decrease, got {steps}")
    if arity not in (2, 3, 4):
        raise ValueError(f"arity must be one of (2, 3, 4), got {arity}")
    n_bits_ext = steps[0]
    if fri_pol.shape != (1 << n_bits_ext,):
        raise ValueError(
            f"fri_pol must have shape {(1 << n_bits_ext,)}, got {fri_pol.shape}"
        )

    tree = merkle_tree(arity)  # arity is fixed across layers, so build once.
    pol = fri_pol
    challenge: Array | None = None
    roots: list[Array] = []
    layers: list[_Layer] = []
    for step, current_bits in enumerate(steps):
        if step > 0:
            # step 0's fold is a no-op (prevBits == currentBits == nBitsExt);
            # the chain folds the committed `f` only from step 1 on.
            pol = fold(pol, challenge, n_bits_ext, steps[step - 1], current_bits)
        if step < len(steps) - 1:
            next_bits = steps[step + 1]
            matrix = _regroup(pol, current_bits, next_bits)
            root, digest_layers = tree.commit(matrix)
            transcript.put(root)  # addTranscript(root, HASH_SIZE)
            roots.append(root)
            layers.append(_Layer(tree, matrix, digest_layers, next_bits))
        else:
            # Final polynomial sent uncompressed: absorb its limbs (addTranscriptGL).
            transcript.put(jnp.array(_cubic_to_base(pol)))
        challenge = _limbs_to_cubic(transcript.get_field())
    return FriProof(roots=roots, final_pol=pol, layers=layers)


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
