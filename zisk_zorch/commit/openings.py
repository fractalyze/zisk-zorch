"""Query-phase opening — pil2-stark's `MerkleTreeGL::getGroupProof` on zorch's tree.

A group proof is one flat array [row..., mp...]: the committed row, then per
level (leaf-first) the (arity-1) sibling digests in group order with the node's
own slot skipped — mp slot j holds group position j + (j >= pos). zorch's
`MerkleTree.open` packs `Opening.path` in exactly that order, so serializing is
flatten-and-concatenate with no re-ordering (pinned by the merkle_proof golden);
the only shape glue is arity 2, where zorch keeps its historical single-sibling
`(digest_elems,)` path entries. That branch is unreachable from this repo —
`trace_commit.merkle_tree` builds arity 4 only — and stays because `MerkleTree`
is zorch's general type, whose arity-2 contract is not ZisK's to narrow.

https://github.com/0xPolygonHermez/pil2-proofman/blob/v1.0.0-alpha/pil2-stark/src/starkpil/merkleTree/merkleTreeGL.cpp#L145-L175
"""

from __future__ import annotations

from functools import partial

import frx
import frx.numpy as fnp
from frx import Array
from zorch.commit.merkle import MerkleTree, Opening


def group_proof(
    tree: MerkleTree, matrix: Array, digest_layers: list[Array], index: int | Array
) -> Array:
    """`getGroupProof`: serialize the opening at `index` to pil2's flat array."""
    opening = tree.open(matrix, digest_layers, index)
    return fnp.concatenate([opening.row, *(fnp.ravel(s) for s in opening.path)])


@partial(frx.jit, static_argnums=(0,))
def batched_group_proof(
    tree: MerkleTree, matrix: Array, digest_layers: list[Array], index: Array
) -> Array:
    """`getGroupProof` at every index of `index`, as one compiled kernel.

    The path walk is orchestration — per level a modulus, a group base, and a
    sibling gather — so a bare `frx.vmap` over `group_proof` leaves several
    eager dispatches per level times the tree depth, tens of round trips whose
    data is a handful of rows. Under `jit` the whole walk is one executable: at
    recursion shapes (2^20 leaves, arity 4, 73 queries) that is ~8 ms per tree
    down to well under one, and a prove opens a dozen trees.

    `tree` is static — `MerkleTree` carries value equality so instances of the
    same configuration share the zone rather than re-tracing.

    Byte-identical to the eager form — the same gathers and concatenate, no
    arithmetic to reassociate."""
    return frx.vmap(lambda i: group_proof(tree, matrix, digest_layers, i))(index)


def verify_group_proof(
    tree: MerkleTree, root: Array, index: int, proof: Array, n_cols: int
) -> bool:
    """`verifyGroupProof`: split the flat array back into the row + per-level
    sibling groups, rebuild the root (leaf hash + k-ary fold), compare to `root`."""
    row, siblings = proof[:n_cols], proof[n_cols:]
    per_level = (tree.arity - 1) * tree.digest_elems
    if siblings.size % per_level:
        raise ValueError(
            f"proof tail ({siblings.size} elements) is not whole sibling "
            f"groups of {per_level}"
        )
    groups = siblings.reshape(-1, tree.arity - 1, tree.digest_elems)
    path = [g[0] if tree.arity == 2 else g for g in groups]
    return tree.verify(root, index, Opening(row=row, path=path))
