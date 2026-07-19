"""Query-phase opening — pil2-stark's `MerkleTreeGL::getGroupProof` on zorch's tree.

A group proof is one flat array [row..., mp...]: the committed row, then per
level (leaf-first) the (arity-1) sibling digests in group order with the node's
own slot skipped — mp slot j holds group position j + (j >= pos). zorch's
`MerkleTree.open` packs `Opening.path` in exactly that order, so serializing is
flatten-and-concatenate with no re-ordering (pinned by the merkle_proof golden);
the only shape glue is arity 2, where zorch keeps its historical single-sibling
`(digest_elems,)` path entries.

https://github.com/0xPolygonHermez/pil2-proofman/blob/v1.0.0-alpha/pil2-stark/src/starkpil/merkleTree/merkleTreeGL.cpp#L145-L175
"""

from __future__ import annotations

import frx.numpy as fnp
from frx import Array

from zorch.commit.merkle import MerkleTree, Opening


def group_proof(
    tree: MerkleTree, matrix: Array, digest_layers: list[Array], index: int | Array
) -> Array:
    """`getGroupProof`: serialize the opening at `index` to pil2's flat array."""
    opening = tree.open(matrix, digest_layers, index)
    return fnp.concatenate([opening.row, *(fnp.ravel(s) for s in opening.path)])


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
