"""pil2's flat-proof writer — `proof2pointer` (`proof_stark.hpp`) in numpy.

The wire proof is one u64 vector:

    airgroupValues*3 | airValues*3 | roots (nStages+1)*4 | evals*3
    | const tree query block | per-custom query block | cm1..cmQ query blocks
    | FRI tree roots (steps-1)*4 | per-FRI-step query block
    | final pol*3 | nonce

where a tree's query block is [every query's leaf row][every query's sibling
path, truncated to ``nSiblings = ceil(nBits/log2(arity)) - llv`` levels]
[the tree's last-verified level, zero-padded to ``arity^llv`` digests].
Stage-1 airgroup/air values serialize as ``[v, 0, 0]`` — every map entry
takes three words on the wire regardless of its dumped packing.

`open_tree` produces a block's pieces from zorch tree data: `group_proof`
already emits pil2's per-query ``[row..., mp...]`` order (pinned by the
merkle_proof golden), so the writer only regroups values-first/paths-second
across queries and slices the last-verified level out of `digest_layers`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import frx
import numpy as np
from frx import Array

from zisk_zorch.commit.openings import group_proof
from zisk_zorch.harness.pil2 import value_offsets
from zisk_zorch.transcript.transcript import DIGEST


def _n_siblings(n_bits: int, arity: int, llv: int) -> int:
    return math.ceil(n_bits / math.log2(arity)) - llv


@dataclass(frozen=True)
class TreeOpening:
    """One committed tree's query-phase wire data."""

    rows: np.ndarray  # (nq, width) leaf rows at the query positions
    paths: np.ndarray  # (nq, n_siblings, (arity-1)*4), already truncated
    last_level: np.ndarray  # (arity^llv * 4,) zero-padded digests


def open_tree_dispatch(
    tree, matrix: Array, digest_layers: list[Array], idx: Array
) -> Array:
    """The device half of `open_tree`: the batched group proofs, no host
    sync — so a caller opening several trees can dispatch them all before
    the first device→host copy blocks."""
    return frx.vmap(lambda i: group_proof(tree, matrix, digest_layers, i))(idx)


def open_tree_convert(
    flat_dev: Array,
    digest_layers: list[Array],
    width: int,
    *,
    n_bits: int,
    arity: int,
    llv: int,
) -> TreeOpening:
    """The host half of `open_tree`: copy the batched proofs over and shape
    the pil2 wire pieces."""
    flat = np.asarray(flat_dev).astype(np.uint64)
    rows = flat[:, :width]
    per_level = (arity - 1) * DIGEST
    path = flat[:, width:].reshape(flat.shape[0], -1, per_level)
    n_sib = _n_siblings(n_bits, arity, llv)
    last_level = _last_level(digest_layers, arity, llv)
    return TreeOpening(rows=rows, paths=path[:, :n_sib], last_level=last_level)


def open_tree(
    tree,
    matrix: Array,
    digest_layers: list[Array],
    positions: np.ndarray,
    *,
    n_bits: int,
    arity: int,
    llv: int,
) -> TreeOpening:
    """Open `matrix`'s tree at `positions` and shape the pil2 wire pieces."""
    idx = frx.numpy.asarray(np.asarray(positions, dtype=np.uint64))
    flat_dev = open_tree_dispatch(tree, matrix, digest_layers, idx)
    return open_tree_convert(
        flat_dev,
        digest_layers,
        int(matrix.shape[1]),
        n_bits=n_bits,
        arity=arity,
        llv=llv,
    )


def _last_level(digest_layers: list[Array], arity: int, llv: int) -> np.ndarray:
    """The first level (walking up from the leaves) with at most ``arity^llv``
    nodes, zero-padded to exactly that many — `setProof`'s walk."""
    if llv == 0:
        return np.zeros((0,), dtype=np.uint64)
    cap = arity**llv
    n = int(digest_layers[0].shape[0])
    level = 0
    while n > cap:
        n = (n + arity - 1) // arity
        level += 1
    nodes = np.asarray(digest_layers[level]).astype(np.uint64)[:n]
    out = np.zeros((cap, DIGEST), dtype=np.uint64)
    out[: nodes.shape[0]] = nodes
    return out.reshape(-1)


def _values3(words: np.ndarray, vmap: list) -> np.ndarray:
    """Dumped value packing (stage-1 one word, else three) -> wire 3-per."""
    out = np.zeros((len(vmap), 3), dtype=np.uint64)
    off = width = 0
    for i, off, width in value_offsets(vmap):
        out[i, :width] = words[off : off + width]
    assert off + width == len(words), (off + width, len(words))
    return out.reshape(-1)


def _tree_block(opening: TreeOpening) -> list[np.ndarray]:
    return [
        opening.rows.reshape(-1),
        opening.paths.reshape(-1),
        opening.last_level,
    ]


def serialize_proof(
    si: dict,
    *,
    airgroup_values: np.ndarray,
    air_values: np.ndarray,
    roots: list[np.ndarray],
    evals: np.ndarray,
    const_opening: TreeOpening,
    custom_openings: list[TreeOpening],
    stage_openings: list[TreeOpening],
    fri_roots: list[np.ndarray],
    fri_openings: list[TreeOpening],
    final_pol: np.ndarray,
    nonce: int,
) -> np.ndarray:
    """Assemble the flat wire proof. `airgroup_values`/`air_values` ride in
    their DUMPED packing (this expands them); everything else is wire-shaped
    already. `stage_openings` is cm1..cmQ in stage order; `fri_openings` one
    per fold step."""
    parts = [
        _values3(airgroup_values, si["airgroupValuesMap"] or []),
        _values3(air_values, si["airValuesMap"] or []),
        *[np.asarray(r, dtype=np.uint64).reshape(-1) for r in roots],
        np.asarray(evals, dtype=np.uint64).reshape(-1),
        *_tree_block(const_opening),
    ]
    for opening in custom_openings:
        parts.extend(_tree_block(opening))
    for opening in stage_openings:
        parts.extend(_tree_block(opening))
    parts.extend(np.asarray(r, dtype=np.uint64).reshape(-1) for r in fri_roots)
    for opening in fri_openings:
        parts.extend(_tree_block(opening))
    parts.append(np.asarray(final_pol, dtype=np.uint64).reshape(-1))
    parts.append(np.array([nonce], dtype=np.uint64))
    return np.concatenate(parts)
