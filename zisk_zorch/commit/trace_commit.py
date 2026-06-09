"""Stage-1 trace commit — pil2-stark's `extendAndMerkelize` on zorch blocks.

extend: the trace columns are evaluations on the order-N subgroup; INTT each
column to coefficients (the native NTT's inverse carries the 1/N), then
RS-encode onto the blown-up domain shifted by coset 7 (`Goldilocks::SHIFT`) —
exactly pil2's `NTT_Goldilocks::extendPol` (INTT with coset-power scaling, NTT
on 2^nBitsExt).

commit: leaf-hash every extended row with pil2's chained linear hash and fold
the k-ary Poseidon2 tree (arity 2/3/4 -> node width 8/12/16, 4-element root) —
`MerkleTreeGL::merkelize`. The tree is zorch's MerkleTree; everything pil2 about
it (hash family, leaf convention, arity-to-width map) rides in via the blocks.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array, lax
from zk_dtypes import goldilocks_mont as F

from zisk_zorch.commit.linear_hash import DIGEST_ELEMS, LinearHash
from zisk_zorch.poseidon2.goldilocks import goldilocks_perm
from zorch.coding.reed_solomon import ReedSolomon
from zorch.commit.merkle import MerkleTree
from zorch.hash.compression import Compression, CompressionParams

# Goldilocks::SHIFT — the LDE coset generator pil2-stark evaluates on.
COSET_SHIFT = 7

# zk_dtypes' Goldilocks two-adic generator (Plonky3's 1753635133440165772) and
# pil2-stark's (`Goldilocks::W[32]` = 7277203076849721926) span the same 2^32
# subgroup but are different elements: pil2_root = zk_root^_ROOT_DLOG. The
# native NTT therefore indexes the domain in a different order than pil2, and
# the mismatch cannot be folded into one output shuffle — the coset scaling
# binds the coefficient index — so `extend` re-indexes rows on the way in
# (placing trace row r at the zk index of the same domain point) and on the
# way out (reading pil2 row r off the zk index of shift*w_pil2^r). The
# gather-free fix is a root-parameterized native NTT (zkx follow-up).
_ROOT_DLOG = 4168946053

# starkinfo's merkleTreeArity -> the Poseidon2 width hashing that tree
# (MerkleTreeGL::merkelize switches arity {2,3,4} to Poseidon2Goldilocks
# {8,12,16} for both the leaf linear hash and the node hash).
_ARITY_WIDTHS = {2: 8, 3: 12, 4: 16}


def _root_reindex(n: int, scale: int) -> Array:
    """Row map between the pil2 and zk domain orders: index j -> scale*j mod n."""
    return (jnp.arange(n) * (scale % n)) % n


def extend(trace: Array, blowup: int) -> Array:
    """LDE a (N, n_cols) evaluation matrix to (N*blowup, n_cols) on coset 7,
    rows in pil2's domain order (`extendPol` semantics)."""
    if trace.ndim != 2:
        raise ValueError(f"trace must be 2-D, got ndim={trace.ndim}")
    n = trace.shape[0]
    rs = ReedSolomon(n, blowup, F, coset_shift=jnp.array(COSET_SHIFT, F))
    # In: the value at zk index j is the trace row of the same domain point,
    # r = _ROOT_DLOG^-1 * j (mod n).
    zk_evals = trace[_root_reindex(n, pow(_ROOT_DLOG, -1, n))] if n > 1 else trace
    coeffs = lax.fft(zk_evals.T, "IFFT", n)  # per-column INTT (includes 1/N)
    zk_extended = rs.encode(coeffs).T
    # Out: pil2 row r = P(shift*w_pil2^r) sits at zk index _ROOT_DLOG * r.
    return zk_extended[_root_reindex(n * blowup, _ROOT_DLOG)]


def merkle_tree(arity: int) -> MerkleTree:
    """pil2's tree for `arity`: linear-hash leaves + arity-to-1 Poseidon2 nodes,
    one width-`4*arity` permutation for both."""
    if arity not in _ARITY_WIDTHS:
        raise ValueError(f"arity must be one of {sorted(_ARITY_WIDTHS)}, got {arity}")
    perm = goldilocks_perm(_ARITY_WIDTHS[arity])
    return MerkleTree(
        LinearHash(perm),
        Compression(perm, CompressionParams(arity=arity, chunk=DIGEST_ELEMS)),
    )


@dataclass(frozen=True)
class TraceCommitment:
    """Stage-1 output: the 4-element root, the digest layers (for the query
    phase's openings), and the extended matrix (the FRI witness)."""

    root: Array
    digest_layers: list[Array]
    extended: Array


def commit_trace(trace: Array, *, blowup: int, arity: int) -> TraceCommitment:
    """pil2-stark `extendAndMerkelize`: LDE the trace, merkelize the rows."""
    extended = extend(trace, blowup)
    root, digest_layers = merkle_tree(arity).commit(extended)
    return TraceCommitment(root=root, digest_layers=digest_layers, extended=extended)
