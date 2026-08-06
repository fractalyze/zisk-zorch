"""Poseidon1-Goldilocks parameters — the hash family native ZisK commits with.

pil2-proofman's `DEFAULT_HASH_ID` is `Poseidon1`, and the installed ziskup
v1.0.0-alpha proving key sets no `hash` in `pilout.globalInfo.json`, so native
ZisK builds its stage-1 merkle trees with Poseidon1 — the *optimized-sparse*
(Hades) form, distinct from the Poseidon2 this repo also models. This module
carries pil2's Poseidon1-Goldilocks constants at width 16, the leaf/node width
of arity 4, on zorch's optimized-sparse `SparsePoseidon` core. Arity 4 is the
only one ZisK proves at — `starkStruct.merkleTreeArity` is a single per-key
value governing the stage, constant and FRI trees alike, and ZisK's generated
`MERKLE_TREE_ARITY` fixes it at 4.

The constants are loaded from `goldilocks_constants.json`, a fixture-gen dump of
pil2's own `Poseidon1Constants` (`tools/fixture-gen/`), so nothing is
hand-transcribed. pil2 stores the full-round matrices row-major in the convention
`out[i] = sum_j state[j] * M[j*W + i]`; zorch applies `M @ state`
(`out[i] = sum_j M[i][j] * state[j]`), so `M`/`P` are transposed on the way in.
Reference: https://github.com/0xPolygonHermez/pil2-proofman/blob/v1.0.0-alpha/fields/src/poseidon1.rs
"""

from __future__ import annotations

import functools
import json
import pathlib

import frx
import frx.numpy as fnp
import numpy as np
from frx import Array
from hash_frx.poseidon.params import SparsePoseidonParams
from hash_frx.poseidon.sparse import SparsePoseidon
from zk_dtypes import goldilocks as F

WIDTHS = (16,)
CAPACITY = 4
RATE = {16: 12}
_ALPHA = 7

_CONSTANTS_PATH = pathlib.Path(__file__).with_name("goldilocks_constants.json")


@functools.lru_cache(maxsize=1)
def _constants() -> dict[int, dict]:
    """pil2's Poseidon1-Goldilocks constants, keyed by width. Parsed once — the
    JSON stores each u64 as a decimal string (a JSON number cannot carry a 64-bit
    value exactly)."""
    doc = json.loads(_CONSTANTS_PATH.read_text())
    return {w["width"]: w for w in doc["widths"]}


def _u64(values: list[str]) -> np.ndarray:
    return np.array([int(v) for v in values], dtype=np.uint64)


def _fld(canon: np.ndarray) -> Array:
    """Canonical u64 ints -> a goldilocks array (the dtype cast Montgomery-encodes)."""
    return fnp.array(canon, dtype=F)


def goldilocks_params(width: int) -> SparsePoseidonParams:
    """pil2-stark's Poseidon1-Goldilocks parameterization for `width`.

    Slices pil2's flat `C` into the schedule's ARC groups and its flat `S` into
    each partial round's dot row + column update, and transposes `M`/`P` into
    zorch's `M @ state` convention.
    """
    if width not in WIDTHS:
        raise ValueError(f"width must be one of {WIDTHS}, got {width}")
    c = _constants()[width]
    w = width
    half = c["half_full_rounds"]
    n_partial = c["n_partial_rounds"]

    cc = _u64(c["c"])
    # C layout, in order: initial(W), full-pre((H-1)*W), transition(W),
    # partial(NP), full-post((H-1)*W).
    o = 0
    initial_arc = cc[o : o + w]
    o += w
    full_rc_pre = cc[o : o + (half - 1) * w].reshape(half - 1, w)
    o += (half - 1) * w
    transition_rc = cc[o : o + w]
    o += w
    partial_rc = cc[o : o + n_partial]
    o += n_partial
    full_rc_post = cc[o : o + (half - 1) * w].reshape(half - 1, w)
    o += (half - 1) * w
    if o != len(cc):
        raise ValueError(f"width {w}: C length {len(cc)} != consumed {o}")

    # pil2 stores mat[j*W + i]; transpose to zorch's mds[i][j] = mat[j*W + i].
    mds = _u64(c["m"]).reshape(w, w).T
    transition_matrix = _u64(c["p"]).reshape(w, w).T

    # S: per round r, [dot row (W)][col update (W-1)] contiguous.
    s = _u64(c["s"]).reshape(n_partial, 2 * w - 1)
    partial_dot = s[:, :w]
    partial_col = s[:, w:]

    return SparsePoseidonParams(
        width=w,
        dtype=F,
        alpha=_ALPHA,
        half_full_rounds=half,
        n_partial_rounds=n_partial,
        initial_arc=_fld(initial_arc),
        full_rc_pre=_fld(full_rc_pre),
        transition_rc=_fld(transition_rc),
        partial_rc=_fld(partial_rc),
        full_rc_post=_fld(full_rc_post),
        mds=_fld(mds),
        transition_matrix=_fld(transition_matrix),
        partial_dot=_fld(partial_dot),
        partial_col=_fld(partial_col),
    )


@functools.lru_cache(maxsize=len(WIDTHS))
def goldilocks_perm(width: int) -> SparsePoseidon:
    """The pil2-stark Poseidon1 permutation for `width` on zorch's optimized-sparse
    core. Cached: the params carry constant arrays whose host-side value-key is
    rebuilt per construction, and one commit builds the leaf/node permutation
    repeatedly. Construction is forced eager so a cache miss under an ambient
    trace stores concrete constants, not that trace's tracers (the block
    composite commits before anything else has warmed the cache)."""
    with frx.ensure_compile_time_eval():
        return SparsePoseidon(goldilocks_params(width))
