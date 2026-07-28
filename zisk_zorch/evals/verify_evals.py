"""Byte-match the evals stage against a real pil2-stark CUDA dump.

The dump (``golden/pil2_dump/pil2_dump_evals.cu``) runs pil2's verbatim
``computeEvals_v2`` + ``computeEvalsReduction`` kernels — the OOD openings
``eval_m = Σ_i LEv[i]·col_m[i·2^extendBits]`` — on LCG-seeded tiled buffers
and writes the evals vector. This runnable regenerates the inputs, opens the
columns with the production ``zorch.pcs.deep.open_columns``, and byte-compares
every eval. Field sums are exact, so the two reduction orders must agree
bit-for-bit.

Run: python -m zisk_zorch.evals.verify_evals --dump=<dir>
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import frx
import frx.numpy as fnp
from zk_dtypes import goldilocks as F, goldilocksx3 as F3

from zorch.pcs.deep import open_columns
from zorch.utils.field import split_coeffs

P = 0xFFFFFFFF00000001
_A = 6364136223846793005
_C = 1442695040888963407


def _lcg_stream(seed: int, count: int, block: int = 1 << 22) -> np.ndarray:
    """The dump harness's LCG, vectorized: within a block, state i is the
    affine jump ``a^i·s + c·(1 + a + … + a^(i−1))`` mod 2^64 — uint64 wrap-
    around IS the modulus, so cumprod/cumsum compute both tables."""
    apow = np.empty(block, dtype=np.uint64)
    apow[0] = 1
    apow[1:] = _A
    np.cumprod(apow, out=apow)  # a^0 .. a^(block-1), wrapping mod 2^64
    q = np.roll(np.cumsum(apow, dtype=np.uint64), 1)  # Σ_{j<i} a^j
    q[0] = 0
    out = np.empty(count, dtype=np.uint64)
    s = np.uint64(seed)
    a_blk = apow[-1] * np.uint64(_A)
    q_blk = q[-1] * np.uint64(_A) + np.uint64(1)  # Σ_{j<block} a^j
    done = 0
    while done < count:
        n = min(block, count - done)
        states = apow[:n] * s + np.uint64(_C) * q[:n]
        # the harness emits AFTER stepping: state_{i+1} = a·state_i + c
        emitted = np.uint64(_A) * states + np.uint64(_C)
        out[done : done + n] = (emitted >> np.uint64(11)) % np.uint64(P)
        s = a_blk * s + np.uint64(_C) * q_blk if n == block else s
        done += n
    return out


def _untile(flat: np.ndarray, n: int, cols: int) -> np.ndarray:
    """pil2's getBufferOffset tile layout -> (n, cols) row-major (256x4 tiles,
    column-major within a tile, ragged last column-block)."""
    r = np.arange(n)[:, None]
    c = np.arange(cols)[None, :]
    by, cb = c // 4, c % 4
    w = np.minimum(cols - 4 * by, 4)
    idx = by * 4 * n + (r // 256) * w * 256 + cb * 256 + (r % 256)
    return flat[idx]


def main() -> int:
    dump = None
    for a in sys.argv[1:]:
        if a.startswith("--dump="):
            dump = pathlib.Path(a.split("=", 1)[1])
    assert dump is not None, "--dump=<dir> required"
    meta = json.loads((dump / "evals.json").read_text())
    nb, eb = meta["n_bits"], meta["extend_bits"]
    cm1, cm2, cm3 = meta["cm1"], meta["cm2"], meta["cm3"]
    assert meta["n_opening_points"] == 1, "runnable covers the 1-opening dump"
    n, ne = 1 << nb, 1 << (nb + eb)
    b, m = cm1 + cm2, cm1 + cm2 + cm3
    st = _lcg_stream(int(meta["lcg_seed"], 16), ne * (b + 3 * cm3) + n * 3)
    o = 0
    r1 = _untile(st[o : o + ne * cm1], ne, cm1)
    o += ne * cm1
    r2 = _untile(st[o : o + ne * cm2], ne, cm2) if cm2 else np.empty((ne, 0), np.uint64)
    o += ne * cm2
    base_cols = fnp.array(np.hstack([r1, r2]).astype(F))
    ep = _untile(st[o : o + ne * 3 * cm3], ne, 3 * cm3)
    o += ne * 3 * cm3
    ext_cols = fnp.array(
        np.ascontiguousarray(ep.reshape(ne, cm3, 3)).astype(F).view(F3).reshape(ne, cm3)
    )
    lev = _untile(st[o : o + n * 3], n, 3)
    weights = fnp.array(np.ascontiguousarray(lev).astype(F).view(F3).reshape(n, 1))
    got_arr = frx.jit(open_columns, static_argnames=("opening_pos", "stride"))(
        base_cols, ext_cols, weights, (0,) * m, stride=1 << eb
    )
    got = np.asarray(split_coeffs(got_arr)).reshape(-1)
    want = np.fromfile(dump / "evals.bin", dtype=np.uint64)
    ok = got.shape == want.shape and bool(np.array_equal(got, want))
    print(f"dump: N=2^{nb} ext=2^{nb+eb} M={m} ({b} base + {cm3} cubic) ({dump})")
    print(("OK       " if ok else "MISMATCH ") + "evals vs pil2 evmap kernels")
    if not ok and got.shape == want.shape:
        bad = np.nonzero(got != want)[0]
        print(f"  first diff at word {bad[0]} of {got.size} ({bad.size} differ)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
