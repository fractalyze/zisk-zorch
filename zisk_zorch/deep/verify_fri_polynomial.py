"""Byte-match the DEEP FRI polynomial against a real pil2-stark CUDA dump.

The dump (``golden/pil2_dump`` via ``friexp_bench --dump``) runs pil2's
``computeFRIExpression`` kernel on inputs drawn from a recorded LCG stream and
writes the full output codeword. This runnable regenerates the identical
inputs from ``deep.json``'s seed, evaluates the composition with
``zorch.pcs.deep``, and byte-compares every row — field arithmetic is exact,
so any algebraically equal form must match bit-for-bit.

Run: python -m zisk_zorch.deep.verify_fri_polynomial --dump=<dir>
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import frx
import frx.numpy as fnp
from zk_dtypes import goldilocks as F, goldilocksx3 as F3

from zorch.pcs.deep import deep_numerator_block
from zorch.utils.field import split_coeffs

P = 0xFFFFFFFF00000001
M64 = (1 << 64) - 1


def _lcg_stream(seed: int, count: int) -> np.ndarray:
    """The dump harness's LCG: s = s*6364136223846793005 + 1442695040888963407;
    emit (s >> 11) % p. Mirrored exactly, order-sensitive."""
    out = np.empty(count, dtype=np.uint64)
    s = seed
    for i in range(count):
        s = (s * 6364136223846793005 + 1442695040888963407) & M64
        out[i] = (s >> 11) % P
    return out


def _untile(flat: np.ndarray, n: int, cols: int) -> np.ndarray:
    """pil2's getBufferOffset tile layout -> plain (n, cols) row-major: 256x4
    tiles, column-major within a tile, ragged last column-block."""
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
    meta = json.loads((dump / "deep.json").read_text())
    nb, cm1, cm2, cm3 = (
        meta["n_bits_ext"], meta["cm1"], meta["cm2"], meta["cm3"],
    )
    assert meta["n_opening_points"] == 1, "runnable covers the 1-opening dump"
    n = 1 << nb
    b, m = cm1 + cm2, cm1 + cm2 + cm3
    # Fill order is the harness's: cmPols regions, x, evals, xi, vf1, vf2.
    total = n * (b + 3 * cm3) + n + 3 * m + 3 + 3 + 3
    st = _lcg_stream(int(meta["lcg_seed"], 16), total)
    o = 0
    base_cols = fnp.array(_untile(st[o : o + n * b], n, b).astype(F)); o += n * b
    ext_planes = _untile(st[o : o + n * 3 * cm3], n, 3 * cm3)
    ext_cols = fnp.array(
        np.ascontiguousarray(
            ext_planes.reshape(n, cm3, 3)
        ).astype(F).view(F3).reshape(n, cm3)
    ); o += n * 3 * cm3
    domain = fnp.array(st[o : o + n].astype(F)); o += n
    evals = fnp.array(st[o : o + 3 * m].astype(F).view(F3).reshape(m)); o += 3 * m
    xis = fnp.array(st[o : o + 3].astype(F).view(F3).reshape(1)); o += 3
    o += 3  # vf1: the between-openings Horner challenge, unused at 1 opening
    vf = fnp.array(st[o : o + 3].astype(F).view(F3).reshape(1))[0]

    # pil2 batches by HORNER in vf2: column 0 carries the HIGHEST power,
    # the last column power 0 (starks_gpu.cu computeFRIExpression) — the
    # reverse of deep_composition's ascending vf^m. Equivalent exact form on
    # the production block primitive: reverse the base block (powers 1..b via
    # one extra vf factor), cubic tail at power 0.
    from zorch.poly.univariate import powers

    def pil2_form(base_cols, ext_cols, evals, xi, vf, domain):
        nb_ = deep_numerator_block(
            base_cols[:, ::-1], evals[:b][::-1], powers(vf, b)
        )
        numer = vf * nb_ + (ext_cols[:, 0] - evals[b])
        return numer / (domain - xi)

    f = frx.jit(pil2_form)(base_cols, ext_cols, evals, xis[0], vf, domain)
    got = np.asarray(split_coeffs(f)).reshape(-1)
    want = np.fromfile(dump / "fri.bin", dtype=np.uint64)
    ok = got.shape == want.shape and bool(np.array_equal(got, want))
    print(f"dump: n_bits_ext={nb} M={m} ({b} base + {cm3} cubic) ({dump})")
    print(("OK       " if ok else "MISMATCH ") + "DEEP codeword vs pil2 dump")
    if not ok and got.shape == want.shape:
        bad = np.nonzero(got != want)[0]
        print(f"  first diff at word {bad[0]} of {got.size} ({bad.size} differ)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
