"""Byte-match the LogUp grand-sum against a real pil2-stark CUDA dump.

The dump (``golden/pil2_dump/pil2_dump_gsum.cu``) chains pil2's per-row cubic
``inv_fold`` (``local[r] = Σ_i num[r,i]·den[r,i]⁻¹``) into pil2's verbatim
Blelloch scan (``accOperationGPU`` from ``hints.cu``) and writes both stages.
This runnable regenerates the inputs and byte-matches the production
``quotient.gsum.grand_sum`` (fused fold + ``cumsum``) against the scanned
column, and its local-term half against ``local.bin`` for localization.

Run: python -m zisk_zorch.quotient.verify_gsum --dump=<dir>
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import frx.numpy as fnp
from zk_dtypes import goldilocks as F, goldilocksx3 as F3

from zorch.utils.field import split_coeffs

from zisk_zorch.quotient.gsum import grand_sum

P = 0xFFFFFFFF00000001
_A = 6364136223846793005
_C = 1442695040888963407


def _lcg_stream(seed: int, count: int, block: int = 1 << 22) -> np.ndarray:
    """The gsum harness's LCG: like the other dumps' but emitting
    ``(s >> 11) % p + 1`` — the +1 keeps denominators nonzero. Vectorized via
    affine jumps; uint64 wraparound is the mod-2^64 arithmetic."""
    apow = np.empty(block, dtype=np.uint64)
    apow[0] = 1
    apow[1:] = _A
    np.cumprod(apow, out=apow)
    q = np.roll(np.cumsum(apow, dtype=np.uint64), 1)
    q[0] = 0
    out = np.empty(count, dtype=np.uint64)
    s = np.uint64(seed)
    a_blk = apow[-1] * np.uint64(_A)
    q_blk = q[-1] * np.uint64(_A) + np.uint64(1)
    done = 0
    while done < count:
        n = min(block, count - done)
        states = apow[:n] * s + np.uint64(_C) * q[:n]
        emitted = np.uint64(_A) * states + np.uint64(_C)
        out[done : done + n] = (emitted >> np.uint64(11)) % np.uint64(P) + np.uint64(1)
        s = a_blk * s + np.uint64(_C) * q_blk if n == block else s
        done += n
    return out


def _check(name: str, got_arr, want_path: pathlib.Path) -> bool:
    got = np.asarray(split_coeffs(got_arr)).reshape(-1)
    want = np.fromfile(want_path, dtype=np.uint64)
    ok = got.shape == want.shape and bool(np.array_equal(got, want))
    print(("OK       " if ok else "MISMATCH ") + name)
    if not ok and got.shape == want.shape:
        bad = np.nonzero(got != want)[0]
        print(f"  first diff at word {bad[0]} of {got.size} ({bad.size} differ)")
    return ok


def main() -> int:
    dump = None
    for a in sys.argv[1:]:
        if a.startswith("--dump="):
            dump = pathlib.Path(a.split("=", 1)[1])
    assert dump is not None, "--dump=<dir> required"
    meta = json.loads((dump / "gsum.json").read_text())
    nb, I = meta["n_bits"], meta["interactions"]
    n = 1 << nb
    st = _lcg_stream(int(meta["lcg_seed"], 16), 2 * n * I * 3)
    num = fnp.array(st[0::2].astype(F).view(F3).reshape(n, I))
    den = fnp.array(st[1::2].astype(F).view(F3).reshape(n, I))
    print(f"dump: N=2^{nb} I={I} ({dump})")

    local = num[:, 0] / den[:, 0]
    for i in range(1, I):
        local = local + num[:, i] / den[:, i]
    ok = _check("local inv+fold vs pil2 kernel", local, dump / "local.bin")
    ok &= _check(
        "grand-sum column vs pil2 Blelloch scan",
        grand_sum(num, den),
        dump / "gsum.bin",
    )
    print("LogUp grand-sum byte-match: " + ("ALL OK" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
