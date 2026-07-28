"""Byte-match the FRI fold chain against a real pil2-stark CUDA dump.

The dump (``golden/pil2_dump/pil2_dump_fri.cu``) runs pil2's verbatim ``fold``
kernel over the production schedule on an LCG-seeded codeword with a fresh
cubic challenge per layer, and writes every folded layer. This runnable
regenerates the identical inputs from ``fri.json``'s seed, folds with the
production ``fri.fold``, and byte-compares each layer — exact field
arithmetic, so equal or wrong, never close.

Run: python -m zisk_zorch.fri.verify_fri_prove --dump=<dir>
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import frx
import frx.numpy as fnp
from zk_dtypes import goldilocks as F, goldilocksx3 as F3

from zorch.utils.field import split_coeffs

from zisk_zorch.fri.fold import fold

P = 0xFFFFFFFF00000001
M64 = (1 << 64) - 1


def _lcg_stream(seed: int, count: int) -> np.ndarray:
    """The dump harness's LCG (same constants as the DEEP dump's):
    s = s*6364136223846793005 + 1442695040888963407; emit (s >> 11) % p."""
    out = np.empty(count, dtype=np.uint64)
    s = seed
    for i in range(count):
        s = (s * 6364136223846793005 + 1442695040888963407) & M64
        out[i] = (s >> 11) % P
    return out


def main() -> int:
    dump = None
    for a in sys.argv[1:]:
        if a.startswith("--dump="):
            dump = pathlib.Path(a.split("=", 1)[1])
    assert dump is not None, "--dump=<dir> required"
    meta = json.loads((dump / "fri.json").read_text())
    nb_ext, steps = meta["n_bits_ext"], meta["steps"]
    n = 1 << nb_ext
    nfolds = len(steps) - 1
    st = _lcg_stream(int(meta["lcg_seed"], 16), 3 * n + 3 * nfolds)
    pol = fnp.array(st[: 3 * n].astype(F).view(F3).reshape(n))
    chals = st[3 * n :].astype(F).view(F3).reshape(nfolds)
    print(f"dump: schedule {steps} ({dump})")
    ok_all = True
    for k in range(1, len(steps)):
        challenge = fnp.array(chals[k - 1 : k])[0].reshape(())
        pol = fold(pol, challenge, nb_ext, steps[k - 1], steps[k])
        got = np.asarray(split_coeffs(pol)).reshape(-1)
        want = np.fromfile(dump / f"layer{k}.bin", dtype=np.uint64)
        ok = got.shape == want.shape and bool(np.array_equal(got, want))
        ok_all &= ok
        print(
            ("OK       " if ok else "MISMATCH ")
            + f"layer {k} (nBits={steps[k]}) vs pil2 fold kernel"
        )
        if not ok:
            break
    print("FRI fold byte-match: " + ("ALL OK" if ok_all else "FAILED"))
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
