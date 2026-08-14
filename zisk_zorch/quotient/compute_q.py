"""pil2's ``computeQ``: the quotient evaluations -> the committed Q section.

The composite quotient q = cExp/Z_H has degree < qDeg·N, so it commits as
qDeg polynomials of degree < N: INTT the coset evaluations at the extended
length, split the coefficients into qDeg chunks, scale chunk p by
``shift^(-pN)``, and coset-evaluate each chunk back on the extended domain.
The per-chunk scale is what makes the plain re-NTT land on the coset: the
INTT of coset evaluations leaves ``c_k·shift^k``, and stripping only the
``shift^(pN)`` part keeps the in-chunk ``shift^r`` that a coset evaluation
needs — which is also why qDeg = 1 reduces to the identity (S[0] = 1), the
shape every inner AIR has and the reason this transform went unmodeled until
the recursion circuits (qDeg 4-7, fractalyze/zisk-zorch#112).

Layout matches ``proof_stark``'s section: row i carries ``Q0..Q_{qDeg-1}``,
three gl64 lanes each.
"""

from __future__ import annotations

import frx.numpy as fnp
import numpy as np
from frx import lax
from zk_dtypes import goldilocks as F
from zk_dtypes import pfinfo

from zisk_zorch.quotient.zerofier import _PIL2_GENERATOR

_P = int(pfinfo(F).modulus)
_SHIFT = 7


def compute_q(q_ext_cols, nb: int, nbe: int, q_deg: int):
    """``(Ne, 3)`` quotient evaluation lanes -> the ``(Ne, 3*q_deg)``
    committed section, both in pil2's domain order."""
    n, ne = 1 << nb, 1 << nbe
    coeffs = lax.ntt(
        q_ext_cols.T, ntt_type="INTT", ntt_length=ne, generator=_PIL2_GENERATOR
    )
    shift_in = pow(pow(_SHIFT, _P - 2, _P), n, _P)
    cols = []
    s = 1
    for p in range(q_deg):
        chunk = coeffs[:, p * n : (p + 1) * n] * fnp.array(np.uint64(s).astype(F))
        pad = fnp.zeros((3, ne - n), F)
        cols.append(
            lax.ntt(
                fnp.concatenate([chunk, pad], axis=1),
                ntt_type="NTT",
                ntt_length=ne,
                generator=_PIL2_GENERATOR,
            )
        )
        s = (s * shift_in) % _P
    out = fnp.stack(cols, axis=0)
    return fnp.transpose(out, (2, 0, 1)).reshape(ne, 3 * q_deg)
