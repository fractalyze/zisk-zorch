"""pil2's computeQ coefficient split, checked against direct chunk evaluation.

q has degree < qDeg·N, so the committed section's chunk p must equal the
coset evaluation of the coefficient chunk ``c[pN:(p+1)N]``. Both sides here
share only the definition of coset evaluation (the shift^k-scaled NTT that
compute_q's INTT inverts); the transform under test supplies the chunk
scale, the re-NTT, and the row-major chunk interleave. qDeg = 1 must
reduce to the identity — the prover commits the evaluations directly there.
"""

from __future__ import annotations

import frx.numpy as fnp
import numpy as np
from absl.testing import absltest
from frx import lax
from zk_dtypes import goldilocks as F
from zk_dtypes import pfinfo

from zisk_zorch.quotient.compute_q import compute_q
from zisk_zorch.quotient.zerofier import _PIL2_GENERATOR

_P = int(pfinfo(F).modulus)
_SHIFT = 7
_NB, _NBE = 3, 5


def _coset_eval(coeff_lanes: np.ndarray, ne: int):
    """``(3, k)`` coefficient lanes -> their ``(3, ne)`` coset evaluation:
    the shift^k scale rides the coefficients, python-int math because the
    lane product overflows u64."""
    k = coeff_lanes.shape[1]
    shift_pow = np.array([pow(_SHIFT, i, _P) for i in range(k)], dtype=object)
    scaled = (coeff_lanes.astype(object) * shift_pow) % _P
    lanes = np.zeros((3, ne), dtype=np.uint64)
    lanes[:, :k] = scaled.astype(np.uint64)
    return lax.ntt(
        fnp.array(lanes.astype(F)),
        ntt_type="NTT",
        ntt_length=ne,
        generator=_PIL2_GENERATOR,
    )


class ComputeQTest(absltest.TestCase):
    def _check(self, q_deg: int) -> None:
        n, ne = 1 << _NB, 1 << _NBE
        rng = np.random.default_rng(q_deg)
        coeffs = rng.integers(0, _P, size=(3, q_deg * n), dtype=np.uint64)
        q_cols = fnp.transpose(_coset_eval(coeffs, ne), (1, 0))
        got = np.asarray(compute_q(q_cols, _NB, _NBE, q_deg)).astype(np.uint64)
        self.assertEqual(got.shape, (ne, 3 * q_deg))
        for p in range(q_deg):
            want = np.asarray(_coset_eval(coeffs[:, p * n : (p + 1) * n], ne))
            np.testing.assert_array_equal(
                got[:, 3 * p : 3 * p + 3],
                want.astype(np.uint64).T,
                err_msg=f"chunk {p}",
            )

    def test_qdeg_1_is_the_identity(self):
        self._check(1)

    def test_qdeg_2(self):
        self._check(2)

    def test_qdeg_4(self):
        self._check(4)


if __name__ == "__main__":
    absltest.main()
