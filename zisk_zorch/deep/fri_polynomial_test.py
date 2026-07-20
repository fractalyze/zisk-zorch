"""DEEP composition verified by the FRI low-degree property.

Not a pil2 golden (that needs a proving-key `friExp` op list). Instead: build
random low-degree columns, open them *correctly* at the OOD point, and confirm
the DEEP polynomial `f` is low degree — every summand `(p(x) − p(ξ))/(x − ξ)` is
a genuine polynomial of degree `< N − 1`, so `f`'s coset INTT vanishes above that
bound. A *wrong* opening leaves a rational remainder, so `f` is full degree — the
negative control.
"""

from __future__ import annotations

import frx
import frx.numpy as jnp
import numpy as np
from absl.testing import absltest
from zk_dtypes import goldilocks as F
from zk_dtypes import goldilocksx3 as F3

from zorch.poly.univariate import powers

from zisk_zorch.deep.fri_polynomial import _ood_points, deep_composition
from zisk_zorch.fri.fold import intt
from zorch.utils.field import to_limb_rows
from zisk_zorch.quotient.zerofier import _coset_points

_N_BITS = 4
_BLOWUP_BITS = 1
_N_BASE = 3  # base committed columns (the #69 8-byte path)
_N_CUBIC = 2  # cubic committed columns (e.g. the quotient)
_N_COLS = _N_BASE + _N_CUBIC


def _rand_cubic(shape, seed: int) -> jnp.ndarray:
    n = int(np.prod(shape))
    flat = np.random.default_rng(seed).integers(0, 1 << 30, (n, 3)).astype(np.uint64)
    return jnp.array(flat.astype(F).view(F3).reshape(n)).reshape(shape)


def _rand_base(shape, seed: int) -> jnp.ndarray:
    return jnp.asarray(
        np.random.default_rng(seed).integers(0, 1 << 30, shape).astype(np.uint64).view(F)
    )


def _coset_evals(coeffs: jnp.ndarray) -> jnp.ndarray:
    """`(M, N)` cubic coeffs -> `(N_ext, M)` cubic evals on the extended coset."""
    x = _coset_points(_N_BITS, _BLOWUP_BITS)  # (N_ext,) base
    mat = [jnp.ones_like(x)]
    for _ in range(coeffs.shape[1] - 1):
        mat.append(mat[-1] * x)
    xpow = jnp.stack(mat, axis=1)  # (N_ext, N) base
    # col[i, m] = sum_d coeffs[m, d] * x_i^d
    cols = [jnp.sum(coeffs[m][None, :] * xpow, axis=1) for m in range(coeffs.shape[0])]
    return jnp.stack(cols, axis=1)  # (N_ext, M) cubic


def _poly_eval(coeffs_row: jnp.ndarray, point: jnp.ndarray) -> jnp.ndarray:
    return jnp.sum(coeffs_row * powers(point, coeffs_row.shape[0]))


def _high_coeffs(f: jnp.ndarray) -> np.ndarray:
    """The extended-coset INTT coefficients of cubic `f` at index >= N-1 (the
    degrees a valid DEEP polynomial must not reach), as canonical limbs."""
    n_ext = f.shape[0]
    base = to_limb_rows(f)
    coeffs = intt(base, _N_BITS + _BLOWUP_BITS)  # (N_ext, 3) base
    return np.asarray(coeffs[(1 << _N_BITS) - 1:])


class DeepCompositionTest(absltest.TestCase):
    def setUp(self):
        n = 1 << _N_BITS
        # Mixed committed columns, base then cubic (the DEEP batching order): base
        # columns exercise #69's 8-byte path, cubic ones stand in for the quotient.
        base_coeffs = _rand_base((_N_BASE, n), seed=1)  # deg < N, base
        cubic_coeffs = _rand_cubic((_N_CUBIC, n), seed=4)  # deg < N, cubic
        self.base_cols = _coset_evals(base_coeffs)  # (N_ext, B) base
        self.cubic_cols = _coset_evals(cubic_coeffs)  # (N_ext, C) cubic
        self.z = _rand_cubic((1,), seed=2)[0]
        self.vf = _rand_cubic((1,), seed=3)[0]
        self.opening_pos = [0] * _N_COLS
        self.xis = _ood_points(_limbs(self.z), (0,), _N_BITS)  # (1,) = [z]
        self.evals = jnp.stack(
            [_poly_eval(base_coeffs[m], self.z) for m in range(_N_BASE)]
            + [_poly_eval(cubic_coeffs[m], self.z) for m in range(_N_CUBIC)]
        )

    def _compose(self, evals):
        return deep_composition(
            self.base_cols, self.cubic_cols, evals, self.xis, self.opening_pos,
            self.vf, n_bits=_N_BITS, blowup_bits=_BLOWUP_BITS,
        )

    def test_correct_opening_is_low_degree(self):
        high = _high_coeffs(self._compose(self.evals))
        self.assertTrue(np.all(high == 0), "DEEP polynomial exceeded its degree bound")

    def test_wrong_opening_is_not_low_degree(self):
        bad = self.evals.at[0].set(self.evals[0] + self.evals[1])
        high = _high_coeffs(self._compose(bad))
        self.assertFalse(np.all(high == 0), "wrong opening should break low-degreeness")


def _limbs(cubic_scalar: jnp.ndarray) -> jnp.ndarray:
    return frx.lax.bitcast_convert_type(cubic_scalar, F).reshape(3)


if __name__ == "__main__":
    absltest.main()
