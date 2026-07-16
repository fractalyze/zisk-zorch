"""OOD-opening round trip: `Σ_k LEv[k]·p(shift·g^k) == p(ξ)`.

Not a pil2 golden — the identity itself pins `compute_lev`/`open_columns` to
pil2's `computeLEv`/`evmap` formulas. Build a random low-degree polynomial, its
evaluations on the extended coset (the "committed column"), and confirm the
LEv-weighted sum recovers a direct evaluation at each opening point `ξ = z·g^p`.
"""

from __future__ import annotations

import frx.numpy as jnp
import numpy as np
from absl.testing import absltest
from zk_dtypes import goldilocks as F
from zk_dtypes import goldilocksx3 as F3

from zorch.poly.univariate import powers

from zisk_zorch.deep.opening import compute_lev, open_columns
from zisk_zorch.quotient.zerofier import _coset_points, _root

_N_BITS = 5
_BLOWUP_BITS = 1


def _rand_cubic(n: int, seed: int) -> jnp.ndarray:
    """`n` random cubic elements (golden `u64x3` numpy path — the fork-safe
    `.view`)."""
    flat = np.random.default_rng(seed).integers(0, 1 << 30, (n, 3)).astype(np.uint64)
    return jnp.array(flat.astype(F).view(F3).reshape(n))


def _poly_eval(coeffs: jnp.ndarray, point: jnp.ndarray) -> jnp.ndarray:
    """`Σ_d coeffs[d]·point^d` for a cubic scalar `point`."""
    return jnp.sum(coeffs * powers(point, coeffs.shape[0]))


def _coset_evals(coeffs: jnp.ndarray, n_bits: int, blowup_bits: int) -> jnp.ndarray:
    """The polynomial evaluated on the extended coset `x_i = shift·w(nBitsExt)^i`
    — the committed column the opening reads."""
    x = _coset_points(n_bits, blowup_bits)  # (N_ext,) base
    mat = [jnp.ones_like(x)]
    for _ in range(coeffs.shape[0] - 1):
        mat.append(mat[-1] * x)
    xpow = jnp.stack(mat, axis=1)  # (N_ext, N) base
    return jnp.sum(coeffs[None, :] * xpow, axis=1)  # (N_ext,) cubic


class OpeningTest(absltest.TestCase):
    def test_opening_recovers_direct_eval(self):
        n = 1 << _N_BITS
        coeffs = _rand_cubic(n, seed=1)  # deg < N
        column = _coset_evals(coeffs, _N_BITS, _BLOWUP_BITS)[:, None]  # (N_ext, 1)
        z = _rand_cubic(1, seed=2)[0]

        # openingPoints [0, 1]: open at z and at z·g (the wrapped next-row point).
        opening_points = (0, 1)
        lev = compute_lev(_limbs(z), opening_points, _N_BITS)
        g = _root(_N_BITS)
        for o, p in enumerate(opening_points):
            xi = z * jnp.power(g, p)
            evals = open_columns(
                column, lev, [o], n_bits=_N_BITS, blowup_bits=_BLOWUP_BITS
            )
            self.assertTrue(
                _cubic_eq(evals[0], _poly_eval(coeffs, xi)),
                f"opening {p} did not recover p(z·g^{p})",
            )


def _limbs(cubic_scalar: jnp.ndarray) -> jnp.ndarray:
    """A cubic scalar as its 3 Goldilocks limbs — `compute_lev`'s `z` input."""
    import frx

    return frx.lax.bitcast_convert_type(cubic_scalar, F).reshape(3)


def _cubic_eq(a: jnp.ndarray, b: jnp.ndarray) -> bool:
    import frx

    la = np.asarray(frx.lax.bitcast_convert_type(a, F).reshape(3))
    lb = np.asarray(frx.lax.bitcast_convert_type(b, F).reshape(3))
    return bool(np.array_equal(la, lb))


if __name__ == "__main__":
    absltest.main()
