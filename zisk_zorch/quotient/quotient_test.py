"""Stage-2 quotient: zerofier byte-match + the `C / Z_H` division.

The inverse zerofier is pinned against pil2's `buildZHInv` golden (exact). The
division and the `constraint_eval` composition are checked by round-trip: build
`Z_H` independently from the coset formula, form `C = Q * Z_H`, and confirm
`compute_quotient(C)` recovers `Q`.
"""

from __future__ import annotations

import pathlib

import frx.numpy as fnp
import numpy as np
from absl.testing import absltest
from zk_dtypes import goldilocks as F
from zk_dtypes import goldilocksx3 as F3

from zisk_zorch.golden import load, u64
from zisk_zorch.quotient.quotient import compute_quotient, quotient_from_constraints
from zisk_zorch.quotient.zerofier import (
    _root,
    inv_frame_zerofier,
    inv_one_row_zerofier,
    inv_zerofier,
)

_TESTDATA = pathlib.Path(__file__).parent / "testdata" / "golden"
_MODULUS = 0xFFFFFFFF00000001
_COSET_SHIFT = 7
_TWO_ADIC_ROOT = 7277203076849721926


def _base(values: list[int]) -> fnp.ndarray:
    return fnp.array(np.array([v % _MODULUS for v in values], dtype=np.uint64), dtype=F)


def _cubic(limbs: list[int]) -> fnp.ndarray:
    """Flat canonical limbs (3 per element) -> a goldilocksx3 array."""
    flat = np.array([v % _MODULUS for v in limbs], dtype=np.uint64).reshape(-1, 3)
    return fnp.array(flat.astype(F).view(F3).reshape(flat.shape[0]))


def _zh_evals(n_bits: int, blowup_bits: int) -> fnp.ndarray:
    """`Z_H(x) = x^N - 1` on the coset, built independently of the impl."""
    extend = 1 << blowup_bits
    n_ext = 1 << (n_bits + blowup_bits)
    sn = pow(_COSET_SHIFT, 1 << n_bits, _MODULUS)
    w_ext = pow(_TWO_ADIC_ROOT, 1 << (32 - blowup_bits), _MODULUS)
    period = [(sn * pow(w_ext, i, _MODULUS) - 1) % _MODULUS for i in range(extend)]
    return _base(period * (n_ext // extend))


class ZerofierTest(absltest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.golden = load(_TESTDATA / "zerofier_inv.json")

    def test_root_rejects_bits_past_the_two_adic_ceiling(self) -> None:
        # pil2's generator is W[32], so there is no 2^33-th root; the guard must
        # name that rather than let `1 << (32 - bits)` fail as a Python shift.
        self.assertIsNotNone(_root(32))
        for bits in (33, 64, -1):
            with self.subTest(bits=bits), self.assertRaisesRegex(ValueError, "W\\[32\\]"):
                _root(bits)

    def test_every_row_matches_pil2_build_zh_inv(self) -> None:
        for case in self.golden["every_row"]:
            with self.subTest(n_bits=case["n_bits"], blowup_bits=case["blowup_bits"]):
                got = inv_zerofier(case["n_bits"], case["blowup_bits"])
                self.assertTrue(bool(fnp.array_equal(got, u64(case["zi"]))))

    def test_one_row_matches_pil2_build_one_row_zerofier_inv(self) -> None:
        for case in self.golden["one_row"]:
            with self.subTest(blowup_bits=case["blowup_bits"], row_index=case["row_index"]):
                got = inv_one_row_zerofier(
                    case["n_bits"], case["blowup_bits"], case["row_index"]
                )
                self.assertTrue(bool(fnp.array_equal(got, u64(case["zi"]))))

    def test_frame_matches_pil2_build_frame_zerofier_inv(self) -> None:
        for case in self.golden["frame"]:
            with self.subTest(offset_min=case["offset_min"], offset_max=case["offset_max"]):
                got = inv_frame_zerofier(
                    case["n_bits"], case["blowup_bits"],
                    case["offset_min"], case["offset_max"],
                )
                self.assertTrue(bool(fnp.array_equal(got, u64(case["zi"]))))


class QuotientTest(absltest.TestCase):
    def test_quotient_recovers_q_from_composite(self) -> None:
        # C = Q * Z_H, so Q = C / Z_H = C * Zi must recover Q exactly.
        n_bits, blowup_bits = 3, 2
        n_ext = 1 << (n_bits + blowup_bits)
        q = _cubic([(7 * i + 3) % _MODULUS for i in range(3 * n_ext)])
        composite = q * _zh_evals(n_bits, blowup_bits)
        got = compute_quotient(composite, n_bits, blowup_bits)
        self.assertTrue(bool(fnp.array_equal(got, q)))

    def test_quotient_from_constraints_folds_then_divides(self) -> None:
        # The composed path equals constraint_eval (via compute_quotient) — the
        # alpha-fold composite divided by the zerofier.
        from zorch.constraint_eval import constraint_eval

        n_bits, blowup_bits = 3, 1
        n_ext = 1 << (n_bits + blowup_bits)
        trace = _base([(i * i + 1) % _MODULUS for i in range(n_ext)]).reshape(n_ext, 1)
        alpha = _cubic([2, 0, 0, 5, 0, 0])  # K=2 cubic fold weights

        def eval_fn(t):  # two constraints in the trailing axis
            return fnp.concatenate([t, t * t], axis=-1)

        got = quotient_from_constraints(eval_fn, trace, alpha, n_bits, blowup_bits)
        want = compute_quotient(constraint_eval(eval_fn, trace, alpha), n_bits, blowup_bits)
        self.assertTrue(bool(fnp.array_equal(got, want)))


if __name__ == "__main__":
    absltest.main()
