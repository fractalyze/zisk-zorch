"""The DEEP batch's row windows, on a synthetic AIR.

The batch is evaluated one row window per dispatch so the evMap's cubic
columns are never alive over the whole extended domain at once (#241). It is
elementwise over that domain, so the windows concatenated must equal the
whole-domain result exactly -- these are field elements, not floats, and
there is no rounding to hide behind. Needs no proving key and no device.
"""

from __future__ import annotations

import frx
import frx.numpy as fnp
import numpy as np
from absl.testing import absltest, parameterized
from zk_dtypes import goldilocks as F
from zk_dtypes import goldilocksx3 as F3
from zorch.utils.field import join_coeffs

from zisk_zorch.harness.pil2 import committed_column, row_windows
from zisk_zorch.harness.pil2_prover import _DEEP_ROW_CHUNKS, Pil2OpeningProver

_NB, _NBE = 4, 5
_NE = 1 << _NBE

# One entry per shape `committed_column` can produce -- a base committed
# column, a cubic one, a constant, a custom commit -- spread over two
# opening points so the batch's vf1-Horner across groups is exercised too.
_CM_POLS_MAP = [
    {"stage": 1, "stagePos": 0, "dim": 1},
    {"stage": 1, "stagePos": 1, "dim": 3},
    {"stage": 2, "stagePos": 0, "dim": 3},
]
_EV_MAP = [
    {"type": "cm", "id": 0, "openingPos": 0},
    {"type": "cm", "id": 1, "openingPos": 1},
    {"type": "cm", "id": 2, "openingPos": 0},
    {"type": "const", "id": 1, "openingPos": 1},
    {"type": "custom", "commitId": 0, "id": 0, "openingPos": 0},
]
_SI = {
    "cmPolsMap": _CM_POLS_MAP,
    "evMap": _EV_MAP,
    "openingPoints": [0, 1],
}


class _Role:
    """The attribute surface `deep_fn` reads. The real role wants a proving
    key and a device to construct; the batch itself wants neither."""

    _columns = Pil2OpeningProver._columns
    deep_fn = Pil2OpeningProver.deep_fn

    def __init__(self, si: dict, nb: int, nbe: int) -> None:
        self._si, self._nb, self._nbe = si, nb, nbe


def _sections(seed: int) -> dict:
    rng = np.random.default_rng(seed)

    def section(width: int):
        return fnp.asarray(rng.integers(0, 1 << 32, size=(_NE, width)).astype(F))

    return {
        ("cm", 1): section(4),
        ("cm", 2): section(3),
        ("const", 0): section(2),
        ("custom", 0): section(2),
    }


def _cubic_scalars(seed: int, n: int):
    limbs = np.random.default_rng(seed).integers(0, 1 << 32, size=(n, 3)).astype(F)
    return join_coeffs(fnp.asarray(limbs), F3).reshape(n)


class RowWindowsTest(parameterized.TestCase):
    @parameterized.named_parameters(
        ("divides", 32, 8),
        ("does_not_divide", 30, 8),
        ("one_window", 32, 1),
        ("more_windows_than_rows", 3, 8),
    )
    def test_the_windows_tile_the_domain_in_order(self, n: int, count: int):
        windows = row_windows(n, count)
        self.assertEqual(windows[0][0], 0)
        self.assertEqual(windows[-1][1], n)
        for (_, hi), (lo, _) in zip(windows, windows[1:]):
            self.assertEqual(hi, lo, "the windows leave a gap or overlap")
        self.assertTrue(all(lo < hi for lo, hi in windows), "an empty window")


class CommittedColumnWindowTest(parameterized.TestCase):
    @parameterized.named_parameters(*((f"evmap_{i}", i) for i in range(len(_EV_MAP))))
    def test_a_window_is_the_whole_column_sliced(self, entry: int):
        bufs = _sections(7)
        whole = committed_column(_EV_MAP[entry], _CM_POLS_MAP, bufs)
        for lo, hi in row_windows(_NE, _DEEP_ROW_CHUNKS):
            window = committed_column(
                _EV_MAP[entry], _CM_POLS_MAP, bufs, rows=(lo, hi - lo)
            )
            np.testing.assert_array_equal(np.asarray(window), np.asarray(whole[lo:hi]))


class DeepRowWindowTest(absltest.TestCase):
    def test_the_windows_concatenated_equal_the_whole_domain_batch(self):
        bufs = _sections(11)
        evals = _cubic_scalars(13, len(_EV_MAP))
        domain = fnp.asarray(
            np.random.default_rng(17).integers(0, 1 << 32, size=_NE).astype(F)
        )
        xi, vf1, vf2 = _cubic_scalars(19, 3)
        role = _Role(_SI, _NB, _NBE)

        # Through `jit` with the argument split `Pil2OpeningProver` builds:
        # a traced `start` and a static `size`. Calling `deep_fn` directly
        # cannot see that split, and a `size` that traces fails only here --
        # a slice's width has to be known at trace time.
        chunk = frx.jit(role.deep_fn, static_argnames=("size",))
        # The composition `Pil2OpeningProver.prove` and `driver::prove` both
        # perform: one dispatch per window, concatenated in domain order.
        windowed = fnp.concatenate(
            [
                chunk(bufs, evals, domain, xi, vf1, vf2, lo, size=hi - lo)
                for lo, hi in row_windows(_NE, _DEEP_ROW_CHUNKS)
            ]
        )
        whole = frx.jit(role.deep_fn)(bufs, evals, domain, xi, vf1, vf2)

        self.assertEqual(windowed.shape, whole.shape)
        np.testing.assert_array_equal(np.asarray(windowed), np.asarray(whole))


if __name__ == "__main__":
    absltest.main()
