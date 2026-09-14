"""The openings' row windows, and the quotient's window count, on a
synthetic AIR.

Both openings programs are evaluated one row window per dispatch so the
evMap's columns are never alive over the whole extended domain at once
(#241, #243). Composed back -- the DEEP batch concatenated, `evals` added --
they must equal the whole-domain result exactly: these are field elements,
not floats, and there is no rounding to hide behind. Needs no proving key
and no device.
"""

from __future__ import annotations

import os

import frx
import frx.numpy as fnp
import numpy as np
from absl.testing import absltest, parameterized
from zk_dtypes import goldilocks as F
from zk_dtypes import goldilocksx3 as F3
from zorch.utils.field import join_coeffs

from zisk_zorch.harness import pil2_prover
from zisk_zorch.harness.pil2 import add_windows, committed_column, row_windows
from zisk_zorch.harness.pil2_prover import (
    _OPENING_ROW_CHUNKS,
    _Q_MAX_CHUNKS,
    Pil2OpeningProver,
    _row_chunks,
)

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
    """The attribute surface the openings read. The real role wants a proving
    key and a device to construct; the openings themselves want neither."""

    _columns = Pil2OpeningProver._columns
    deep_fn = Pil2OpeningProver.deep_fn
    evals_fn = Pil2OpeningProver.evals_fn

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
        for lo, hi in row_windows(_NE, _OPENING_ROW_CHUNKS):
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
                for lo, hi in row_windows(_NE, _OPENING_ROW_CHUNKS)
            ]
        )
        whole = frx.jit(role.deep_fn)(bufs, evals, domain, xi, vf1, vf2)

        self.assertEqual(windowed.shape, whole.shape)
        np.testing.assert_array_equal(np.asarray(windowed), np.asarray(whole))


class EvalsRowWindowTest(absltest.TestCase):
    def test_the_windows_added_equal_the_whole_domain_openings(self):
        bufs = _sections(23)
        lev = join_coeffs(
            fnp.asarray(
                np.random.default_rng(29)
                .integers(0, 1 << 32, size=(1 << _NB, len(_SI["openingPoints"]), 3))
                .astype(F)
            ),
            F3,
        )
        role = _Role(_SI, _NB, _NBE)

        # The same argument split `Pil2OpeningProver` jits with: a traced
        # `start` and a static `size`.
        chunk = frx.jit(role.evals_fn, static_argnames=("size",))
        # The composition `Pil2OpeningProver.prove`, `evals_sum` and
        # `driver::prove` all perform: one dispatch per window, added.
        windowed = add_windows(
            chunk(bufs, lev, lo, size=hi - lo)
            for lo, hi in row_windows(1 << _NB, _OPENING_ROW_CHUNKS)
        )
        whole = frx.jit(role.evals_fn)(bufs, lev)

        self.assertEqual(windowed.shape, whole.shape)
        np.testing.assert_array_equal(np.asarray(windowed), np.asarray(whole))


class QuotientRowChunkTest(parameterized.TestCase):
    """The cExp's window count against the clients sharing the card. The
    ceiling prices dispatch against cache with one client owning it, so N
    clients -- each with 1/N of the card -- relax it N-fold."""

    def _chunks(self, clients: str | None, per_row: int) -> int:
        env = {k: v for k, v in os.environ.items() if not k.startswith("ZISK_")}
        if clients is not None:
            env["ZISK_CLIENTS"] = clients
        with (
            absltest.mock.patch.dict(os.environ, env, clear=True),
            absltest.mock.patch.object(
                pil2_prover, "live_bytes_per_row", return_value=per_row
            ),
        ):
            return _row_chunks([], [], 1 << 20)

    @parameterized.named_parameters(
        ("one_client", "1", _Q_MAX_CHUNKS),
        ("two_clients", "2", 2 * _Q_MAX_CHUNKS),
        ("four_clients", "4", 4 * _Q_MAX_CHUNKS),
    )
    def test_a_hungry_air_takes_the_ceiling_its_clients_allow(self, clients, want):
        # A per-row cost whose whole-domain set wants far more windows than
        # any of these ceilings, so the ceiling is what answers.
        self.assertEqual(self._chunks(clients, 1 << 20), want)

    def test_one_client_is_the_default(self):
        self.assertEqual(self._chunks(None, 1 << 20), _Q_MAX_CHUNKS)

    def test_a_cheap_air_is_not_moved_by_the_client_count(self):
        # Under the live-set target at one window, so no ceiling applies.
        for clients in ("1", "2", "4"):
            self.assertEqual(self._chunks(clients, 1), 1)

    def test_a_client_count_below_one_is_refused(self):
        with self.assertRaisesRegex(ValueError, "ZISK_CLIENTS"):
            self._chunks("0", 1 << 20)


if __name__ == "__main__":
    absltest.main()
