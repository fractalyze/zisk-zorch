"""The circom witness chain's host-side halves, without a proving key.

`_add_waves` and the sMap placement are the two places where this module
departs from proofman's own evaluation order for speed — the waves batch the
exec adds by depth instead of walking them in index order, and the gather runs
on the device off a memoized sMap upload. Both are pinned here against the
straightforward form they replace, on a hand-built calculator: the real one
needs a circuit ``.so``/``.dat`` pair no CI host carries.
"""

from __future__ import annotations

import numpy as np
from absl.testing import absltest
from zk_dtypes import goldilocks as F

from zisk_zorch.harness.recursion import CircomCalc, P, _add_waves


def _adds(rows: list[tuple[int, int, int, int]]) -> np.ndarray:
    """Exec add rows ``(operand_a, operand_b, coeff_a, coeff_b)``."""
    return np.array(rows, dtype=np.uint64).reshape(-1, 4)


def _calc(size_witness: int, adds: np.ndarray, smap_flat: np.ndarray, n_smap: int):
    """A `CircomCalc` with its ctypes half left unbuilt — everything below
    `witness()` is pure host arithmetic over the exec data."""
    calc = CircomCalc.__new__(CircomCalc)
    calc.size_witness = size_witness
    calc.n_adds = len(adds)
    calc.adds = adds
    calc.smap_flat = smap_flat.astype(np.int64)
    calc.n_smap = n_smap
    calc._add_waves = _add_waves(adds, size_witness)
    calc._smap_dev = None
    calc._smap_cols = None
    return calc


def _settled_in_index_order(
    w: np.ndarray, adds: np.ndarray, size_witness: int
) -> np.ndarray:
    """proofman's own form: the adds evaluated strictly in index order."""
    w = w.copy()
    for i, (a, b, ca, cb) in enumerate(adds):
        w[size_witness + i] = (int(w[a]) * int(ca) + int(w[b]) * int(cb)) % P
    w[0] = 0
    return w % np.uint64(P)


class AddWavesTest(absltest.TestCase):
    def test_waves_group_a_serial_chain_one_row_deep(self):
        # add1 reads add0's slot, add2 reads add1's — one add per wave.
        adds = _adds([(1, 2, 2, 3), (4, 3, 1, 5), (5, 4, 1, 1)])
        waves = _add_waves(adds, size_witness=4)
        self.assertEqual([w.tolist() for w in waves], [[0], [1], [2]])

    def test_independent_adds_share_one_wave(self):
        adds = _adds([(1, 2, 1, 1), (2, 3, 1, 1), (1, 3, 1, 1)])
        waves = _add_waves(adds, size_witness=4)
        self.assertEqual([w.tolist() for w in waves], [[0, 1, 2]])

    def test_no_adds_is_no_waves(self):
        self.assertEqual(_add_waves(_adds([]), size_witness=4), [])

    def test_an_add_reading_a_later_slot_is_rejected(self):
        # The reordering is only equivalent to index order while every operand
        # is an earlier slot; index order would read slot 5 as the buffer's
        # initial zero here, a wave would read add1's output.
        adds = _adds([(5, 1, 1, 1), (2, 3, 1, 1)])
        with self.assertRaisesRegex(AssertionError, "later add's slot"):
            _add_waves(adds, size_witness=4)

    def test_settled_matches_index_order_evaluation(self):
        adds = _adds([(1, 2, 2, 3), (4, 3, 1, 5), (5, 4, 7, 11)])
        calc = _calc(4, adds, np.arange(7), n_smap=7)
        w = np.array([9, 1, 2, P - 1, 0, 0, 0], dtype=np.uint64)
        np.testing.assert_array_equal(
            calc._settled(w), _settled_in_index_order(w, adds, 4)
        )


class CommittedTraceTest(absltest.TestCase):
    def _two_col_calc(self):
        adds = _adds([(1, 2, 2, 3)])
        # 3 rows x 2 columns of sMap; index 0 is the unmapped cell.
        return _calc(4, adds, np.array([1, 2, 3, 4, 0, 1]), n_smap=3)

    def test_device_gather_matches_the_host_gather(self):
        calc = self._two_col_calc()
        w = np.array([9, 1, 2, 5, 0], dtype=np.uint64)
        np.testing.assert_array_equal(
            np.asarray(calc.committed_trace(w, 3, 2)).view(np.uint64),
            calc.committed_pols(w, 3, 2),
        )

    def test_padding_rows_match_the_host_gather(self):
        calc = self._two_col_calc()
        w = np.array([9, 1, 2, 5, 0], dtype=np.uint64)
        np.testing.assert_array_equal(
            np.asarray(calc.committed_trace(w, 8, 2)).view(np.uint64),
            calc.committed_pols(w, 8, 2),
        )

    def test_a_second_width_does_not_reuse_the_first_upload(self):
        # The sMap upload is memoized, but `n_cols` is a per-call parameter:
        # a memo that ignored it would answer the second call at the first
        # call's width instead of failing the way `committed_pols` does.
        calc = self._two_col_calc()
        w = np.array([9, 1, 2, 5, 0], dtype=np.uint64)
        calc.committed_trace(w, 3, 2)
        with self.assertRaises(ValueError):
            calc.committed_pols(w, 3, 3)
        with self.assertRaises(ValueError):
            calc.committed_trace(w, 3, 3)

    def test_the_trace_is_field_typed(self):
        calc = self._two_col_calc()
        w = np.array([9, 1, 2, 5, 0], dtype=np.uint64)
        self.assertEqual(calc.committed_trace(w, 3, 2).dtype, F)


if __name__ == "__main__":
    absltest.main()
