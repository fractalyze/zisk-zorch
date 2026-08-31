"""`TraceStager` may not change a single witness bit vs the plain upload.

The staged path exists for the transfer schedule, not the values — #144's
rule that staging may not change a proved byte starts here, one hop below
the block driver. Runs on any backend: the CPU PJRT client also exposes a
``pinned_host`` space, so CI exercises the real two-hop path, and the
fallback arm is covered by constructing the stager degraded.
"""

from __future__ import annotations

import frx.numpy as fnp
import numpy as np
from absl.testing import absltest
from zk_dtypes import goldilocks as F, pfinfo

from zisk_zorch.harness.staging import TraceStager


def _canonical_words(shape, seed=7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    modulus = pfinfo(F).modulus
    words = rng.integers(0, modulus, size=shape, dtype=np.uint64)
    # Pin the boundary rather than hoping the draw hits it: the largest
    # canonical word is exactly where a stray reduction would show.
    words.flat[0] = modulus - 1
    words.flat[-1] = 0
    return words


class TraceStagerTest(absltest.TestCase):
    def test_staged_bits_match_plain_upload(self):
        words = _canonical_words((64, 6))
        staged = TraceStager().stage(words)
        ref = fnp.array(words.view(F))
        self.assertEqual(staged.dtype, ref.dtype)
        np.testing.assert_array_equal(
            np.asarray(staged).view(np.uint64), np.asarray(ref).view(np.uint64)
        )

    def test_staged_lands_in_device_memory(self):
        stager = TraceStager()
        if stager._route is None:
            self.skipTest("backend has no pinned_host memory space")
        staged = stager.stage(_canonical_words((8, 3)))
        self.assertEqual(staged.sharding.memory_kind, "device")

    def test_fallback_path_matches_too(self):
        stager = TraceStager()
        stager._route = None
        words = _canonical_words((16, 4), seed=11)
        staged = stager.stage(words)
        np.testing.assert_array_equal(
            np.asarray(staged).view(np.uint64), words
        )

    def test_source_buffer_may_be_dropped_in_flight(self):
        # The block driver releases a source while its upload may still be
        # in flight; the staged array must not read freed memory. The
        # runtime keeps the host buffer referenced until copied — pin that
        # by dropping every caller-side reference before the first read.
        stager = TraceStager()
        words = _canonical_words((32, 5), seed=13)
        expect = words.copy()
        staged = stager.stage(words)
        del words
        np.testing.assert_array_equal(np.asarray(staged).view(np.uint64), expect)


if __name__ == "__main__":
    absltest.main()
