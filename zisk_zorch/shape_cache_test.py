"""`shape_cache` interns like `functools.cache` and releases on demand.

The release path is the whole point of the wrapper — a block run's residency
bound depends on it reaching caches no key-scoped release can (see
`harness.block_composite`), so it is pinned here rather than left to the one
integration run that needs a card.
"""

from __future__ import annotations

from absl.testing import absltest

from zisk_zorch.evals.lev import _lev_constants
from zisk_zorch.quotient.zerofier import _coset_points
from zisk_zorch.shape_cache import release_shape_caches, shape_cache


class ShapeCacheTest(absltest.TestCase):
    def test_a_shape_is_built_once_and_again_after_release(self):
        calls = []

        @shape_cache
        def build(n_bits: int) -> int:
            calls.append(n_bits)
            return n_bits

        build(3)
        build(3)
        self.assertEqual(calls, [3])
        release_shape_caches()
        build(3)
        self.assertEqual(calls, [3, 3])

    def test_release_reaches_the_prove_stages_own_caches(self):
        # The two device-array caches a block run must be able to reclaim:
        # neither is keyed by a `Pil2Key`, so `release_device_sections` cannot
        # see them.
        release_shape_caches()
        _coset_points(3, 1)
        _lev_constants((0, 1), 3)
        self.assertEqual(_coset_points.cache_info().currsize, 1)
        self.assertEqual(_lev_constants.cache_info().currsize, 1)
        release_shape_caches()
        self.assertEqual(_coset_points.cache_info().currsize, 0)
        self.assertEqual(_lev_constants.cache_info().currsize, 0)


if __name__ == "__main__":
    absltest.main()
