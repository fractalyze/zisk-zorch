"""Byte-match of FRI query-position sampling against the pil2 `fields` crate:
replay the seeded transcript to the fold-loop tail, derive the positions, and
compare to the golden. The discarded `pre_challenge` squeeze is cross-checked so
a wrong pre-derivation transcript state can't pass on the positions alone."""

from __future__ import annotations

import pathlib

import jax.numpy as jnp
import numpy as np
from absl.testing import absltest

from zisk_zorch.fri.queries import sample_query_positions
from zisk_zorch.golden import load, u64, u64x3
from zisk_zorch.transcript.transcript import Transcript

_GOLDEN = pathlib.Path(__file__).parent / "testdata" / "golden" / "query_sample.json"


class QuerySampleTest(absltest.TestCase):
    def test_matches_pil2_reference(self) -> None:
        for case in load(_GOLDEN)["cases"]:
            width = case["width"]
            t = Transcript(width)
            t.put(u64(case["seed_absorb"]))

            pre = t.get_field()  # fold-loop tail challenge, discarded by the prover
            self.assertTrue(
                bool(jnp.array_equal(pre, u64(case["pre_challenge"]))),
                msg=f"width {width} pre_challenge",
            )

            positions = sample_query_positions(
                t,
                u64x3(case["final_pol"]),
                int(case["nonce"]),
                n_queries=case["n_queries"],
                n_bits_ext=case["n_bits_ext"],
            )
            expected = np.array([int(v) for v in case["positions"]], dtype=np.uint64)
            self.assertTrue(
                bool(np.array_equal(positions, expected)),
                msg=f"width {width} positions",
            )


if __name__ == "__main__":
    absltest.main()
