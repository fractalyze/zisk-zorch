"""Byte-match the re-authored Binary quotient against the cExp reference `q`.

`reauthor.reauthor_binary_quotient` assembles `q` from the Binary AIR's row-local
constraints + the ingested `std_sum` interactions (rw's typed `Interaction`s),
folded in proving-key order. Equality with the `cexp_eval` golden `q` (generated
by interpreting pil2's composite-constraint SSA) is the end-to-end byte-match that
verifies rw's authored interactions for Binary.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import pathlib  # noqa: E402

import jax.numpy as jnp  # noqa: E402
from absl.testing import absltest  # noqa: E402

from zisk_zorch.constraints.chip_loader import load_zisk_chips  # noqa: E402
from zisk_zorch.golden import load, u64x3  # noqa: E402
from zisk_zorch.quotient.reauthor import reauthor_binary_quotient  # noqa: E402

_GOLDEN = pathlib.Path(__file__).parent / "testdata" / "golden" / "cexp_eval.json"


class ReauthorBinaryTest(absltest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.chip = load_zisk_chips("v1", ["binary"])["binary"]
        self.cases = [c for c in load(_GOLDEN)["cases"] if c["air"] == "Binary"]

    def test_reauthored_q_matches_cexp_reference(self) -> None:
        self.assertTrue(self.cases, "no Binary cases in the cexp_eval golden")
        for case in self.cases:
            with self.subTest(n_bits=case["n_bits"], blowup_bits=case["blowup_bits"]):
                got = reauthor_binary_quotient(self.chip, case)
                self.assertTrue(bool(jnp.array_equal(got, u64x3(case["q"]))))


if __name__ == "__main__":
    absltest.main()
