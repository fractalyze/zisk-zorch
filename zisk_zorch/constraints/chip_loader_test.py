"""ZisK chip ingestion from the bundled ``rw_constraints`` export.

Not a pil2 byte-match — that comes when stage-2 evaluates these constraints and
the result is pinned against pil2-proofman. This guards the ingestion seam: the
``rw-constraints`` wheel actually bundles ``constraints/zisk/v1`` (the gap
fractalyze/riscv-witness#1889 closed), and every ZisK chip loads with the
Goldilocks field bound and is evaluable on a trace of its declared width.
"""

from __future__ import annotations

import jax

# rw's exported chip code materializes field constants via
# `jnp.full(..., dtype=jnp.uint64).view(FIELD_DTYPE)`, which truncates (and then
# fails the view) unless JAX x64 is on — the same u64 trap zisk-zorch's
# golden path sidesteps by constructing in numpy first. Evaluating ingested
# constraints therefore requires x64; set it before any array op.
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from absl.testing import absltest  # noqa: E402
from zk_dtypes import goldilocks_mont  # noqa: E402

from zisk_zorch.constraints.chip_loader import load_zisk_chips  # noqa: E402

# The ZisK v1 chip set exported by riscv-witness (constraints/zisk/v1).
_EXPECTED_CHIPS = frozenset(
    {
        "add256",
        "arith",
        "arith_eq",
        "arith_eq_384",
        "binary",
        "binary_add",
        "binary_extension",
        "keccak",
        "main",
        "mem",
        "mem_align",
        "mem_align_byte",
        "mem_align_read_byte",
        "mem_align_write_byte",
        "sha256",
    }
)

# Native ZisK bus ids (zisk/pil/opids.pil), the `kind_int` an interaction
# carries — see riscv-witness docs/zisk/conventions/interaction-bus-mapping.md.
_KNOWN_BUS_IDS = frozenset({125, 330, 331, 5000})


class ChipLoaderTest(absltest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.chips = load_zisk_chips()

    def test_loads_the_full_zisk_chip_set(self) -> None:
        self.assertEqual(frozenset(self.chips), _EXPECTED_CHIPS)

    def test_each_chip_evaluates_constraints_on_its_declared_width(self) -> None:
        for name, chip in self.chips.items():
            with self.subTest(chip=name):
                self.assertGreater(chip.num_cols, 0)
                if chip.has_pv:
                    # main ingests public inputs; skip the no-PV smoke path.
                    continue
                trace = jnp.asarray(
                    np.zeros((2, chip.num_cols), dtype=np.uint64), dtype=goldilocks_mont
                )
                violations = chip.eval_constraints(trace)
                self.assertEqual(violations.shape[0], 2)

    def test_chip_name_filter_is_applied(self) -> None:
        only = load_zisk_chips(chip_names=["arith"])
        self.assertEqual(frozenset(only), {"arith"})

    def test_unknown_chip_name_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown ZisK chip names"):
            load_zisk_chips(chip_names=["arith", "nope"])

    def test_arith_carries_typed_lookup_bus_interactions(self) -> None:
        arith = self.chips["arith"]
        sends, receives = arith.get_sends(), arith.get_receives()
        self.assertNotEmpty(sends + receives)
        for info in sends + receives:
            with self.subTest(interaction=info.fn):
                self.assertIn(info.kind_int, _KNOWN_BUS_IDS)
                self.assertEqual(info.kind, "send" if info in sends else "receive")

    def test_interactions_evaluate_to_field_valued_tuples(self) -> None:
        # binary's bus lookups are pure field arithmetic — eval_interactions
        # must run under the Goldilocks interaction dtype (not SP1's uint32).
        binary = self.chips["binary"]
        trace = jnp.asarray(
            np.zeros((2, binary.num_cols), dtype=np.uint64), dtype=goldilocks_mont
        )
        tuples = binary.eval_interactions(trace)
        self.assertNotEmpty(tuples)
        for name, values in tuples.items():
            with self.subTest(interaction=name):
                self.assertEqual(values.shape[0], 2)
                self.assertEqual(values.dtype, goldilocks_mont)


if __name__ == "__main__":
    absltest.main()
