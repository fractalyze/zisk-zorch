"""ZisK chip ingestion from the bundled ``rw_constraints`` export.

Not a pil2 byte-match — that comes when stage-2 evaluates these constraints and
the result is pinned against pil2-proofman. This guards the ingestion seam: the
``rw-constraints`` wheel actually bundles ``constraints/zisk/v1`` (the gap
fractalyze/riscv-witness#1889 closed), and every ZisK chip loads with the
Goldilocks field bound and is evaluable on a trace of its declared width.
"""

from __future__ import annotations

import frx

# rw's exported chip code materializes field constants via
# `fnp.full(..., dtype=fnp.uint64).view(FIELD_DTYPE)`, which truncates (and then
# fails the view) unless JAX x64 is on — the same u64 trap zisk-zorch's
# golden path sidesteps by constructing in numpy first. Evaluating ingested
# constraints therefore requires x64; set it before any array op.
frx.config.update("jax_enable_x64", True)

import frx.numpy as fnp  # noqa: E402
import numpy as np  # noqa: E402
from absl.testing import absltest  # noqa: E402
from zk_dtypes import goldilocks  # noqa: E402

from rw_constraints import PreprocessedColumn  # noqa: E402

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
                # num_cols counts the preprocessed prefix; num_main_cols does
                # not, and the declared columns must account for the gap.
                self.assertEqual(
                    chip.num_cols - chip.num_main_cols,
                    sum(c.width for c in chip.preprocessed_cols),
                )
                if chip.has_pv:
                    # main ingests public inputs; skip the no-PV smoke path.
                    continue
                # Constraint fns evaluate over the combined [prep | main] row,
                # so the smoke trace is num_cols wide (an all-zero prefix is a
                # valid CLK_0 = 0 interior row).
                trace = fnp.asarray(
                    np.zeros((2, chip.num_cols), dtype=np.uint64),
                    dtype=goldilocks,
                )
                violations = chip.eval_constraints(trace)
                self.assertEqual(violations.shape[0], 2)

    def test_preprocessed_columns_are_exactly_the_clk0_set(self) -> None:
        """A tripwire, not an invariant. The rw#2189 Phase C wheel gives the
        four gated chips their `CLK_0`; `full_trace` /
        `load_chip_preprocessed` supply the prefix their constraint fns read.
        The next rw wheel (rw#2342) adds the derived shifted-clock gate
        columns (SEL_LATCH_GATE, CLK_0_BACK_23, IN_USE_LATCHED,
        IN_USE_ACTIVE) and should fail here — extend the expectation AND
        confirm `DERIVED_PREPROCESSED` covers each new name, or the loader
        would try to read them from the proving key and raise.
        """
        clk0 = [PreprocessedColumn(name="CLK_0", index=0, width=1)]
        expected = {
            "arith_eq": clk0,
            "arith_eq_384": clk0,
            "keccak": clk0,
            "sha256": clk0,
        }
        for name, chip in self.chips.items():
            with self.subTest(chip=name):
                self.assertEqual(chip.preprocessed_cols, expected.get(name, []))

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
        trace = fnp.asarray(
            np.zeros((2, binary.num_main_cols), dtype=np.uint64), dtype=goldilocks
        )
        tuples = binary.eval_interactions(trace)
        self.assertNotEmpty(tuples)
        for name, values in tuples.items():
            with self.subTest(interaction=name):
                self.assertEqual(values.shape[0], 2)
                self.assertEqual(values.dtype, goldilocks)


if __name__ == "__main__":
    absltest.main()
