"""Smoke test for the inner-proof bench.

A real run needs a GPU host and sweeps 2^20-row traces, so this asserts nothing
about timings — it only pins the wiring: the module imports, the parser builds,
and `get_ops` still assembles a `BenchmarkOp` zkbench accepts. Without it the
bench has no build-visible consumer and an import error or a zkbench signature
change would only surface when someone ran it by hand.
"""

from __future__ import annotations

import argparse

from absl.testing import absltest

from zisk_zorch.bench_inner_proof import _STAGES, InnerProofBenchmark


def _parse(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    InnerProofBenchmark().add_custom_args(parser)
    return parser.parse_args(argv)


class BenchInnerProofTest(absltest.TestCase):
    def test_parser_defaults_match_the_production_fri_schedule(self) -> None:
        args = _parse([])
        self.assertEqual(args.stages, ",".join(_STAGES))
        self.assertEqual(args.arity, 2)
        # The ZisK v1.0.0-alpha starkStructs fold by 3 down to nBits 5.
        self.assertEqual(args.fold_bits, 3)
        self.assertEqual(args.final_bits, 5)

    def test_config_names_this_implementation(self) -> None:
        config = InnerProofBenchmark().get_config()
        self.assertEqual(config.implementation, "zisk-zorch")

    def test_assembles_a_benchmark_op(self) -> None:
        # `divide` is the cheapest stage to construct: the op's fn, lower thunk
        # and output hash are all lazy, so this builds the BenchmarkOp without
        # running the prover. Tiny sizes keep it on the CPU leg.
        args = _parse(["--n_bits=3", "--n_cols=2", "--stages=divide"])
        ops = list(InnerProofBenchmark().get_ops(args))

        self.assertLen(ops, 1)
        self.assertEqual(ops[0].name, "divide_2p3")
        self.assertEqual(ops[0].metadata["stage"], "divide")
        self.assertEqual(ops[0].throughput_unit, "rows/s")
        self.assertTrue(ops[0].input_hash)

    def test_unknown_stage_is_rejected(self) -> None:
        # get_ops is a generator, so the guard only fires once it is advanced.
        ops = InnerProofBenchmark().get_ops(_parse(["--stages=bogus"]))
        with self.assertRaises(ValueError):
            next(iter(ops))


if __name__ == "__main__":
    absltest.main()
