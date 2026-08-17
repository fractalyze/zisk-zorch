"""Smoke test for the inner-proof bench.

A real run needs a GPU host and sweeps 2^20-row traces, so this asserts nothing
about timings — it only pins the wiring: the module imports, the parser builds,
and `get_ops` still assembles a `BenchmarkOp` zkbench accepts. Without it the
bench has no build-visible consumer and an import error or a zkbench signature
change would only surface when someone ran it by hand.
"""

from __future__ import annotations

import argparse

import frx.numpy as fnp
import numpy as np
from absl.testing import absltest
from zk_dtypes import goldilocks as F

from zisk_zorch.bench_inner_proof import _STAGES, InnerProofBenchmark, _make_eval_fn
from zisk_zorch.commit.trace_commit import merkle_tree
from zisk_zorch.poseidon1.goldilocks import goldilocks_perm as _poseidon1_perm
from zisk_zorch.poseidon2.goldilocks import goldilocks_perm
from zisk_zorch.transcript.transcript import Transcript


def _parse(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    InnerProofBenchmark().add_custom_args(parser)
    return parser.parse_args(argv)


class BenchInnerProofTest(absltest.TestCase):
    def test_parser_defaults_match_the_production_fri_schedule(self) -> None:
        args = _parse([])
        self.assertEqual(args.stages, ",".join(_STAGES))
        self.assertEqual(args.arity, 4)
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

    def test_constraints_are_distinct_products(self) -> None:
        # The quotient proxy is only worth its name if the constraints survive
        # CSE: duplicate column tuples fold into one, so the stage would measure
        # a fraction of --n_constraints. Main's density is 900 over 38 columns.
        fn = _make_eval_fn(n_cols=38, n_constraints=900, degree=9)
        trace = fnp.array(
            np.random.default_rng(0).integers(1, 1 << 30, (8, 38)).astype(np.uint64),
            dtype=F,
        )
        out = np.asarray(fn(trace))  # (rows, 900), field dtype
        # Distinct products of random columns take distinct values, w.h.p.
        self.assertEqual(out.shape, (8, 900))
        self.assertLen({out[:, j].tobytes() for j in range(900)}, 900)

    def test_fri_leg_warms_the_perms_it_traces(self) -> None:
        # `prove` builds its transcript sponge and `merkle_tree(arity)` inside
        # the jit trace, and building a permutation there would hand its
        # external-M4 matrix analysis a tracer instead of concrete constants.
        # So `get_ops` has to populate the permutation caches host-side before
        # jitting; this asserts both seams find their perm already cached —
        # they share one entry today, so a missed seam would only show as a
        # miss, never as a hit count.
        args = _parse(["--stages=fri", "--n_bits=7"])
        goldilocks_perm.cache_clear()
        _poseidon1_perm.cache_clear()
        list(InnerProofBenchmark().get_ops(args))
        before = (goldilocks_perm.cache_info(), _poseidon1_perm.cache_info())
        Transcript()  # the transcript seam
        merkle_tree(args.arity)  # the merkle seam
        after = (goldilocks_perm.cache_info(), _poseidon1_perm.cache_info())
        self.assertEqual([c.misses for c in after], [c.misses for c in before])
        self.assertGreater(
            after[0].hits + after[1].hits, before[0].hits + before[1].hits
        )

    def test_chip_mode_folds_a_real_air(self) -> None:
        # #66: the chip leg replaces the proxy's independent products with a real
        # re-authored AIR's `eval_constraints`. Pin the wiring, not the timing:
        # the eval yields K constraints in the trailing axis, and the quotient op
        # records the AIR's real width — not the proxy's --n_cols/--n_constraints.
        import frx

        frx.config.update("jax_enable_x64", True)  # rw exports view u64→FIELD_DTYPE
        from zisk_zorch.bench_inner_proof import _chip_eval_fn

        eval_fn, n_cols, k = _chip_eval_fn("main")
        self.assertGreater(n_cols, 0)
        self.assertGreater(k, 0)
        out = eval_fn(fnp.zeros((4, n_cols), F))
        self.assertEqual(out.shape, (4, k))

        ops = list(
            InnerProofBenchmark().get_ops(
                _parse(["--stages=quotient", "--chip=main", "--n_bits=3"])
            )
        )
        self.assertLen(ops, 1)
        self.assertEqual(ops[0].metadata["chip"], "main")
        self.assertEqual(ops[0].metadata["n_cols"], n_cols)
        self.assertEqual(ops[0].metadata["n_constraints"], k)

    def test_unknown_stage_is_rejected(self) -> None:
        # get_ops is a generator, so the guard only fires once it is advanced.
        ops = InnerProofBenchmark().get_ops(_parse(["--stages=bogus"]))
        with self.assertRaises(ValueError):
            next(iter(ops))


if __name__ == "__main__":
    absltest.main()
