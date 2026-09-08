"""Pins how `nvtx_programs.py` reads an `nsys` NVTX kernel summary: which
ranges count, what a program's run count means, and the per-prove
normalisation the report is built on. The fixture is a real capture — see
testdata/README.md."""

import contextlib
import io
import pathlib

from absl.testing import absltest, parameterized

from bridge.bench import nvtx_programs

CAPTURE = pathlib.Path("bridge/bench/testdata/nvtx_kern_sum.csv")


class NvtxProgramsTest(parameterized.TestCase):

    def setUp(self):
        super().setUp()
        self.programs = nvtx_programs.read(CAPTURE)

    def test_xlas_own_ranges_are_left_out(self):
        # The fixture carries two TSL ranges around the same kernels; counting
        # them would double every program's device time.
        self.assertNotIn(None, self.programs)
        self.assertCountEqual(
            self.programs,
            [
                "commit1",
                "commit2",
                "quotient_commit",
                "const_setup",
                "quotient_524288",
                "deep",
                "evals",
            ],
        )

    @parameterized.named_parameters(
        ("per_prove", "commit1", 2, 102, 31222145),
        ("per_family", "const_setup", 1, 49, 13637510),
        ("per_quotient_chunk", "quotient_524288", 16, 208, 3773388),
    )
    def test_run_counts_and_totals(self, program, runs, kernels, total_ns):
        p = self.programs[program]
        self.assertEqual((p.runs, p.kernels, p.total_ns), (runs, kernels, total_ns))

    def test_proves_is_the_commonest_run_count(self):
        # Most programs run once per prove; `const_setup` (once per family)
        # and the quotient (once per chunk) must not sway the count.
        self.assertEqual(nvtx_programs.proves(self.programs), 2)

    @parameterized.named_parameters(
        # Once per prove, so halved over the capture's two.
        ("per_prove", "commit1", 2),
        # Once per family: fewer runs than proves, so its own total stands.
        ("per_family", "const_setup", 1),
        # Eight chunks a prove, 16 runs: still halved, so the row is what all
        # eight chunks of one prove cost — not what one chunk cost.
        ("per_quotient_chunk", "quotient_524288", 2),
    )
    def test_the_share_a_program_is_divided_by(self, program, share):
        shares = {n: s for n, _, s in nvtx_programs.per_prove(self.programs, 2)}
        self.assertEqual(shares[program], share)

    def test_per_prove_is_ordered_by_what_one_prove_pays(self):
        rows = nvtx_programs.per_prove(self.programs, 2)
        costs = [p.total_ns / share for _, p, share in rows]
        self.assertEqual(costs, sorted(costs, reverse=True))

    def test_report_prints_a_prove_not_the_capture(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            nvtx_programs.report(CAPTURE, top=1, override=None)
        header, *rows = out.getvalue().splitlines()
        self.assertIn("2 proves, 7 programs, 0.065 s on the device", header)
        self.assertIn("per prove over 301 kernels", header)
        printed = {line.split()[0]: line for line in rows}
        # 3773388 ns and 208 kernels over two proves, with all 16 chunk runs
        # in the one row: a prove pays for its eight chunks, not for one.
        self.assertIn("0.002 s   104 kernels", printed["quotient_524288"])
        # A program that ran fewer times than the capture has proves is
        # reported whole.
        self.assertIn("0.014 s    49 kernels", printed["const_setup"])

    def test_report_says_when_the_feature_was_off(self):
        # Without `--features nvtx` a capture still has XLA's ranges, so an
        # empty table would otherwise look like a profiling mistake.
        csv = pathlib.Path(
            self.create_tempfile(
                content="NVTX Range,NVTX Inst,Kern Inst,Total Time (ns),Kernel Name\n"
                "TSL:XlaModule:#hlo_module=jit_fn#,1,1,1000,sponge_hash_1\n"
            ).full_path
        )
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            nvtx_programs.report(csv, top=1, override=None)
        self.assertEndsWith(
            out.getvalue().strip(),
            "no bridge ranges — was zz_prove built with --features nvtx?",
        )

    @parameterized.parameters(
        (":commit1", "commit1"),
        (":quotient_524288", "quotient_524288"),
        ("TSL:XlaModule:#hlo_module=jit_fn,program_id=1#", None),
        ("TSL:Thunk:#hlo_op=command_buffer_1#", None),
        # Older nsys builds write a domainless range without the separator.
        ("commit1", "commit1"),
    )
    def test_only_the_domainless_ranges_are_the_bridges(self, nvtx_range, expected):
        self.assertEqual(nvtx_programs.program(nvtx_range), expected)

    @parameterized.parameters(
        ("sponge_hash_1", "sponge_hash"),
        ("sponge_hash_14", "sponge_hash"),
        ("loop_add_multiply_fusion", "loop_add_multiply_fusion"),
        ("wrapped_transpose", "wrapped_transpose"),
    )
    def test_family_folds_the_fusion_index(self, kernel, expected):
        self.assertEqual(nvtx_programs.family(kernel), expected)


if __name__ == "__main__":
    absltest.main()
