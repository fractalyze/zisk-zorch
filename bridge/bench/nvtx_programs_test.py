"""Pins how `nvtx_programs.py` reads an `nsys` NVTX kernel summary: which
ranges count, what a program's run count means, and the per-prove
normalisation. The fixture is a real capture — see testdata/README.md."""

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
