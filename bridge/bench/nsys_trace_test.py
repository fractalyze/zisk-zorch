"""Pins the pieces every nsys reader in this directory shares: the column-unit
rule, the span algebra the reports are built out of, and the rule that tells
the two provers' kernels apart. The cases are small enough to check by eye,
which is the point of them living here rather than inside one report's tests —
a change to the attribution rule has to fail one suite, not neither."""

from absl.testing import absltest, parameterized

from bridge.bench import nsys_trace


class ColumnTest(parameterized.TestCase):
    """nsys picks a column's unit from the capture, so a reader has to scale
    by the header rather than assume one."""

    @parameterized.parameters(
        ("Start (ns)", 1),
        ("Start (us)", 1_000),
        ("Start (µs)", 1_000),
        ("Start (ms)", 1_000_000),
        ("Start (s)", 1_000_000_000),
    )
    def test_a_time_column_scales_to_nanoseconds(self, header, scale):
        self.assertEqual(nsys_trace.column([header], "Start"), (header, scale))

    @parameterized.parameters(("Bytes (B)", 1), ("Bytes (MB)", 1_000_000))
    def test_a_size_column_scales_to_bytes(self, header, scale):
        self.assertEqual(
            nsys_trace.column([header], "Bytes", nsys_trace.UNITS_BYTES),
            (header, scale),
        )

    def test_a_unit_outside_the_table_raises(self):
        # Silently scaling by the wrong power of ten, or by zero, would move
        # every published figure without failing anything.
        with self.assertRaises(ValueError):
            nsys_trace.column(["Start (bytes)"], "Start")
        with self.assertRaises(ValueError):
            nsys_trace.column(["Bytes (MB)"], "Bytes")

    def test_a_missing_column_raises(self):
        with self.assertRaises(ValueError):
            nsys_trace.column(["Duration (ns)"], "Start")


class SpansTest(parameterized.TestCase):
    """The interval algebra every report is built on."""

    @parameterized.named_parameters(
        ("disjoint", [(0, 1), (2, 3)], [(0, 1), (2, 3)]),
        ("overlapping", [(0, 2), (1, 3)], [(0, 3)]),
        ("touching", [(0, 1), (1, 2)], [(0, 2)]),
        ("unsorted", [(2, 3), (0, 1)], [(0, 1), (2, 3)]),
        ("unsorted_and_overlapping", [(5, 6), (0, 2), (1, 3)], [(0, 3), (5, 6)]),
        ("nested", [(0, 9), (3, 4)], [(0, 9)]),
    )
    def test_merge(self, spans, want):
        self.assertEqual(nsys_trace.merge(spans), want)

    @parameterized.named_parameters(
        ("partial", [(0, 10)], [(5, 15)], [(5, 10)]),
        ("none", [(0, 5)], [(5, 10)], []),
        ("contained", [(2, 4)], [(0, 9)], [(2, 4)]),
        ("several", [(0, 10)], [(1, 2), (3, 4)], [(1, 2), (3, 4)]),
    )
    def test_intersect(self, a, b, want):
        self.assertEqual(nsys_trace.intersect(a, b), want)
        self.assertEqual(nsys_trace.intersect(b, a), want)

    @parameterized.named_parameters(
        ("no_overlap", [(0, 2)], [(3, 5)], 0),
        ("partial", [(0, 4)], [(2, 9)], 2),
        ("contained", [(2, 4)], [(0, 9)], 2),
        # A transfer spanning two kernels is hidden only where they run.
        ("many_to_one", [(0, 10)], [(1, 2), (4, 6)], 3),
    )
    def test_overlap(self, a, b, want):
        self.assertEqual(nsys_trace.overlap(a, b), want)
        self.assertEqual(nsys_trace.overlap(b, a), want)

    @parameterized.named_parameters(
        ("holes", [(0, 10)], [(2, 4), (6, 8)], [(0, 2), (4, 6), (8, 10)]),
        ("all_covered", [(0, 10)], [(0, 20)], []),
        ("nothing_removed", [(0, 10)], [(20, 30)], [(0, 10)]),
        (
            "cursor_spans_several",
            [(0, 5), (10, 15)],
            [(1, 2), (11, 12)],
            [(0, 1), (2, 5), (10, 11), (12, 15)],
        ),
        ("edge_touching", [(0, 10)], [(0, 3), (7, 10)], [(3, 7)]),
    )
    def test_subtract(self, a, b, want):
        self.assertEqual(nsys_trace.subtract(a, b), want)

    def test_covered_sums_a_cover(self):
        self.assertEqual(nsys_trace.covered([(0, 10), (20, 25)]), 15)

    def test_subtract_and_intersect_partition_the_time(self):
        # host_idle's reconciliation rests on this, and h2d_overlap's
        # critical path is the same identity read the other way: what one
        # cover takes from another plus what it leaves is the whole of it.
        a, b = [(0, 100)], [(10, 20), (30, 90)]
        self.assertEqual(
            nsys_trace.overlap(a, b) + nsys_trace.covered(nsys_trace.subtract(a, b)),
            nsys_trace.covered(a),
        )


class OwnerTest(parameterized.TestCase):
    """Which prover a kernel belongs to. Both reports split their numbers on
    this one rule, which is why it has one home."""

    @parameterized.named_parameters(
        ("xla_fusion", "loop_add_gather_fusion", nsys_trace.BRIDGE),
        ("xla_hash", "sponge_hash_1", nsys_trace.BRIDGE),
        ("pil2_plain", "genMerkleProof(gl64_t *, unsigned long)", nsys_trace.PIL2),
        ("pil2_varargs", "evalTwiddleFirstKernel(gl64_t *, ...)", nsys_trace.PIL2),
        (
            "pil2_templated",
            "void merkleNodeKernel_pos1<(unsigned int)12>(unsigned long)",
            nsys_trace.PIL2,
        ),
    )
    def test_owner(self, kernel, side):
        self.assertEqual(nsys_trace.owner(kernel), side)

    @parameterized.named_parameters(
        ("h2d", "[CUDA memcpy Host-to-Device]"),
        ("d2h", "[CUDA memcpy Device-to-Host]"),
        ("d2d", "[CUDA memcpy Device-to-Device]"),
        ("memset", "[CUDA memset]"),
    )
    def test_owner_refuses_a_memory_operation(self, name):
        # A copy carries no signature, so the `(` rule would silently call
        # every copy in the capture the bridge's — pil2's included, since the
        # two provers share the process under `cargo-zisk`. Wrong answers are
        # worse than errors here: it is how a report can attribute another
        # prover's transfers to this one and nothing looks amiss.
        with self.assertRaises(ValueError):
            nsys_trace.owner(name)


if __name__ == "__main__":
    absltest.main()
