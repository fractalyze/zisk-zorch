"""Pins how `h2d_overlap.py` reads an `nsys` CUDA GPU trace: the interval
arithmetic behind "on the critical path", which prover a kernel and an upload
stream belong to, and the totals that come out of a real capture. The fixture
is one prove's worth of a real run — see testdata/README.md."""

import contextlib
import io
import pathlib

from absl.testing import absltest, parameterized

from bridge.bench import h2d_overlap

CAPTURE = pathlib.Path("bridge/bench/testdata/cuda_gpu_trace.csv")


class IntervalsTest(parameterized.TestCase):
    """The arithmetic, on spans small enough to check by eye."""

    @parameterized.named_parameters(
        ("disjoint", [(0, 1), (3, 4)], [(0, 1), (3, 4)]),
        ("touching_join", [(0, 2), (2, 4)], [(0, 4)]),
        ("nested_absorbed", [(0, 9), (2, 4)], [(0, 9)]),
        ("unsorted_input", [(5, 6), (0, 2), (1, 3)], [(0, 3), (5, 6)]),
    )
    def test_merge(self, spans, want):
        self.assertEqual(h2d_overlap.merge(spans), want)

    @parameterized.named_parameters(
        ("no_overlap", [(0, 2)], [(3, 5)], 0),
        ("partial", [(0, 4)], [(2, 9)], 2),
        ("contained", [(2, 4)], [(0, 9)], 2),
        # A transfer spanning two kernels is hidden only where they run.
        ("many_to_one", [(0, 10)], [(1, 2), (4, 6)], 3),
    )
    def test_overlap(self, a, b, want):
        self.assertEqual(h2d_overlap.overlap(a, b), want)
        self.assertEqual(h2d_overlap.overlap(b, a), want)

    def test_exposed_is_the_cover_minus_the_overlap(self):
        uploads, kernels = [(0, 10), (20, 25)], [(5, 22)]
        self.assertEqual(h2d_overlap.covered(uploads), 15)
        self.assertEqual(h2d_overlap.overlap(uploads, kernels), 7)

    @parameterized.named_parameters(
        ("nanoseconds", "Start (ns)", 1),
        ("microseconds", "Start (µs)", 1_000),
        ("seconds", "Start (s)", 1_000_000_000),
    )
    def test_column_scales_to_nanoseconds(self, header, scale):
        self.assertEqual(h2d_overlap.column([header], "Start"), (header, scale))

    def test_column_rejects_a_header_it_cannot_scale(self):
        with self.assertRaises(ValueError):
            h2d_overlap.column(["Bytes (MB)"], "Bytes")


class OwnerTest(parameterized.TestCase):

    @parameterized.named_parameters(
        ("xla_fusion", "loop_add_gather_fusion", h2d_overlap.BRIDGE),
        ("xla_hash", "sponge_hash_1", h2d_overlap.BRIDGE),
        ("pil2_plain", "genMerkleProof(gl64_t *, unsigned long)", h2d_overlap.PIL2),
        (
            "pil2_templated",
            "void merkleNodeKernel_pos1<(unsigned int)12>(unsigned long)",
            h2d_overlap.PIL2,
        ),
    )
    def test_owner(self, kernel, side):
        self.assertEqual(h2d_overlap.owner(kernel), side)


class CaptureTest(absltest.TestCase):
    """The fixture's own numbers, which are the capture's."""

    def setUp(self):
        super().setUp()
        self.capture = h2d_overlap.read(CAPTURE)
        self.owners = self.capture.upload_owners()

    def test_upload_streams_go_to_the_right_prover(self):
        # 66 carries pil2's kernels and 13 the bridge's, so those two are
        # settled by the stream itself; 14 is a dedicated transfer stream
        # and is settled by the kernels that follow its copies.
        self.assertEqual(self.owners, {"14": "bridge", "13": "bridge", "66": "pil2"})

    def test_a_dedicated_stream_survives_copies_that_vote_the_other_way(self):
        # Stream 14 is the bridge's, but pil2's recursion runs in the gaps,
        # so some of its copies are followed by a pil2 kernel. The majority
        # is what decides, and the fixture carries both votes.
        starts = sorted(
            (span[0], side)
            for side, spans in self.capture.kernels.items()
            for span in spans
        )
        followers = set()
        for up in self.capture.uploads:
            if up.stream != "14":
                continue
            after = [side for start, side in starts if start >= up.span[1]]
            if after:
                followers.add(after[0])
        self.assertEqual(followers, {h2d_overlap.BRIDGE, h2d_overlap.PIL2})
        self.assertEqual(self.owners["14"], h2d_overlap.BRIDGE)

    def test_kernels_split_between_the_two_provers(self):
        self.assertEqual(len(self.capture.kernels[h2d_overlap.BRIDGE]), 348)
        self.assertEqual(len(self.capture.kernels[h2d_overlap.PIL2]), 46)

    def test_the_bridges_uploads(self):
        uploads = [u for u in self.capture.uploads if self.owners[u.stream] == "bridge"]
        self.assertLen(uploads, 79)
        self.assertEqual(sum(u.n_bytes for u in uploads), 1_442_899_000)
        by_kind = {u.kind for u in uploads}
        self.assertEqual(by_kind, {"Pageable", "Pinned"})

    def test_no_upload_of_either_prover_overlaps_its_own_kernels(self):
        # The finding the tool exists to report: the read-ahead never gets a
        # transfer onto the device beside a kernel of the prove it belongs
        # to, so every byte uploaded is time the client stands still.
        for side in (h2d_overlap.BRIDGE, h2d_overlap.PIL2):
            kernels = h2d_overlap.merge(self.capture.kernels[side])
            uploads = h2d_overlap.merge(
                [u.span for u in self.capture.uploads if self.owners[u.stream] == side]
            )
            self.assertEqual(h2d_overlap.overlap(uploads, kernels), 0, side)

    def test_bridge_upload_time_is_all_on_the_critical_path(self):
        kernels = h2d_overlap.merge(self.capture.kernels[h2d_overlap.BRIDGE])
        uploads = h2d_overlap.merge(
            [u.span for u in self.capture.uploads if self.owners[u.stream] == "bridge"]
        )
        self.assertEqual(h2d_overlap.covered(uploads), 74_038_028)
        self.assertEqual(h2d_overlap.covered(kernels), 58_496_094)


class ReportTest(absltest.TestCase):
    """The printed lines, which are where the numbers in docs/bridge.md
    "The uploads, measured" come from."""

    def report(self, path: pathlib.Path, top: int = 4) -> list[str]:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            h2d_overlap.report(path, top=top)
        return out.getvalue().splitlines()

    def csv(self, body: str) -> pathlib.Path:
        header = "Start (ns),Duration (ns),Bytes (MB),SrcMemKd,Strm,Name\n"
        return pathlib.Path(self.create_tempfile(content=header + body).full_path)

    def test_each_side_gets_its_own_block(self):
        lines = self.report(CAPTURE)
        self.assertStartsWith(lines[0], "## ")
        self.assertIn("0.22 s of timeline", lines[0])
        sides = [
            line.split()[0]
            for line in lines
            if line.startswith("   ") and not line.startswith("    ")
        ]
        self.assertEqual(sides, ["bridge", "pil2"])

    def test_the_bridges_numbers(self):
        lines = "\n".join(self.report(CAPTURE))
        self.assertIn("bridge leg  0.18 s, kernels busy  0.06 s over 323 spans", lines)
        self.assertIn("uploads    79 copies,   1.44 GB in  0.07 s at  19.5 GB/s", lines)
        # The headline: exposed time, its share of the leg, and the zero.
        self.assertIn(
            "0.07 s on the critical path, 40 % of the leg;  0.00 s overlapped", lines
        )
        self.assertIn(
            "from Pageable     48 copies,   1.31 GB in  0.07 s at  18.4 GB/s", lines
        )
        self.assertIn(
            "from Pinned       31 copies,   0.13 GB in  0.00 s at  44.1 GB/s", lines
        )

    def test_the_clip_before_the_first_kernel_is_printed_only_where_there_is_one(self):
        lines = self.report(CAPTURE)
        early = [line for line in lines if "before the" in line]
        # pil2's window opens on a transfer; the bridge's opens on a kernel.
        self.assertLen(early, 1)
        self.assertIn("0.02 s of that ran before the leg's first kernel", early[0])

    def test_top_bounds_the_largest_line(self):
        largest = [line for line in self.report(CAPTURE, top=2) if "largest:" in line]
        self.assertLen(largest, 2)
        for line in largest:
            self.assertLen(line.split("largest:")[1].split(","), 2)

    def test_a_capture_with_neither_says_so(self):
        # An empty table would otherwise read as a run with no uploads
        # rather than as a capture taken without -t cuda.
        lines = self.report(self.csv("0,10,,,7,[CUDA memset]\n"))
        self.assertLen(lines, 1)
        self.assertEndsWith(lines[0], "no kernels and no transfers — was -t cuda on?")

    def test_uploads_no_kernel_follows_are_reported_not_dropped(self):
        # A transfer stream resolves off the kernel that starts after its
        # copies; with none, the bytes must still be accounted for.
        lines = self.report(
            self.csv(
                "0,100,,,7,sponge_hash_1\n"
                "500,100,64.0,Pageable,9,[CUDA memcpy Host-to-Device]\n"
            )
        )
        self.assertIn(
            "unattributed 1 copies, 0.06 GB on stream(s) 9: no kernel runs after them",
            "\n".join(lines),
        )


if __name__ == "__main__":
    absltest.main()
