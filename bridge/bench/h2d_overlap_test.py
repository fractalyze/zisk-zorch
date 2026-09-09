"""Pins how `h2d_overlap.py` reads an `nsys` CUDA GPU trace: the interval
arithmetic behind "on the critical path", which prover a kernel and an upload
stream belong to, and the totals that come out of a real capture. The fixture
is one prove's worth of a real run — see testdata/README.md."""

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


if __name__ == "__main__":
    absltest.main()
