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


class IdleGapsTest(parameterized.TestCase):
    """How long the device had been idle when a copy started, on spans small
    enough to check by eye. The rest of the span algebra is pinned in
    nsys_trace_test."""

    @parameterized.named_parameters(
        # A copy inside a kernel waited for nothing; one after it waited
        # from that kernel's end, not from the previous gap.
        ("during_a_kernel", 5, 0),
        ("right_after_one", 10, 0),
        ("into_the_gap", 17, 7),
        ("after_the_second", 30, 5),
    )
    def test_idle_gaps_measures_from_the_last_kernel_end(self, began, want):
        uploads = [h2d_overlap.Upload((began, began + 1), 0, "Pageable", "9")]
        kernels = [(0, 10), (20, 25)]
        self.assertEqual(h2d_overlap.idle_gaps(uploads, kernels), [want])

    def test_idle_gaps_skips_copies_before_the_first_kernel(self):
        # There is no "since the device went idle" before the device has
        # ever been busy, and counting one as a zero would flatter the
        # distribution.
        uploads = [h2d_overlap.Upload((s, s + 1), 0, "Pageable", "9") for s in (1, 12)]
        self.assertEqual(h2d_overlap.idle_gaps(uploads, [(5, 10)]), [2])


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

    def test_the_bridge_uploads_before_its_own_first_kernel(self):
        # 256 ns of copy 1.01 ms ahead of the bridge's first kernel. Small,
        # but not nothing: a report that hides a clip this size below a
        # floor tells the reader the window opens on a kernel when it does
        # not.
        kernels = h2d_overlap.merge(self.capture.kernels[h2d_overlap.BRIDGE])
        uploads = h2d_overlap.merge(
            [u.span for u in self.capture.uploads if self.owners[u.stream] == "bridge"]
        )
        first = kernels[0][0]
        self.assertEqual(
            h2d_overlap.covered([(s, min(e, first)) for s, e in uploads if s < first]),
            256,
        )

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
        # The headline: exposed time and the zero it is all exposed against.
        self.assertIn(
            "0.07 s on the critical path;  0.00 s overlapped by its own kernels",
            lines,
        )
        self.assertIn("0.07 s of that fell inside the leg (40 % of it)", lines)
        self.assertIn(
            "from Pageable     48 copies,   1.31 GB in  0.07 s at  18.4 GB/s", lines
        )
        self.assertIn(
            "from Pinned       31 copies,   0.13 GB in  0.00 s at  44.1 GB/s", lines
        )

    def test_the_share_is_of_the_leg_the_exposure_is_measured_in(self):
        # The numerator and the leg have to be the same window. Every one of
        # pil2's copies runs before its first kernel, so none of its 0.02 s
        # critical path is in its leg; dividing the whole of it by that leg
        # would print 56 % of a leg the copies never touch.
        shares = [
            line for line in self.report(CAPTURE) if "fell inside the leg" in line
        ]
        self.assertLen(shares, 2)
        self.assertIn("0.00 s of that fell inside the leg (0 % of it)", shares[1])

    def test_the_time_outside_the_leg_is_printed_for_both_sides(self):
        # No floor on either half, so the two account for the whole of the
        # critical path that no leg contains.
        outside = [line for line in self.report(CAPTURE) if "before the leg" in line]
        self.assertLen(outside, 2)
        self.assertEndsWith(
            outside[0],
            "0.00 s ran before the leg's first kernel and  0.00 s after" " its last",
        )
        self.assertIn("0.02 s ran before the leg's first kernel", outside[1])

    def test_each_side_is_measured_against_the_other_provers_kernels_too(self):
        # The control for the zero above: the card runs copy and compute
        # together when the two belong to different clients, so pil2's
        # copies do overlap the bridge's kernels in this same window.
        lines = self.report(CAPTURE)
        cross = [line for line in lines if "by the other prover's" in line]
        self.assertLen(cross, 2)
        self.assertEndsWith(cross[0], "0.00 s by the other prover's")
        self.assertEndsWith(cross[1], "0.01 s by the other prover's")

    def test_the_idle_before_each_copy_is_reported(self):
        lines = "\n".join(self.report(CAPTURE))
        # 78 of the bridge's 79 copies have a preceding kernel; over half
        # start into a device that has been idle more than a millisecond.
        self.assertIn("44 of 78 copies (56 %) started more than 1 ms", lines)
        self.assertIn(
            "idle before a copy: median   1.18 ms, p90  14.23 ms, max   17.88 ms",
            lines,
        )

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
