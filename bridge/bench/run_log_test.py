"""Pins the rules more than one reader shares: how the bridge's `ZZ_LOG` line
per instance is read, and whether a run finished. The loose match on the kind
and the two verify wordings are the point of the module, so both are pinned as
behaviour and not left to the regexes."""

from absl.testing import absltest, parameterized

from bridge.bench import run_log


def line(kind: str = "worker", air: str = "VirtualTableZisk0_n21") -> str:
    return (
        f"[zz +  6.917] instance 9 {air} ({kind}): 1.228 s,"
        " of which 0.681 s waiting for the client"
        " (0.671 s of uploads and reads done ahead)"
    )


class RunLogTest(parameterized.TestCase):
    @parameterized.named_parameters(
        ("worker", "worker"),
        ("streamed", "streamed"),
        # The reason this module exists: a reader that matched only the kinds
        # it knew about dropped the streamed instance and misaligned every
        # prove-to-AIR mapping built on the count. A kind nobody has written
        # yet must still arrive as an instance.
        ("a kind nobody has added yet", "conscripted"),
    )
    def test_every_kind_is_an_instance(self, kind):
        (one,) = run_log.instances(line(kind))
        self.assertEqual(one.kind, kind)
        self.assertEqual(one.index, 9)
        self.assertEqual(one.air, "VirtualTableZisk0_n21")

    def test_held_excludes_the_wait_for_a_client(self):
        (one,) = run_log.instances(line())
        self.assertAlmostEqual(one.total, 1.228)
        self.assertAlmostEqual(one.waiting, 0.681)
        self.assertAlmostEqual(one.held, 0.547)

    def test_the_span_is_read_back_from_the_end(self):
        (one,) = run_log.instances(line())
        start, end = one.span
        self.assertAlmostEqual(end, 6.917)
        self.assertAlmostEqual(start, 6.917 - 0.547)

    def test_lines_that_are_not_an_instance_are_ignored(self):
        log = (
            "[zz +  0.309] bridge up: 1 PJRT client(s)\n"
            "[zz +  0.380]   load commit1: 15661 KB, read 6.4 ms\n"
            "[zz +  1.000] fixed sections for Main_n22: 0.1 s under the slot,"
            " 0.2 s ahead of it\n"
        )
        self.assertEmpty(run_log.instances(log))

    def test_every_instance_in_a_log_is_returned(self):
        log = "\n".join([line("worker"), "unrelated", line("streamed")])
        self.assertEqual(
            [one.kind for one in run_log.instances(log)], ["worker", "streamed"]
        )


OOM = (
    "2026-09-11T07:57:55.251173Z proofman::proofman ERROR: zisk-zorch bridge:"
    " instance 1: PJRT error in Event_Await: Out of memory while trying to"
    " allocate 1.88GiB.\n"
)


class FinishedTest(parameterized.TestCase):
    @parameterized.named_parameters(
        # Both wordings are a pass. A prover built before 2026-09-08 prints the
        # first and a later one the second, and neither identifies the stack --
        # a native and a bridged run of one vintage end the same way.
        ("pre_09_08", "··· ✓ Proof verified successfully"),
        ("post_09_08", "··· ✓ Vadcop Final proof was verified"),
    )
    def test_both_verify_wordings_count(self, phrase):
        self.assertTrue(run_log.verified(phrase + "\n"))

    def test_a_rescued_oom_is_a_run_that_finished(self):
        """The bridge catches a read-ahead upload's OOM and uploads under the
        slot; Rust's panic hook printed the message before `catch_unwind` saw
        it. The text is identical to an aborted run's, so `exit=` decides."""
        self.assertTrue(run_log.completed(OOM + "exit=0\n"))
        self.assertFalse(run_log.completed(OOM + "exit=134\n"))

    def test_exit_beats_the_text(self):
        """A log that verified and then failed on the way out did not finish."""
        self.assertFalse(
            run_log.completed("··· ✓ Vadcop Final proof was verified\nexit=1\n")
        )

    def test_without_an_exit_line_a_quiet_log_is_not_called_an_abort(self):
        """A partial capture, or a log not written by run.sh: say aborted only
        when there is also a failure to point at."""
        self.assertTrue(run_log.completed("nothing in particular\n"))
        self.assertFalse(run_log.completed(OOM))


if __name__ == "__main__":
    absltest.main()
