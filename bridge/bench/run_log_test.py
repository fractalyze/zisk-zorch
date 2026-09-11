"""Pins the one rule two readers share: how the bridge's `ZZ_LOG` line per
instance is read. The loose match on the kind is the point of the module, so
it is pinned as behaviour and not left to the regex."""

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


if __name__ == "__main__":
    absltest.main()
