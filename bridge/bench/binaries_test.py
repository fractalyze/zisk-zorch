"""Pins that a run's record names the binary's bytes, not its file.

The case that matters is a binary rebuilt in place: same path, same role, and
a size that need not move. Only the hash separates those two builds, so a
record without one leaves a bench quotable against the wrong source."""

import contextlib
import io
import pathlib
import tempfile

from absl.testing import absltest

from bridge.bench import binaries


def write(tmp: pathlib.Path, name: str, body: bytes) -> pathlib.Path:
    path = tmp / name
    path.write_bytes(body)
    return path


class BinariesTest(absltest.TestCase):
    def setUp(self):
        super().setUp()
        self.tmp = pathlib.Path(tempfile.mkdtemp())

    def test_a_record_round_trips_through_host_txt(self):
        path = write(self.tmp, "cargo-zisk-dev", b"a prover")
        one = binaries.identify(binaries.PROVER, str(path), digest=True)
        self.assertEqual(binaries.parse(one.line()), {binaries.PROVER: one})

    def test_a_rebuilt_binary_at_the_same_path_is_a_different_record(self):
        # The whole point: the path, the role and even the size can be equal
        # across a rebuild, so the hash has to be what separates them.
        path = write(self.tmp, "cargo-zisk-dev", b"build one")
        before = binaries.identify(binaries.PROVER, str(path), digest=True)
        write(self.tmp, "cargo-zisk-dev", b"build two")
        after = binaries.identify(binaries.PROVER, str(path), digest=True)
        self.assertEqual(before.size, after.size)
        self.assertNotEqual(before.sha256, after.sha256)

    def test_the_plugin_is_recorded_on_what_the_compile_cache_keys_on(self):
        path = write(self.tmp, "libpjrt.so", b"a plugin")
        plugin = binaries.identify(binaries.PLUGIN, str(path), digest=False)
        self.assertEqual(plugin.sha256, "")
        self.assertEqual(plugin.size, len(b"a plugin"))
        self.assertIn("mtime=", plugin.line())
        self.assertNotIn("sha256=", plugin.line())

    def test_a_path_with_spaces_survives_the_round_trip(self):
        path = write(self.tmp, "cargo zisk dev", b"a prover")
        one = binaries.identify(binaries.PROVER, str(path), digest=True)
        self.assertIn(" ", one.path)
        self.assertEqual(binaries.parse(one.line())[binaries.PROVER].path, one.path)

    def test_the_host_lines_around_the_records_are_left_alone(self):
        path = write(self.tmp, "cargo-zisk-dev", b"a prover")
        one = binaries.identify(binaries.PROVER, str(path), digest=True)
        host = f"14:02:11 up 6 days,  load average: 0.31\n1024 MiB\n{one.line()}\n"
        self.assertEqual(binaries.parse(host), {binaries.PROVER: one})

    def test_a_missing_binary_is_an_error_rather_than_an_empty_record(self):
        with self.assertRaises(OSError):
            binaries.identify(binaries.PROVER, str(self.tmp / "absent"), digest=True)

    def cli(self, *argv: str) -> pathlib.Path:
        """What run.sh appends to host.txt for these arguments."""
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(binaries.main(["binaries.py", *argv]), 0)
        host = self.tmp / "host.txt"
        host.write_text(out.getvalue())
        return host

    def test_the_cli_writes_both_roles_and_only_the_prover_is_hashed(self):
        prover = write(self.tmp, "cargo-zisk-dev", b"a prover")
        plugin = write(self.tmp, "libpjrt.so", b"a plugin")
        host = self.cli("--prover", str(prover), "--plugin", str(plugin))
        found = binaries.parse(host.read_text())
        self.assertCountEqual(found, [binaries.PROVER, binaries.PLUGIN])
        self.assertTrue(found[binaries.PROVER].sha256)
        self.assertFalse(found[binaries.PLUGIN].sha256)

    def test_a_native_run_records_a_prover_and_no_plugin(self):
        prover = write(self.tmp, "cargo-zisk", b"a prover")
        host = self.cli("--prover", str(prover))
        self.assertIn("plugin none", binaries.describe(host))

    def test_a_line_that_is_not_a_record_cannot_kill_the_read(self):
        # host.txt is the host's file first, and a summary that dies on an
        # unbalanced quote or a torn write in it reports nothing about a run
        # that is otherwise fine.
        path = write(self.tmp, "cargo-zisk-dev", b"a prover")
        one = binaries.identify(binaries.PROVER, str(path), digest=True)
        host = "an unbalanced ' quote\nprover path=/x size=big mtime=z\nplugin\n"
        self.assertEqual(binaries.parse(host + one.line()), {binaries.PROVER: one})

    def test_a_run_with_no_record_says_so(self):
        # Silence here would read as "nothing to add" on a summary whose whole
        # job is to say which binary it describes.
        self.assertEqual(
            binaries.describe(self.tmp / "absent.txt"), "prover not recorded"
        )
        (self.tmp / "host.txt").write_text("14:02:11 up 6 days\n")
        self.assertEqual(
            binaries.describe(self.tmp / "host.txt"), "prover not recorded"
        )


if __name__ == "__main__":
    absltest.main()
