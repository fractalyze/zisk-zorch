"""Pins how `summarize.py` reads a run the bridge rescued.

The bridge catches a read-ahead upload that a full card refused and sends the
same words up under the slot, logging PJRT's message verbatim. That message
looks exactly like the one an aborted run leaves behind, so the summary has to
tell the two apart: a rescued run finished, and #175 is measured with this
tool."""

import contextlib
import io
import pathlib
import tempfile

from absl.testing import absltest

from bridge.bench import summarize

# A rescued run, with the text a real one carries. The two-line block is the
# **default panic hook's** output, captured from this toolchain by running
# `upload_ahead`'s panic case with the hook left in place: the hook prints at
# panic time, before `catch_unwind` catches the unwind, so it lands in the log
# whatever ZZ_LOG says and whatever the bridge does afterwards. An earlier
# fixture carried only the bridge's own line, which is why it passed over a
# log the tool still misread.
HOOK = """\
thread '<unnamed>' (2093326) panicked at src/lib.rs:461:35:
PJRT error in BufferFromHostBuffer: Out of memory while trying to allocate 3.00GiB
note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace
"""

RESCUED = (
    HOOK
    + """\
[zz +  3.120] Main_n22: read-ahead upload gave way to the slot (PJRT error in \
BufferFromHostBuffer: Out of memory while trying to allocate 3.00GiB)
[zz +  3.400] fixed sections for Main_n22: 0.31 s under the slot, 0.00 s ahead of it
INFO: <<< GENERATING_INNER_PROOFS (6345ms)
Elapsed (wall clock) time (h:mm:ss or m:ss): 0:20.29
Vadcop Final proof was verified
exit=0
"""
)

# The same rescue with ZZ_LOG unset: the bridge logs nothing, so the hook's
# output is the only trace of it. The run still finished.
RESCUED_QUIET = (
    HOOK
    + """\
Elapsed (wall clock) time (h:mm:ss or m:ss): 0:20.29
Vadcop Final proof was verified
exit=0
"""
)

# A run the same out-of-memory actually killed.
ABORTED = (
    HOOK
    + """\
[zz +  3.120] instance 0 Main_n22 (basic): 0.50 s, of which 0.10 s waiting
exit=101
"""
)

# What run.sh writes into host.txt: the host's own two lines, then the identity
# of the binaries that made the run.
HOST = "14:02:11 up 6 days,  load average: 0.31\n1024 MiB\n"
PROVER = (
    "prover path=/opt/zisk/cargo-zisk-dev size=184549376"
    " mtime=2026-09-08T13:49:02Z sha256={sha}\n"
)

# Died on the first prove: no instance finished, and no exit line was ever
# written because the harness was killed with it.
ABORTED_BEFORE_ANY_INSTANCE = (
    HOOK
    + """\
INFO: <<< INITIALIZING_PROOFMAN (8712ms)
"""
)


def run(text: str, host: str | None = None) -> str:
    """A run directory holding this log, and the host.txt run.sh writes beside
    it when one is given."""
    out_dir = pathlib.Path(tempfile.mkdtemp())
    path = out_dir / "run.log"
    path.write_text(text)
    if host is not None:
        (out_dir / "host.txt").write_text(host)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        summarize.summarize(path)
    return out.getvalue()


class SummarizeTest(absltest.TestCase):
    def test_a_rescued_read_ahead_upload_is_not_an_abort(self):
        # The panic hook's message is in this log, identical to the aborted
        # one's; only the outcome differs.
        out = run(RESCUED)
        self.assertNotIn("ABORTED", out)
        self.assertIn("read-ahead uploads sent to the slot: 1", out)
        self.assertIn("Main_n22 x1", out)

    def test_a_rescue_is_recognised_with_zz_log_unset(self):
        # No bridge line at all then — the hook's output is the only trace,
        # and it is the same text an abort leaves.
        out = run(RESCUED_QUIET)
        self.assertNotIn("ABORTED", out)
        self.assertIn("read-ahead uploads sent to the slot: 1", out)
        self.assertIn("air not recorded", out)

    def test_a_real_abort_is_still_reported(self):
        self.assertIn("ABORTED: Out of memory", run(ABORTED))

    def test_an_abort_before_the_first_instance_is_still_reported(self):
        self.assertIn("ABORTED: Out of memory", run(ABORTED_BEFORE_ANY_INSTANCE))

    def test_either_verification_wording_is_recognised(self):
        # The wording has changed across prover builds, and it does not track
        # native-vs-bridge: runs of both modes on one build share a phrase.
        # Missing either reports a good run as unverified.
        self.assertIn("verified=True", run("Vadcop Final proof was verified\n"))
        self.assertIn("verified=True", run("Proof verified successfully\n"))
        self.assertIn("verified=False", run("something else entirely\n"))

    def test_two_runs_differ_in_their_summaries_when_only_the_prover_did(self):
        # The acceptance the tool exists for: the logs are the same text, so
        # the only thing that can tell the runs apart is what built them.
        log = "Elapsed (wall clock) time (h:mm:ss or m:ss): 0:20.29\n"
        before, after = (
            run(log, HOST + PROVER.format(sha=sha))
            for sha in ("1f2ae3c4b5d6e7f8", "90a1b2c3d4e5f607")
        )
        self.assertIn("prover cargo-zisk-dev sha 1f2ae3c4b5d6", before)
        self.assertIn("prover cargo-zisk-dev sha 90a1b2c3d4e5", after)
        self.assertNotEqual(before, after)

    def test_a_run_whose_binaries_were_not_recorded_says_so(self):
        # A summary that cannot name its prover has to look different from one
        # that can; silence here is what let a stale binary pass as a fix.
        self.assertIn("prover not recorded", run("exit=0\n"))

    def test_a_run_with_neither_says_nothing_about_either(self):
        out = run("Elapsed (wall clock) time (h:mm:ss or m:ss): 0:20.29\n")
        self.assertNotIn("ABORTED", out)
        self.assertNotIn("read-ahead uploads", out)


if __name__ == "__main__":
    absltest.main()
