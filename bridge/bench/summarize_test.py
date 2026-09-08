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

RESCUED = """\
[zz +  3.120] Main_n22: read-ahead upload gave way to the slot (PJRT error in \
BufferFromHostBuffer: Out of memory while trying to allocate 3.00GiB)
[zz +  3.400] fixed sections for Main_n22: 0.31 s under the slot, 0.00 s ahead of it
INFO: <<< GENERATING_INNER_PROOFS (6345ms)
Elapsed (wall clock) time (h:mm:ss or m:ss): 0:20.29
Proof verified successfully
"""

ABORTED = """\
[zz +  3.120] instance 0 Main_n22 (basic): 0.50 s, of which 0.10 s waiting
PJRT error in Event_Await: Out of memory while trying to allocate 3.00GiB
"""

# A run that died on its first prove: no instance ever finished, so nothing
# the bridge reports per instance is in the log. The abort still has to be
# reported — this is the shape an out-of-memory usually takes.
ABORTED_BEFORE_ANY_INSTANCE = """\
INFO: <<< INITIALIZING_PROOFMAN (8712ms)
PJRT error in Event_Await: Out of memory while trying to allocate 3.00GiB
"""


def run(text: str) -> str:
    with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as f:
        f.write(text)
        path = pathlib.Path(f.name)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        summarize.summarize(path)
    path.unlink()
    return out.getvalue()


class SummarizeTest(absltest.TestCase):
    def test_a_rescued_read_ahead_upload_is_not_an_abort(self):
        out = run(RESCUED)
        self.assertNotIn("ABORTED", out)
        self.assertIn("read-ahead uploads sent to the slot: 1", out)
        self.assertIn("Main_n22 x1", out)

    def test_a_real_abort_is_still_reported(self):
        self.assertIn("ABORTED: Out of memory", run(ABORTED))

    def test_an_abort_before_the_first_instance_is_still_reported(self):
        self.assertIn("ABORTED: Out of memory", run(ABORTED_BEFORE_ANY_INSTANCE))

    def test_both_stacks_verification_phrases_are_recognised(self):
        # The two stacks this tool compares do not share a phrase, and the
        # summary compares them side by side, so missing either reports a good
        # run as unverified.
        self.assertIn("verified=True", run("Vadcop Final proof was verified\n"))
        self.assertIn("verified=True", run("Proof verified successfully\n"))
        self.assertIn("verified=False", run("something else entirely\n"))

    def test_a_run_with_neither_says_nothing_about_either(self):
        out = run("Elapsed (wall clock) time (h:mm:ss or m:ss): 0:20.29\n")
        self.assertNotIn("ABORTED", out)
        self.assertNotIn("read-ahead uploads", out)


if __name__ == "__main__":
    absltest.main()
