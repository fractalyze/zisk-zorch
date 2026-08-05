"""Runs the per-stage byte-gates (`gates.GATES`) over a host-provided
capture bundle.

Each gate pins ONE stage's math with its inputs read from the dump, so a
composed-runnable mismatch localizes here to a single link. The reference
is a host-provided native artifact: without a ``ZISK_PIL2_CAPTURE`` bundle
(``tools/pil2-dump`` regenerates one) the suite skips loudly — CI hosts
carry no capture, so these effectively run on provisioned hosts, like the
``verify_inner_proof`` runnable they back."""

from __future__ import annotations

import os
import pathlib

from absl.testing import absltest

from zisk_zorch.harness.capture import (
    CAPTURE_ENV,
    FIXTURE_INSTANCE,
    FIXTURE_STARKINFO,
    Capture,
)
from zisk_zorch.harness.gates import GATES


class GatesTest(absltest.TestCase):
    def test_every_stage_gate_byte_matches_the_capture(self):
        bundle = os.environ.get(CAPTURE_ENV, "")
        if not bundle or not pathlib.Path(bundle).is_dir():
            self.skipTest(
                f"no capture: set {CAPTURE_ENV} to a tools/pil2-dump bundle "
                "directory to run the per-stage byte-gates"
            )
        dump = pathlib.Path(bundle)
        cap = Capture(dump, FIXTURE_INSTANCE, dump / FIXTURE_STARKINFO)
        for gate in GATES:
            with self.subTest(gate.__name__):
                # None = the AIR has nothing to gate there (a legitimate
                # skip); only an explicit False is a byte mismatch.
                self.assertIsNot(
                    gate(cap), False, f"{gate.__name__} diverged from the dump"
                )


if __name__ == "__main__":
    absltest.main()
