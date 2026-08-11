"""The own-prove wire proof byte-matches the native flat proof.

One composed prove from the capture's dumped claim (so the schedule
replays the native transcript exactly), serialized through `emit_wire` +
`emit_wire_proof`, compared word-for-word against the bundle's native
``proof`` section — query rows, sibling paths, last-level digests, and
nonce included. Like `gates_test`, the suite skips loudly without a
``ZISK_PIL2_CAPTURE`` bundle; the bundle must carry the native flat proof
(a CPU ``tools/pil2-dump`` capture does)."""

from __future__ import annotations

import os
import pathlib

import frx.numpy as fnp
import numpy as np
from absl.testing import absltest

from zisk_zorch.harness.block_composite import emit_wire_proof
from zisk_zorch.harness.capture import (
    CAPTURE_ENV,
    FIXTURE_INSTANCE,
    FIXTURE_STARKINFO,
    Capture,
)
from zisk_zorch.harness.pil2 import transcript_width
from zisk_zorch.harness.pil2_prover import Pil2InnerProver
from zisk_zorch.transcript.transcript import Transcript
from zisk_zorch.types import InnerWitness


class WireProofTest(absltest.TestCase):
    def test_own_prove_wire_proof_matches_native(self):
        bundle = os.environ.get(CAPTURE_ENV, "")
        if not bundle or not pathlib.Path(bundle).is_dir():
            self.skipTest(
                f"no capture: set {CAPTURE_ENV} to a tools/pil2-dump bundle "
                "directory (with the native flat proof) to run"
            )
        dump = pathlib.Path(bundle)
        cap = Capture(dump, FIXTURE_INSTANCE, dump / FIXTURE_STARKINFO)
        if cap.gpu_dump:
            self.skipTest("GPU staging buffers truncate the native proof")

        prover = Pil2InnerProver(cap.pil2_key, emit_wire=True)
        transcript = Transcript(
            transcript_width(cap.si["starkStruct"]), cap.hash_family
        )
        quotient_claim = opening = None
        for name, result in prover.prove_stages(
            cap.pil2_claim(), InnerWitness(fnp.asarray(cap.trace)), transcript
        ):
            if name == "quotient":
                quotient_claim = result.reduced_claim
            elif name == "opening":
                opening = result.reduction_proof

        out = pathlib.Path(self.create_tempdir().full_path)
        path = emit_wire_proof(cap, out, quotient_claim, opening)
        np.testing.assert_array_equal(np.load(path), np.load(cap.path("proof")))


if __name__ == "__main__":
    absltest.main()
