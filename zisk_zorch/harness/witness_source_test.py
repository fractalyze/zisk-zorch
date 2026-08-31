"""The `WitnessSource` seam: protocol shape, and `Capture` conforming to it.

The full conformance proof is `block_composite_test` driving a real
`Capture` through `prove_block` on a bundle; this suite pins the seam
itself so a drift (a renamed member, a widened driver dependency) fails
here, on a bare CI host, instead of only on a capture-provisioned one.
"""

from __future__ import annotations

import json
import os
import pathlib

import numpy as np
from absl.testing import absltest

from zisk_zorch.harness.capture import (
    CAPTURE_ENV,
    FIXTURE_INSTANCE,
    FIXTURE_STARKINFO,
    Capture,
)
from zisk_zorch.harness.witness_source import WitnessSource


class _StubSource:
    """The minimal object the protocol admits — what an rw-backed source
    must supply, with no dump bundle behind it."""

    instance = "stub_inst0"
    si: dict = {"name": "Stub"}
    hash_family = "Poseidon2"

    # isinstance on a runtime_checkable protocol probes data members with
    # hasattr, so the stub's key must resolve, not raise.
    pil2_key = None

    def pil2_claim(self):
        raise NotImplementedError

    def trace_words(self) -> np.ndarray:
        return np.zeros((4, 2), dtype=np.uint64)

    def release(self) -> None:
        pass


class WitnessSourceTest(absltest.TestCase):
    def test_stub_conforms(self):
        self.assertIsInstance(_StubSource(), WitnessSource)

    def test_capture_class_carries_the_surface(self):
        # `instance`/`si` are set in __init__, so class-level presence can
        # only be asserted for the methods and descriptors.
        for member in (
            "pil2_key",
            "pil2_claim",
            "trace_words",
            "release",
            "hash_family",
        ):
            self.assertTrue(hasattr(Capture, member), member)

    def test_capture_conforms_on_a_bundle(self):
        bundle = os.environ.get(CAPTURE_ENV, "")
        if not bundle or not pathlib.Path(bundle).is_dir():
            self.skipTest(
                f"no capture: set {CAPTURE_ENV} to a tools/pil2-dump bundle "
                "directory to run"
            )
        bundle = pathlib.Path(bundle)
        inst = os.environ.get("ZISK_PIL2_INSTANCE", FIXTURE_INSTANCE)
        key = os.environ.get("ZISK_PROVING_KEY", "")
        # The fixture bundle copies its starkinfo in-dir; a block dump
        # doesn't, so there the starkinfo resolves through the proving key
        # like block_composite_test's — without one, skip loudly.
        if inst == FIXTURE_INSTANCE and (bundle / FIXTURE_STARKINFO).exists():
            starkinfo = bundle / FIXTURE_STARKINFO
        elif key and pathlib.Path(key).is_dir():
            from zisk_zorch.harness.verify_proof_layout import starkinfo_for

            key_dir = pathlib.Path(key)
            gi = json.loads((key_dir / "pilout.globalInfo.json").read_text())
            starkinfo = starkinfo_for(key_dir, gi, inst)
        else:
            self.skipTest(
                "bundle carries no fixture starkinfo; set ZISK_PROVING_KEY "
                "(and ZISK_PIL2_INSTANCE) to resolve it from the key"
            )
        cap = Capture(bundle, inst, starkinfo)
        self.assertIsInstance(cap, WitnessSource)
        words = cap.trace_words()
        self.assertEqual(words.dtype, np.uint64)
        self.assertEqual(words.shape, (cap.n, cap.n_cols))
        # The device trace is the same words reinterpreted, not a copy pass.
        self.assertTrue(
            np.array_equal(np.asarray(cap.trace).view(np.uint64), words)
        )


if __name__ == "__main__":
    absltest.main()
