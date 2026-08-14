"""`prove_block` drives the block phases end to end on a capture bundle.

The composite's own orchestration — stage-1 commits to contributions, the
derived global challenge, every instance's composed prove under that seed,
then the global-constraint binding — has no other caller: the byte-gates
enter through `on_stage`, so nothing else exercises the phase wiring
itself. This runs it as a one-instance block, which is the smallest shape
that still walks all four phases.

Like `gates_test` and `wire_proof_test`, the suite skips loudly without a
``ZISK_PIL2_CAPTURE`` bundle.
"""

from __future__ import annotations

import json
import os
import pathlib

import numpy as np
from absl.testing import absltest

from zisk_zorch.harness.block_composite import prove_block
from zisk_zorch.harness.capture import CAPTURE_ENV, FIXTURE_INSTANCE, Capture
from zisk_zorch.harness.verify_proof_layout import starkinfo_for


class BlockCompositeTest(absltest.TestCase):
    def test_prove_block_phases_on_one_instance(self):
        bundle = os.environ.get(CAPTURE_ENV, "")
        key = os.environ.get("ZISK_PROVING_KEY", "")
        if not bundle or not pathlib.Path(bundle).is_dir():
            self.skipTest(
                f"no capture: set {CAPTURE_ENV} to a tools/pil2-dump bundle "
                "directory to run"
            )
        if not key or not pathlib.Path(key).is_dir():
            self.skipTest(
                "no proving key: set ZISK_PROVING_KEY to the provingKey "
                "directory (the verkey and globalInfo the block phases need)"
            )
        key_dir = pathlib.Path(key)
        gi = json.loads((key_dir / "pilout.globalInfo.json").read_text())
        # Block bundles carry their own instances (the fixture bundle's
        # `FIXTURE_INSTANCE` is only one of them), so the instance is
        # selectable — any single instance walks the same four phases.
        inst = os.environ.get("ZISK_PIL2_INSTANCE", FIXTURE_INSTANCE)
        if not (pathlib.Path(bundle) / f"{inst}_trace.npy").exists():
            self.skipTest(f"{inst} not in {bundle}; set ZISK_PIL2_INSTANCE")
        cap = Capture(pathlib.Path(bundle), inst, starkinfo_for(key_dir, gi, inst))
        family = cap.si["name"]
        root = key_dir / "zisk" if (key_dir / "zisk").is_dir() else key_dir / "build"
        verkey_files = sorted(root.rglob(f"{family}.verkey.json"))
        if not verkey_files:
            self.skipTest(f"no verkey for {family} under {root}")
        vk = np.array(json.loads(verkey_files[0].read_text()), dtype=np.uint64)

        stages = []
        seed, results, global_values = prove_block(
            [(family, cap, vk)],
            global_info=gi,
            global_constraints=[],
            on_stage=lambda inst, name, _r: stages.append((inst, name)),
        )

        # Phase 2 produced a cubic seed, phase 3 ran every stage of the one
        # instance, and phase 4 returned its airgroup values.
        self.assertEqual(np.asarray(seed).shape, (3,))
        self.assertLen(results, 1)
        self.assertContainsSubset(
            ["logup_witness", "quotient", "opening"], [n for _, n in stages]
        )
        self.assertEqual(global_values, [])


if __name__ == "__main__":
    absltest.main()
