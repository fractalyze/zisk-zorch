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
from zisk_zorch.harness.contributions import (
    aggregate_contributions,
    global_challenge,
    instance_contribution,
    stage1_values,
)
from zisk_zorch.harness.pil2 import hint_value
from zisk_zorch.harness.recursion import key_root
from zisk_zorch.harness.verify_proof_layout import starkinfo_for


class _Unsettled:
    """`source` with its ``witness_calc`` columns zeroed — the trace a
    witness generator hands over (`StageOneWitness`'s contract), which no
    capture carries because a capture is dumped after pil2 filled them.

    Delegates everything else, so the block phases see the capture."""

    def __init__(self, source, columns: list[int]) -> None:
        self._source = source
        self._columns = columns

    def __getattr__(self, name):
        return getattr(self._source, name)

    def trace_words(self) -> np.ndarray:
        words = self._source.trace_words().copy()
        words[:, self._columns] = 0
        return words


def _witness_calc_columns(key) -> list[int]:
    """Where the key's ``witness_calc`` hints write, as stage-1 column
    indices."""
    cmp_ = key.starkinfo["cmPolsMap"]
    return [
        cmp_[hint_value(h, "reference")["id"]]["stagePos"]
        for h in key.expressionsinfo["hintsInfo"]
        if h["name"] == "witness_calc"
    ]


class BlockCompositeTest(absltest.TestCase):
    def _instance(self):
        """The bundle's selected instance as ``(globalInfo, capture,
        family, verkey)``, or a loud skip when the environment has no
        bundle, no proving key, or no verkey for the family."""
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
        root = key_root(key_dir)
        verkey_files = sorted(root.rglob(f"{family}.verkey.json"))
        if not verkey_files:
            self.skipTest(f"no verkey for {family} under {root}")
        vk = np.array(json.loads(verkey_files[0].read_text()), dtype=np.uint64)
        return gi, cap, family, vk

    def test_prove_block_phases_on_one_instance(self):
        gi, cap, family, vk = self._instance()

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

    def test_the_seed_folds_the_settled_root1(self):
        """Phase 1 must commit the trace phase 3 proves.

        For an AIR with ``witness_calc`` hints fed a trace whose hinted
        columns are unset, a phase 1 that commits the trace as handed folds
        a root1 the proof never carries: the verifier re-derives the seed
        from the proof's root1, every challenge diverges, and the proof is
        rejected. The check is on the seed because that is where the two
        roots meet."""
        gi, cap, family, vk = self._instance()
        columns = _witness_calc_columns(cap.pil2_key)
        if not columns:
            self.skipTest(
                f"{family} has no witness_calc hints; set ZISK_PIL2_INSTANCE "
                "to an instance of an AIR that has them"
            )
        claim = cap.pil2_claim()
        roots = []

        def keep_root(_instance, name, result):
            if name == "trace_commit":
                roots.append(np.asarray(result.root).astype(np.uint64))

        seed, _, _ = prove_block(
            [(family, _Unsettled(cap, columns), vk)],
            global_info=gi,
            global_constraints=[],
            on_stage=keep_root,
        )

        # Fold phase 3's own root1 — the one the proof carries and the
        # verifier re-derives from — the way phase 2 does, and require the
        # seed the block actually used to be that.
        av1 = (
            stage1_values(claim.airvalues, cap.si["airValuesMap"])
            if cap.si.get("airValuesMap")
            else np.zeros(0, dtype=np.uint64)
        )
        want = global_challenge(
            claim.publics,
            stage1_values(claim.proofvalues, gi["proofValuesMap"]),
            aggregate_contributions(
                [
                    instance_contribution(
                        vk,
                        roots[0],
                        av1,
                        hash_family=gi["hash"],
                        lattice_size=gi["latticeSize"],
                    )
                ]
            ),
            hash_family=gi["hash"],
        )
        np.testing.assert_array_equal(np.asarray(seed), np.asarray(want))


if __name__ == "__main__":
    absltest.main()
