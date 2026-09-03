"""The exported artifacts ARE the Python prover: one random instance,
proved by `Pil2InnerProver` and by the artifact replay, must serialize to
the same flat proof bit-for-bit.

Needs a proving key (`ZISK_PROVING_KEY`) and a GPU; skips otherwise.
`ZISK_EXPORT_AIR` picks the AIR (default RomData, the smallest basic AIR
with every stage present: stage-2 air values, an airgroup value, a
chunked quotient). `ZISK_ARTIFACTS` reuses an existing export directory
instead of exporting into the test's temp dir.
"""

from __future__ import annotations

import os
import pathlib
import tempfile
from dataclasses import replace

import frx
import frx.numpy as fnp
import numpy as np
from absl.testing import absltest
from zk_dtypes import goldilocks as F

from zisk_zorch.commit.trace_commit import extend
from zisk_zorch.export import replay
from zisk_zorch.export.cases import random_case
from zisk_zorch.export.export_air import export_air, export_key
from zisk_zorch.export.runtime import Artifact
from zisk_zorch.harness.block_composite import emit_wire_proof
from zisk_zorch.harness.pil2 import transcript_width
from zisk_zorch.harness.pil2_prover import Pil2Claim, Pil2InnerProver
from zisk_zorch.transcript.transcript import Transcript
from zisk_zorch.types import InnerWitness


class _Source:
    """The `WitnessSource` surface `emit_wire_proof` reads."""

    def __init__(self, si: dict) -> None:
        self.si = si
        self.instance = "replay"


class ArtifactReplayTest(absltest.TestCase):
    def test_replay_matches_the_python_prover(self):
        key_dir = os.environ.get("ZISK_PROVING_KEY", "")
        if not key_dir or not pathlib.Path(key_dir).is_dir():
            self.skipTest("no proving key: set ZISK_PROVING_KEY")
        if frx.devices()[0].platform != "gpu":
            self.skipTest("the artifacts are exported for the GPU backend")
        air = os.environ.get("ZISK_EXPORT_AIR", "RomData")
        key = export_key(pathlib.Path(key_dir), air)
        si = key.starkinfo
        nb = si["starkStruct"]["nBits"]

        artifacts = os.environ.get("ZISK_ARTIFACTS", "")
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        if artifacts and (pathlib.Path(artifacts) / f"{air}_n{nb}").is_dir():
            art_dir = pathlib.Path(artifacts) / f"{air}_n{nb}"
        else:
            art_dir = export_air(pathlib.Path(key_dir), air, pathlib.Path(tmp.name))

        case = random_case(key, seed=166)
        inst, sections, custom_base = (
            case.instance,
            case.sections,
            case.sections.custom_base,
        )
        w1 = si["mapSectionsN"]["cm1"]

        # The reference: the Python composite over the same key and sections.
        prover_key = key
        if custom_base:
            blowup = 1 << (si["starkStruct"]["nBitsExt"] - nb)
            prover_key = replace(
                key,
                custom_base={ci: b.view(F) for ci, b in custom_base.items()},
                custom_ext={
                    ci: np.asarray(extend(fnp.array(b.view(F)), blowup))
                    for ci, b in custom_base.items()
                },
            )
        prover = Pil2InnerProver(prover_key, emit_wire=True)
        claim = Pil2Claim(
            n_bits=nb,
            n_cols=w1,
            publics=inst.publics,
            airvalues=inst.airvalues,
            proofvalues=inst.proofvalues,
            global_challenge=inst.global_challenge,
        )
        transcript = Transcript(transcript_width(si["starkStruct"]), key.hash_family)
        results = dict(
            prover.prove_stages(
                claim, InnerWitness(fnp.asarray(inst.trace.view(F))), transcript
            )
        )
        expected_path = emit_wire_proof(
            _Source(si),
            pathlib.Path(tmp.name),
            results["quotient"].reduced_claim,
            results["opening"].reduction_proof,
        )
        expected = np.load(expected_path)

        got, _ = replay.prove(Artifact(art_dir), sections, inst)

        self.assertEqual(got.shape, expected.shape)
        mismatch = np.flatnonzero(got != expected)
        self.assertEqual(
            mismatch.size, 0, f"{mismatch.size} words differ, first at {mismatch[:8]}"
        )


if __name__ == "__main__":
    absltest.main()
