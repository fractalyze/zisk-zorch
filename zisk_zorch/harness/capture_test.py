"""`Capture.release` drops every cached array and later reads reload.

The block composite walks ~40 captures in one process; the per-instance
caches must not outlive their instance (the traces alone exceed host RAM),
so `release` is load-bearing for block-scale runs."""

from __future__ import annotations

import json
import pathlib

import numpy as np
from absl.testing import absltest

from zisk_zorch.harness.capture import Capture

# The minimal starkinfo `Capture.__init__` reads: a 2-stage AIR, 4 rows,
# blowup 2, two committed columns, no customs.
STARKINFO = {
    "nStages": 2,
    "boundaries": [{"name": "everyRow"}],
    "starkStruct": {
        "nBits": 2,
        "nBitsExt": 3,
        "merkleTreeArity": 2,
        "steps": [{"nBits": 3}],
    },
    "cmPolsMap": [],
    "evMap": [],
    "openingPoints": [0],
    "mapSectionsN": {"cm1": 2, "cm2": 0, "cm3": 1},
    "qDim": 1,
    "nConstants": 1,
}


class CaptureReleaseTest(absltest.TestCase):
    def _capture(self) -> tuple[Capture, np.ndarray]:
        dump = pathlib.Path(self.create_tempdir().full_path)
        si = dump / "Test.starkinfo.json"
        si.write_text(json.dumps(STARKINFO))
        words = np.arange(8, dtype=np.uint64)  # n=4 rows x cm1=2 cols
        np.save(dump / "inst_trace.npy", words)
        # A per-stage root file marks the bundle as a CPU (row-major) dump.
        np.save(dump / "inst_root1.npy", np.zeros(4, dtype=np.uint64))
        return Capture(dump, "inst", si), words

    def test_release_drops_caches_and_rereads(self):
        cap, words = self._capture()
        first = cap.u64("trace")
        self.assertIs(cap.u64("trace"), first)  # cached
        trace = cap.trace
        self.assertIn("trace", cap.__dict__)

        cap.release()

        self.assertEqual(cap._sections, {})
        self.assertNotIn("trace", cap.__dict__)
        reread = cap.u64("trace")
        self.assertIsNot(reread, first)
        np.testing.assert_array_equal(reread, words)
        np.testing.assert_array_equal(
            np.asarray(cap.trace).astype(np.uint64),
            np.asarray(trace).astype(np.uint64),
        )


if __name__ == "__main__":
    absltest.main()
