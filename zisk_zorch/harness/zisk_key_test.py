"""`zisk_pil2_key` builds the same key `Capture.pil2_key` dumps.

The synthetic-tree cases pin the mechanics on a bare CI host: the
extended section must invert back to the ``.const`` bytes (exact field
arithmetic — equal or wrong), and the guards fail loudly. The bundle case
byte-compares against the capture's dumped ``const_ext`` — the actual
"key-only == dump-fed" claim a non-capture witness source rests on.
"""

from __future__ import annotations

import json
import os
import pathlib

import frx.numpy as fnp
import numpy as np
from absl.testing import absltest
from zk_dtypes import goldilocks as F, pfinfo

from zisk_zorch.commit.trace_commit import unextend
from zisk_zorch.harness.capture import CAPTURE_ENV, FIXTURE_INSTANCE, Capture
from zisk_zorch.harness.verify_proof_layout import starkinfo_for
from zisk_zorch.harness.zisk_key import (
    zisk_hash_family,
    zisk_pil2_key,
)

_NB, _NBE, _NCONST = 3, 5, 2


def _write_key_tree(root: pathlib.Path, air: str, *, custom_commits=None) -> dict:
    """A minimal example-layout key: ``build/<g>/airs/<air>/air/<air>.*``."""
    base = root / "build" / "g0" / "airs" / air / "air"
    base.mkdir(parents=True)
    si = {
        "starkStruct": {"nBits": _NB, "nBitsExt": _NBE},
        "nConstants": _NCONST,
    }
    if custom_commits is not None:
        si["customCommits"] = custom_commits
    (base / f"{air}.starkinfo.json").write_text(json.dumps(si))
    (base / f"{air}.expressionsinfo.json").write_text(json.dumps({"marker": air}))
    rng = np.random.default_rng(3)
    words = rng.integers(0, pfinfo(F).modulus, size=(1 << _NB, _NCONST), dtype=np.uint64)
    words.tofile(base / f"{air}.const")
    return {"air_groups": ["g0"], "airs": [[{"name": air}]]}


class ZiskKeyTest(absltest.TestCase):
    def test_const_ext_inverts_to_the_const_bytes(self):
        root = pathlib.Path(self.create_tempdir().full_path)
        gi = _write_key_tree(root, "TinyAir")
        key = zisk_pil2_key(root, gi, "TinyAir")
        self.assertEqual(key.expressionsinfo, {"marker": "TinyAir"})
        self.assertEqual(key.hash_family, "Poseidon2")
        blowup = 1 << (_NBE - _NB)
        self.assertEqual(key.const_ext.shape, ((1 << _NB) * blowup, _NCONST))
        recovered = np.asarray(unextend(fnp.array(key.const_ext), blowup))
        np.testing.assert_array_equal(
            recovered.view(np.uint64), np.asarray(key.const_base).view(np.uint64)
        )

    def test_custom_commit_air_fails_loudly(self):
        root = pathlib.Path(self.create_tempdir().full_path)
        gi = _write_key_tree(root, "RomLike", custom_commits=[{"name": "rom"}])
        with self.assertRaises(NotImplementedError):
            zisk_pil2_key(root, gi, "RomLike")

    def test_hash_family_guard(self):
        self.assertEqual(zisk_hash_family({}), "Poseidon2")
        self.assertEqual(zisk_hash_family({"hash": "Poseidon1"}), "Poseidon1")
        with self.assertRaises(ValueError):
            zisk_hash_family({"hash": "Keccak"})

    def test_matches_the_captures_dumped_key(self):
        bundle = os.environ.get(CAPTURE_ENV, "")
        key_dir = os.environ.get("ZISK_PROVING_KEY", "")
        if not bundle or not pathlib.Path(bundle).is_dir():
            self.skipTest(f"no capture: set {CAPTURE_ENV} to a bundle directory")
        if not key_dir or not pathlib.Path(key_dir).is_dir():
            self.skipTest("no proving key: set ZISK_PROVING_KEY")
        key_dir = pathlib.Path(key_dir)
        gi = json.loads((key_dir / "pilout.globalInfo.json").read_text())
        inst = os.environ.get("ZISK_PIL2_INSTANCE", FIXTURE_INSTANCE)
        starkinfo = starkinfo_for(key_dir, gi, inst)
        cap = Capture(pathlib.Path(bundle), inst, starkinfo)
        air = cap.si["name"]
        if cap.si.get("customCommits"):
            self.skipTest(f"{air} carries custom commits (#115 question 5)")
        key = zisk_pil2_key(key_dir, gi, air)
        self.assertEqual(key.hash_family, cap.hash_family)
        np.testing.assert_array_equal(
            np.asarray(key.const_base).view(np.uint64),
            np.asarray(cap.const_base).view(np.uint64),
        )
        np.testing.assert_array_equal(
            np.asarray(key.const_ext).view(np.uint64),
            np.asarray(cap.bufs[("const", 0)]).view(np.uint64),
        )


if __name__ == "__main__":
    absltest.main()
