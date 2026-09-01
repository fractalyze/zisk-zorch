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

# The smallest starkinfo `Capture.__init__` accepts — enough to reach
# `hash_family`, which reads only the key tree above the starkinfo path.
_MINIMAL_STARKINFO = {
    "nStages": 2,
    "boundaries": ["everyRow"],
    "starkStruct": {
        "nBits": _NB,
        "nBitsExt": _NBE,
        "merkleTreeArity": 4,
        "steps": [{"nBits": _NBE}],
    },
    "cmPolsMap": [],
    "evMap": [],
    "openingPoints": [0, 1],
    "mapSectionsN": {"cm1": 1, "cm3": 1},
    "nConstants": _NCONST,
    "qDim": 3,
}


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
    words = rng.integers(
        0, pfinfo(F).modulus, size=(1 << _NB, _NCONST), dtype=np.uint64
    )
    words.tofile(base / f"{air}.const")
    return {"air_groups": ["g0"], "airs": [[{"name": air}]]}


class ZiskKeyTest(absltest.TestCase):
    def test_const_ext_inverts_to_the_const_bytes(self):
        root = pathlib.Path(self.create_tempdir().full_path)
        gi = _write_key_tree(root, "TinyAir")
        key = zisk_pil2_key(root, gi, "TinyAir")
        self.assertEqual(key.expressionsinfo, {"marker": "TinyAir"})
        # `_write_key_tree`'s globalInfo carries no `hash`, i.e. pil2's
        # DEFAULT_HASH_ID (see test_hash_family_guard).
        self.assertEqual(key.hash_family, "Poseidon1")
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
        # An absent `hash` is pil2's DEFAULT_HASH_ID, not a guess: the
        # shipped ziskup v1.0.0-alpha key omitted it and native committed
        # with Poseidon1 (see poseidon1/goldilocks.py). Defaulting to
        # Poseidon2 here would byte-mismatch every tree of such a prove.
        self.assertEqual(zisk_hash_family({}), "Poseidon1")
        self.assertEqual(zisk_hash_family({"hash": "Poseidon1"}), "Poseidon1")
        self.assertEqual(zisk_hash_family({"hash": "Poseidon2"}), "Poseidon2")
        with self.assertRaises(ValueError):
            zisk_hash_family({"hash": "Keccak"})

    def test_hash_family_agrees_with_the_capture_path(self):
        # A key-only Pil2Key and a dump-fed one must resolve the same key to
        # the same sponge, or the two paths prove differently off one key.
        root = pathlib.Path(self.create_tempdir().full_path)
        starkinfo = root / "airs" / "Air" / "air" / "Air.starkinfo.json"
        starkinfo.parent.mkdir(parents=True)
        starkinfo.write_text(json.dumps(_MINIMAL_STARKINFO))
        for gi, expect in (({"hash": "Poseidon2"}, "Poseidon2"), ({}, "Poseidon1")):
            (root / "pilout.globalInfo.json").write_text(json.dumps(gi))
            cap = Capture(root, "probe", starkinfo)
            self.assertEqual(cap.hash_family, expect)
            self.assertEqual(zisk_hash_family(gi), expect)

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
        # Block bundles carry their own instances, so the default fixture one
        # is usually absent — skip rather than fail deep inside the loader.
        if not (pathlib.Path(bundle) / f"{inst}_const_ext.npy").exists():
            self.skipTest(f"{inst} not in {bundle}; set ZISK_PIL2_INSTANCE")
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
        # `u64`, not `bufs`: the dumped const section is the only one this
        # compares, and `bufs` materializes every other extended section to
        # reach it (cm1_ext alone is 2.4 GB on a Main-width block bundle,
        # and a bundle that dumps only some sections would raise instead).
        np.testing.assert_array_equal(
            np.asarray(key.const_ext).view(np.uint64).reshape(-1),
            cap.u64("const_ext").reshape(-1),
        )


if __name__ == "__main__":
    absltest.main()
