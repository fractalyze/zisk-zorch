"""`RwBlockWitness` joins rw bundles to the key's AIRs (#115).

The join, the claim assembly and the shape guards are pinned on stub
bundles — they are pure Python over the proving key's own metadata, and a
stub is the only way to exercise the failure paths (a chip the key does
not have, a trace of the wrong width) that a real generator never emits.

The env-gated case is the one that matters for the contract: a source
built from a real key must be a `WitnessSource`, and its trace must commit
to the same root the capture path does.
"""

from __future__ import annotations

import json
import os
import pathlib

import numpy as np
from absl.testing import absltest
from zk_dtypes import goldilocks as F, pfinfo

from zisk_zorch.harness.capture import CAPTURE_ENV, Capture
from zisk_zorch.harness.rw_source import RwBlockWitness, RwInstanceWitness
from zisk_zorch.harness.verify_proof_layout import starkinfo_for
from zisk_zorch.harness.witness_source import WitnessSource

_NB, _NBE, _NCONST, _NCOLS = 3, 5, 2, 4


class _StubTrace:
    def __init__(self, chip_name: str, words: np.ndarray):
        self.chip_name = chip_name
        self.data = words
        self.num_rows, self.num_cols = words.shape


class _StubBundle:
    """The duck-typed surface `RwBlockWitness` reads off `ZiskTraceBundle`."""

    def __init__(self, traces, num_reals=None, main_air_values=()):
        self.traces = list(traces)
        # rw lists only the chips that actually fired rows.
        self.chip_names = sorted(t.chip_name for t in self.traces if t.num_rows)
        self.num_reals = num_reals or {t.chip_name: t.num_rows for t in self.traces}
        self.main_air_values = np.asarray(main_air_values, dtype=np.uint64)


def _write_key(root: pathlib.Path, air: str, *, air_values=0) -> dict:
    """A minimal example-layout key holding one basic AIR."""
    base = root / "build" / "g0" / "airs" / air / "air"
    base.mkdir(parents=True)
    si = {
        "starkStruct": {"nBits": _NB, "nBitsExt": _NBE},
        "nConstants": _NCONST,
        "mapSectionsN": {"cm1": _NCOLS},
    }
    if air_values:
        si["airValuesMap"] = [{"stage": 1} for _ in range(air_values)]
    (base / f"{air}.starkinfo.json").write_text(json.dumps(si))
    (base / f"{air}.expressionsinfo.json").write_text(json.dumps({}))
    rng = np.random.default_rng(11)
    rng.integers(
        0, pfinfo(F).modulus, size=(1 << _NB, _NCONST), dtype=np.uint64
    ).tofile(base / f"{air}.const")
    (base / f"{air}.verkey.json").write_text(json.dumps([1, 2, 3, 4]))
    return {"air_groups": ["g0"], "airs": [[{"name": air}]]}


def _words(rows=1 << _NB, cols=_NCOLS) -> np.ndarray:
    rng = np.random.default_rng(5)
    return rng.integers(0, pfinfo(F).modulus, size=(rows, cols), dtype=np.uint64)


class RwSourceTest(absltest.TestCase):
    def _key(self, air="TinyAir", **kw):
        root = pathlib.Path(self.create_tempdir().full_path)
        return root, _write_key(root, air, **kw)

    def test_joins_chips_to_airs_and_orders_by_family(self):
        root, gi = self._key()
        # Two segments of the same chip: both must appear, in segment order.
        bundles = [
            _StubBundle([_StubTrace("TinyAir", _words())]) for _ in range(2)
        ]
        srcs = RwBlockWitness(
            bundles,
            root,
            gi,
            publics=np.zeros(1, dtype=np.uint64),
            proofvalues=np.zeros(0, dtype=np.uint64),
        ).sources()
        self.assertLen(srcs, 2)
        self.assertEqual([f for f, _, _ in srcs], ["TinyAir", "TinyAir"])
        self.assertEqual(
            [w.instance for _, w, _ in srcs], ["rw_air0_seg0", "rw_air0_seg1"]
        )
        np.testing.assert_array_equal(srcs[0][2], np.array([1, 2, 3, 4], np.uint64))

    def test_source_satisfies_the_witness_protocol(self):
        root, gi = self._key()
        (_, witness, _), = RwBlockWitness(
            [_StubBundle([_StubTrace("TinyAir", _words())])],
            root,
            gi,
            publics=np.zeros(1, dtype=np.uint64),
            proofvalues=np.zeros(0, dtype=np.uint64),
        ).sources()
        self.assertIsInstance(witness, WitnessSource)
        claim = witness.pil2_claim()
        self.assertEqual(claim.n_bits, _NB)
        self.assertEqual(claim.n_cols, _NCOLS)
        self.assertEqual(claim.airvalues.size, 0)

    def test_trace_words_is_the_bundle_buffer_not_a_copy(self):
        # The whole point of the intake: no read, no copy.
        root, gi = self._key()
        words = _words()
        (_, witness, _), = RwBlockWitness(
            [_StubBundle([_StubTrace("TinyAir", words)])],
            root,
            gi,
            publics=np.zeros(1, dtype=np.uint64),
            proofvalues=np.zeros(0, dtype=np.uint64),
        ).sources()
        self.assertIs(witness.trace_words(), words)

    def test_release_then_read_fails_loudly(self):
        root, gi = self._key()
        (_, witness, _), = RwBlockWitness(
            [_StubBundle([_StubTrace("TinyAir", _words())])],
            root,
            gi,
            publics=np.zeros(1, dtype=np.uint64),
            proofvalues=np.zeros(0, dtype=np.uint64),
        ).sources()
        witness.release()
        # An rw source cannot go back to a dump, so a post-release read is
        # a driver bug, not something to paper over with an empty matrix.
        with self.assertRaises(RuntimeError):
            witness.trace_words()

    def test_empty_traces_are_skipped_via_chip_names(self):
        root, gi = self._key()
        bundle = _StubBundle(
            [_StubTrace("TinyAir", _words()), _StubTrace("Ghost", _words(0, _NCOLS))]
        )
        srcs = RwBlockWitness(
            [bundle],
            root,
            gi,
            publics=np.zeros(1, dtype=np.uint64),
            proofvalues=np.zeros(0, dtype=np.uint64),
        ).sources()
        # `Ghost` fired no rows, so it is absent from chip_names and must
        # not be joined — a key lookup for it would fail.
        self.assertLen(srcs, 1)

    def test_unknown_chip_fails_loudly(self):
        root, gi = self._key()
        with self.assertRaises(KeyError):
            RwBlockWitness(
                [_StubBundle([_StubTrace("NotAnAir", _words())])],
                root,
                gi,
                publics=np.zeros(1, dtype=np.uint64),
                proofvalues=np.zeros(0, dtype=np.uint64),
            ).sources()

    def test_wrong_trace_width_fails_loudly(self):
        root, gi = self._key()
        with self.assertRaises(ValueError):
            RwBlockWitness(
                [_StubBundle([_StubTrace("TinyAir", _words(cols=_NCOLS + 1))])],
                root,
                gi,
                publics=np.zeros(1, dtype=np.uint64),
                proofvalues=np.zeros(0, dtype=np.uint64),
            ).sources()

    def test_non_main_airvalues_is_not_silently_zeroed(self):
        # An AIR that needs stage-1 air values and has no rw supply must
        # fail, not prove against zeros (rw#2360 is Main-only).
        root, gi = self._key(air="Other", air_values=3)
        with self.assertRaises(NotImplementedError):
            RwBlockWitness(
                [_StubBundle([_StubTrace("Other", _words())])],
                root,
                gi,
                publics=np.zeros(1, dtype=np.uint64),
                proofvalues=np.zeros(0, dtype=np.uint64),
            ).sources()

    def test_commits_to_the_same_root_as_the_capture_path(self):
        """The contract claim: same trace words in, same stage-1 root out."""
        bundle_dir = os.environ.get(CAPTURE_ENV, "")
        key_dir = os.environ.get("ZISK_PROVING_KEY", "")
        if not bundle_dir or not pathlib.Path(bundle_dir).is_dir():
            self.skipTest(f"no capture: set {CAPTURE_ENV}")
        if not key_dir or not pathlib.Path(key_dir).is_dir():
            self.skipTest("no proving key: set ZISK_PROVING_KEY")
        inst = os.environ.get("ZISK_PIL2_INSTANCE", "")
        if not inst or not (pathlib.Path(bundle_dir) / f"{inst}_trace.npy").exists():
            self.skipTest("set ZISK_PIL2_INSTANCE to an instance in the bundle")

        import frx.numpy as fnp

        from zisk_zorch.commit.trace_commit import commit_trace

        key_dir = pathlib.Path(key_dir)
        gi = json.loads((key_dir / "pilout.globalInfo.json").read_text())
        cap = Capture(
            pathlib.Path(bundle_dir), inst, starkinfo_for(key_dir, gi, inst)
        )
        air = cap.si["name"]
        if cap.si.get("airValuesMap") and air != "Main":
            self.skipTest(f"{air} needs air values rw does not supply yet")

        words = cap.trace_words()
        bundle = _StubBundle(
            [_StubTrace(air, words)],
            main_air_values=(
                cap.u64("airvalues") if cap.si.get("airValuesMap") else ()
            ),
        )
        (_, witness, _), = RwBlockWitness(
            [bundle],
            key_dir,
            gi,
            publics=cap.u64("publics"),
            proofvalues=cap.u64("proofvalues"),
        ).sources()
        self.assertIsInstance(witness, WitnessSource)
        self.assertEqual(witness.hash_family, cap.hash_family)

        ss = cap.si["starkStruct"]
        roots = [
            commit_trace(
                fnp.array(w.view(F)),
                blowup=1 << (ss["nBitsExt"] - ss["nBits"]),
                arity=ss["merkleTreeArity"],
                hash_family=cap.hash_family,
            ).root
            for w in (witness.trace_words(), words)
        ]
        self.assertTrue(bool(fnp.array_equal(*roots)), "rw-source root differs")


if __name__ == "__main__":
    absltest.main()
