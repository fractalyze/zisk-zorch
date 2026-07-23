"""Byte-match of the stage-1 commit against a real ZisK program trace.

Unlike `trace_commit_test` (synthetic seeded fixtures at toy heights), this
consumes a full-program witness trace produced by the native ZisK state
machines — the go-program-hello-world guest's InputData AIR at its real
proving-key shape (2^21 rows x 9 cols, blowup 2, arity 4) — and matches the
root the native pil2-stark prover committed for that exact trace.

The fixture pair under `testdata/fullprogram/<air>/` comes from the
fractalyze/zisk fork's `rw-fixture-gen` (see the README's "Real-program
stage-1 fixtures" recipe): `expected_*_trace.npy.gz` is the trace dump whose
payload is the fixture's `golden_sha256` preimage, and `stage1_commit.json`
carries the native `commit_witness` root plus the starkStruct params. Roots
are captured with `--hash-family Poseidon2` — the family this repo models;
the installed ZisK proving key's own default is currently Poseidon1 (its
globalInfo carries no `hash` field), which zisk-zorch does not implement.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import pathlib
import struct

import frx.numpy as fnp
import numpy as np
from absl.testing import absltest
from zk_dtypes import goldilocks as F

from zisk_zorch.commit.trace_commit import commit_trace
from zisk_zorch.golden import u64

_TESTDATA = pathlib.Path(__file__).parent / "testdata" / "fullprogram"


def _load_trace_npy_gz(path: pathlib.Path, rows: int, cols: int) -> tuple[np.ndarray, str]:
    """Decompress a rw-fixture-gen trace dump; return (u64 matrix, payload sha256).

    The payload after the v1 npy header is row-major canonical u64 LE — exactly
    the bytes the fixture's `golden_sha256` hashes.
    """
    raw = gzip.open(path, "rb").read()
    assert raw[:6] == b"\x93NUMPY" and raw[6] == 1, f"{path} is not a v1 .npy"
    (header_len,) = struct.unpack("<H", raw[8:10])
    payload = raw[10 + header_len :]
    assert len(payload) == rows * cols * 8, (
        f"{path}: {len(payload)} payload bytes, expected {rows}x{cols} u64s"
    )
    sha = hashlib.sha256(payload).hexdigest()
    return np.frombuffer(payload, dtype="<u8").reshape(rows, cols), sha


class FullProgramCommitTest(absltest.TestCase):
    def test_matches_native_stage1_roots(self) -> None:
        fixture_dirs = sorted(d for d in _TESTDATA.iterdir() if d.is_dir())
        self.assertNotEmpty(fixture_dirs)
        for fixture in fixture_dirs:
            with self.subTest(fixture.name):
                self._match_one(fixture)

    def _match_one(self, fixture: pathlib.Path) -> None:
        meta = json.loads((fixture / "fixture_metadata.json").read_text())
        s1 = json.loads((fixture / "stage1_commit.json").read_text())

        # The native root was hashed with this family; a mismatch means the
        # fixture was regenerated without --hash-family Poseidon2 and cannot
        # match this repo's Poseidon2 pipeline.
        self.assertEqual(s1["hash_family"], "Poseidon2")
        # A committed-trace hash differing from the dump means the native
        # prover's witness_calc hints rewrote columns before the LDE — the
        # dump would no longer be the committed matrix.
        self.assertEqual(s1["committed_trace_sha256"], s1["trace_sha256"])

        rows, cols = s1["trace_rows"], s1["trace_cols"]
        trace_u64, payload_sha = _load_trace_npy_gz(
            next(fixture.glob("expected_*_trace.npy.gz")), rows, cols
        )
        self.assertEqual(payload_sha, meta["golden_sha256"], msg="stale trace dump")

        trace = fnp.array(trace_u64, dtype=F)
        commitment = commit_trace(
            trace,
            blowup=1 << s1["blowup_bits"],
            arity=s1["merkle_tree_arity"],
        )
        self.assertTrue(
            bool(fnp.array_equal(commitment.root, u64(s1["root"]))),
            msg=f"{fixture.name}: root mismatch vs native {s1['air']}",
        )


if __name__ == "__main__":
    absltest.main()
