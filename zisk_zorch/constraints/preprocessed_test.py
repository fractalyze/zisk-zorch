# Copyright 2026 The zisk-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Preprocessed-column reader, against fibonacci-square's `SpecifiedRanges` AIR.

The `.const` input is built by `const_fixture` rather than committed, so the
fixture is readable code instead of an opaque 6 KB blob. Not a ZisK AIR — none
commits a fixed column yet (riscv-witness#2189) and no ZisK key is checked in —
but `.const` is pil2-stark's format, not a per-key one.

`__L1__` decoding as the row-0 Lagrange basis pins the reader's canonical,
row-major reading of what it is handed.

What a generated input cannot pin is the correspondence to pil2 itself — both
the values and the on-disk byte order, since the reader only ever parses what
`const_fixture` wrote. Both belong to `scripts/extract_const_fixture.py
--check`, which diffs the generated payload against a real pil2 bundle.

There is deliberately no LDE assertion here. `extend`'s agreement with pil2's
`extendPol` is already byte-matched against a pil2-derived golden in
`commit/trace_commit_test.py` (`lde.json`, from `tools/fixture-gen`), which is
the package that owns the transform; repeating it here would pin the same
function twice and needed a prover dump to do it.
"""

from __future__ import annotations

import json
import pathlib
import tempfile
from types import SimpleNamespace

import frx

# rw chip code views uint64 as the field dtype; x64 must be on before any array op.
frx.config.update("jax_enable_x64", True)

import frx.numpy as fnp  # noqa: E402
import numpy as np  # noqa: E402
from absl.testing import absltest  # noqa: E402
from zk_dtypes import goldilocks as F  # noqa: E402
from zk_dtypes import goldilocksx3 as F3  # noqa: E402

from zisk_zorch.constraints import const_fixture  # noqa: E402
from zisk_zorch.constraints.preprocessed import (  # noqa: E402
    Preprocessed,
    load_preprocessed,
)
from zisk_zorch.logup.bus import LogUpBus  # noqa: E402

_AIR = "SpecifiedRanges"
_NAMES = const_fixture.COL_NAMES
_N = const_fixture.N_ROWS


def _key_root(tmp: pathlib.Path, const: bytes | None = None) -> pathlib.Path:
    """A generated proving-key root for `_AIR`. `const` overrides the payload,
    for the corruption case."""
    air_dir = tmp / _AIR / "air"
    air_dir.mkdir(parents=True, exist_ok=True)
    (air_dir / f"{_AIR}.starkinfo.json").write_text(
        json.dumps(const_fixture.build_starkinfo())
    )
    payload = const_fixture.build_const_bytes() if const is None else const
    (air_dir / f"{_AIR}.const").write_bytes(payload)
    return tmp


class PreprocessedTest(absltest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.key = _key_root(pathlib.Path(self._tmp.name))

    def test_reads_the_key_in_constpolsmap_order(self) -> None:
        prep = load_preprocessed(_AIR, root=self.key)
        self.assertEqual(prep.names, _NAMES)
        self.assertEqual(prep.values.shape, (_N, len(_NAMES)))
        self.assertEqual(prep.values.dtype, F)

    def test_l1_decodes_as_the_row_0_lagrange_basis(self) -> None:
        l1 = load_preprocessed(_AIR, root=self.key).column("__L1__")
        want = np.zeros(_N, dtype=np.uint64)
        want[0] = 1
        self.assertTrue(bool(fnp.array_equal(l1, fnp.array(want, dtype=F))))

    def test_columns_selects_and_reorders(self) -> None:
        full = load_preprocessed(_AIR, root=self.key)
        picked = load_preprocessed(_AIR, root=self.key, columns=["__L1__", _NAMES[0]])
        self.assertEqual(picked.names, ("__L1__", _NAMES[0]))
        self.assertEqual(picked.values.shape, (_N, 2))
        self.assertTrue(
            bool(fnp.array_equal(picked.column("__L1__"), full.column("__L1__")))
        )
        self.assertTrue(
            bool(fnp.array_equal(picked.column(_NAMES[0]), full.column(_NAMES[0])))
        )

    def test_unknown_column_names_raise(self) -> None:
        with self.assertRaises(KeyError):
            load_preprocessed(_AIR, root=self.key, columns=["CLK_0"])
        with self.assertRaises(KeyError):
            load_preprocessed(_AIR, root=self.key).column("CLK_0")

    def test_size_mismatch_raises(self) -> None:
        """A key that disagrees with its starkinfo must fail, not shear."""
        with tempfile.TemporaryDirectory() as tmp:
            truncated = const_fixture.build_const_bytes()[:-8]
            bogus = _key_root(pathlib.Path(tmp), const=truncated)
            with self.assertRaisesRegex(ValueError, "bytes, expected"):
                load_preprocessed(_AIR, root=bogus)

    def test_feeds_eval_pair_col_as_a_preprocessed_trace(self) -> None:
        """The loop this module closes: #116 raises on a flagged term with no
        preprocessed trace, and what `load_preprocessed` returns is the trace
        that satisfies it."""
        prep = load_preprocessed(_AIR, root=self.key)
        self.assertIsInstance(prep, Preprocessed)
        main = fnp.array(np.zeros((_N, 1), dtype=np.uint64), dtype=F)
        # `1 * prep[__L1__]`, the flag set — so a fallback to `main` could not
        # produce this: column 2 is out of range there.
        vpc = SimpleNamespace(
            constant="0",
            column_weights=[(_NAMES.index("__L1__"), True, "1")],
            column_products=[],
        )

        got = LogUpBus.eval_pair_col(vpc, main, prep.values)
        want = prep.column("__L1__").astype(F3)
        self.assertTrue(bool(fnp.array_equal(got, want)))

        with self.assertRaisesRegex(ValueError, "preprocessed"):
            LogUpBus.eval_pair_col(vpc, main)


if __name__ == "__main__":
    absltest.main()
