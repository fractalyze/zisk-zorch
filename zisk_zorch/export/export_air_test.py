"""Which families `--air=recursion` resolves to, over a proving-key tree
laid out on disk. No key and no GPU: the selection is a question about file
layout and the shape rule, not about lowering anything.
"""

from __future__ import annotations

import json
import pathlib

from absl.testing import absltest

from zisk_zorch.export.export_air import recursion_families, unservable_family


def _key_tree(root: pathlib.Path, families: list[str]) -> pathlib.Path:
    """A proving key carrying `families` for its one AIR, as `circuit_base`
    reads the layout: the constants are what says a family is there."""
    (root / "pilout.globalInfo.json").write_text(
        json.dumps({"air_groups": ["Zisk"], "airs": [[{"name": "Main"}]]})
    )
    air = root / "zisk" / "Zisk" / "airs" / "Main"
    bases = {
        "compressor": air / "compressor" / "compressor",
        "recursive1": air / "recursive1" / "recursive1",
        "recursive2": root / "zisk" / "Zisk" / "recursive2" / "recursive2",
    }
    for family in families:
        base = bases[family]
        base.parent.mkdir(parents=True, exist_ok=True)
        base.with_name(base.name + ".const").write_bytes(b"")
    return root


class RecursionFamiliesTest(absltest.TestCase):
    def test_a_family_the_key_does_not_ship_is_not_offered(self):
        key = _key_tree(pathlib.Path(self.create_tempdir().full_path), ["recursive2"])
        self.assertEqual(recursion_families(key), ["recursive2"])

    def test_a_family_that_is_not_one_shape_is_left_out_of_the_selection(self):
        # A compressor's starkinfo is per AIR, so `--air=recursion` has to
        # export the families it can rather than refuse the whole key over
        # the one it cannot.
        key = _key_tree(
            pathlib.Path(self.create_tempdir().full_path),
            ["compressor", "recursive1", "recursive2"],
        )
        self.assertEqual(recursion_families(key), ["recursive1", "recursive2"])
        self.assertIsNotNone(unservable_family("compressor"))
        self.assertIsNone(unservable_family("recursive1"))


if __name__ == "__main__":
    absltest.main()
