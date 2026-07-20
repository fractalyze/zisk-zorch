"""Byte-match of the pil2 transcript against pil2-proofman's `Transcript`,
replaying the scripted absorb/squeeze sequence per width and comparing every
squeezed value."""

from __future__ import annotations

import pathlib

import frx.numpy as fnp
import numpy as np
from absl.testing import absltest

from zisk_zorch.golden import load, u64
from zisk_zorch.transcript.transcript import Transcript

_GOLDEN = pathlib.Path(__file__).parent / "testdata" / "golden" / "transcript.json"


class TranscriptTest(absltest.TestCase):
    def test_matches_pil2_reference(self) -> None:
        for entry in load(_GOLDEN)["widths"]:
            width = entry["width"]
            t = Transcript(width)
            for step in entry["steps"]:
                op = step["op"]
                if op == "put":
                    t.put(u64(step["values"]))
                elif op == "get_field":
                    out = t.get_field()
                    self.assertTrue(
                        bool(fnp.array_equal(out, u64(step["output"]))),
                        msg=f"width {width} get_field",
                    )
                elif op == "get_fields1_x5":
                    out = fnp.stack([t.get_fields1() for _ in range(5)])
                    self.assertTrue(
                        bool(fnp.array_equal(out, u64(step["output"]))),
                        msg=f"width {width} get_fields1",
                    )
                elif op == "get_permutations":
                    out = t.get_permutations(step["n"], step["n_bits"])
                    expected = np.array(
                        [int(v) for v in step["output"]], dtype=np.uint64
                    )
                    self.assertTrue(
                        bool(np.array_equal(out, expected)),
                        msg=f"width {width} get_permutations",
                    )
                elif op == "get_state":
                    out = t.get_state()
                    self.assertTrue(
                        bool(fnp.array_equal(out, u64(step["output"]))),
                        msg=f"width {width} get_state",
                    )
                else:
                    self.fail(f"unknown golden op {op}")


if __name__ == "__main__":
    absltest.main()
