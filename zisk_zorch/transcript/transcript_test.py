"""Byte-match of the pil2 transcript against pil2-proofman's `Transcript`,
replaying the scripted absorb/squeeze sequence per width and comparing every
squeezed value."""

from __future__ import annotations

import pathlib

import frx.numpy as fnp
import numpy as np
from absl.testing import absltest

from zisk_zorch.golden import load, u64
from zisk_zorch.transcript.transcript import (
    DIGEST,
    WIDTH,
    Transcript,
    transcript_hash,
)

_GOLDEN = pathlib.Path(__file__).parent / "testdata" / "golden" / "transcript.json"


class TranscriptTest(absltest.TestCase):
    def test_matches_pil2_reference(self) -> None:
        for entry in load(_GOLDEN)["widths"]:
            width = entry["width"]
            family = entry.get("hash_family", "Poseidon2")
            t = Transcript(width, family)
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

    def test_transcript_hash_matches_eager_transcript(self) -> None:
        # The scan-jit path must reproduce the eager put/get_state discipline
        # at every block-boundary shape: empty, mid-block, exact multiples,
        # one past a multiple, and a section-sized buffer.
        rng = np.random.default_rng(7)
        rate = WIDTH - DIGEST
        for family in ("Poseidon2", "Poseidon1"):
            for n in (0, 1, rate - 1, rate, rate + 1, 3 * rate, 462):
                values = u64(
                    rng.integers(0, (1 << 64) - (1 << 32), n, dtype=np.uint64)
                )
                t = Transcript(WIDTH, family)
                t.put(values)
                expected = t.get_state()[:DIGEST]
                got = transcript_hash(values, WIDTH, family)
                self.assertTrue(
                    bool(fnp.array_equal(got, expected)),
                    msg=f"{family} n={n}",
                )

    def test_rejects_a_width_pil2_never_sponges(self) -> None:
        # Width 4 is the one that has to raise rather than build: Poseidon2
        # carries it for the grinding predicate, so the permutation accepts it
        # and the sponge would come back with rate 0 and never flush.
        with self.assertRaises(ValueError):
            Transcript(4)
        with self.assertRaises(ValueError):
            transcript_hash(u64(np.arange(3, dtype=np.uint64)), 4)


if __name__ == "__main__":
    absltest.main()
