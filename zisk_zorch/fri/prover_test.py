"""Byte-match of the full FRI prover against pil2's fold/merkelize/query loop.

The golden runs the reference fold chain over one random FRI polynomial,
driving challenges through the pil2 transcript (a fixed seed absorb stands in
for the proof state preceding FRI). Each case checks the layer roots, the final
polynomial sent in clear, and every layer opening at the fixed queries.
"""

from __future__ import annotations

import pathlib

import frx.numpy as jnp
from absl.testing import absltest

from zisk_zorch.fri.prover import prove, prove_queries
from zisk_zorch.golden import load, u64, u64x3
from zisk_zorch.transcript.transcript import Transcript

_TESTDATA = pathlib.Path(__file__).parent / "testdata" / "golden"


class FriProveTest(absltest.TestCase):
    def test_matches_pil2_fri_prover(self) -> None:
        for case in load(_TESTDATA / "fri_prove.json")["cases"]:
            with self.subTest(steps=case["steps"], arity=case["arity"]):
                transcript = Transcript()  # width 12 == pil2 transcriptArity 3
                transcript.put(u64(case["seed"]))
                proof = prove(
                    u64x3(case["init_pol"]),
                    case["steps"],
                    arity=case["arity"],
                    transcript=transcript,
                )

                self.assertTrue(
                    bool(jnp.array_equal(proof.final_pol, u64x3(case["final_pol"]))),
                    msg="final polynomial",
                )
                self.assertEqual(len(proof.roots), len(case["roots"]))
                for layer, (got, want) in enumerate(zip(proof.roots, case["roots"])):
                    self.assertTrue(
                        bool(jnp.array_equal(got, u64(want))), msg=f"root {layer}"
                    )

                openings = prove_queries(proof, [q["query"] for q in case["queries"]])
                for q, per_layer in zip(case["queries"], openings):
                    self.assertEqual(len(per_layer), len(q["layers"]))
                    for layer, (got, want) in enumerate(zip(per_layer, q["layers"])):
                        self.assertTrue(
                            bool(jnp.array_equal(got, u64(want["proof"]))),
                            msg=f"query {q['query']} layer {layer}",
                        )


if __name__ == "__main__":
    absltest.main()
