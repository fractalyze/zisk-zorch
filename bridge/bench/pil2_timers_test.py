"""Pins how `pil2_timers.py` reads pil2's `-vv` timer blocks: telling a basic
instance's proof from the recursive proofs over it, keeping commit and proof
apart, and naming the airs. The fixture is a real run — see
testdata/README.md."""

import pathlib

from absl.testing import absltest, parameterized

from bridge.bench import pil2_timers

LOG = pathlib.Path("bridge/bench/testdata/pil2_prove.log")
GLOBAL_INFO = pathlib.Path("bridge/bench/testdata/pilout.globalInfo.json")

MAIN = (0, 0)
ROM_DATA = (0, 4)


class Pil2TimersTest(parameterized.TestCase):

    def setUp(self):
        super().setUp()
        self.basic, self.recursive = pil2_timers.parse(LOG.read_text())

    def test_only_the_airs_proofman_proved_as_basic_instances(self):
        self.assertCountEqual(self.basic, [MAIN, ROM_DATA])

    @parameterized.named_parameters(
        ("main", MAIN, 0.150212, 0.546010),
        ("rom_data", ROM_DATA, 0.019420, 0.073162),
    )
    def test_commit_and_proof_are_kept_apart(self, air, commit, proof):
        self.assertAlmostEqual(self.basic[air].seconds["COMMIT"], commit)
        self.assertAlmostEqual(self.basic[air].seconds["PROOF"], proof)

    def test_the_later_proofs_of_an_instance_are_recursive(self):
        # Main's air carries four blocks: a commit, the basic proof, and the
        # Recursive1 and Recursive2 proofs above it, all under the same
        # `ID 1 [0:0]` header.
        self.assertAlmostEqual(self.recursive.seconds["PROOF"], 1.417352)
        self.assertEqual(self.recursive.seconds["COMMIT"], 0)

    def test_categories_are_summed_under_their_instance(self):
        main = self.basic[MAIN]
        self.assertAlmostEqual(main.categories["MERKLE_TREE"], 0.2706)
        self.assertAlmostEqual(main.categories["EXPRESSIONS"], 0.2404)
        self.assertEqual(main.hottest(2), "MERKLE_TREE 0.271, EXPRESSIONS 0.240")

    def test_air_names_come_from_the_global_info(self):
        names = pil2_timers.air_names(GLOBAL_INFO)
        self.assertEqual(names[MAIN], "Main")
        self.assertEqual(names[ROM_DATA], "RomData")

    def test_no_global_info_means_no_names(self):
        self.assertEqual(pil2_timers.air_names(None), {})

    @parameterized.parameters(
        ("Using 1 streams per GPU for basic proofs and 0 streams", 1),
        ("Using 3 streams per GPU for basic proofs and 0 streams", 3),
        ("no such line", None),
    )
    def test_streams_is_read_for_the_interleaving_warning(self, log, expected):
        self.assertEqual(pil2_timers.streams(log), expected)

    def test_the_fixture_is_a_single_stream_run(self):
        self.assertEqual(pil2_timers.streams(LOG.read_text()), 1)


if __name__ == "__main__":
    absltest.main()
