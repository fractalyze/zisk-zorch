"""Pins how `pil2_timers.py` reads pil2's `-vv` timer blocks: telling a basic
instance's proof from the recursive proofs over it, keeping commit and proof
apart, keeping the instances of one air apart, and naming the airs. The
fixture is a real run — see testdata/README.md."""

import contextlib
import io
import pathlib

from absl.testing import absltest, parameterized

from bridge.bench import pil2_timers

LOG = pathlib.Path("bridge/bench/testdata/pil2_prove.log")
GLOBAL_INFO = pathlib.Path("bridge/bench/testdata/pilout.globalInfo.json")

MAIN = (0, 0)
ROM_DATA = (0, 4)

# Two instances of one air, the case the fixture cannot carry: the guest it
# came from runs one instance per air, while the sha-hasher workload of
# docs/bridge.md runs 13 Main. Shaped like the fixture's blocks, trimmed to
# one phase and one category each.
TWO_MAIN_INSTANCES = """\
proofman_common::proof_ctx INFO: Using 1 streams per GPU for basic proofs
[TRACE] PilStark: TIMERS FOR INSTANCE ID 0 [0:0]
[TRACE] PilStark: <-- STARK_GPU_COMMIT : 0.100000 s
[TRACE] PilStark:      KERNELS CONTRIBUTIONS:
[TRACE] PilStark:         MERKLE_TREE    :  0.1000s (100.00%)
[TRACE] PilStark: TIMERS FOR INSTANCE ID 1 [0:0]
[TRACE] PilStark: <-- STARK_GPU_COMMIT : 0.300000 s
[TRACE] PilStark:      KERNELS CONTRIBUTIONS:
[TRACE] PilStark:         MERKLE_TREE    :  0.3000s (100.00%)
2026-09-07T16:49:47.137926Z proofman::proofman DEBUG: >>> GEN_PROOF_0 [0:0]
2026-09-07T16:49:47.328433Z proofman::proofman DEBUG: >>> GEN_PROOF_1 [0:0]
"""


class Pil2TimersTest(parameterized.TestCase):

    def setUp(self):
        super().setUp()
        self.basic, self.recursive = pil2_timers.parse(LOG.read_text())

    def test_only_the_instances_proofman_proved_as_basic_are_kept(self):
        self.assertCountEqual(self.basic, [(1, MAIN), (3, ROM_DATA)])

    @parameterized.named_parameters(
        ("main", (1, MAIN), 0.150212, 0.546010),
        ("rom_data", (3, ROM_DATA), 0.019420, 0.073162),
    )
    def test_commit_and_proof_are_kept_apart(self, key, commit, proof):
        self.assertAlmostEqual(self.basic[key].seconds["COMMIT"], commit)
        self.assertAlmostEqual(self.basic[key].seconds["PROOF"], proof)

    def test_the_later_proofs_of_an_instance_are_recursive(self):
        # Main's instance carries four blocks: a commit, the basic proof, and
        # the Recursive1 and Recursive2 proofs above it, all under the same
        # `ID 1 [0:0]` header.
        self.assertAlmostEqual(self.recursive.seconds["PROOF"], 1.417352)
        self.assertEqual(self.recursive.seconds["COMMIT"], 0)

    def test_categories_are_summed_under_their_instance(self):
        main = self.basic[(1, MAIN)]
        self.assertAlmostEqual(main.categories["MERKLE_TREE"], 0.2706)
        self.assertAlmostEqual(main.categories["EXPRESSIONS"], 0.2404)
        self.assertEqual(main.hottest(2), "MERKLE_TREE 0.271, EXPRESSIONS 0.240")

    def test_instances_of_one_air_stay_apart(self):
        basic, _ = pil2_timers.parse(TWO_MAIN_INSTANCES)
        self.assertCountEqual(basic, [(0, MAIN), (1, MAIN)])
        self.assertAlmostEqual(basic[(0, MAIN)].total(), 0.1)
        self.assertAlmostEqual(basic[(1, MAIN)].total(), 0.3)

    def test_by_air_keeps_the_instances_a_row_sums(self):
        basic, _ = pil2_timers.parse(TWO_MAIN_INSTANCES)
        rows = pil2_timers.by_air(basic)
        self.assertCountEqual(rows, [MAIN])
        self.assertLen(rows[MAIN], 2)

    def test_a_row_carries_its_instance_count_and_average(self):
        # Without the count the 0.400 s sum reads as one Main instance's cost.
        log = pathlib.Path(self.create_tempfile(content=TWO_MAIN_INSTANCES).full_path)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            pil2_timers.report(log, {MAIN: "Main"}, 1)
        header, row, whole = out.getvalue().splitlines()
        self.assertEndsWith(header, "2 basic instances")
        self.assertIn("Main                 x2   commit 0.400 s", row)
        self.assertIn("total 0.400 s (0.200/instance)", row)
        self.assertIn("all 2", whole)

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
