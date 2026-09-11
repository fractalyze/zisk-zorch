"""What `pil2_layout.py` must get right about pil2's per-stream buffer.

The tool re-implements vendor arithmetic, which is only safe while it is
checked, so the cases here are the ways it could drift and still print a
plausible table: an overlap counted as an addition, the wrong constant-tree
branch, a stage-1 value sized as an extension element."""

from absl.testing import absltest

from bridge.bench import pil2_layout

GIB = 1 << 30


def starkinfo(*, n_bits=10, n_constants=3, cm1=38, cm2=24, cm3=6, arity=3):
    """A minimal starkinfo with the fields the layout reads."""
    return {
        "name": "Fake",
        "nStages": 2,
        "nConstants": n_constants,
        "nPublics": 68,
        "qDeg": 2,
        "mapSectionsN": {"const": n_constants, "cm1": cm1, "cm2": cm2, "cm3": cm3},
        "customCommits": [],
        "boundaries": [{"name": "everyRow"}],
        "openingPoints": [-1, 0, 1],
        "evMap": [{}] * 61,
        "challengesMap": [{}] * 6,
        "proofValuesMap": [{"stage": 1}] * 8,
        "airgroupValuesMap": [{"stage": 2}],
        "airValuesMap": [{"stage": 1}] * 197,
        "starkStruct": {
            "nBits": n_bits,
            "nBitsExt": n_bits + 1,
            "merkleTreeArity": arity,
            "lastLevelVerification": 1,
            "nQueries": 230,
            "steps": [
                {"nBits": n_bits + 1},
                {"nBits": n_bits - 2},
                {"nBits": n_bits - 5},
            ],
        },
    }


class OverlapTest(absltest.TestCase):
    def test_the_base_trace_is_placed_where_the_next_extended_section_goes(self):
        # This is the whole point of the comparison: pil2 does not release the
        # base trace before building `cm2_ext`, it writes `cm2_ext` over it.
        # Two sections at one offset is what a bridge lifetime fix has to
        # reproduce by other means.
        offsets = {
            name: off for name, off, _ in pil2_layout.Layout(starkinfo()).sections
        }
        self.assertEqual(offsets["cm1 (base, shares cm2_ext)"], offsets["cm2_ext"])
        self.assertEqual(
            offsets["cm2 (base, shares cm3_ext)"], offsets["cm3_ext (qsec)"]
        )

    def test_the_buffer_is_smaller_than_the_sections_it_holds(self):
        # The consequence of those overlaps, and the one number an attribution
        # is built on. A layout that summed the sections would report a pil2
        # need larger than pil2's own figure, making the bridge's excess look
        # smaller than it is.
        layout = pil2_layout.Layout(starkinfo())
        self.assertLess(layout.total, sum(size for _, _, size in layout.sections))

    def test_a_base_trace_wider_than_what_covers_it_does_grow_the_buffer(self):
        # The other side of the same `max`: the overlap is free only up to the
        # extended section's length. A tool that always took the extended
        # section would under-report a cm1-dominated AIR.
        covered = pil2_layout.Layout(starkinfo(cm1=24, cm2=24))
        overflowing = pil2_layout.Layout(starkinfo(cm1=200, cm2=24))
        self.assertGreater(overflowing.total, covered.total)


class ConstTreeTest(absltest.TestCase):
    def test_the_tree_branch_is_worth_the_whole_tree(self):
        # pil2 decides per AIR whether the constant tree lives in every
        # stream's buffer or once per GPU outside it. Assuming one branch is
        # wrong by the tree, which for VirtualTableZisk0 is 2.9 GiB.
        si = starkinfo(n_constants=88)
        outside = pil2_layout.Layout(si, const_tree_in_buffer=False)
        inside = pil2_layout.Layout(si, const_tree_in_buffer=True)
        tree = (1 << 11) * 88 + pil2_layout.num_nodes_mt(1 << 11, 3)
        self.assertEqual(inside.total - outside.total, tree)

    def test_the_report_says_which_branch_it_used(self):
        # A table without it cannot be compared to anything: the same AIR has
        # two different per-stream needs depending on the branch.
        self.assertIn("once per GPU", pil2_layout.Layout(starkinfo()).report())
        self.assertIn(
            "in every stream's buffer",
            pil2_layout.Layout(starkinfo(), const_tree_in_buffer=True).report(),
        )


class ValuesSizeTest(absltest.TestCase):
    def test_stage_one_values_are_one_element_and_later_ones_three(self):
        self.assertEqual(pil2_layout.values_size([{"stage": 1}, {"stage": 2}]), 1 + 3)


class NumNodesTest(absltest.TestCase):
    def test_a_tree_over_one_leaf_is_one_hash(self):
        self.assertEqual(pil2_layout.num_nodes_mt(1, 3), 4)

    def test_levels_are_padded_up_to_the_arity(self):
        # 4 leaves at arity 3: 4 -> pad to 6 -> 2 -> pad to 3 -> 1.
        # Dropping the padding is a silent undercount that grows with height.
        self.assertEqual(pil2_layout.num_nodes_mt(4, 3), (4 + 2 + 2 + 1 + 1) * 4)


class ExpectTest(absltest.TestCase):
    def _path(self, si):
        import json

        return self.create_tempfile("si.json", content=json.dumps(si)).full_path

    def test_a_figure_matching_neither_branch_prints_no_table(self):
        # The failure mode this guard exists for: a drifted layout still
        # prints a well-formed section table, and a table is believed.
        self.assertEqual(
            pil2_layout.main([self._path(starkinfo()), "--expect", "99GiB"]), 1
        )

    def test_pil2s_own_figure_selects_the_branch_rather_than_a_flag(self):
        # Full size, so the two branches differ by far more than the
        # tolerance -- at toy sizes both match and the choice is not tested.
        si = starkinfo(n_bits=21, n_constants=88)
        inside = pil2_layout.Layout(si, const_tree_in_buffer=True)
        expect = f"{inside.total * 8 / GIB:.2f}GiB"
        self.assertEqual(pil2_layout.main([self._path(si), "--expect", expect]), 0)


if __name__ == "__main__":
    absltest.main()
