"""What `buffer_assignment.py` must get right about XLA's dump.

The ways a reader of this dump can print a plausible table and be wrong:

- **Charging an input to the transient.** A program's parameters are buffers
  the client already holds; counting them as what the execution adds would
  double the term the whole unit exists to size.
- **Dropping an allocation kind.** The dump has five, and a regex that misses
  one understates the transient silently. The guard is XLA's own printed
  total, which the parse must reproduce.
- **Taking the platform tag for part of the program's name.** The tag carries
  a dot (`sm_12.0a`), so a name split on "." keeps half of it.
- **Guessing which program a module is.** Every exported module is named
  `jit_fn`, so the name cannot say, and the id is a compile-order artefact.
  The manifest's declared shapes are the signature; a signature two programs
  share has to be reported as ambiguous rather than resolved, because putting
  one program's transient under another's name is the whole quantity wrong.
- **Reporting a live range with no sequence.** `3-4` is a position in an
  instruction sequence, not a duration; without the sequence length it cannot
  be read.
"""

import pathlib
import shutil

from absl.testing import absltest

from bridge.bench import buffer_assignment

MIB = 1 << 20
# The fixture: the shipped plugin's own dump of a 512x512 matmul-plus-scale,
# see testdata/README.md. Small on purpose -- what is under test is the
# format, which belongs to the plugin rather than to the program.
DUMP = pathlib.Path(__file__).parent / "testdata" / "xla_dump"
MODULE_ID = 5  # module_0005 in the fixture's filenames
# The authoritative fixture: one real bridge executable, VirtualTableZisk0's
# `commit2`, laid out as the per-program tree the dump is taken in. It carries
# what the toy above cannot -- a returned tuple and its index table, 28
# load-time constants, and an arena whose regions are an NTT's stage buffers.
TREE = pathlib.Path(__file__).parent / "testdata" / "xla_dump_tree"
COMMIT2_IN = [603979776]
COMMIT2_OUT = [
    32,
    32,
    128,
    512,
    2048,
    8192,
    32768,
    131072,
    524288,
    2097152,
    8388608,
    33554432,
    134217728,
    1207959552,
]


class ParseTest(absltest.TestCase):
    def test_modules_are_keyed_by_id_not_name(self):
        # Every bridge program's module is called `jit_fn`, so the id is the
        # only thing that tells 34 executables apart.
        self.assertEqual(buffer_assignment.modules(DUMP), [MODULE_ID])

    def test_module_name_excludes_the_platform_tag(self):
        # `sm_12.0a` carries a dot, so a name split on "." keeps half of it.
        self.assertEqual(
            buffer_assignment.parse_executable(DUMP, MODULE_ID).module, "jit__lambda"
        )

    def test_allocations_reproduce_xlas_own_total(self):
        # The fixture's report says 6291496 B; parse_executable raises unless
        # the allocations sum to it, so reaching this line is the check.
        executable = buffer_assignment.parse_executable(DUMP, MODULE_ID)
        self.assertEqual(executable.reported_total, 6291496)
        self.assertEqual(
            sum(a.size for a in executable.allocations), executable.reported_total
        )

    def test_a_dropped_allocation_is_refused_not_reported(self):
        # Mutate the real fixture rather than check in a corrupted one: drop
        # the temp arena's header, which is what a regex miss would do.
        d = pathlib.Path(self.create_tempdir().full_path)
        for path in DUMP.iterdir():
            shutil.copy(path, d)
        assignment = next(d.glob("*-buffer-assignment.txt"))
        kept = [
            line
            for line in assignment.read_text().splitlines()
            if "preallocated-temp" not in line
        ]
        assignment.write_text("\n".join(kept) + "\n")
        with self.assertRaisesRegex(ValueError, "XLA's own report"):
            buffer_assignment.parse_executable(d, MODULE_ID)

    def test_inputs_are_not_counted_as_what_the_execution_adds(self):
        executable = buffer_assignment.parse_executable(DUMP, MODULE_ID)
        # One 1 MiB parameter, a 5 MiB temp arena, a 4 B output.
        self.assertEqual(executable.parameter_bytes, 1 * MIB)
        self.assertEqual(executable.temp_bytes, 5 * MIB)
        self.assertEqual(executable.output_bytes, 4)
        self.assertEqual(
            executable.added_bytes, executable.temp_bytes + executable.output_bytes
        )

    def test_load_time_allocations_are_not_what_an_execution_adds(self):
        # The fixture's nine thread-local allocations, and a bridge program's
        # constants, are placed once for the executable rather than per run.
        # Counting them in `added_bytes` left the reconciliation short by
        # exactly their total.
        executable = buffer_assignment.parse_executable(DUMP, MODULE_ID)
        self.assertGreater(executable.other_bytes, 0)
        self.assertEqual(
            executable.added_bytes, executable.temp_bytes + executable.output_bytes
        )
        # Still in the sum that reproduces XLA's own total.
        self.assertEqual(
            sum(a.size for a in executable.allocations),
            executable.added_bytes
            + executable.parameter_bytes
            + executable.other_bytes,
        )

    def test_shared_allocation_is_not_the_sum_of_its_values(self):
        executable = buffer_assignment.parse_executable(DUMP, MODULE_ID)
        temp = next(a for a in executable.allocations if a.is_temp)
        # Seven values share the arena at overlapping offsets, so their sizes
        # sum past it; the allocation's own size is the one to quote.
        self.assertGreater(sum(v.size for v in temp.values), temp.size)
        self.assertEqual(temp.size, 5 * MIB)

    def test_live_range_comes_with_the_sequence_it_indexes(self):
        executable = buffer_assignment.parse_executable(DUMP, MODULE_ID)
        dot = next(
            v for v, _ in executable.temp_values() if v.name == "gemm_fusion_dot"
        )
        self.assertEqual(executable.live_range(dot), (3, 4))
        self.assertEqual(len(executable.sequence), 12)
        self.assertEqual(executable.sequence[3], "gemm_fusion_dot")

    def test_a_warm_cache_dump_is_named_as_such(self):
        empty = pathlib.Path(self.create_tempdir().full_path)
        with self.assertRaisesRegex(FileNotFoundError, "ZZ_COMPILE_CACHE"):
            buffer_assignment.parse_executable(empty, MODULE_ID)


def spec(nbytes: int, dtype: str = "uint64") -> dict:
    """One array declaration of exactly `nbytes`, in the given dtype."""
    return {"dtype": dtype, "dims": [nbytes // buffer_assignment.DTYPE_BYTES[dtype]]}


def manifest(**programs) -> dict:
    """A manifest holding only what `declared` reads: each program's input and
    output declarations. Sizes are given in bytes, with the dtype whose width
    divides them."""
    return {
        "programs": {
            name: {"inputs": list(ins), "outputs": list(outs)}
            for name, (ins, outs) in programs.items()
        }
    }


class IdentifyTest(absltest.TestCase):
    """Giving a `jit_fn` module its program name back."""

    def setUp(self):
        super().setUp()
        self.executable = buffer_assignment.parse_executable(DUMP, MODULE_ID)
        # The fixture: one 1 MiB parameter, one 4 B output.
        self.signature = ([spec(1 * MIB)], [spec(4, "uint32")])

    def test_a_32_bit_declaration_is_not_sized_as_64(self):
        # Three of an AIR's arrays are 32-bit; sizing them at 8 bytes would
        # double them and move the signature off the program.
        m = manifest(rows=([spec(32, "int32")], []))
        self.assertEqual(buffer_assignment.declared(m), {"rows": ((32,), ())})

    def test_an_unknown_dtype_is_refused_not_guessed(self):
        m = {
            "programs": {
                "p": {"inputs": [{"dtype": "f16", "dims": [4]}], "outputs": []}
            }
        }
        with self.assertRaisesRegex(KeyError, "unknown dtype"):
            buffer_assignment.declared(m)

    def test_declared_sizes_are_bytes_not_elements(self):
        m = manifest(commit2=([spec(768 * MIB)], [spec(1536 * MIB)]))
        self.assertEqual(
            buffer_assignment.declared(m), {"commit2": ((768 * MIB,), (1536 * MIB,))}
        )

    def test_a_unique_signature_names_the_module(self):
        m = manifest(
            fits=self.signature,
            other=([spec(2 * MIB)], [spec(4, "uint32")]),
        )
        matched, ambiguous = buffer_assignment.identify({MODULE_ID: self.executable}, m)
        self.assertEqual(matched, {MODULE_ID: "fits"})
        self.assertEqual(ambiguous, {})

    def test_a_shared_signature_is_reported_not_resolved(self):
        m = manifest(one=self.signature, two=self.signature)
        matched, ambiguous = buffer_assignment.identify({MODULE_ID: self.executable}, m)
        self.assertEqual(matched, {})
        self.assertEqual(ambiguous, {MODULE_ID: ["one", "two"]})

    def test_no_match_leaves_the_module_unnamed(self):
        m = manifest(other=([spec(999 * MIB)], [spec(4, "uint32")]))
        matched, ambiguous = buffer_assignment.identify({MODULE_ID: self.executable}, m)
        self.assertEqual((matched, ambiguous), ({}, {}))


class TreeTest(absltest.TestCase):
    """A dump/<AIR>/<program>/ tree, and the check that a directory's name is
    the program it holds."""

    def commit2_manifest(self, **override) -> dict:
        outs = override.get("outputs", COMMIT2_OUT)
        ins = override.get("inputs", COMMIT2_IN)
        return manifest(commit2=([spec(n) for n in ins], [spec(n) for n in outs]))

    def test_the_directory_name_is_the_program(self):
        found = buffer_assignment.per_program(TREE)
        self.assertEqual(list(found), ["commit2"])

    def test_a_directory_holding_two_modules_is_refused(self):
        # Two compiles sharing a dump directory collide on `module_NNNN`, and
        # the directory's name can then only be right for one of them.
        d = pathlib.Path(self.create_tempdir().full_path)
        inner = d / "commit2"
        inner.mkdir()
        for path in (TREE / "commit2").iterdir():
            shutil.copy(path, inner)
            shutil.copy(path, inner / path.name.replace("module_0001", "module_0002"))
        with self.assertRaisesRegex(ValueError, "must hold one"):
            buffer_assignment.per_program(d)

    def test_every_value_in_the_arena_has_a_live_range(self):
        # A tuple element is `name{index}` in both files; a plain value is
        # `name` in the assignment and `name{}` in the live ranges. Looking up
        # the assignment's spelling unchanged silently finds nothing, and the
        # largest arenas here are built from tuple elements.
        e = buffer_assignment.per_program(TREE)["commit2"]
        missing = [v.name for v, _ in e.temp_values() if e.live_range(v) is None]
        self.assertEqual(missing, [])
        self.assertTrue(any("{" in v.name for v, _ in e.temp_values()))

    def test_a_real_executable_matches_what_the_air_declares(self):
        found = buffer_assignment.per_program(TREE)
        self.assertEqual(buffer_assignment.verify(found, self.commit2_manifest()), [])

    def test_the_output_tuple_table_is_not_charged_to_the_program(self):
        # 14 outputs, so XLA allocates 112 B of pointers beside them. The
        # manifest does not declare it; counting it as an output made every
        # multi-output program fail verification.
        e = buffer_assignment.per_program(TREE)["commit2"]
        tables = [a for a in e.allocations if a.is_tuple_table]
        self.assertEqual(len(tables), 1)
        self.assertEqual(tables[0].size, 8 * len(COMMIT2_OUT))

    def test_a_tuple_table_of_the_wrong_width_is_reported(self):
        # The table's size is a statement about the output count, so it is
        # checked rather than skipped: a program returning a different number
        # of arrays than the manifest says would otherwise pass.
        found = buffer_assignment.per_program(TREE)
        wrong = buffer_assignment.verify(
            found, self.commit2_manifest(outputs=COMMIT2_OUT[:-1])
        )
        self.assertTrue(any("outputs" in w for w in wrong), wrong)

    def test_a_mislabelled_directory_is_reported(self):
        found = buffer_assignment.per_program(TREE)
        renamed = {"deep": found["commit2"]}
        wrong = buffer_assignment.verify(renamed, self.commit2_manifest())
        self.assertTrue(any("no such program" in w for w in wrong), wrong)


class ReconcileTest(absltest.TestCase):
    """The identity `temp = peak_during - in_use_after`, and what breaks it."""

    def setUp(self):
        super().setUp()
        self.executable = buffer_assignment.parse_executable(DUMP, MODULE_ID)

    def test_the_arena_is_the_peak_above_the_reading_after_the_program(self):
        temp = self.executable.temp_bytes
        after = 500 * MIB
        r = buffer_assignment.reconcile(self.executable, after, after + temp)
        self.assertEqual(r.temp_measured, temp)
        self.assertEqual(r.unexplained, 0)

    def test_outputs_and_inputs_do_not_enter_the_identity(self):
        # Both are live on the line after the program -- outputs because the
        # program produced them, inputs because the caller held them -- so
        # they cancel. An identity that counted either would move with the
        # AIR's section sizes rather than with the arena.
        temp = self.executable.temp_bytes
        for base in (0, 8 * MIB, 4096 * MIB):
            r = buffer_assignment.reconcile(self.executable, base, base + temp)
            self.assertEqual(r.unexplained, 0, f"base {base}")

    def test_a_concurrent_upload_shows_as_unexplained_rather_than_absorbed(self):
        temp = self.executable.temp_bytes
        after = 500 * MIB
        r = buffer_assignment.reconcile(self.executable, after, after + temp + 64 * MIB)
        self.assertEqual(r.unexplained, 64 * MIB)

    def test_a_peak_this_program_never_raised_reads_negative(self):
        # `peak` is a monotonic client high-water: for every prove after the
        # binding one it is an older number, and the difference is not this
        # program's arena. The negative is the signal to check whether the
        # peak rose here at all.
        r = buffer_assignment.reconcile(self.executable, 4096 * MIB, 4096 * MIB)
        self.assertLess(r.unexplained, 0)


if __name__ == "__main__":
    absltest.main()
