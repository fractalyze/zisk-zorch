#!/usr/bin/env python3
"""Tests for the page-cache census and the reset built on it.

Two of these pin arithmetic a hand-rolled census gets wrong (a file's last page
is partial, an empty file has no pages at all), and two pin the set: that it
names what proofman's init reads, and that it leaves out the two families that
are most of a proving key. Getting the set wrong is the silent failure -- the
warm still succeeds, it just warms the wrong files, and the run it was supposed
to protect reads from disk anyway.

The residency assertions only go one way on purpose. Reading a file makes its
pages resident on any filesystem, so `warm` can be asserted exactly; dropping
them is advice the kernel may decline -- on tmpfs, which is where a test's
scratch directory often lives, there is no backing store to drop to and the
eviction is a no-op. A test that demanded a zero there would fail on the
sandbox and pass on the rig, which is the wrong way round for a rule about
measurement.
"""

import os
import pathlib

from absl.testing import absltest

from bridge.bench import pagecache


class CensusTest(absltest.TestCase):
    def setUp(self):
        super().setUp()
        self.root = pathlib.Path(self.create_tempdir().full_path)

    def write(self, name: str, size: int) -> pathlib.Path:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"\xa5" * size)
        return path

    def test_counts_the_files_bytes_not_the_pages(self):
        """A one-byte file holds one byte, however large the page holding it."""
        path = self.write("small.const_gpu", 1)
        pagecache.warm(path)
        entry = pagecache.census(path)
        self.assertEqual(entry.total, 1)
        self.assertEqual(entry.resident, 1)

    def test_partial_last_page_does_not_round_up(self):
        size = pagecache.PAGE + 17
        path = self.write("ragged.dat", size)
        pagecache.warm(path)
        self.assertEqual(pagecache.census(path), pagecache.Census(size, size))

    def test_empty_file_is_neither_resident_nor_absent(self):
        """An empty file has no pages, so mincore is never asked about it."""
        self.assertEqual(
            pagecache.census(self.write("empty.bin", 0)), pagecache.Census(0, 0)
        )

    def test_warm_reads_the_whole_file(self):
        """Including a file longer than one read chunk, which is the case a
        loop that reads once silently truncates."""
        size = pagecache.READ_CHUNK + 4096
        path = self.write("long.exec", size)
        pagecache.evict(path)
        pagecache.warm(path)
        self.assertEqual(pagecache.census(path), pagecache.Census(size, size))

    def test_evict_leaves_the_file_intact(self):
        """Whether the pages go is the kernel's call; the bytes are not."""
        path = self.write("key.const_gpu", 8192)
        pagecache.warm(path)
        pagecache.evict(path)
        self.assertEqual(path.read_bytes(), b"\xa5" * 8192)
        self.assertEqual(pagecache.census(path).total, 8192)

    def test_share_of_an_empty_group_is_not_a_division_by_zero(self):
        self.assertEqual(pagecache.Census(0, 0).share, 0.0)

    def test_censuses_add(self):
        self.assertEqual(
            pagecache.Census(1, 2) + pagecache.Census(30, 40), pagecache.Census(31, 42)
        )


class ExpandTest(absltest.TestCase):
    def setUp(self):
        super().setUp()
        self.root = pathlib.Path(self.create_tempdir().full_path)
        for name in ("a/one.const_gpu", "a/b/two.exec", "three.dat"):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"x")

    def test_a_directory_walks_it(self):
        self.assertLen(list(pagecache.expand(str(self.root))), 3)

    def test_a_glob_recurses(self):
        found = list(pagecache.expand(f"{self.root}/**/*.exec"))
        self.assertLen(found, 1)
        self.assertTrue(found[0].endswith("two.exec"))

    def test_directories_are_not_files(self):
        """`**` matches directories too, and a census of one raises."""
        self.assertEqual(
            sorted(os.path.basename(p) for p in pagecache.expand(f"{self.root}/**")),
            ["one.const_gpu", "three.dat", "two.exec"],
        )


class InitSetTest(absltest.TestCase):
    def test_names_what_init_reads(self):
        labels = [label for label, _ in pagecache.init_set("/pk")]
        self.assertContainsSubset(["const_gpu", "exec", "dat"], labels)

    def test_leaves_out_the_two_families_init_does_not_read(self):
        """`.const` and `.consttree_gpu` are most of a proving key by size and
        belong to the phases after init. Warming them would evict the set this
        reset exists to protect, so a pattern that catches them is the bug."""
        patterns = [pattern for _, pattern in pagecache.init_set("/pk")]
        self.assertNotIn("/pk/**/*.const", patterns)
        self.assertNotIn("/pk/**/*.consttree_gpu", patterns)
        for pattern in patterns:
            self.assertFalse(pathlib.PurePath("/pk/x/y.const").match(pattern), pattern)
            self.assertFalse(
                pathlib.PurePath("/pk/x/y.consttree_gpu").match(pattern), pattern
            )

    def test_patterns_hang_off_the_proving_key_given(self):
        self.assertTrue(
            all(
                p.startswith("/somewhere/pk/")
                for _, p in pagecache.init_set("/somewhere/pk")
            )
        )


class ApplyTest(absltest.TestCase):
    def setUp(self):
        super().setUp()
        self.root = pathlib.Path(self.create_tempdir().full_path)
        (self.root / "one.const_gpu").write_bytes(b"x" * 4096)
        (self.root / "two.const_gpu").write_bytes(b"y" * 2048)
        (self.root / "skip.const").write_bytes(b"z" * 8192)

    def test_a_group_sums_its_files_and_only_its_files(self):
        rows = pagecache.apply([("cg", f"{self.root}/*.const_gpu")], pagecache.warm)
        self.assertEqual(rows, [("cg", pagecache.Census(6144, 6144))])

    def test_the_action_runs_before_the_census(self):
        """Otherwise a --warm run reports the state it was called on rather
        than the state it left, and reads warm when it is not."""
        path = self.root / "one.const_gpu"
        pagecache.evict(path)
        rows = pagecache.apply([("cg", str(path))], pagecache.warm)
        self.assertEqual(rows[0][1].resident, 4096)

    def test_a_group_that_matches_nothing_is_still_a_row(self):
        self.assertEqual(
            pagecache.apply([("none", f"{self.root}/*.missing")]),
            [("none", pagecache.Census(0, 0))],
        )


class MainTest(absltest.TestCase):
    def test_a_key_with_no_init_files_is_an_error(self):
        """The silent failure this guard exists for: a mistyped or moved key
        censuses to a clean table of zeroes and exits 0, so a caller warming
        before a timed run gets no warm and no complaint."""
        root = self.create_tempdir().full_path
        pathlib.Path(root, "notes.txt").write_text("not a proving key")
        with self.assertRaises(SystemExit):
            pagecache.main(["pagecache.py", "--warm", "--proofman-init", root])

    def test_a_key_with_init_files_is_not(self):
        """The guard keys on the set being empty, not on the files being cold:
        a key present but wholly evicted is the case the cold arm needs."""
        root = self.create_tempdir().full_path
        path = pathlib.Path(root, "air", "one.const_gpu")
        path.parent.mkdir(parents=True)
        path.write_bytes(b"x" * 4096)
        pagecache.evict(path)
        self.assertEqual(pagecache.main(["pagecache.py", "--proofman-init", root]), 0)

    def test_warm_and_evict_are_not_both(self):
        with self.assertRaises(SystemExit):
            pagecache.main(["pagecache.py", "--warm", "--evict", "x=/tmp"])

    def test_nothing_to_census_is_an_error_not_an_empty_report(self):
        with self.assertRaises(SystemExit):
            pagecache.main(["pagecache.py"])


if __name__ == "__main__":
    absltest.main()
