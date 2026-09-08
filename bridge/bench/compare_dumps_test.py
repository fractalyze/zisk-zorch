"""The plan guard in compare_dumps: two runs that put different airs on the
same instance id must be refused rather than compared byte for byte."""

import pathlib

import numpy as np
from absl.testing import absltest

from bridge.bench import compare_dumps

HELLO = "\n".join(f">>> GEN_PROOF_{i} [0:{a}]" for i, a in enumerate([1, 0, 5, 4]))
SHA = "\n".join(f">>> GEN_PROOF_{i} [0:{a}]" for i, a in enumerate([1, 0, 0, 0]))


def _run(root: pathlib.Path, name: str, log: str | None, proofs: dict[int, list[int]]):
    d = root / name
    (d / "dumps").mkdir(parents=True)
    if log is not None:
        (d / "run.log").write_text(log)
    for inst, words in proofs.items():
        np.array(words, dtype=np.uint64).tofile(d / "dumps" / f"{inst}.bin")
    return d / "dumps"


class CompareDumpsTest(absltest.TestCase):

    def setUp(self):
        super().setUp()
        self.root = pathlib.Path(self.create_tempdir().full_path)
        self.proofs = {0: [1, 2], 1: [3, 4]}

    def test_identical_dumps_under_the_same_plan_pass(self):
        a = _run(self.root, "native", HELLO, self.proofs)
        b = _run(self.root, "bridge", HELLO, self.proofs)
        self.assertEqual(compare_dumps.main(["", str(a), str(b)]), 0)

    def test_differing_dumps_under_the_same_plan_fail(self):
        a = _run(self.root, "native", HELLO, self.proofs)
        b = _run(self.root, "bridge", HELLO, {0: [1, 2], 1: [3, 9]})
        self.assertEqual(compare_dumps.main(["", str(a), str(b)]), 1)

    def test_a_different_plan_is_refused_not_compared(self):
        # Byte-identical dumps, but instance 2 is a different air in each run:
        # the answer must be "refused", never "identical".
        a = _run(self.root, "native", HELLO, self.proofs)
        b = _run(self.root, "sha", SHA, self.proofs)
        self.assertEqual(compare_dumps.main(["", str(a), str(b)]), 2)

    def test_a_missing_log_falls_back_to_comparing_by_id(self):
        a = _run(self.root, "native", None, self.proofs)
        b = _run(self.root, "bridge", HELLO, self.proofs)
        self.assertEqual(compare_dumps.main(["", str(a), str(b)]), 0)


if __name__ == "__main__":
    absltest.main()
