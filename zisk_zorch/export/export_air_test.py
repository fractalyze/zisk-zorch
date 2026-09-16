"""`lower`'s guards against a program that closed over a device array.

An interned array from an earlier prove in this process does not vanish
when the exporter traces the program that reads it: under the pinned frx
it lowers as a dense `stablehlo.constant`, which ships inside the artifact
and stays resident for the executable's life. Needs no proving key and no
device — the guards read the lowered module, not a run.
"""

from __future__ import annotations

import frx
import frx.numpy as fnp
import numpy as np
from absl.testing import absltest

from zisk_zorch.export.export_air import MAX_CONSTANT_ELEMENTS, lower
from zisk_zorch.export.stages import Program, Spec

_N = MAX_CONSTANT_ELEMENTS * 2


def _program(fn, dims=(8,)) -> Program:
    return Program(
        name="probe",
        fn=fn,
        inputs=[Spec("x", "uint64", dims)],
        output_names=["y"],
    )


class LowerGuardTest(absltest.TestCase):
    def test_an_in_trace_computation_lowers(self):
        got = lower(_program(lambda x: (x + fnp.arange(8, dtype=np.uint64),)))
        self.assertNotEmpty(got)

    def test_a_closed_over_device_array_is_refused(self):
        pack = frx.device_put(np.arange(_N, dtype=np.uint64))
        with self.assertRaisesRegex(RuntimeError, "closes over a device array"):
            lower(_program(lambda x: (x + pack,), dims=(_N,)))

    def test_a_literal_at_the_budget_lowers(self):
        small = frx.device_put(np.arange(MAX_CONSTANT_ELEMENTS, dtype=np.uint64))
        got = lower(_program(lambda x: (x + small,), dims=(MAX_CONSTANT_ELEMENTS,)))
        self.assertNotEmpty(got)


if __name__ == "__main__":
    absltest.main()
