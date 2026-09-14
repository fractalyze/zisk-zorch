"""What the extend's input re-layout costs the arena, from the compiled module.

The LDE transforms a column while a section is stored by row, so each column
block is transposed on the way in. Those transposes are emitted per block and
the codeword does not depend on them, so every golden passes whether they stay
per block or not — `trace_commit_test` is blind to this by construction, the
same hole `fusion_test` exists for. What decides it is the whole-section field
view: take it before the blocks are sliced off and XLA sinks the slices below
the bitcast and merges the transposes into one the size of the section, live
from the first block to the last. `extend_words` takes the view per block.

GPU-only (`tags = ["gpu"]`): the merge is what the pinned plugin's GPU backend
does with the graph, and a CPU compile answers for a backend the bridge never
runs on.
"""

from __future__ import annotations

import frx
import frx.numpy as fnp
import numpy as np
from absl.testing import absltest
from frx import lax
from zk_dtypes import goldilocks as F

from zisk_zorch.commit.trace_commit import extend, extend_words

# Small on purpose: the merge is a property of the graph's shape, not its size.
# The block budget is four columns of the extended domain, so 16 columns split
# 4 + 4 + 4 + 4 and the loop has blocks to merge across.
_N = 1 << 12
_COLS = 16
_BLOWUP = 2
_BLOCK_BYTES = (_N * _BLOWUP) * 8 * 4


def _temp_bytes(fn) -> tuple[int, int]:
    """`fn`'s temp arena and argument size, as the compiled executable
    accounts for them. Traced under x64, as the export lowers."""
    with frx.enable_x64():
        lowered = frx.jit(fn).lower(frx.ShapeDtypeStruct((_N, _COLS), np.uint64))
        m = lowered.compile().memory_analysis()
    return m.temp_size_in_bytes, m.argument_size_in_bytes


def _view_whole_section(words):
    """The section viewed as field elements before the blocks are sliced."""
    trace = lax.bitcast_convert_type(words, F)
    return lax.bitcast_convert_type(
        extend(trace, _BLOWUP, block_bytes=_BLOCK_BYTES), fnp.uint64
    )


def _view_per_block(words):
    return lax.bitcast_convert_type(
        extend_words(words, _BLOWUP, block_bytes=_BLOCK_BYTES), fnp.uint64
    )


class LdeArenaTest(absltest.TestCase):
    def test_per_block_view_keeps_the_section_out_of_the_arena(self) -> None:
        whole, section_bytes = _temp_bytes(_view_whole_section)
        per_block, _ = _temp_bytes(_view_per_block)
        # The merged transpose is the section's own size, so taking the view
        # per block has to recover at least that much. Anything less means the
        # merge survived in some other shape.
        self.assertGreaterEqual(whole - per_block, section_bytes)

    def test_gpu_backend(self) -> None:
        # The target is GPU-tagged; a CPU run here means the tag filter is off.
        self.assertEqual(frx.default_backend(), "gpu")


if __name__ == "__main__":
    absltest.main()
