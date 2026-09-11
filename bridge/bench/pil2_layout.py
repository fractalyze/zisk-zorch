#!/usr/bin/env python3
"""What pil2 holds on the device for one AIR, section by section, from the
proving key.

The bridge's per-stage inventory (`mem_stages.py`) says which buffers a bridged
prove has alive at each stage. This is the other half of that comparison: the
same question asked of pil2, whose answer is not observable the same way
because pil2 does not allocate per section. It takes **one** buffer per stream,
`mapTotalN` Goldilocks elements wide, and places every section at an offset
inside it -- so the sections that are dead by the time a later one is written
share their bytes with it, and the buffer is sized by whichever overlap is
worst rather than by the sum.

That is the whole of the difference this reader exists to show. `cm1` base
lives at the same offset as `cm2_ext`; `cm2` base at the same offset as `cm3`
(the quotient section) and its tree. pil2 does not release those sections --
it never allocated them separately to begin with.

**Where the numbers come from.** The layout is `StarkInfo::setMapOffsets` in
`pil2-stark/src/starkpil/stark_info.cpp`, reproduced here against the AIR's
own `<air>.starkinfo.json`. Reproducing vendor arithmetic is worth doing only
if it is checked, so `--expect` takes pil2's own figure for the AIR -- the
`TOTAL PROVER MEMORY USAGE` line of a `-vv` run, which is
`prover_buffer_size * 8` and `prover_buffer_size` is `get_map_totaln_c` -- and
the tool fails unless it reproduces it. A section table from an unchecked
re-implementation is a guess with a table's authority.

**pil2's "GB" is GiB.** `common/src/utils.rs`'s `format_bytes` divides by 1024
and labels the units KB/MB/GB, so the `6.03 GB` it prints for Main is 6.03
GiB. Comparing it to a figure in real GB understates the bridge's excess by
7%.

Usage:
  pil2_layout.py <provingKey>/zisk/Zisk/airs/<Air>/air/<Air>.starkinfo.json
                 [--expect 6.03GiB]
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

FIELD_EXTENSION = 3
HASH_SIZE = 4
# poseidon2_goldilocks.hpp: (1 << 19) nonces over 512 blocks.
NONCES_LAUNCH_GRID_SIZE = ((1 << 19) + 512 - 1) // 512
GIB = 1 << 30


def num_nodes_mt(height: int, arity: int) -> int:
    """`StarkInfo::getNumNodesMT`: the element count of a Merkle tree's nodes
    over `height` leaves, padded to the arity at every level."""
    nodes = height
    level = height
    while level > 1:
        nodes += (arity - (level % arity)) % arity
        level = (level + arity - 1) // arity
        nodes += level
    return nodes * HASH_SIZE


def values_size(value_map: list[dict]) -> int:
    """Stage-1 values are one element; later stages are extension-wide."""
    return sum(1 if v["stage"] == 1 else FIELD_EXTENSION for v in value_map)


class Layout:
    """pil2's one buffer for one AIR, as `setMapOffsets` builds it for a GPU
    prove of a basic (non-recursive) AIR.

    `const_tree_in_buffer` is `setMapOffsets`'s `!preallocate` branch and it is
    decided per AIR, not per run: an AIR whose constant tree pil2 preloaded
    once for the whole GPU (`Constant polynomials (separate)` in a -vv log)
    keeps it out of the per-stream buffer and shares that one copy across
    streams; an AIR whose tree did not make the preload budget carries it
    inside every stream's buffer instead. For hello-world the split falls
    either side of these two shapes -- Main's extended constants are 192 MiB
    and VirtualTableZisk0's are 2.75 GiB -- so a tool that assumed one branch
    would be 2.9 GiB wrong on the other. `--expect` is what settles it."""

    def __init__(self, si: dict, const_tree_in_buffer: bool = False):
        self.const_tree_in_buffer = const_tree_in_buffer
        ss = si["starkStruct"]
        self.si = si
        self.arity = ss["merkleTreeArity"]
        self.n = 1 << ss["nBits"]
        self.n_ext = 1 << ss["nBitsExt"]
        self.n_queries = ss["nQueries"]
        self.steps = [s["nBits"] for s in ss["steps"]]
        self.widths = si["mapSectionsN"]
        self.num_nodes = num_nodes_mt(self.n_ext, self.arity)
        # (name, elements) in the order pil2 places them, with the running
        # total after each. Overlaps are recorded as placements at an offset
        # already in use rather than as additions.
        self.sections: list[tuple[str, int, int]] = []
        self.total = self._build(ss, si)

    def _place(self, name: str, offset: int, elements: int) -> None:
        self.sections.append((name, offset, elements))

    def _build(self, ss: dict, si: dict) -> int:
        total = 0
        if self.const_tree_in_buffer:
            # The extended constants and their tree, first in the buffer. Over
            # 512 MiB of extended constants pil2 also computes them here
            # rather than reading them (`calculateFixedExtended`), which is
            # this AIR's `const_ext` by another route.
            tree = self.n_ext * si["nConstants"] + self.num_nodes
            self._place("const_ext + tree (in buffer)", total, tree)
            total += tree
        # The GPU scalars and the query-proof staging, all before the trace
        # sections and none of them overlapping.
        head = [
            ("const (base)", si["nConstants"] * self.n),
            ("custom_fixed", 0),
            ("publics", si["nPublics"]),
            ("proofvalues", values_size(si["proofValuesMap"])),
            ("airgroupvalues", values_size(si["airgroupValuesMap"])),
            ("airvalues", values_size(si["airValuesMap"])),
            ("challenge", HASH_SIZE),
            ("nonce", 1),
            ("nonce_blocks", NONCES_LAUNCH_GRID_SIZE),
            ("input_hash_nonce", HASH_SIZE),
            ("evals", len(si["evMap"]) * FIELD_EXTENSION),
            ("challenges", len(si["challengesMap"]) * FIELD_EXTENSION),
            ("xdivxsub", len(si["openingPoints"]) * FIELD_EXTENSION),
            ("fri_queries", self.n_queries),
            ("proof_queries", self._queries_proof_size()),
        ]
        for name, elements in head:
            self._place(name, total, elements)
            total += elements

        # The trace sections. Each `max` below is an overlap: the base section
        # is placed at an offset a later extended section is also written to,
        # and the buffer only has to be large enough for whichever is longer.
        self._place("cm1_ext", total, self.n_ext * self.widths["cm1"])
        total += self.n_ext * self.widths["cm1"]
        self._place("mt1", total, self.num_nodes)
        total += self.num_nodes

        cm1_base = total  # placed, not advanced: cm2_ext is written over it
        self._place("cm1 (base, shares cm2_ext)", cm1_base, self.n * self.widths["cm1"])
        self._place("cm2_ext", total, self.n_ext * self.widths["cm2"])
        total += self.n_ext * self.widths["cm2"]
        self._place("mt2", total, self.num_nodes)
        total += self.num_nodes
        total = max(cm1_base + self.n * self.widths["cm1"], total)

        cm2_base = total
        self._place("cm2 (base, shares cm3_ext)", cm2_base, self.n * self.widths["cm2"])
        self._place("cm3_ext (qsec)", total, self.n_ext * self.widths["cm3"])
        total += self.n_ext * self.widths["cm3"]
        self._place("mt3", total, self.num_nodes)
        total += self.num_nodes
        total = max(cm2_base + self.n * self.widths["cm2"], total)

        self._place("q/f", total, self.n_ext * FIELD_EXTENSION)
        q = total
        total += self.n_ext * FIELD_EXTENSION

        # zi/x and the expression scratch sit at the far end, and `lev` is
        # written over `q`. Each is a candidate for the maximum rather than an
        # addition to it.
        max_total = total + len(si["boundaries"]) * self.n_ext
        lev = q + min(len(si["openingPoints"]), 4) * self.n * FIELD_EXTENSION
        lev += FIELD_EXTENSION * self.n + len(si["openingPoints"]) * FIELD_EXTENSION
        max_total = max(max_total, lev)

        self._place("buff_helper", total, self.n_ext * FIELD_EXTENSION)
        total += self.n_ext * FIELD_EXTENSION
        max_total = max(max_total, q + 2 * self.n_ext * FIELD_EXTENSION + si["qDeg"])

        for step, n_bits in enumerate(self.steps[1:]):
            height = 1 << n_bits
            width = ((1 << self.steps[step]) // height) * FIELD_EXTENSION
            self._place(f"fri_{step + 1}", total, height * width)
            total += height * width
            self._place(f"mt_fri_{step + 1}", total, num_nodes_mt(height, self.arity))
            total += num_nodes_mt(height, self.arity)

        return max(total, max_total)

    def _queries_proof_size(self) -> int:
        """The staging the openings are gathered into, sized by the widest
        tree any of them can come from."""
        widths = [w for name, w in self.widths.items() if name != "const"]
        widths.append(self.widths["const"])
        for i, n_bits in enumerate(self.steps[:-1]):
            groups = 1 << self.steps[i + 1]
            widths.append(((1 << n_bits) // groups) * FIELD_EXTENSION)
        max_tree_width = max(widths)
        ss = self.si["starkStruct"]
        n_siblings = (
            math.ceil(ss["nBitsExt"] / math.log2(self.arity))
            - ss["lastLevelVerification"]
        )
        max_proof = n_siblings * (self.arity - 1) * HASH_SIZE
        n_trees = 1 + (self.si["nStages"] + 1) + len(self.si["customCommits"])
        n_trees_fri = len(self.steps) - 1
        return (n_trees + n_trees_fri) * (max_tree_width + max_proof) * self.n_queries

    def report(self) -> str:
        lines = [
            f"{self.si['name']}  n=2^{self.si['starkStruct']['nBits']}"
            f"  ext=2^{self.si['starkStruct']['nBitsExt']}"
            f"  widths const/cm1/cm2/cm3 ="
            f" {self.widths['const']}/{self.widths['cm1']}"
            f"/{self.widths['cm2']}/{self.widths['cm3']}",
            "",
            f"{'section':<30} {'offset MiB':>12} {'size MiB':>10}",
        ]
        mib = 1 << 20
        for name, offset, elements in self.sections:
            if elements * 8 < mib:
                continue
            lines.append(
                f"{name:<30} {offset * 8 / mib:>12,.0f} {elements * 8 / mib:>10,.0f}"
            )
        where = (
            "in every stream's buffer"
            if self.const_tree_in_buffer
            else "preloaded once per GPU, shared across streams"
        )
        lines += [
            "",
            f"constant tree: {where}",
            f"mapTotalN = {self.total} elements"
            f" = {self.total * 8 / GIB:.2f} GiB per stream",
        ]
        return "\n".join(lines)


def parse_size(text: str) -> float:
    """`6.03GiB` -> bytes. pil2 prints GiB and calls them GB; either spelling
    is read as GiB here, which is what its `format_bytes` means by both."""
    number = float(text.rstrip("GiBb"))
    return number * GIB


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("starkinfo", type=pathlib.Path)
    ap.add_argument(
        "--expect",
        help="pil2's own figure for this AIR (the -vv TOTAL PROVER MEMORY"
        " USAGE line), e.g. 6.03GiB. The tool fails unless it reproduces it.",
    )
    ap.add_argument(
        "--const-tree",
        choices=("separate", "in-buffer"),
        default="separate",
        help="where pil2 keeps this AIR's constant tree, when no --expect is"
        " given to settle it",
    )
    ap.add_argument(
        "--tolerance",
        type=float,
        default=0.01,
        help="GiB the reproduction may differ by, for pil2's 2-decimal print",
    )
    args = ap.parse_args(argv)

    si = json.loads(args.starkinfo.read_text())
    if not args.expect:
        print(Layout(si, const_tree_in_buffer=args.const_tree == "in-buffer").report())
        return 0

    # With pil2's own figure in hand, which branch it took is not a guess:
    # the two differ by the whole constant tree, so at most one can match.
    want = parse_size(args.expect)
    matched = [
        layout
        for layout in (Layout(si, in_buffer) for in_buffer in (False, True))
        if abs(want - layout.total * 8) / GIB <= args.tolerance
    ]
    if not matched:
        both = " / ".join(f"{Layout(si, b).total * 8 / GIB:.2f}" for b in (False, True))
        print(
            f"MISMATCH: pil2 reports {want / GIB:.2f} GiB, this layout gives"
            f" {both} GiB (tree out of / in the buffer). No section table is"
            " printed -- the layout here has drifted from"
            " StarkInfo::setMapOffsets and a table from it would be a guess.",
            file=sys.stderr,
        )
        return 1
    layout = matched[0]
    print(layout.report())
    print(f"\nreproduces pil2's own {want / GIB:.2f} GiB for this AIR")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
