"""Byte-match `serialize_proof` against a capture's native flat proof.

Rebuilds every committed tree from the capture's own extended sections,
replays the transcript for the grinding seed, reads the nonce off the native
proof's tail, draws the query positions, opens every tree, serializes — and
compares the result to the dumped ``*_proof.npy`` region by region, so a
layout error names the block it lives in.

    python -m zisk_zorch.harness.verify_proof_serializer --dump=<dir> \
        --instance=ag0_air2_inst5 --starkinfo=<starkinfo.json>
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import frx.numpy as fnp
import numpy as np
from zk_dtypes import goldilocks as F
from zorch.pcs.fold import to_base_field

from zisk_zorch.commit.trace_commit import merkle_tree
from zisk_zorch.fri.queries import grind_is_valid, query_positions_for
from zisk_zorch.fri.seam import Pil2FriCode
from zisk_zorch.harness.capture import Capture, cubic
from zisk_zorch.harness.pil2 import transcript_width
from zisk_zorch.harness.proof_serializer import (
    TreeOpening,
    open_tree,
    serialize_proof,
)
from zisk_zorch.harness.schedule import replay_challenges


def _region_report(got: np.ndarray, want: np.ndarray, layout: list[tuple[str, int]]):
    ok_all = True
    off = 0
    for name, size in layout:
        g, w = got[off : off + size], want[off : off + size]
        ok = np.array_equal(g, w)
        ok_all &= ok
        if not ok:
            first = int(np.nonzero(g != w)[0][0])
            print(f"MISMATCH {name} (+{first} of {size})")
        else:
            print(f"OK       {name} ({size} words)")
        off += size
    if off != len(want):
        print(f"MISMATCH total length: laid out {off}, native {len(want)}")
        ok_all = False
    return ok_all


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", type=pathlib.Path, required=True)
    ap.add_argument("--instance", required=True)
    ap.add_argument("--starkinfo", type=pathlib.Path, required=True)
    args = ap.parse_args()
    cap = Capture(args.dump, args.instance, args.starkinfo)
    si = cap.si
    ss = si["starkStruct"]
    arity, llv = ss["merkleTreeArity"], ss.get("lastLevelVerification", 0)
    nbe = ss["nBitsExt"]
    steps = cap.steps
    native = np.load(cap.path("proof"))

    # The grinding seed is the schedule's last squeeze; every earlier
    # challenge must already replay (the composed gate pins this).
    *_, (label, seed, want_seed) = replay_challenges(cap)
    assert label == "grinding seed" and np.array_equal(
        seed.astype(np.uint64), want_seed
    ), "schedule replay must be green before the serializer can be judged"
    nonce = int(native[-1])
    pow_bits = ss["powBits"]
    assert grind_is_valid(fnp.array(seed.astype(np.uint64), dtype=F), nonce, pow_bits)
    positions = query_positions_for(
        fnp.array(seed.astype(np.uint64), dtype=F),
        transcript_width(ss),
        nonce,
        n_queries=ss["nQueries"],
        n_bits_ext=nbe,
    )

    mt = merkle_tree(arity)

    def committed(section: str, cols: int) -> TreeOpening:
        matrix = fnp.array(cap.u64(section).astype(F).reshape(1 << nbe, cols))
        root, layers = mt.commit(matrix)
        return root, open_tree(
            mt, matrix, layers, positions, n_bits=nbe, arity=arity, llv=llv
        )

    _, const_open = committed("const_ext", si["nConstants"])
    customs = []
    for ci, c in enumerate(si.get("customCommits", [])):
        _, opening = committed(f"custom{ci}_ext", si["mapSectionsN"][c["name"] + "0"])
        customs.append(opening)
    stage_opens = []
    for s in range(si["nStages"] + 1):
        sec = {0: "cm1_ext", 1: "cm2_ext"}.get(s, "quotient_cm")
        root, opening = committed(sec, si["mapSectionsN"][f"cm{s + 1}"])
        want_root = cap.u64({0: "root1", 1: "root2"}.get(s, "rootQ"))
        assert np.array_equal(
            np.asarray(root).astype(np.uint64), want_root
        ), f"cm{s + 1} root"
        stage_opens.append(opening)

    code = Pil2FriCode(tuple(steps))
    fri_roots, fri_opens = [], []
    for k in range(1, len(steps)):
        grouped = to_base_field(code.group_leaves(cubic(cap.u64(f"fri_layer{k - 1}"))))
        root, layers = mt.commit(grouped)
        fri_roots.append(np.asarray(root).astype(np.uint64))
        fri_opens.append(
            open_tree(
                mt,
                grouped,
                layers,
                np.asarray(positions) % (1 << steps[k]),
                n_bits=steps[k],
                arity=arity,
                llv=llv,
            )
        )

    got = serialize_proof(
        si,
        airgroup_values=cap.u64("airgroupvalues"),
        air_values=cap.u64("airvalues"),
        roots=[cap.u64("root1"), cap.u64("root2"), cap.u64("rootQ")],
        evals=cap.u64("evals"),
        const_opening=const_open,
        custom_openings=customs,
        stage_openings=stage_opens,
        fri_roots=fri_roots,
        fri_openings=fri_opens,
        final_pol=cap.u64(f"fri_layer{len(steps) - 1}"),
        nonce=nonce,
    )

    def tree_sizes(name: str, opening: TreeOpening) -> list[tuple[str, int]]:
        return [
            (f"{name} rows", opening.rows.size),
            (f"{name} paths", opening.paths.size),
            (f"{name} last-level", opening.last_level.size),
        ]

    layout = [
        ("airgroupvalues", 3 * len(si["airgroupValuesMap"] or [])),
        ("airvalues", 3 * len(si["airValuesMap"] or [])),
        ("roots", (si["nStages"] + 1) * 4),
        ("evals", 3 * len(si["evMap"])),
        *tree_sizes("const", const_open),
    ]
    for ci, opening in enumerate(customs):
        layout += tree_sizes(f"custom{ci}", opening)
    for s, opening in enumerate(stage_opens):
        layout += tree_sizes(f"cm{s + 1}", opening)
    layout.append(("fri roots", 4 * len(fri_roots)))
    for k, opening in enumerate(fri_opens):
        layout += tree_sizes(f"fri step {k + 1}", opening)
    layout += [("final pol", 3 << steps[-1]), ("nonce", 1)]

    print(f"{args.instance}: serialized {got.size} vs native {native.size}")
    ok = _region_report(got, native, layout)
    print("flat-proof serialization: " + ("BYTE-MATCH" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
