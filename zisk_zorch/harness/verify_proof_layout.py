"""Validate the flat-proof layout spec against a capture's proofs.

``proof2pointer`` (pil2-stark ``proof_stark.hpp``) serializes a proof as:

    airgroupValues*3 | airValues*3 | roots (nStages+1)*4 | evals*3
    | const-tree query values + merkle paths [| last levels]
    | per-custom-commit query values + paths [| last levels]
    | per-stage (cm1..cmQ) query values + paths [| last levels]
    | FRI tree roots (len(steps)-1)*4
    | per-FRI-step query values + paths [| last levels]
    | final pol*3 | nonce

Every ``*_proof.npy`` must parse to exactly its length by this formula,
and the extracted airgroupValues / roots / evals must equal the capture's
independently dumped sections — pinning the offsets `serialize_proof`
reproduces. ``--proofs`` points at a separate proof directory (e.g. the
composite's own emitted proofs) while the sections still come from
``--dump``; recursion-stage proofs whose sections were never dumped
length-check only.

Key layouts: the ziskup tree (``zisk/<group>/airs/<air>/{air,compressor}``,
``zisk/<group>/recursive2``, ``zisk/vadcop_final``; recursive1 shares
recursive2's shape) and the example-build tree (``build/<group>/...``).

    python -m zisk_zorch.harness.verify_proof_layout \
        --dump=<capture dir> --key=<provingKey dir> [--proofs=<proof dir>]
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import re
import sys

import numpy as np

P = 0xFFFFFFFF00000001
_HASH = 4


def _canon(a: np.ndarray) -> np.ndarray:
    return np.where(a >= np.uint64(P), a - np.uint64(P), a)


def parse_proof(si: dict, buf: np.ndarray) -> dict:
    """Section views of a flat proof, plus the total the layout accounts for."""
    ss = si["starkStruct"]
    arity = ss["merkleTreeArity"]
    llv = ss.get("lastLevelVerification", 0)
    n_stages, nq = si["nStages"], ss["nQueries"]
    sib_per = (arity - 1) * _HASH
    last_levels = (arity**llv) * _HASH if llv else 0

    def tree(width: int, n_bits: int) -> int:
        n_sib = math.ceil(n_bits / math.log2(arity)) - llv
        return nq * width + nq * n_sib * sib_per + last_levels

    p = 0
    out = {}
    n_agv = len(si["airgroupValuesMap"] or [])
    out["airgroupvalues"] = buf[p : p + 3 * n_agv]
    p += 3 * n_agv
    p += 3 * len(si["airValuesMap"] or [])
    out["roots"] = [
        buf[p + i * _HASH : p + (i + 1) * _HASH] for i in range(n_stages + 1)
    ]
    p += (n_stages + 1) * _HASH
    out["evals"] = buf[p : p + 3 * len(si["evMap"])]
    p += 3 * len(si["evMap"])
    steps = ss["steps"]
    p += tree(si["nConstants"], steps[0]["nBits"])
    for c in si.get("customCommits", []):
        p += tree(si["mapSectionsN"][c["name"] + "0"], steps[0]["nBits"])
    for s in range(n_stages + 1):
        p += tree(si["mapSectionsN"][f"cm{s + 1}"], steps[0]["nBits"])
    p += (len(steps) - 1) * _HASH
    for i in range(1, len(steps)):
        width = 3 * (1 << (steps[i - 1]["nBits"] - steps[i]["nBits"]))
        p += tree(width, steps[i]["nBits"])
    p += 3 * (1 << steps[-1]["nBits"])
    p += 1
    out["consumed"] = p
    return out


def _starkinfo_for(key: pathlib.Path, gi: dict, inst: str) -> pathlib.Path:
    group0 = gi["air_groups"][0]
    airs = [a["name"] for a in gi["airs"][0]]
    # ziskup layout first, example-build layout as the fallback.
    roots = [key / "zisk", key / "build"]
    root = next((r for r in roots if r.is_dir()), roots[0])
    vadcop = (
        root / "vadcop_final"
        if (root / "vadcop_final").is_dir()
        else key / "build" / "vadcop_final"
    )
    if inst.startswith("compressor_"):
        air = airs[int(re.search(r"_air(\d+)_", inst).group(1))]
        return root / group0 / "airs" / air / "compressor" / "compressor.starkinfo.json"
    if inst.startswith(("recursive1_", "recursive2_")):
        return root / group0 / "recursive2" / "recursive2.starkinfo.json"
    if inst.startswith("vadcop_final_"):
        return vadcop / "vadcop_final.starkinfo.json"
    air = airs[int(re.search(r"_air(\d+)_", inst).group(1))]
    return root / group0 / "airs" / air / "air" / f"{air}.starkinfo.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", type=pathlib.Path, required=True)
    ap.add_argument("--key", type=pathlib.Path, required=True)
    ap.add_argument(
        "--proofs",
        type=pathlib.Path,
        default=None,
        help="proof directory when separate from --dump (own emitted proofs)",
    )
    args = ap.parse_args()
    proofs_dir = args.proofs or args.dump

    gi = json.loads((args.key / "pilout.globalInfo.json").read_text())

    ok = total = 0
    proofs = sorted(proofs_dir.glob("*_proof.npy"))
    if not proofs:
        print("SKIP     no *_proof sections")
        return 0
    for f in proofs:
        inst = re.sub(r"_proof\.npy$", "", f.name)
        si = json.loads(_starkinfo_for(args.key, gi, inst).read_text())
        buf = np.load(f)
        o = parse_proof(si, buf)
        this = o["consumed"] == len(buf)
        note = ""
        if (args.dump / f"{inst}_root1.npy").exists():
            for i, name in enumerate(["1", "2"]):
                this &= np.array_equal(
                    o["roots"][i], _canon(np.load(args.dump / f"{inst}_root{name}.npy"))
                )
            this &= np.array_equal(
                o["roots"][si["nStages"]],
                _canon(np.load(args.dump / f"{inst}_rootQ.npy")),
            )
            this &= np.array_equal(
                _canon(o["evals"]), _canon(np.load(args.dump / f"{inst}_evals.npy"))
            )
            if len(o["airgroupvalues"]):
                this &= np.array_equal(
                    _canon(o["airgroupvalues"]),
                    _canon(np.load(args.dump / f"{inst}_airgroupvalues.npy")),
                )
        else:
            note = " (length only — no dumped sections)"
        total += 1
        ok += bool(this)
        print(
            ("OK       " if this else "MISMATCH ") + f"{inst} ({len(buf)} words){note}"
        )
    print(
        f"flat-proof layout: {ok}/{total} "
        + ("— ALL OK" if ok == total else "— FAILED")
    )
    return 0 if ok == total else 1


if __name__ == "__main__":
    sys.exit(main())
