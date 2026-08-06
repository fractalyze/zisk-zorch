#!/usr/bin/env python3
"""Generate the Main continuation golden from a real block capture.

`zisk_zorch/constraints/continuation.py` hand-ports Main's 96 direct bus
updates; the proving key carries the same updates as `im_airval` hints
(one stage-2 scalar `±mult / D` each). This script evaluates the key's
own hint SSA — numerator, denominator, and every tuple slot — over each
dumped segment's stage-1 air values and the dumped `std_alpha` /
`std_gamma`, then writes the per-interaction results as the golden that
`continuation_test.py` matches the port against, tuple for tuple. The
reference is pil2's compiled expressions, so a transcription slip in the
port cannot survive the diff.

The dump is a `PIL2_DUMP_DIR` capture of a real eth-block prove (the 12
Main segments of block 21740136); nothing here needs the trace sections,
only the small scalar files.

Every segment gets its inputs and the summed `direct_term` — the Σ pins
every tuple, sign, selector, and fold, so that alone is the byte match.
The 96 per-interaction fractions are emitted only for the FIRST instance:
they add failure localization, not strength, and carrying them for all
segments would quintuple the committed golden for identical structure.

Usage (any venv carrying frx + zk_dtypes, repo root on PYTHONPATH):

    python scripts/extract_continuation_fixture.py \
        --key=$ZISK_PROVING_KEY/zisk/Zisk/airs/Main/air \
        --dump=/path/to/blockdump --instances=ag0_air0_inst1,... \
        --out=zisk_zorch/constraints/testdata/golden/main_continuation.json
"""

from __future__ import annotations

import argparse
import json
import pathlib

import frx.numpy as fnp
import numpy as np
from zk_dtypes import goldilocksx3 as F3

from zisk_zorch.golden import u64x3
from zisk_zorch.quotient.cexp_ref import _run_block as run_block

_P = 0xFFFFFFFF00000001


def _field(hint: dict, name: str) -> dict:
    return next(f for f in hint["fields"] if f["name"] == name)


def _values(hint: dict, name: str) -> list[dict]:
    f = _field(hint, name)
    return sorted(f["values"], key=lambda v: v.get("pos", [0]))


def _cubic(words: list[int]):
    return u64x3([str(w) for w in words]).reshape(())


def _limbs(x) -> list[str]:
    flat = np.asarray(fnp.array(x, dtype=F3).reshape(1)).view(np.uint64)
    return [str(w) for w in flat.reshape(-1)[:3]]


def scalar_env(starkinfo: dict, airvalues: np.ndarray, challenges: np.ndarray) -> dict:
    """The SSA interpreter env for scalar-only expressions: air values
    id-keyed off `airValuesMap` (stage-1 packs one word, stage >= 2 three)
    and the dumped challenges."""
    av, off = {}, 0
    for i, v in enumerate(starkinfo["airValuesMap"]):
        if v["stage"] == 1:
            av[i] = _cubic([int(airvalues[off]), 0, 0])
            off += 1
        else:
            av[i] = _cubic([int(w) for w in airvalues[off : off + 3]])
            off += 3
    ch = challenges.reshape(-1, 3)
    return {
        "airvalues": av,
        "challenges": {i: _cubic([int(w) for w in ch[i]]) for i in range(len(ch))},
        "cm": {},
        "const": {},
        "custom": {},
        "publics": {},
        "proofvalues": {},
        "airgroupvalues": {},
        "zi": {},
    }


def eval_operand(v: dict, env: dict, exps: dict):
    if v["op"] == "number":
        return _cubic([int(v["value"]) % _P, 0, 0])
    if v["op"] == "airvalue":
        return env["airvalues"][v["id"]]
    if v["op"] == "tmp":
        return run_block(exps[v["id"]], env, 1)
    raise NotImplementedError(f"direct operand op {v['op']!r}")


def named_stage1(starkinfo: dict, airvalues: np.ndarray) -> dict[str, list[int]]:
    """name -> stage-1 words, repeated names appended in map order (the
    oracle's binding shape)."""
    out: dict[str, list[int]] = {}
    off = 0
    for v in starkinfo["airValuesMap"]:
        if v["stage"] == 1:
            out.setdefault(v["name"], []).append(int(airvalues[off]))
            off += 1
        else:
            off += 3
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", type=pathlib.Path, required=True)
    ap.add_argument("--dump", type=pathlib.Path, required=True)
    ap.add_argument("--instances", required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()

    si = json.loads((args.key / "Main.starkinfo.json").read_text())
    ei = json.loads((args.key / "Main.expressionsinfo.json").read_text())
    exps = {e["expId"]: e["code"] for e in ei["expressionsCode"]}
    im_hints = [h for h in ei["hintsInfo"] if h["name"] == "im_airval"]

    segments = []
    for k, inst in enumerate(args.instances.split(",")):
        airvalues = np.load(args.dump / f"{inst}_airvalues.npy")
        challenges = np.load(args.dump / f"{inst}_challenges.npy")
        env = scalar_env(si, airvalues, challenges)
        fractions = []
        direct = _cubic([0, 0, 0])
        for h in im_hints:
            num = eval_operand(_values(h, "numerator")[0], env, exps)
            den = eval_operand(_values(h, "denominator")[0], env, exps)
            direct = direct + num / den
            fractions.append({"numerator": _limbs(num), "denominator": _limbs(den)})
        ch = challenges.reshape(-1, 3)
        seg = {
            "instance": inst,
            "airvalues_named": named_stage1(si, airvalues),
            "alpha": [str(int(w)) for w in ch[0]],
            "gamma": [str(int(w)) for w in ch[1]],
            "direct_term": _limbs(direct),
        }
        if k == 0:
            seg["interactions"] = fractions
        segments.append(seg)
        print(f"{inst}: {len(fractions)} im_airval hints evaluated")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {"n_bits": si["starkStruct"]["nBits"], "segments": segments}, indent=1
        )
        + "\n"
    )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
