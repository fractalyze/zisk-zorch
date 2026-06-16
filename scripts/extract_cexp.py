#!/usr/bin/env python3
"""Extract a ZisK AIR's stage-2 cExp fragment + per-constraint SSAs from the
proving key into the quotient testdata.

The stage-2 quotient byte-match (`zisk_zorch/quotient`) is pinned by two vendored
proving-key artifacts per AIR:

  testdata/<air>_cexp.json        the composite cExp fragment the golden harness
                                  interprets (cExpId + the *PolsMap maps + the
                                  flat `code` SSA) -> the reference `q`.
  testdata/<air>_constraints.json the individual `constraints[]` (per-constraint
                                  SSAs) the AIR-general fold consumes
                                  (`cexp_ref.evaluate_from_constraints`).

These were hand-extracted for Binary + MemAlignReadByte; this reproduces that
extraction so a new chip (Arith, BinaryAdd, ...) is one command. Source: the
ziskup proving key (`$ZISK_PROVING_KEY`, else ~/.zisk/provingKey/zisk/Zisk/airs/),
files `<Air>.starkinfo.json` + `<Air>.expressionsinfo.json`.

Recipe (reverse-engineered, validated byte-identical against the vendored pair):
  - cExpId/qDim/qDeg/boundaries/openingPoints + challengesMap/airValuesMap/
    airgroupValuesMap : verbatim from starkinfo.
  - cmPolsMap    : starkinfo entries projected to {name, stage, dim, stagePos}.
  - constPolsMap : starkinfo entries projected to {name, dim} (constants carry no
                   stage/stagePos).
  - code         : the expressionsCode[*] entry whose expId == cExpId.
  - constraints  : expressionsinfo.constraints[] projected to
                   {code, boundary, stage, imPol, line}.

The generic fold (`evaluate_from_constraints`) only handles all-`everyRow` AIRs
today; check the printed boundaries before wiring a new chip's golden case.

Usage:
  python scripts/extract_cexp.py Arith [BinaryAdd ...]   # write <air>_{cexp,constraints}.json
  python scripts/extract_cexp.py --check                 # regen Binary+MemAlign, diff vs vendored
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_PK = Path(
    os.environ.get("ZISK_PROVING_KEY", Path.home() / ".zisk/provingKey/zisk/Zisk/airs")
)
_TESTDATA = Path(__file__).resolve().parent.parent / "zisk_zorch/quotient/testdata"

# AIR name -> testdata file stem, for names that aren't a plain lowercase.
_STEM = {"MemAlignReadByte": "memalign_readbyte"}

_FRAG_VERBATIM = (
    "cExpId",
    "qDim",
    "qDeg",
    "boundaries",
    "openingPoints",
    "challengesMap",
    "airValuesMap",
    "airgroupValuesMap",
)
_CM_KEYS = ("name", "stage", "dim", "stagePos")
_CONST_KEYS = ("name", "dim")
_CONSTRAINT_KEYS = ("code", "boundary", "stage", "imPol", "line")


def _load(air: str, ext: str) -> dict:
    return json.loads((_PK / air / "air" / f"{air}.{ext}.json").read_text())


def _project(cols: list[dict], keys: tuple[str, ...]) -> list[dict]:
    return [{k: c[k] for k in keys if k in c} for c in cols]


def extract_fragment(air: str) -> dict:
    """The composite cExp fragment for `air` — preserves the vendored key order so
    a regenerated file matches byte-for-byte."""
    si = _load(air, "starkinfo")
    exp = _load(air, "expressionsinfo")
    code = next(e["code"] for e in exp["expressionsCode"] if e["expId"] == si["cExpId"])
    out = {"air": air}
    out["cExpId"] = si["cExpId"]
    out["qDim"] = si["qDim"]
    out["qDeg"] = si["qDeg"]
    out["boundaries"] = si["boundaries"]
    out["openingPoints"] = si["openingPoints"]
    out["cmPolsMap"] = _project(si["cmPolsMap"], _CM_KEYS)
    out["constPolsMap"] = _project(si["constPolsMap"], _CONST_KEYS)
    out["challengesMap"] = si["challengesMap"]
    out["airValuesMap"] = si["airValuesMap"]
    out["airgroupValuesMap"] = si["airgroupValuesMap"]
    out["code"] = code
    return out


def extract_constraints(air: str) -> dict:
    """The per-constraint SSAs for `air` (the AIR-general fold's input)."""
    exp = _load(air, "expressionsinfo")
    return {"air": air, "constraints": _project(exp["constraints"], _CONSTRAINT_KEYS)}


def _stem(air: str) -> str:
    return _STEM.get(air, air.lower())


def _write(air: str) -> None:
    stem = _stem(air)
    frag, cons = extract_fragment(air), extract_constraints(air)
    for suffix, data in (("cexp", frag), ("constraints", cons)):
        (_TESTDATA / f"{stem}_{suffix}.json").write_text(json.dumps(data, indent=1))
    boundaries = sorted({c["boundary"] for c in cons["constraints"]})
    print(
        f"{air} -> {stem}_{{cexp,constraints}}.json "
        f"({len(frag['code'])} cExp ops, {len(cons['constraints'])} constraints, "
        f"boundaries={boundaries})"
    )
    if boundaries != ["everyRow"]:
        print(
            f"  WARNING: {air} has non-everyRow boundaries — "
            "evaluate_from_constraints raises NotImplementedError until those are wired."
        )


def _check() -> int:
    bad = 0
    for air in ("Binary", "MemAlignReadByte"):
        stem = _stem(air)
        for suffix, got in (("cexp", extract_fragment(air)), ("constraints", extract_constraints(air))):
            vendored = json.loads((_TESTDATA / f"{stem}_{suffix}.json").read_text())
            ok = got == vendored
            print(f"{'OK ' if ok else 'BAD'} {stem}_{suffix}.json")
            bad += not ok
    return bad


if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv == ["--check"]:
        sys.exit(_check())
    if not argv:
        sys.exit(__doc__)
    for name in argv:
        _write(name)
