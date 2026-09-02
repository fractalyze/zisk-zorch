"""Random instances for the export byte-gates, and their on-disk form for
the Rust bridge's `zz_prove`.

    FRX_PLATFORMS=cuda python -m zisk_zorch.export.cases \\
        --proving_key=<provingKey> --artifacts=<dir> --air=RomData --out=<case>

writes the instance and the key's fixed sections as raw little-endian u64
files plus `case.json`, and `expected_proof.bin`: the Python replay's flat
proof over the same artifacts. `zz_prove <artifacts> <case>` must
reproduce it word for word — the same gate `stages_test` runs between the
replay and the Python prover, one link further down.
"""

from __future__ import annotations

import argparse
import json
import pathlib
from dataclasses import dataclass

import numpy as np
from zk_dtypes import goldilocks as F

from zisk_zorch.export import replay
from zisk_zorch.export.export_air import export_key
from zisk_zorch.export.runtime import Artifact
from zisk_zorch.harness.pil2 import Pil2Key, value_offsets

_ORDER = int(F(0) - F(1)) + 1


def _words(rng: np.random.Generator, shape) -> np.ndarray:
    return rng.integers(0, _ORDER, size=shape, dtype=np.uint64)


def packed_width(values_map: list) -> int:
    offs = value_offsets(values_map)
    return offs[-1][1] + offs[-1][2] if offs else 0


@dataclass(frozen=True)
class Case:
    key: Pil2Key
    instance: replay.Instance
    sections: replay.KeySections


def random_case(key: Pil2Key, seed: int = 166) -> Case:
    """A uniformly random instance of `key`'s AIR (trace, scalars, seed)
    and random custom sections — no constraint holds, which is fine for a
    byte-gate: every stage is a deterministic function of its inputs."""
    si = key.starkinfo
    rng = np.random.default_rng(seed)
    n = 1 << si["starkStruct"]["nBits"]
    instance = replay.Instance(
        trace=_words(rng, (n, si["mapSectionsN"]["cm1"])),
        publics=_words(rng, (si["nPublics"],)),
        airvalues=_words(rng, (packed_width(si.get("airValuesMap") or []),)),
        proofvalues=_words(rng, (packed_width(si.get("proofValuesMap") or []),)),
        global_challenge=_words(rng, (3,)),
    )
    sections = replay.KeySections(
        const_base=np.asarray(key.const_base).view(np.uint64),
        custom_base={ci: _words(rng, buf.shape) for ci, buf in key.custom_base.items()},
    )
    return Case(key, instance, sections)


def write_case(
    case: Case, air: str, out: pathlib.Path, expected_proof: np.ndarray | None
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    si = case.key.starkinfo
    inst, sec = case.instance, case.sections
    files = {
        "trace": inst.trace,
        "publics": inst.publics,
        "airvalues": inst.airvalues,
        "proofvalues": inst.proofvalues,
        "global_challenge": inst.global_challenge,
        "const_base": sec.const_base,
    }
    for ci, buf in sec.custom_base.items():
        files[f"custom_base_{ci}"] = buf
    if expected_proof is not None:
        files["expected_proof"] = expected_proof
    for name, arr in files.items():
        np.ascontiguousarray(arr, dtype=np.uint64).tofile(out / f"{name}.bin")
    (out / "case.json").write_text(
        json.dumps({"air": air, "n_bits": si["starkStruct"]["nBits"]})
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--proving_key", required=True, type=pathlib.Path)
    ap.add_argument("--artifacts", required=True, type=pathlib.Path)
    ap.add_argument("--air", required=True)
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--seed", type=int, default=166)
    args = ap.parse_args()
    key = export_key(args.proving_key, args.air)
    case = random_case(key, args.seed)
    nb = key.starkinfo["starkStruct"]["nBits"]
    proof, _ = replay.prove(
        Artifact(args.artifacts / f"{args.air}_n{nb}"), case.sections, case.instance
    )
    write_case(case, args.air, args.out, proof)
    print(f"wrote {args.out} ({proof.size}-word expected proof)")


if __name__ == "__main__":
    main()
