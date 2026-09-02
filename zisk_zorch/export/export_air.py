"""Export one AIR's schedule as StableHLO artifacts plus a manifest.

    FRX_PLATFORMS=cuda python -m zisk_zorch.export.export_air \\
        --proving_key=<provingKey> --air=Main --out=artifacts

writes ``<out>/<Air>_n<nBits>/<program>.mlirbc`` for every program of
`stages.AirPrograms` and ``manifest.json`` beside them: the schedule facts
and, per program, every input and output by name, dtype and shape in
parameter order. The C side (`bridge/`) binds buffers by those names.

Export against the backend you will run on — `lax.ntt` and the hash
regions lower shape-specialized per backend. The artifact is one `(air,
nBits)`: the proving key fixes a basic AIR's height, so the directory is
named for both.

The key loads through `zisk_key.zisk_pil2_key`'s recipe (constants from
``<Air>.const``, extended by our own LDE). An AIR with custom commits (Rom)
gets zero-filled custom sections here: only their SHAPES trace into the
programs — the real section arrives at runtime through `StepsParams`, and
the ``custom_setup`` program extends and commits it on device.
"""

from __future__ import annotations

import argparse
import io
import json
import pathlib

import numpy as np
from zk_dtypes import goldilocks as F

import frx
from zisk_zorch.export.stages import AirPrograms, Program, raw_boundary
from zisk_zorch.harness.pil2 import Pil2Key
from zisk_zorch.harness.pil2_prover import Pil2InnerProver
from zisk_zorch.harness.recursion import const_pil2_key
from zisk_zorch.harness.zisk_key import zisk_air_base, zisk_hash_family


def export_key(proving_key: pathlib.Path, air: str) -> Pil2Key:
    """The AIR's `Pil2Key` for export: real constants, shape-only custom
    sections (see the module docstring)."""
    gi = json.loads((proving_key / "pilout.globalInfo.json").read_text())
    base = zisk_air_base(proving_key, gi, air)
    si = json.loads(pathlib.Path(f"{base}.starkinfo.json").read_text())
    key = const_pil2_key(
        si,
        json.loads(pathlib.Path(f"{base}.expressionsinfo.json").read_text()),
        f"{base}.const",
        zisk_hash_family(gi),
    )
    customs = si.get("customCommits") or []
    if not customs:
        return key
    n, ne = 1 << si["starkStruct"]["nBits"], 1 << si["starkStruct"]["nBitsExt"]
    custom_base, custom_ext = {}, {}
    for ci, c in enumerate(customs):
        width = c["stageWidths"][0]
        custom_base[ci] = np.zeros((n, width), dtype=np.uint64).view(F)
        custom_ext[ci] = np.zeros((ne, width), dtype=np.uint64).view(F)
    return Pil2Key(
        starkinfo=key.starkinfo,
        expressionsinfo=key.expressionsinfo,
        const_base=key.const_base,
        const_ext=key.const_ext,
        custom_ext=custom_ext,
        hash_family=key.hash_family,
        custom_base=custom_base,
    )


def lower(program: Program) -> bytes:
    """The program's StableHLO bytecode. `keep_unused` keeps the lowered
    signature equal to the manifest's: jit's default prunes parameters a
    program never reads (a scalar section no expression references), and
    the bridge binds by position."""
    with frx.enable_x64():
        lowered = frx.jit(program.fn, keep_unused=True).lower(*program.avals())
    module = lowered.compiler_ir(dialect="stablehlo")
    n_params = _entry_arity(module)
    if n_params != len(program.inputs):
        # A device array the program closed over (an interned constant pack
        # from an earlier prove in this process) lowers as a hoisted entry
        # parameter; the manifest would then lie to the bridge.
        raise RuntimeError(
            f"{program.name}: lowered entry takes {n_params} parameters, "
            f"manifest lists {len(program.inputs)} — the program closes over "
            "a device array; compute it in-trace instead"
        )
    buf = io.BytesIO()
    module.operation.write_bytecode(buf)
    return buf.getvalue()


def _entry_arity(module) -> int:
    """How many parameters the module's public ``main`` takes."""
    for op in module.body.operations:
        if op.operation.name == "func.func" and op.sym_name.value == "main":
            return len(op.type.inputs)
    raise RuntimeError("lowered module has no main")


def spec_json(spec) -> dict:
    return {"name": spec.name, "dtype": spec.dtype, "dims": list(spec.dims)}


def export_air(proving_key: pathlib.Path, air: str, out: pathlib.Path) -> pathlib.Path:
    """Lower every program of `air` and write the artifact directory."""
    key = export_key(proving_key, air)
    programs = AirPrograms(key, Pil2InnerProver(key))
    nb = programs.nb
    art = out / f"{air}_n{nb}"
    art.mkdir(parents=True, exist_ok=True)
    manifest = {
        "air": air,
        "artifact_version": 1,
        **programs.schedule(),
        "programs": {},
    }
    for program in map(raw_boundary, programs.programs()):
        outputs = program.outputs()  # traces once for the manifest
        code = lower(program)  # lowers the same function for the bytecode
        (art / f"{program.name}.mlirbc").write_bytes(code)
        manifest["programs"][program.name] = {
            "file": f"{program.name}.mlirbc",
            "inputs": [spec_json(s) for s in program.inputs],
            "outputs": [spec_json(s) for s in outputs],
        }
        print(f"wrote {art / program.name}.mlirbc ({len(code)} B)")
    (art / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return art


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--proving_key", type=pathlib.Path, required=True)
    ap.add_argument("--air", action="append", required=True, help="AIR name; repeatable, or 'all'")
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("artifacts"))
    args = ap.parse_args()
    airs = args.air
    if airs == ["all"]:
        gi = json.loads((args.proving_key / "pilout.globalInfo.json").read_text())
        airs = [a["name"] for a in gi["airs"][0]]
    for air in airs:
        art = export_air(args.proving_key, air, args.out)
        print(f"exported {air} -> {art}")


if __name__ == "__main__":
    main()
