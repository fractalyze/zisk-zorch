"""A basic ZisK AIR's `Pil2Key` from the proving key alone (#115).

`Capture.pil2_key` sources the extended constant and custom sections from
dump sections, so it exists only where a native capture does. A witness
source that hands traces over in memory has just the proving-key
directory; like `recursion_pil2_key`, the base constants come from
``<Air>.const`` and the extended section from the prover's own coset LDE
— exact field arithmetic, so it is equal to the dumped section or wrong.
"""

from __future__ import annotations

import json
import pathlib

import frx.numpy as fnp
import numpy as np
from zk_dtypes import goldilocks as F

from zisk_zorch.commit.trace_commit import extend
from zisk_zorch.harness.pil2 import Pil2Key


def zisk_air_base(key: pathlib.Path, gi: dict, air: str) -> pathlib.Path:
    """``<root>/<group0>/airs/<air>/air/<air>`` — the basic-AIR artifact
    stem the key's JSON/const files hang off (native ziskup keys root at
    ``zisk/``, the example builds at ``build/``)."""
    root = key / "zisk" if (key / "zisk").is_dir() else key / "build"
    return root / gi["air_groups"][0] / "airs" / air / "air" / air


def zisk_hash_family(gi: dict) -> str:
    """`pilout.globalInfo.json`'s sponge selection, defaulted like
    `Capture.hash_family`: a key that ships no ``hash`` entry is an
    example key, and those are Poseidon2."""
    family = gi.get("hash", "Poseidon2")
    if family not in ("Poseidon1", "Poseidon2"):
        raise ValueError(
            f"unknown hash family {family!r} — expected 'Poseidon1' or 'Poseidon2'"
        )
    return family


def zisk_pil2_key(key: pathlib.Path, gi: dict, air: str) -> Pil2Key:
    """The AIR's proving-key artifacts with both constant domains
    materialized. Custom commits (Rom) are not resolvable from the key
    alone yet — who supplies the rom section is #115 contract question 5 —
    so a custom-bearing AIR fails loudly here instead of proving with an
    empty section."""
    base = zisk_air_base(key, gi, air)
    si = json.loads(pathlib.Path(f"{base}.starkinfo.json").read_text())
    if si.get("customCommits"):
        raise NotImplementedError(
            f"{air}: custom commits are not resolvable from the proving key "
            "alone (zisk-zorch#115 contract question 5) — use a capture"
        )
    ss = si["starkStruct"]
    n = 1 << ss["nBits"]
    blowup = 1 << (ss["nBitsExt"] - ss["nBits"])
    const_base = np.fromfile(f"{base}.const", dtype=np.uint64).reshape(
        n, si["nConstants"]
    )
    const_ext = np.asarray(
        extend(fnp.array(const_base.view(F)), blowup), dtype=np.uint64
    ).view(F)
    return Pil2Key(
        starkinfo=si,
        expressionsinfo=json.loads(
            pathlib.Path(f"{base}.expressionsinfo.json").read_text()
        ),
        const_base=const_base.view(F),
        const_ext=const_ext,
        custom_ext={},
        custom_base={},
        hash_family=zisk_hash_family(gi),
    )
