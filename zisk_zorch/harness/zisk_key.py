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
import sys

from zisk_zorch.harness.pil2 import Pil2Key
from zisk_zorch.harness.recursion import const_pil2_key, key_root


def zisk_air_base(key: pathlib.Path, gi: dict, air: str) -> pathlib.Path:
    """``<root>/<group0>/airs/<air>/air/<air>`` — the basic-AIR artifact
    stem the key's JSON/const files hang off (`recursion.key_root` probes
    the native-ziskup ``zisk/`` vs example ``build/`` root)."""
    return key_root(key) / gi["air_groups"][0] / "airs" / air / "air" / air


def zisk_hash_family(gi: dict) -> str:
    """`pilout.globalInfo.json`'s sponge selection. A key that ships no
    ``hash`` entry is assumed to be an example key (those are Poseidon2) —
    looser than `Capture.hash_family`, which raises when the globalInfo
    file exists but omits the entry; the assumption is announced on stderr
    because a wrong guess byte-mismatches the whole prove with nothing
    pointing here."""
    family = gi.get("hash")
    if family is None:
        print(
            "pilout.globalInfo.json ships no 'hash' entry; assuming Poseidon2",
            file=sys.stderr,
        )
        return "Poseidon2"
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
    return const_pil2_key(
        si,
        json.loads(pathlib.Path(f"{base}.expressionsinfo.json").read_text()),
        f"{base}.const",
        zisk_hash_family(gi),
    )
