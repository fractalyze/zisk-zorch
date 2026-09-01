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

from zisk_zorch.harness.pil2 import Pil2Key
from zisk_zorch.harness.recursion import const_pil2_key, key_root


def zisk_air_base(key: pathlib.Path, gi: dict, air: str) -> pathlib.Path:
    """``<root>/<group0>/airs/<air>/air/<air>`` — the basic-AIR artifact
    stem the key's JSON/const files hang off (`recursion.key_root` probes
    the native-ziskup ``zisk/`` vs example ``build/`` root)."""
    return key_root(key) / gi["air_groups"][0] / "airs" / air / "air" / air


def zisk_hash_family(gi: dict) -> str:
    """`pilout.globalInfo.json`'s sponge selection, resolved the way pil2
    resolves it: an absent ``hash`` means ``DEFAULT_HASH_ID``, which is
    Poseidon1 (see `zisk_zorch/poseidon1/goldilocks.py` — the shipped
    ziskup v1.0.0-alpha key omitted the entry and native committed its
    stage-1 trees with Poseidon1 because of exactly this default).

    A present-but-unrecognized value still raises: that is a malformed key,
    not a default. An explicitly wrong family byte-mismatches every tree,
    transcript, and grind of the prove with nothing pointing here."""
    family = gi.get("hash", "Poseidon1")
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
