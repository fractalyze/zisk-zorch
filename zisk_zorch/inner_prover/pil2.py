"""pil2's value-packing and buffer-layout conventions, in one place.

The pil2-mode prover roles (`quotient/witness_std.py`, `quotient/prover.py`,
`opening/prover.py`) and the capture loader (`inner_prover/capture.py`) read
the same proving-key maps and pack the same buffers, so the conventions live
here below both: how scalar value sections pack (stage-1 values as one
Goldilocks word, stage>=2 as three), how a ``cmPolsMap``/``evMap`` entry
resolves to a committed column (cubic entries joined from 3 contiguous
lanes), and how a stage's challenges squeeze off the transcript in
``challengesMap`` order. A packing edit here lands in the prover and the
byte-gates' reference reads at once — they cannot drift apart.

`Pil2Key` bundles the proving-key artifacts the roles are configured with;
instance data (publics, the trace, the global challenge) rides the claims in
`types.py` instead.
"""

from __future__ import annotations

from dataclasses import dataclass

import frx.numpy as fnp
import numpy as np
from frx import Array
from zk_dtypes import goldilocks as F
from zk_dtypes import goldilocksx3 as F3
from zorch.utils.field import join_coeffs

from zisk_zorch.transcript.transcript import Transcript


@dataclass(frozen=True)
class Pil2Key:
    """One AIR's proving-key artifacts — static configuration for the
    pil2-mode roles, fixed across instances of the AIR.

    `starkinfo` / `expressionsinfo` are the key's JSON maps as parsed dicts
    (shape metadata, the challenge/column/eval maps, the SSA code blocks).
    The constant columns come in both domains because the stage-2 witness
    expressions evaluate over the base domain while the quotient and the
    openings read the extended sections; `custom_ext` holds each custom
    commit's extended section keyed by ``commitId``."""

    starkinfo: dict
    expressionsinfo: dict
    const_base: np.ndarray
    const_ext: np.ndarray
    custom_ext: dict[int, np.ndarray]


def to_field(words) -> Array:
    """u64 words -> a base-field device array."""
    return fnp.array(np.asarray(words, dtype=np.uint64).astype(F))


def absorb_words(transcript: Transcript, words) -> None:
    """Absorb dumped u64 words as base-field elements."""
    transcript.put(to_field(words))


def cubic_scalar(words) -> Array:
    """3 u64 limbs -> a cubic scalar."""
    return join_coeffs(to_field(words).reshape(3), F3).reshape(())


def values_env(words: np.ndarray, vmap: list) -> dict[int, Array]:
    """A dumped value section as the SSA interpreter's scalar environment:
    pil2 packs stage-1 values as one word, stage>=2 as three."""
    out, off = {}, 0
    for i, v in enumerate(vmap):
        if v["stage"] == 1:
            out[i] = cubic_scalar([words[off], 0, 0])
            off += 1
        else:
            out[i] = cubic_scalar(words[off : off + 3])
            off += 3
    return out


def publics_env(words: np.ndarray) -> dict[int, Array]:
    """The publics section as the interpreter's scalar environment — one
    base word per public, embedded cubic."""
    return {i: cubic_scalar([words[i], 0, 0]) for i in range(len(words))}


def stage_challenge_ids(challenges_map: list, stage: int) -> list[int]:
    """The ``challengesMap`` indices squeezed at `stage`'s boundary, in map
    (= transcript) order."""
    return [i for i, c in enumerate(challenges_map) if c.get("stage") == stage]


def squeeze_stage_challenges(
    transcript: Transcript, challenges_map: list, stage: int
) -> dict[int, Array]:
    """Squeeze `stage`'s challenges off the transcript — one cubic scalar per
    ``challengesMap`` entry of that stage, keyed by the entry's map index (the
    id the SSA operands reference)."""
    return {
        i: join_coeffs(transcript.get_field().reshape(-1, 3), F3).reshape(())
        for i in stage_challenge_ids(challenges_map, stage)
    }


def named_challenge(challenges_map: list, name: str) -> int:
    """The ``challengesMap`` index of the challenge called `name`."""
    return next(i for i, c in enumerate(challenges_map) if c.get("name") == name)


def committed_column(entry: dict, cmp_map: list, bufs: dict) -> Array:
    """An ``evMap``-shaped entry's committed column, cubic entries joined
    from their 3 contiguous lanes. `bufs` maps ``(kind, stage-or-commitId)``
    to the section matrix (device or host; host slices copy on convert)."""
    if entry["type"] == "cm":
        pm = cmp_map[entry["id"]]
        buf = bufs[("cm", pm["stage"])]
        if pm["dim"] == 1:
            return fnp.asarray(buf[:, pm["stagePos"]])
        lanes = buf[:, pm["stagePos"] : pm["stagePos"] + 3]
        if isinstance(lanes, np.ndarray):
            lanes = np.ascontiguousarray(lanes)
        return join_coeffs(fnp.asarray(lanes), F3)
    if entry["type"] == "const":
        return fnp.asarray(bufs[("const", 0)][:, entry["id"]])
    return fnp.asarray(bufs[("custom", entry["commitId"])][:, entry["id"]])


def cm_env(cmp_map: list, bufs: dict) -> dict[int, Array]:
    """Every committed column present in `bufs`, keyed by ``cmPolsMap``
    index. Entries whose stage has no buffer yet (the quotient's own section
    while it is being computed) are simply absent, so a stray SSA reference
    to one fails loudly."""
    return {
        i: committed_column({"type": "cm", "id": i}, cmp_map, bufs)
        for i, pm in enumerate(cmp_map)
        if ("cm", pm["stage"]) in bufs
    }


def const_env(bufs: dict, n_constants: int) -> dict[int, Array]:
    """The constant columns as the interpreter's ``const`` environment."""
    return {i: fnp.asarray(bufs[("const", 0)][:, i]) for i in range(n_constants)}


def custom_env(bufs: dict, custom_commits: list) -> dict[tuple[int, int], Array]:
    """The custom-commit columns keyed ``(commitId, column)``."""
    return {
        (ci, j): fnp.asarray(bufs[("custom", ci)][:, j])
        for ci in range(len(custom_commits))
        for j in range(bufs[("custom", ci)].shape[1])
    }
