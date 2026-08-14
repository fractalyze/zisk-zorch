"""pil2's value-packing and buffer-layout conventions, in one place.

The pil2-mode prover roles and composite (`pil2_prover.py`), the capture
loader (`capture.py`), the schedule replay (`schedule.py`), and the gates
read the same proving-key maps and pack the same buffers, so the conventions
live here below all of them: how scalar value sections pack (stage-1 values
as one Goldilocks word, stage>=2 as three), how a ``cmPolsMap``/``evMap``
entry resolves to a committed column (cubic entries joined from 3 contiguous
lanes), how a stage's challenges squeeze off the transcript in
``challengesMap`` order, and the evMap opening / two-challenge DEEP forms
pil2's opening phase computes. A packing edit here lands in the prover and
the byte-gates' reference reads at once — they cannot drift apart.

`Pil2Key` bundles the proving-key artifacts the roles are configured with;
instance data (publics, the trace, the global challenge) rides the claims in
`pil2_prover.py` instead.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field

import frx
import frx.numpy as fnp
import numpy as np
from frx import Array
from zk_dtypes import goldilocks as F
from zk_dtypes import goldilocksx3 as F3
from zk_dtypes import pfinfo
from zorch.pcs.deep import open_columns
from zorch.utils.field import join_coeffs

from zisk_zorch.evals.lev import _TWO_ADIC_ROOT
from zisk_zorch.transcript.transcript import DIGEST, Transcript

MODULUS = int(pfinfo(F).modulus)


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
    # `pilout.globalInfo.json`'s `hash` — the sponge every tree, transcript,
    # and grind of this key runs ("Poseidon1" on native ZisK keys).
    hash_family: str = "Poseidon2"
    # Base-domain custom sections, keyed like `custom_ext`. The stage-2
    # witness expressions evaluate over the base domain (Rom's read the
    # `rom` custom commit there); recovered from the extended sections via
    # `unextend` where the source provides only those.
    custom_base: dict[int, np.ndarray] = field(default_factory=dict)


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


def device_sections(key: Pil2Key, domain: str) -> tuple[Array, dict[int, Array]]:
    """The key's `domain` ("base"/"ext") constant and custom sections as
    device arrays, uploaded once per key and shared by every role holding
    it — per-role copies double the resident footprint, which matters
    beside a chunked-quotient working set that already fills the card."""
    cache = getattr(key, "_device_sections", None)
    if cache is None:
        cache = {}
        object.__setattr__(key, "_device_sections", cache)
    if domain not in cache:
        const = key.const_base if domain == "base" else key.const_ext
        customs = key.custom_base if domain == "base" else key.custom_ext
        cache[domain] = (
            fnp.asarray(const),
            {ci: fnp.asarray(buf) for ci, buf in customs.items()},
        )
    return cache[domain]


def release_device_sections(key: Pil2Key) -> None:
    """Drop `key`'s uploaded sections. The cache rides the key, and the key
    outlives the roles (`Capture.release` keeps it), so dropping a family's
    prover frees its device residency only through this."""
    cache = getattr(key, "_device_sections", None)
    if cache is not None:
        cache.clear()


def scalar_env(
    starkinfo: dict,
    *,
    publics: np.ndarray,
    airvalues: np.ndarray,
    proofvalues: np.ndarray,
    challenges: dict,
    airgroupvalues: dict,
) -> dict:
    """The SSA interpreter's scalar operand classes, packed once for every
    environment builder — challenges and airgroup values arrive already
    keyed (they may be squeezed/computed rather than dumped)."""
    return {
        "challenges": challenges,
        "publics": publics_env(publics),
        "airvalues": values_env(airvalues, starkinfo["airValuesMap"]),
        "airgroupvalues": airgroupvalues,
        "proofvalues": values_env(proofvalues, starkinfo["proofValuesMap"]),
    }


def absorb_stage2_airvalues(
    transcript: Transcript, words: np.ndarray, values_map: list
) -> None:
    """Absorb the stage-2 air values off the packed section: the schedule
    absorbs only the stage-2 triples, walking past one word per stage-1
    entry (`values_env`'s packing)."""
    off = 0
    for v in values_map:
        if v["stage"] == 1:
            off += 1
        elif v["stage"] == 2:
            absorb_words(transcript, words[off : off + 3])
            off += 3


def hint_value(hint: dict, name: str) -> dict:
    """A hint's named field's (single) value entry."""
    return next(f for f in hint["fields"] if f["name"] == name)["values"][0]


def transcript_width(stark_struct: dict) -> int:
    """The prove transcript's width: ``DIGEST`` lanes per transcript arity."""
    return DIGEST * stark_struct.get("transcriptArity", 3)


def stage_challenge_ids(challenges_map: list, stage: int) -> list[int]:
    """The ``challengesMap`` indices squeezed at `stage`'s boundary, in map
    (= transcript) order."""
    return [i for i, c in enumerate(challenges_map) if c.get("stage") == stage]


@functools.lru_cache(maxsize=None)
def _squeeze_leg_jit(n: int):
    # The transcript rides the boundary as a pytree; its squeeze cursor is
    # static aux, so the flush-or-read branch inside get_field traces
    # concretely and each cursor state compiles its own tiny zone.
    def leg(t: Transcript) -> tuple[Transcript, Array]:
        return t, fnp.stack([t.get_field() for _ in range(n)])

    return frx.jit(leg)


def squeeze_stage_challenges(
    transcript: Transcript, challenges_map: list, stage: int
) -> dict[int, Array]:
    """Squeeze `stage`'s challenges off the transcript — one cubic scalar per
    ``challengesMap`` entry of that stage, keyed by the entry's map index (the
    id the SSA operands reference). The whole leg runs as one jit zone; the
    caller's transcript advances in place."""
    ids = stage_challenge_ids(challenges_map, stage)
    if not ids:
        return {}
    advanced, limbs = _squeeze_leg_jit(len(ids))(transcript)
    transcript._state = advanced._state
    transcript._out = advanced._out
    transcript._pending = list(advanced._pending)
    transcript._out_cursor = advanced._out_cursor
    values = join_coeffs(limbs, F3)
    return {ch: values[j].reshape(()) for j, ch in enumerate(ids)}


def challenge_id(challenges_map: list, name: str) -> int:
    """The ``challengesMap`` index of the challenge called `name`."""
    idx = next((i for i, c in enumerate(challenges_map) if c.get("name") == name), None)
    if idx is None:
        raise KeyError(f"challengesMap has no challenge named {name!r}")
    return idx


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


def open_evmap_columns(columns: list, ev_map: list, lev: Array, *, stride: int):
    """pil2 STEP_EVALS' opening of the ``evMap`` columns: `open_columns`
    wants the base and extension columns as two blocks, so the split
    re-permutes back to evMap order afterwards — the layout the absorbed
    ``evals`` section carries."""
    is_base = [col.dtype == F for col in columns]
    order = [i for i, b in enumerate(is_base) if b] + [
        i for i, b in enumerate(is_base) if not b
    ]
    base = [columns[i] for i in order if is_base[i]]
    ext = [columns[i] for i in order[len(base) :]]
    n_ext = columns[0].shape[0]
    split = open_columns(
        fnp.stack(base, axis=1) if base else fnp.zeros((n_ext, 0), F),
        fnp.stack(ext, axis=1) if ext else fnp.zeros((n_ext, 0), F3),
        lev,
        [ev_map[i]["openingPos"] for i in order],
        stride=stride,
    )
    return fnp.zeros_like(split).at[np.array(order)].set(split)


def deep_two_challenge(
    columns: list,
    evals: Array,
    domain: Array,
    xi: Array,
    vf1: Array,
    vf2: Array,
    *,
    ev_map: list,
    openings: list,
    n_bits: int,
) -> Array:
    """pil2 computeFRIExpression: vf2-Horner within an opening group (evMap
    order), one reciprocal per group, vf1-Horner across groups. zorch's
    ``deep_composition`` batches with one challenge's powers, so it cannot
    express this two-challenge form. Traceable — the coset `domain` must
    arrive as an argument (an in-trace coset feeding the cubic reciprocal
    is #67's compiler-crash trigger), and the subgroup generator is host
    arithmetic so tracing never touches it."""
    g = pow(_TWO_ADIC_ROOT, 1 << (32 - n_bits), MODULUS)
    n = 1 << n_bits
    fri = None
    for k, prime in enumerate(openings):
        acc = None
        for i, e in enumerate(ev_map):
            if e["openingPos"] != k:
                continue
            term = columns[i] - evals[i]
            acc = term if acc is None else acc * vf2 + term
        if acc is None:
            # An opening point no evMap entry references is a zero group;
            # pil2 accumulates from zero, so the Horner step still scales
            # the running sum by vf1.
            if fri is not None:
                fri = fri * vf1
            continue
        shift = fnp.array(np.uint64(pow(g, prime % n, MODULUS)).astype(F))
        group = acc / (domain - xi * shift)
        fri = group if fri is None else fri * vf1 + group
    return fri
