# Copyright 2026 The zisk-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Assemble the cExp interpreter's operand environment from real prover data.

`cexp_ref` interprets a proving key's SSA against an `env` of operand pools;
`cexp_ref._load_inputs` builds that env from a golden case's synthetic inputs.
This module builds the same env from the arrays an actual prove has — the
committed columns, the key's constants, the transcript's challenges — so the
same interpreter can serve a real proof rather than only a reference check
(fractalyze/zisk-zorch#107).

The packing conventions here are pil2's, and each one is load-bearing:
air/airgroup/proof value buffers pack a stage-1 entry as one word and a
stage>=2 entry as three; `publics` are base-field scalars widened to the cubic
extension; committed columns are indexed by `cmPolsMap` position, with a cubic
column occupying three consecutive lanes of its stage's buffer.
"""

from __future__ import annotations

import frx.numpy as fnp
import numpy as np
from frx import Array
from zk_dtypes import goldilocks as F, goldilocksx3 as F3


def cubic_scalar(words) -> Array:
    """A single cubic element from three canonical u64 lanes."""
    return fnp.array(np.asarray(words, dtype=np.uint64).astype(F).view(F3))[0]


def values_env(words: np.ndarray, value_map: list[dict]) -> dict[int, Array]:
    """One of pil2's air/airgroup/proof value buffers, split per its map.

    Stage-1 entries occupy one word (a base-field value widened here), later
    stages three — reading them uniformly silently shifts every entry after the
    first stage-1 one.
    """
    out: dict[int, Array] = {}
    off = 0
    for i, v in enumerate(value_map):
        if v["stage"] == 1:
            out[i] = cubic_scalar([words[off], 0, 0])
            off += 1
        else:
            out[i] = cubic_scalar(words[off : off + 3])
            off += 3
    return out


def column(buf: np.ndarray, entry: dict) -> Array:
    """A committed column from its stage buffer: a cubic column joins the three
    consecutive lanes at its `stagePos`, a base column is taken as is."""
    pos = entry["stagePos"]
    if entry["dim"] == 3:
        lanes = np.ascontiguousarray(buf[:, pos : pos + 3])
        return fnp.array(lanes.view(F3).reshape(buf.shape[0]))
    return fnp.array(buf[:, pos])


def build_env(
    starkinfo: dict,
    *,
    stage_bufs: dict[int, np.ndarray],
    const: np.ndarray,
    challenges: Array,
    publics: np.ndarray,
    airvalues: np.ndarray,
    airgroupvalues: np.ndarray,
    proofvalues: np.ndarray,
    inv_zerofier: Array,
    custom_bufs: dict[int, np.ndarray] | None = None,
) -> dict:
    """The operand env `cexp_ref._run_block` resolves against.

    `stage_bufs` maps a stage number to that stage's committed columns over the
    domain the expression is evaluated on; `const` and `custom_bufs` likewise.
    `challenges` is the transcript's cubic challenge vector, indexed by
    `challengesMap` position.
    """
    cmp_map = starkinfo["cmPolsMap"]
    custom_bufs = custom_bufs or {}
    return {
        # A base-domain env legitimately carries only the stages computed so
        # far, so a column whose stage has no buffer is simply absent rather
        # than an error — the caller binds it later if an expression needs it.
        "cm": {
            i: column(stage_bufs[e["stage"]], e)
            for i, e in enumerate(cmp_map)
            if e["stage"] in stage_bufs
        },
        "const": {i: fnp.array(const[:, i]) for i in range(starkinfo["nConstants"])},
        "custom": {
            (ci, j): fnp.array(buf[:, j])
            for ci, buf in custom_bufs.items()
            for j in range(buf.shape[1])
        },
        "challenges": {
            i: fnp.array(challenges[i : i + 1])[0] for i in range(len(challenges))
        },
        "publics": {i: cubic_scalar([publics[i], 0, 0]) for i in range(len(publics))},
        "airvalues": values_env(airvalues, starkinfo["airValuesMap"]),
        "airgroupvalues": values_env(airgroupvalues, starkinfo["airgroupValuesMap"]),
        "proofvalues": values_env(proofvalues, starkinfo["proofValuesMap"]),
        "zi": {0: inv_zerofier},
    }
