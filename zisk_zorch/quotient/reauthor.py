"""Re-author the Binary stage-2 quotient from rw constraints + interactions.

Produces `q = (Σ_i constraint[i] · vc^(N−1−i)) · Zi` — pil2's composite-constraint
quotient — WITHOUT interpreting pil2's cExp SSA in the prover. The row-local
constraints are authored from the Binary AIR directly; the `std_sum` (LogUp
bus/gsum) constraints are authored from the chip's typed `Interaction`s. Byte-
matching this against the cExp reference `q` (`cexp_ref` / the `cexp_eval` golden)
is what verifies rw's authored interactions — the per-chip CPU test can't, since
interactions are CPU-erased there.

Binary-specific for now (the constraint set, the bus-cluster grouping, and the
row-local formulas are wired for the `binary` AIR); the `gsum_e` reconstruction
and the fold are generic and will lift to other AIRs later. See
docs/stage2-constraint-ingest.md and
https://github.com/0xPolygonHermez/pil2-proofman/blob/v0.18.0/pil2-components/lib/std/pil/std_sum.pil
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jax import Array
from zk_dtypes import goldilocks_mont as F
from zk_dtypes import goldilocksx3_mont as F3

from zisk_zorch.quotient.zerofier import inv_zerofier

_P = 0xFFFFFFFF00000001

# Binary AIR layout (proving-key cmPolsMap): 39 stage-1 base cols, then the
# stage-2 witness cubic cols gsum / im_cluster×4, then the stage-3 quotient.
_N_STAGE1 = 39
_GSUM = 39
_IM_CLUSTER = (40, 41, 42, 43)
# Each std_sum cluster is the pair of interactions whose LogUp denominators it
# combines; the 4th pairs byte-table send 7 with the operation receive.
_CLUSTERS = ((0, 1, 2), (1, 3, 4), (2, 5, 6), (3, 7, 8))  # (im_cluster_slot, iact_i, iact_j)
_TRANSITION_IACT = 0  # gsum_e[0] (byte-table send 0) gates the gsum transition


def _embed(values: list[str]) -> Array:
    """Base canonical-u64 decimals -> `F3` array of `(b, 0, 0)` embeddings (the
    numpy-level base→cubic lift; the zkx CPU emitter crashes on cubic bitcast)."""
    limbs = np.array([[int(v), 0, 0] for v in values], dtype=np.uint64)
    return jnp.array(limbs.astype(F).view(F3).reshape(limbs.shape[0]))


def _cubic(values: list[str]) -> Array:
    flat = np.array([int(v) for v in values], dtype=np.uint64).reshape(-1, 3)
    return jnp.array(flat.astype(F).view(F3).reshape(flat.shape[0]))


def _rotate(col: Array, shift: int) -> Array:
    """`out[i] = col[(i + shift) mod n]` — the extended-domain image of a
    next/previous-row opening (slice+concat; no `jnp.roll` in the zkx fork)."""
    n = col.shape[0]
    s = shift % n
    return col if s == 0 else jnp.concatenate([col[s:], col[:s]])


def _eval_pair_col(vpc, trace: Array) -> Array:
    """Materialize a rw `VirtualPairCol` (affine `const + Σ wᵢ·colᵢ`) on the base
    `trace` `(N, n_cols)`, embedded to `F3`. Binary's bus tuples are affine
    (no column products)."""
    n = trace.shape[0]
    acc = jnp.array(np.full(n, int(vpc.constant) % _P, dtype=np.uint64), dtype=F)
    for col, _is_pre, weight in vpc.column_weights:
        acc = acc + jnp.array(np.full(n, int(weight) % _P, dtype=np.uint64), dtype=F) * trace[:, col]
    return _embed_base(acc)


def _embed_base(base: Array) -> Array:
    """An `F` base array -> `F3` `(b, 0, 0)` (numpy-level, like `_embed`)."""
    u = np.asarray(base.astype(jnp.uint64))
    z = np.zeros_like(u)
    return jnp.array(np.stack([u, z, z], axis=1).astype(F).view(F3).reshape(u.shape[0]))


def gsum_e(interaction, trace: Array, alpha: Array) -> Array:
    """The LogUp bus denominator `gsum_e` (before `+ std_gamma`) for one
    interaction: reverse-α-Horner over its tuple (`Interaction.values` as
    `VirtualPairCol`s on `trace`, last component at the highest α power), then
    `· α + kind_int` (the native bus id appended at the low end). This is pil2's
    `std_sum` order — the REVERSE of `gsum.bus_denominator`'s tuple[0]-highest
    convention, and it omits γ (added in the constraint body)."""
    vals = [_eval_pair_col(v, trace) for v in interaction.values]
    den = vals[-1]
    for v in reversed(vals[:-1]):
        den = den * alpha + v
    return den * alpha + _embed([str(interaction.kind)])


def reauthor_binary_quotient(chip, case: dict) -> Array:
    """Re-author Binary's stage-2 quotient `q` on the synthetic `case` columns
    (a `cexp_eval` golden case), authoring std_sum from `chip`'s interactions."""
    n_bits, blowup_bits = case["n_bits"], case["blowup_bits"]
    n = 1 << (n_bits + blowup_bits)
    extend = 1 << blowup_bits

    cm: dict[int, Array] = {}
    for col in case["cm"]:
        cm[col["id"]] = (_embed if col["dim"] == 1 else _cubic)(col["values"])
    l1 = _embed(case["const"][0]["values"])
    alpha, gamma, vc = (_cubic(case["challenges"][i]["value"]) for i in (0, 1, 2))
    airvalues = {a["id"]: _cubic(a["value"]) for a in case["airvalues"]}
    gsum_result = _cubic(case["airgroupvalues"][0]["value"])

    # Base stage-1 trace for VirtualPairCol evaluation (cm id == column index).
    trace = jnp.stack([_base_col(case, j) for j in range(_N_STAGE1)], axis=1)

    sends = chip.get_sends()
    recvs = chip.get_receives()
    iacts = [s.interaction for s in sends] + [r.interaction for r in recvs]
    ge = [gsum_e(it, trace, alpha) for it in iacts]
    # LogUp multiplicity sign: send (assume) → −1, receive (prove) → +1.
    neg_one = _embed([str(_P - 1)])
    one = _embed(["1"])
    mult = [neg_one if it.is_send else one for it in iacts]

    c: list[Array] = [None] * 14  # type: ignore[list-item]
    # Row-local (0..6) — authored from the Binary AIR.
    c[0] = cm[33] * (one - cm[33])  # mode32 booleanity
    c[1] = cm[32] * (one - cm[32])  # carry[7] booleanity
    c[2] = cm[34] * (one - cm[34])  # result_is_a
    c[3] = cm[35] * (one - cm[35])  # use_first_byte
    c[4] = cm[36] * (one - cm[36])  # c_is_signed
    c[5] = cm[37] - (cm[33] * ((cm[36] + _embed(["512"])) - cm[0]) + cm[0])  # b_op_or_sext
    c[6] = cm[38] - cm[33] * cm[36]  # mode32_and_c_is_signed

    # std_sum im_cluster (7..10): im·∏(gsum_e+γ) − Σ mult·∏_{k≠·}(gsum_e+γ).
    for ci, (slot, i, j) in enumerate(_CLUSTERS, start=7):
        di, dj = ge[i] + gamma, ge[j] + gamma
        c[ci] = cm[_IM_CLUSTER[slot]] * (di * dj) - (mult[i] * dj + mult[j] * di)
    # gsum transition (11): ((gsum − 'gsum·(1−L1)) − Σ im_cluster)·(gsum_e[0]+γ) + 1.
    gsum_prev = _rotate(cm[_GSUM], -extend)
    sum_im = cm[40] + cm[41] + cm[42] + cm[43]
    c[11] = ((cm[_GSUM] - gsum_prev * (one - l1)) - sum_im) * (ge[_TRANSITION_IACT] + gamma) + one
    # im_direct (12): a constant operation-bus descriptor (10·α + 5000).
    direct = _embed(["10"]) * alpha + _embed(["5000"])
    c[12] = airvalues[1] * (direct + gamma) - (jnp.zeros(n, F3) - airvalues[0])
    # boundary (13): __L1__'·(gsum_result − gsum − im_direct).
    c[13] = _rotate(l1, extend) * (gsum_result - cm[_GSUM] - airvalues[1])

    composite = c[0]
    for i in range(1, 14):
        composite = composite * vc + c[i]
    return composite * inv_zerofier(n_bits, blowup_bits)


def _base_col(case: dict, col_id: int) -> Array:
    """Stage-1 column `col_id` from the golden case as an `F` base array."""
    entry = next(col for col in case["cm"] if col["id"] == col_id)
    return jnp.array(np.array([int(v) for v in entry["values"]], dtype=np.uint64), dtype=F)
