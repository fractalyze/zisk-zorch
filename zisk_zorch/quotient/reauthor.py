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
https://github.com/0xPolygonHermez/pil2-proofman/blob/v1.0.0-alpha/pil2-components/lib/std/pil/std_sum.pil
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array
from zk_dtypes import goldilocksx3_mont as F3

from zisk_zorch.golden import u64x3
from zisk_zorch.quotient.field_io import base_trace, embed, rotate
from zisk_zorch.quotient.gsum import _P, gsum_e
from zisk_zorch.quotient.zerofier import inv_zerofier

# Binary AIR layout (proving-key cmPolsMap): 39 stage-1 base cols, then the
# stage-2 witness cubic cols gsum / im_cluster×4, then the stage-3 quotient.
_N_STAGE1 = 39
_GSUM = 39
_IM_CLUSTER = (40, 41, 42, 43)
# Each std_sum cluster is the pair of interactions whose LogUp denominators it
# combines; the 4th pairs byte-table send 7 with the operation receive.
_CLUSTERS = ((0, 1, 2), (1, 3, 4), (2, 5, 6), (3, 7, 8))  # (im_cluster_slot, iact_i, iact_j)
_TRANSITION_IACT = 0  # gsum_e[0] (byte-table send 0) gates the gsum transition


def reauthor_binary_quotient(chip, case: dict) -> Array:
    """Re-author Binary's stage-2 quotient `q` on the synthetic `case` columns
    (a `cexp_eval` golden case), authoring std_sum from `chip`'s interactions."""
    n_bits, blowup_bits = case["n_bits"], case["blowup_bits"]
    n = 1 << (n_bits + blowup_bits)
    extend = 1 << blowup_bits

    cm: dict[int, Array] = {}
    for col in case["cm"]:
        cm[col["id"]] = (embed if col["dim"] == 1 else u64x3)(col["values"])
    l1 = embed(case["const"][0]["values"])
    alpha, gamma, vc = (u64x3(case["challenges"][i]["value"]) for i in (0, 1, 2))
    airvalues = {a["id"]: u64x3(a["value"]) for a in case["airvalues"]}
    gsum_result = u64x3(case["airgroupvalues"][0]["value"])

    # Base stage-1 trace for VirtualPairCol evaluation (cm id == column index).
    trace = base_trace(case, _N_STAGE1)

    sends = chip.get_sends()
    recvs = chip.get_receives()
    iacts = [s.interaction for s in sends] + [r.interaction for r in recvs]
    ge = [gsum_e(it, trace, alpha) for it in iacts]
    # LogUp multiplicity sign: send (assume) → −1, receive (prove) → +1.
    neg_one = embed([str(_P - 1)])
    one = embed(["1"])
    mult = [neg_one if it.is_send else one for it in iacts]

    c: list[Array] = [None] * 14  # type: ignore[list-item]
    # Row-local (0..6) — authored from the Binary AIR.
    c[0] = cm[33] * (one - cm[33])  # mode32 booleanity
    c[1] = cm[32] * (one - cm[32])  # carry[7] booleanity
    c[2] = cm[34] * (one - cm[34])  # result_is_a
    c[3] = cm[35] * (one - cm[35])  # use_first_byte
    c[4] = cm[36] * (one - cm[36])  # c_is_signed
    c[5] = cm[37] - (cm[33] * ((cm[36] + embed(["512"])) - cm[0]) + cm[0])  # b_op_or_sext
    c[6] = cm[38] - cm[33] * cm[36]  # mode32_and_c_is_signed

    # std_sum im_cluster (7..10): im·∏(gsum_e+γ) − Σ mult·∏_{k≠·}(gsum_e+γ).
    for ci, (slot, i, j) in enumerate(_CLUSTERS, start=7):
        di, dj = ge[i] + gamma, ge[j] + gamma
        c[ci] = cm[_IM_CLUSTER[slot]] * (di * dj) - (mult[i] * dj + mult[j] * di)
    # gsum transition (11): ((gsum − 'gsum·(1−L1)) − Σ im_cluster)·(gsum_e[0]+γ) + 1.
    gsum_prev = rotate(cm[_GSUM], -extend)
    sum_im = cm[40] + cm[41] + cm[42] + cm[43]
    c[11] = ((cm[_GSUM] - gsum_prev * (one - l1)) - sum_im) * (ge[_TRANSITION_IACT] + gamma) + one
    # im_direct (12): a constant operation-bus descriptor (10·α + 5000).
    direct = embed(["10"]) * alpha + embed(["5000"])
    c[12] = airvalues[1] * (direct + gamma) - (jnp.zeros(n, F3) - airvalues[0])
    # boundary (13): __L1__'·(gsum_result − gsum − im_direct).
    c[13] = rotate(l1, extend) * (gsum_result - cm[_GSUM] - airvalues[1])

    composite = c[0]
    for i in range(1, 14):
        composite = composite * vc + c[i]
    return composite * inv_zerofier(n_bits, blowup_bits)
