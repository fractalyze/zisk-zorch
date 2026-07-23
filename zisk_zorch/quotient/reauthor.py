"""Re-author a stage-2 quotient from rw constraints + interactions.

Produces `q = (Σ_i constraint[i] · vc^(N−1−i)) · Zi` — pil2's composite-constraint
quotient. The `std_sum` (LogUp bus/gsum) constraints are authored from the chip's
typed `Interaction`s; byte-matching against the cExp reference `q` (`cexp_ref` /
the `cexp_eval` golden) is what verifies rw's authored interactions — the
per-chip CPU test can't, since interactions are CPU-erased there.

The two AIR paths differ in how they source the row-local constraints:

- **Binary** (`reauthor_binary_quotient`) authors its 7 row-local constraints
  from the AIR directly (small enough to hand-write), then the std_sum.
- **Arith** (`reauthor_arith_quotient`) has 49 row-local constraints whose
  pil2 imPol factoring does not align with rw's `eval_constraints` (33 columns,
  4/49 match a single eval column), so re-deriving them is out of scope here;
  it sources row-local from the cExp reference (those are already covered by
  rw's per-chip CPU test) and reconstructs only the 16 std_sum constraints from
  the interactions — the unique verification value, since interactions are
  CPU-erased. See docs/stage2-constraint-ingest.md and
  https://github.com/0xPolygonHermez/pil2-proofman/blob/v1.0.0-alpha/pil2-components/lib/std/pil/std_sum.pil
"""

from __future__ import annotations

import frx.numpy as fnp
from frx import Array
from zk_dtypes import goldilocksx3 as F3

from zisk_zorch.golden import base_trace, embed, u64x3
from zisk_zorch.quotient.cexp_ref import _load_inputs, _run_block
from zisk_zorch.quotient.gsum import _P, eval_pair_col, gsum_e
from zisk_zorch.quotient.zerofier import inv_zerofier

# Binary AIR layout (proving-key cmPolsMap): 39 stage-1 base cols, then the
# stage-2 witness cubic cols gsum / im_cluster×4, then the stage-3 quotient.
_N_STAGE1 = 39
_GSUM = 39
_IM_CLUSTER = (40, 41, 42, 43)
# Each std_sum cluster is the pair of interactions whose LogUp denominators it
# combines; the 4th pairs byte-table send 7 with the operation receive.
_CLUSTERS = (
    (0, 1, 2),
    (1, 3, 4),
    (2, 5, 6),
    (3, 7, 8),
)  # (im_cluster_slot, iact_i, iact_j)
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
    c[5] = cm[37] - (
        cm[33] * ((cm[36] + embed(["512"])) - cm[0]) + cm[0]
    )  # b_op_or_sext
    c[6] = cm[38] - cm[33] * cm[36]  # mode32_and_c_is_signed

    # std_sum im_cluster (7..10): im·∏(gsum_e+γ) − Σ mult·∏_{k≠·}(gsum_e+γ).
    for ci, (slot, i, j) in enumerate(_CLUSTERS, start=7):
        di, dj = ge[i] + gamma, ge[j] + gamma
        c[ci] = cm[_IM_CLUSTER[slot]] * (di * dj) - (mult[i] * dj + mult[j] * di)
    # gsum transition (11): ((gsum − 'gsum·(1−L1)) − Σ im_cluster)·(gsum_e[0]+γ) + 1.
    gsum_prev = fnp.roll(cm[_GSUM], extend)
    sum_im = cm[40] + cm[41] + cm[42] + cm[43]
    c[11] = ((cm[_GSUM] - gsum_prev * (one - l1)) - sum_im) * (
        ge[_TRANSITION_IACT] + gamma
    ) + one
    # im_direct (12): a constant operation-bus descriptor (10·α + 5000).
    direct = embed(["10"]) * alpha + embed(["5000"])
    c[12] = airvalues[1] * (direct + gamma) - (fnp.zeros(n, F3) - airvalues[0])
    # boundary (13): __L1__'·(gsum_result − gsum − im_direct).
    c[13] = fnp.roll(l1, -extend) * (gsum_result - cm[_GSUM] - airvalues[1])

    composite = c[0]
    for i in range(1, 14):
        composite = composite * vc + c[i]
    return composite * inv_zerofier(n_bits, blowup_bits)


# Arith stage-2 witness layout (proving-key cmPolsMap): 44 stage-1 base cols,
# then the cubic witness cols gsum / im_cluster×11 / im_single×3, then the
# quotient. cExp std_sum constraint idx -> (im witness col, the interaction
# indices grouped into that LogUp denominator). Discovered by matching each
# std_sum constraint against the interaction-reconstructed im definition;
# interaction order is `get_sends()` ++ `get_receives()`.
_ARITH_N_STAGE1 = 44
_ARITH_GSUM = 44
_ARITH_IM = {
    49: (45, (2, 3)),
    50: (46, (4, 5)),
    51: (47, (6, 7)),
    52: (48, (0, 16)),
    53: (49, (17, 18)),
    54: (50, (19, 20)),
    55: (51, (21, 22)),
    56: (52, (8, 23)),
    57: (53, (10, 12)),
    58: (54, (9, 14)),
    59: (55, (11, 13)),
    60: (56, (15,)),
    61: (57, (25,)),
    62: (58, (24,)),
}
_ARITH_TRANSITION_GATE = 1  # gsum_e of this interaction gates the gsum transition


def _signed_multiplicity(interaction, trace: Array) -> Array:
    """pil2's signed std_sum multiplicity for one interaction. The arith table
    (kind 331) and range-check (330) lookup sends use multiplicity 1; only the
    operation bus (5000) carries a multiplicity column (cm41). rw exports cm41 as
    the multiplicity for *every* arith bus interaction, so the lookups are forced
    to 1 to match pil2's cExp (which uses the literal constant there). `send` →
    −mul, `receive` → +mul."""
    mul = (
        embed(["1"])
        if interaction.kind in (330, 331)
        else eval_pair_col(interaction.multiplicity, trace)
    )
    return (embed([str(_P - 1)]) * mul) if interaction.is_send else mul


def _im_constraint(
    im: Array, group: tuple[int, ...], d: list[Array], m: list[Array]
) -> Array:
    """One cExp std_sum im constraint, cleared of denominators:
    `im·∏_k d_k − Σ_k m_k·∏_{j≠k} d_j` (im is defined as Σ_k m_k/d_k)."""
    prod_all = d[group[0]]
    for k in group[1:]:
        prod_all = prod_all * d[k]
    s: Array | None = None
    for k in group:
        term = m[k]
        for j in group:
            if j != k:
                term = term * d[j]
        s = term if s is None else s + term
    return im * prod_all - s


def reauthor_arith_quotient(
    chip, case: dict, row_local_constraints: list[dict]
) -> Array:
    """Re-author Arith's stage-2 quotient `q` on a `cexp_eval` golden case.

    Row-local (cExp constraints 0..48) is sourced from the proving-key
    `row_local_constraints` SSA — already covered by rw's per-chip CPU test, and
    not re-derivable from rw's mismatched-granularity `eval_constraints`. The 16
    std_sum constraints (49..64) are reconstructed from `chip`'s interactions:
    this is the part the byte-match against the reference `q` verifies, since
    interactions are CPU-erased in rw's own test."""
    n_bits, blowup_bits = case["n_bits"], case["blowup_bits"]
    extend = 1 << blowup_bits

    # `_load_inputs` is the one case→F3 loader (also feeding the row-local SSA
    # eval below); read cm / const / challenges / airgroupvalue off its env
    # instead of re-materializing them.
    env = _load_inputs(case)
    cm = env["cm"]
    l1 = env["const"][0]
    alpha, gamma, vc = (env["challenges"][i] for i in (0, 1, 2))
    gsum_result = env["airgroupvalues"][0]
    trace = base_trace(case, _ARITH_N_STAGE1)

    iacts = [s.interaction for s in chip.get_sends()] + [
        r.interaction for r in chip.get_receives()
    ]
    d = [gsum_e(it, trace, alpha) + gamma for it in iacts]
    m = [_signed_multiplicity(it, trace) for it in iacts]

    cols = [
        _run_block(row_local_constraints[i]["code"], env, extend) for i in range(49)
    ]
    for ci in range(49, 63):
        slot, group = _ARITH_IM[ci]
        cols.append(_im_constraint(cm[slot], group, d, m))
    # gsum transition (63): ((gsum − 'gsum·(1−L1)) − Σ im)·(gsum_e[gate]+γ) + 1.
    one = embed(["1"])
    gsum_prev = fnp.roll(cm[_ARITH_GSUM], extend)
    sum_im = cm[45]
    for col in range(46, 59):
        sum_im = sum_im + cm[col]
    base = (cm[_ARITH_GSUM] - gsum_prev * (one - l1)) - sum_im
    cols.append(base * d[_ARITH_TRANSITION_GATE] + one)
    # boundary (64): __L1__'·(gsum_result − gsum).
    cols.append(fnp.roll(l1, -extend) * (gsum_result - cm[_ARITH_GSUM]))

    composite = cols[0]
    for col in cols[1:]:
        composite = composite * vc + col
    return composite * inv_zerofier(n_bits, blowup_bits)
