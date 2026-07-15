"""Reference evaluator for pil2's composite-constraint SSA (`cExp`).

pil2's `calculateQuotientPolynomial` evaluates one compiled expression — the
proving key's `expressionsCode[cExpId]` — per extended-domain row to produce the
quotient `q = C / Z_H`. This module interprets that SSA op list directly over the
Goldilocks cubic extension: the same computation the golden's Rust VM performs,
re-expressed as vectorized field ops.

It is the **reference** `q` (it interprets pil2's compiled expression). The
stage-2 prover does NOT interpret the SSA — it re-authors `q` from the ingested
rw constraints plus the generated `std_sum` constraints, folded in proving-key
`constraints[]` order (see quotient.py), and must byte-match this. Used by tests.

`evaluate_from_constraints` is the AIR-general middle ground between the two:
it folds the proving key's *individual* `constraints[]` (per-constraint SSAs)
generically — Horner in `std_vc` × the everyRow zerofier — rather than the
pre-folded composite, byte-matching the same `q` for any all-everyRow AIR with no
per-air wiring. It is the seam reauthor's rw-authored constraints slot into,
one constraint at a time.

The op list is N-independent (one straight-line program per row), so it runs on
any extended domain; tests drive it on a small synthetic one. Every operand is
evaluated in the cubic extension `F3` — a base operand embeds as `(b, 0, 0)`, so
base x cubic stays exact scalar multiplication. The base->cubic embed happens at
the numpy level.
"""

from __future__ import annotations

import frx.numpy as jnp
from frx import Array

from zisk_zorch.golden import u64x3
from zisk_zorch.quotient.field_io import embed, rotate
from zisk_zorch.quotient.zerofier import inv_zerofier


def _load_inputs(case: dict) -> dict:
    """Materialize every cExp operand source from a golden case as `F3`."""
    cm: dict[int, Array] = {}
    for col in case["cm"]:
        cm[col["id"]] = (embed if col["dim"] == 1 else u64x3)(col["values"])
    const_cols: dict[int, Array] = {}
    for col in case["const"]:
        const_cols[col["id"]] = (embed if col["dim"] == 1 else u64x3)(col["values"])
    challenges = {c["id"]: u64x3(c["value"]) for c in case["challenges"]}
    airvalues = {a["id"]: u64x3(a["value"]) for a in case["airvalues"]}
    airgroupvalues = {a["id"]: u64x3(a["value"]) for a in case["airgroupvalues"]}
    return {
        "cm": cm,
        "const": const_cols,
        "challenges": challenges,
        "airvalues": airvalues,
        "airgroupvalues": airgroupvalues,
        "zi": embed(case["zi"]),
    }


def _operand(s: dict, env: dict, tmp: dict[int, Array], extend: int) -> Array:
    """Resolve one cExp SSA operand to its `F3` value. `cm`/`const` carry a `prime`
    rotation of ±`extend` rows — the extended-domain image of pil2's ±1
    next/previous-row opening."""
    t = s["type"]
    if t == "number":
        return embed([s["value"]])
    if t == "cm":
        return rotate(env["cm"][s["id"]], s["prime"] * extend)
    if t == "const":
        return rotate(env["const"][s["id"]], s["prime"] * extend)
    if t == "challenge":
        return env["challenges"][s["id"]]
    if t == "airvalue":
        return env["airvalues"][s["id"]]
    if t == "airgroupvalue":
        return env["airgroupvalues"][s["id"]]
    if t == "tmp":
        return tmp[s["id"]]
    if t == "Zi":
        return env["zi"]
    raise ValueError(f"unhandled cExp operand type {t!r}")


def _run_block(code: list[dict], env: dict, extend: int) -> Array:
    """Interpret one straight-line SSA block — pil2's full composite `cExp` or a
    single `constraints[]` body — over `F3`, returning the value its final op
    writes (the `q` dest for the composite; the last tmp for a lone constraint)."""
    tmp: dict[int, Array] = {}
    q: Array | None = None
    result: Array | None = None
    for op in code:
        a = _operand(op["src"][0], env, tmp, extend)
        b = _operand(op["src"][1], env, tmp, extend)
        kind = op["op"]
        if kind == "add":
            result = a + b
        elif kind == "sub":
            result = a - b
        elif kind == "mul":
            result = a * b
        else:
            raise ValueError(f"unhandled cExp op {kind!r}")
        dest = op["dest"]
        if dest["type"] == "tmp":
            tmp[dest["id"]] = result
        elif dest["type"] == "q":
            q = result
        else:
            raise ValueError(f"unhandled cExp dest {dest['type']!r}")
    if result is None:
        raise ValueError("empty cExp block")
    return q if q is not None else result


def evaluate(fragment: dict, case: dict) -> Array:
    """Interpret the full composite cExp SSA on the golden case, returning the
    `[n_ext]` cubic quotient column `q`. `fragment` is the vendored proving-key
    cExp fragment (`code` + maps); `case` carries the synthetic inputs and shape."""
    n_bits, blowup_bits = case["n_bits"], case["blowup_bits"]
    env = _load_inputs(case)
    q = _run_block(fragment["code"], env, 1 << blowup_bits)
    return jnp.broadcast_to(q, (1 << (n_bits + blowup_bits),))


def evaluate_from_constraints(constraints: list[dict], case: dict) -> Array:
    """Reproduce the cExp quotient `q` from the proving key's **individual**
    `constraints[]` (per-constraint SSAs) folded generically, instead of pil2's
    pre-folded composite `code`. The fold is Horner in the quotient challenge
    `std_vc` (challenge 2) — constraint 0 at the highest power `vc^(N−1)`, the last
    constraint the constant term — times the everyRow inverse zerofier.

    Unlike reauthor.py (Binary-wired), nothing here is per-air, so it byte-matches
    `evaluate` / the cExp golden for any all-everyRow AIR. This is the
    generalization seam the rw-authored constraints slot into per-constraint."""
    extend = 1 << case["blowup_bits"]
    env = _load_inputs(case)
    if any(c["boundary"] != "everyRow" for c in constraints):
        raise NotImplementedError("only everyRow-boundary constraints are folded today")
    cols = [_run_block(c["code"], env, extend) for c in constraints]
    vc = env["challenges"][2]
    composite = cols[0]
    for col in cols[1:]:
        composite = composite * vc + col
    return composite * inv_zerofier(case["n_bits"], case["blowup_bits"])
