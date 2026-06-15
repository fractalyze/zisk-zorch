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

The op list is N-independent (one straight-line program per row), so it runs on
any extended domain; tests drive it on a small synthetic one. Every operand is
evaluated in the cubic extension `F3` — a base operand embeds as `(b, 0, 0)`, so
base x cubic stays exact scalar multiplication. The base->cubic embed happens at
the numpy level: the zkx CPU emitter crashes on cubic bitcast/`view`.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jax import Array
from zk_dtypes import goldilocks_mont as F
from zk_dtypes import goldilocksx3_mont as F3

from zisk_zorch.golden import u64x3


def _embed(values: list[str]) -> Array:
    """Base canonical-u64 decimals -> `F3` array of `(b, 0, 0)` embeddings.

    The decimals are already canonical (`< p`, the golden's `as_canonical_u64` /
    pil2's field literals), so `astype(F)` value-converts each straight into the
    Montgomery field — no explicit reduction needed."""
    limbs = np.array([[int(v), 0, 0] for v in values], dtype=np.uint64)
    return jnp.array(limbs.astype(F).view(F3).reshape(limbs.shape[0]))


def _rotate(col: Array, shift: int) -> Array:
    """`out[i] = col[(i + shift) mod n]` — the extended-domain image of a
    next/previous-row opening. Built from slice+concat (no `jnp.roll` in the
    zkx jax fork)."""
    n = col.shape[0]
    s = shift % n
    if s == 0:
        return col
    return jnp.concatenate([col[s:], col[:s]])


def _load_inputs(case: dict) -> dict:
    """Materialize every cExp operand source from a golden case as `F3`."""
    cm: dict[int, Array] = {}
    for col in case["cm"]:
        cm[col["id"]] = (_embed if col["dim"] == 1 else u64x3)(col["values"])
    const_cols: dict[int, Array] = {}
    for col in case["const"]:
        const_cols[col["id"]] = (_embed if col["dim"] == 1 else u64x3)(col["values"])
    challenges = {c["id"]: u64x3(c["value"]) for c in case["challenges"]}
    airvalues = {a["id"]: u64x3(a["value"]) for a in case["airvalues"]}
    airgroupvalues = {a["id"]: u64x3(a["value"]) for a in case["airgroupvalues"]}
    return {
        "cm": cm,
        "const": const_cols,
        "challenges": challenges,
        "airvalues": airvalues,
        "airgroupvalues": airgroupvalues,
        "zi": _embed(case["zi"]),
    }


def evaluate(fragment: dict, case: dict) -> Array:
    """Interpret the cExp SSA on the golden case, returning the `[n_ext]` cubic
    quotient column `q`. `fragment` is the vendored proving-key cExp fragment
    (`code` + maps); `case` carries the synthetic inputs and domain shape."""
    n_bits, blowup_bits = case["n_bits"], case["blowup_bits"]
    extend = 1 << blowup_bits
    env = _load_inputs(case)
    cm, const_cols = env["cm"], env["const"]
    challenges, zi = env["challenges"], env["zi"]
    airvalues, airgroupvalues = env["airvalues"], env["airgroupvalues"]

    def operand(s: dict) -> Array:
        t = s["type"]
        if t == "number":
            return _embed([s["value"]])
        if t == "cm":
            return _rotate(cm[s["id"]], s["prime"] * extend)
        if t == "const":
            return _rotate(const_cols[s["id"]], s["prime"] * extend)
        if t == "challenge":
            return challenges[s["id"]]
        if t == "airvalue":
            return airvalues[s["id"]]
        if t == "airgroupvalue":
            return airgroupvalues[s["id"]]
        if t == "tmp":
            return tmp[s["id"]]
        if t == "Zi":
            return zi
        raise ValueError(f"unhandled cExp operand type {t!r}")

    tmp: dict[int, Array] = {}
    q: Array | None = None
    for op in fragment["code"]:
        a, b = operand(op["src"][0]), operand(op["src"][1])
        kind = op["op"]
        if kind == "add":
            r = a + b
        elif kind == "sub":
            r = a - b
        elif kind == "mul":
            r = a * b
        else:
            raise ValueError(f"unhandled cExp op {kind!r}")
        dest = op["dest"]
        if dest["type"] == "tmp":
            tmp[dest["id"]] = r
        elif dest["type"] == "q":
            q = r
        else:
            raise ValueError(f"unhandled cExp dest {dest['type']!r}")

    if q is None:
        raise ValueError("cExp SSA produced no `q` destination")
    return jnp.broadcast_to(q, (1 << (n_bits + blowup_bits),))
