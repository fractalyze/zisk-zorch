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

import frx.numpy as fnp
from frx import Array

from zisk_zorch.golden import embed, u64x3
from zisk_zorch.quotient.zerofier import inv_zerofier

_ELEM_BYTES = 8


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
        "zi": {0: embed(case["zi"])},
    }


def _column(col: Array, prime: int, extend: int, rows) -> Array:
    """A column operand's value: the whole (rolled) column, or — under row
    chunking — the gathered `rows` window. The wrap stays cyclic, matching
    `roll`, so chunked and whole evaluations agree bytewise."""
    if rows is None:
        return fnp.roll(col, -(prime * extend))
    if prime:
        return col[(rows + prime * extend) % col.shape[0]]
    return col[rows]


def _operand(s: dict, env: dict, tmp: dict[int, Array], extend: int, rows) -> Array:
    """Resolve one cExp SSA operand to its `F3` value. `cm`/`const` carry a `prime`
    rotation of ±`extend` rows — the extended-domain image of pil2's ±1
    next/previous-row opening."""
    t = s["type"]
    if t == "number":
        return embed([s["value"]])
    if t == "cm":
        return _column(env["cm"][s["id"]], s["prime"], extend, rows)
    if t == "const":
        return _column(env["const"][s["id"]], s["prime"], extend, rows)
    if t == "challenge":
        return env["challenges"][s["id"]]
    if t == "airvalue":
        return env["airvalues"][s["id"]]
    if t == "airgroupvalue":
        return env["airgroupvalues"][s["id"]]
    if t == "public":
        return env["publics"][s["id"]]
    if t == "proofvalue":
        return env["proofvalues"][s["id"]]
    if t == "custom":
        return _column(
            env["custom"][(s.get("commitId", 0), s["id"])],
            s.get("prime", 0),
            extend,
            rows,
        )
    if t == "tmp":
        return tmp[s["id"]]
    if t == "Zi":
        return _column(env["zi"][s.get("boundaryId", 0)], 0, extend, rows)
    raise ValueError(f"unhandled cExp operand type {t!r}")


def live_bytes_per_row(code: list[dict], column_dim) -> int:
    """Peak bytes of live SSA temporaries per evaluated row.

    `run_block` is straight-line, so a temporary is live from the op that
    writes it to the op that last reads it; walking the block with those
    last-use points gives the widest simultaneous set. `column_dim(src)`
    resolves a `cm` operand's extension degree (1 or 3) — everything else is
    fixed by the operand class.

    Two properties of an operand decide what it costs, and they are
    independent (`_shape` carries both):

    - **Extension degree**, which fixes the bytes an element occupies. Every
      scalar class is CUBIC, `number` and `public` included: `_operand` embeds
      both through `golden.embed` / the packed cubic publics section, so their
      values are base but their dtype is `F3` and any product with one widens
      to 24 bytes an element. Reading them as base under-reports a temporary
      threefold, which is the direction that OOMs.
    - **Whether the value varies per row.** Only the column classes do; the
      scalar classes are 0-d (or shape-(1,)) and cost a fixed handful of bytes
      no matter how many rows are evaluated. A temporary folded purely from
      scalars — a Horner chain over the quotient challenge, say — is one
      scalar, so charging it per row would pin the window at its ceiling for a
      working set that does not exist.

    This is the quantity that sizes the row window: the interpreter's
    temporaries are the working set the evaluation streams through, and the
    window controls how much of it is resident at once."""
    last: dict[int, int] = {}
    for i, op in enumerate(code):
        for s in op["src"]:
            if s["type"] == "tmp":
                last[s["id"]] = i

    shape: dict[int, tuple[int, bool]] = {}

    def _shape(s: dict) -> tuple[int, bool]:
        """One operand's `(extension degree, varies per row)`. Mirrors
        `_operand`'s dispatch, so an unhandled class fails here too rather
        than being silently sized as something it is not."""
        t = s["type"]
        if t == "cm":
            return column_dim(s), True
        if t == "tmp":
            return shape[s["id"]]
        if t in ("const", "custom", "Zi"):
            return 1, True
        if t in (
            "number",
            "public",
            "challenge",
            "airvalue",
            "airgroupvalue",
            "proofvalue",
        ):
            return 3, False
        raise ValueError(f"unhandled cExp operand type {t!r}")

    live: set[int] = set()
    peak = 0
    for i, op in enumerate(code):
        (dim_a, row_a), (dim_b, row_b) = _shape(op["src"][0]), _shape(op["src"][1])
        result = (3 if 3 in (dim_a, dim_b) else 1, row_a or row_b)
        if op["dest"]["type"] == "tmp":
            shape[op["dest"]["id"]] = result
            live.add(op["dest"]["id"])
        live = {t for t in live if last.get(t, -1) > i}
        peak = max(peak, sum(_ELEM_BYTES * shape[t][0] for t in live if shape[t][1]))
    return peak


def run_block(code: list[dict], env: dict, extend: int, rows=None) -> Array:
    """Interpret one straight-line SSA block — pil2's full composite `cExp` or a
    single `constraints[]` body — over `F3`, returning the value its final op
    writes (the `q` dest for the composite; the last tmp for a lone constraint).

    `rows` (a device index vector) evaluates only that row window — the
    tmp working set shrinks with the window, which is what lets a
    3000-column AIR's composite fit device memory (zz#138's Keccakf)."""
    tmp: dict[int, Array] = {}
    q: Array | None = None
    result: Array | None = None
    for op in code:
        a = _operand(op["src"][0], env, tmp, extend, rows)
        b = _operand(op["src"][1], env, tmp, extend, rows)
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
    q = run_block(fragment["code"], env, 1 << blowup_bits)
    return fnp.broadcast_to(q, (1 << (n_bits + blowup_bits),))


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
    cols = [run_block(c["code"], env, extend) for c in constraints]
    vc = env["challenges"][2]
    composite = cols[0]
    for col in cols[1:]:
        composite = composite * vc + col
    return composite * inv_zerofier(case["n_bits"], case["blowup_bits"])
