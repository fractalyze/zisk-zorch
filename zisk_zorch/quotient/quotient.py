"""Stage-2 quotient `Q = C / Z_H` on the extended (coset) domain.

pil2's `calculateQuotientPolynomial` evaluates the composite constraint
expression `cExp`: the AIR (and bus / boundary) constraints folded by powers of
the stage-`nStages+1` challenge, divided by the zerofier. The fold is the
scheme-agnostic part — zorch's `constraint_eval` marks
`sum_k alpha_k * eval_fn(trace)_k` as one fused `zorch.constraint_eval`
composite (the same primitive sp1-zorch's zerocheck builds on) — and this module
adds the eSTARK-specific division by the inverse zerofier (`zerofier.py`).

Byte-match note: the alpha-power assignment / constraint order must follow pil2's
eSTARK convention (the proving key's pilout / expressionsinfo, including `imPols`
intermediates), NOT rw_constraints' `constraint_order` (which is SP1
`eval_block` / zerocheck indexing). The caller supplies an `eval_fn` whose
trailing-axis constraint order matches pil2's.

`Q` commits each cubic row as its 3 contiguous Goldilocks limbs — pil2's
`FIELD_EXTENSION` memory order, which is also what the FRI seam reads.

https://github.com/0xPolygonHermez/pil2-proofman/blob/v1.0.0-alpha/pil2-stark/src/starkpil/starks.hpp#L415-L417
"""

from __future__ import annotations

from collections.abc import Callable

from frx import Array
from zorch.constraint_eval import constraint_eval

from zisk_zorch.quotient.zerofier import inv_zerofier


def compute_quotient(composite_evals: Array, n_bits: int, blowup_bits: int) -> Array:
    """`Q = C * Zi` on the extended domain: the composite constraint column
    `composite_evals` (cubic, length `2^(n_bits+blowup_bits)`, natural order)
    times the inverse zerofier. Pointwise, since `Zi` is precomputed per row."""
    zi = inv_zerofier(n_bits, blowup_bits)  # validates n_bits/blowup_bits, length n_ext
    if composite_evals.shape[0] != zi.shape[0]:
        raise ValueError(
            f"composite_evals length {composite_evals.shape[0]} != extended domain "
            f"{zi.shape[0]} for n_bits={n_bits}, blowup_bits={blowup_bits}"
        )
    return composite_evals * zi


def quotient_from_constraints(
    eval_fn: Callable[[Array], Array],
    trace_ext: Array,
    alpha: Array,
    n_bits: int,
    blowup_bits: int,
) -> Array:
    """Fold `eval_fn(trace_ext)`'s constraints by powers of `alpha` into the
    composite (one fused `zorch.constraint_eval`), then divide by the zerofier.

    `eval_fn(trace_ext)` produces constraints in the trailing axis (`[..., K]`),
    matching `alpha`'s trailing length; `trace_ext` is the trace on the extended
    (coset) domain (the stage-1 LDE)."""
    composite = constraint_eval(eval_fn, trace_ext, alpha)
    return compute_quotient(composite, n_bits, blowup_bits)
