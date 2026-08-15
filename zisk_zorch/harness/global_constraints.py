"""pil2's global constraints — the cross-instance statement over one block.

After every instance proves, its airgroup values (the LogUp grand-sum
results) aggregate across instances — ``aggTypes`` picks add or mul per
value — and the key's ``pilout.globalConstraints`` SSA blocks must all
evaluate to zero over the aggregate, the proof-level publics/proof values,
and the global stage-2 challenges. This is bus conservation for the whole
block: every per-instance gsum is free to be nonzero as long as the block
sums to zero (zz#109's cross-instance half).

The blocks reuse the cExp SSA vocabulary, so `run_block` interprets them;
only the aggregation and the scalar environment live here.
"""

from __future__ import annotations

import numpy as np

from zisk_zorch.harness.pil2 import (
    MODULUS,
    cubic_scalar,
    limbs,
    publics_env,
    values_env,
)
from zisk_zorch.quotient.cexp_ref import run_block


def aggregate_airgroupvalues(
    per_instance: list[np.ndarray], agg_types: list[dict]
) -> dict[int, object]:
    """One airgroup's values summed (aggType 0) or multiplied (aggType 1)
    across instances, as the interpreter's cubic scalars keyed by value id.
    Cubic addition is limbwise; a mul aggregation would need the F3 product,
    which no ZisK value uses — fail loudly if a key ever does."""
    out = {}
    for vid, at in enumerate(agg_types):
        if at["aggType"] != 0:
            raise NotImplementedError(f"aggType {at['aggType']} (mul) unwired")
        acc = np.zeros(3, dtype=object)
        for words in per_instance:
            acc = (acc + words[3 * vid : 3 * vid + 3].astype(object)) % MODULUS
        out[vid] = cubic_scalar(acc.astype(np.uint64))
    return out


def check_global_constraints(
    constraints: list[dict],
    *,
    publics: np.ndarray,
    proofvalues: np.ndarray,
    proof_values_map: list,
    challenges: dict[int, object],
    airgroupvalues: dict[int, object],
) -> list[np.ndarray]:
    """Every constraint's value (its final tmp), as dumped u64 limbs — the
    caller asserts zeros. The environment is scalar-only: a global
    constraint reaching for a column class fails loudly."""
    env = {
        "cm": {},
        "const": {},
        "custom": {},
        "zi": {},
        "challenges": challenges,
        "publics": publics_env(publics),
        "airvalues": {},
        "airgroupvalues": airgroupvalues,
        "proofvalues": values_env(proofvalues, proof_values_map),
    }
    results = []
    for c in constraints:
        value = run_block(c["code"], env, 1)
        results.append(limbs(value).astype(np.uint64))
    return results
