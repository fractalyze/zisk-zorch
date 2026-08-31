"""The witness-source seam: what `prove_block` needs from one AIR instance.

`Capture` (a pil2 dump bundle) is one implementation; #115's rw intake — the
witness generator handing traces over in memory instead of `.npy` dumps — is
the other. The protocol is deliberately narrowed to exactly the surface the
block driver consumes, so a second source implements the contract, not the
dump-bundle accidents (file naming, GPU tiling, canonicalize-on-read).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from zisk_zorch.harness.pil2 import Pil2Key
from zisk_zorch.harness.pil2_prover import Pil2Claim


@runtime_checkable
class WitnessSource(Protocol):
    """One AIR instance's witness and statement, block-driver view.

    `instance` names the instance (wire-proof files and gate logs key on
    it), `si` is the AIR's starkinfo dict, and `hash_family` the key's
    Poseidon family.
    """

    instance: str
    si: dict
    hash_family: str

    @property
    def pil2_key(self) -> Pil2Key:
        """The AIR's proving-key artifacts (shared per family — the driver
        builds one prover per family from the family's first source)."""
        ...

    def pil2_claim(self) -> Pil2Claim:
        """The instance's statement. `global_challenge` may be a
        placeholder: phase 3 `replace`s it with the self-derived seed."""
        ...

    def trace_words(self) -> np.ndarray:
        """The settled stage-1 witness as canonical Goldilocks u64 words,
        row-major ``(2**n_bits, n_cols)``, host-resident.

        Callable once per phase (commit, prove); the returned view is valid
        until `release()`. Canonicality is the source's contract — the
        driver reinterprets the words as field lanes without a reduction
        pass (`view`, not `astype`; #144)."""
        ...

    def release(self) -> None:
        """Drop per-instance residency (host sections, device arrays). A
        block walks ~40 instances in one process; nothing per-instance may
        survive its iteration. The next `trace_words()` re-materializes."""
        ...
