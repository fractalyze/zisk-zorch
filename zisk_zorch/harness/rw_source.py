"""Witness sources backed by riscv-witness trace bundles (#115's intake).

`Capture` reads a settled trace out of a pil2 `.npy` dump; this reads one
the witness generator just produced, straight out of the bundle's own host
memory. That removes the last native input from the prove path — and with
it the dump read, which is the commit leg's dominant cost (#144).

The bundle types are duck-typed rather than imported. `riscv_witness` is a
heavy compiled wheel that only a witness-generating caller needs, so the
join, the claim assembly, and the shape guards here stay testable — and
this module stays importable — without it.

What rw supplies, and what it does not:

- The **trace** is the settled stage-1 matrix, padded to the AIR's row
  count, canonical Goldilocks, row-major (rw#2360 contract questions 2-4).
- **Main's stage-1 air values** ride beside it as a flat vector in
  airValuesMap order (rw#2368). An AIR that declares `airValuesMap` and is
  not Main has no rw supply yet and fails loudly rather than proving
  against zeros.
- **Publics and proof values are block-level**, not per-segment, and rw
  does not carry them (#115 contract question 1). The caller supplies them
  once; they are the same words `Capture` reads from `publics.npy` /
  `proofvalues.npy`.
"""

from __future__ import annotations

import json
import pathlib
from typing import Iterable

import numpy as np

from zisk_zorch.harness.pil2 import Pil2Key
from zisk_zorch.harness.pil2_prover import Pil2Claim
from zisk_zorch.harness.recursion import key_root
from zisk_zorch.harness.zisk_key import zisk_air_base, zisk_pil2_key


class RwInstanceWitness:
    """One chip's trace from one rw segment, as a `WitnessSource`.

    Holds the bundle's zero-copy view: no read, no un-tile, no
    canonicalize pass. The per-AIR artifacts (`si`, `pil2_key`) are shared
    across every instance of the AIR — they are the proving key's, not the
    instance's — so `release` drops only this instance's trace handle.
    """

    def __init__(
        self,
        instance: str,
        si: dict,
        hash_family: str,
        key: Pil2Key,
        claim: Pil2Claim,
        words: np.ndarray,
    ) -> None:
        self.instance = instance
        self.si = si
        self.hash_family = hash_family
        self._key = key
        self._claim = claim
        self._words: np.ndarray | None = words

    @property
    def pil2_key(self) -> Pil2Key:
        return self._key

    def pil2_claim(self) -> Pil2Claim:
        return self._claim

    def trace_words(self) -> np.ndarray:
        if self._words is None:
            # The bundle's buffer is gone and there is nothing to re-read
            # from — unlike `Capture`, which can go back to the dump. A
            # driver that calls this after `release` has a scheduling bug,
            # and proving against a stale or empty matrix would hide it.
            raise RuntimeError(
                f"{self.instance}: trace_words() after release(); an rw "
                "source cannot re-materialize (hold the bundle longer)"
            )
        return self._words

    def release(self) -> None:
        """Drop this instance's handle on the bundle's buffer.

        Only the reference goes. The memory stays valid while anything
        else holds it — the caller's bundle, and any array the driver
        still has in flight — which is what `WitnessSource.trace_words`
        requires of a source that releases mid-upload.
        """
        self._words = None


def _chip_air_index(gi: dict) -> dict[str, int]:
    """`{air name: index}` over the ZisK airgroup — the join rw's
    `chip_names` land on. The index is what instance names carry and what
    `verify_proof_layout.starkinfo_for` parses back out."""
    return {air["name"]: i for i, air in enumerate(gi["airs"][0])}


class RwBlockWitness:
    """Joins rw trace bundles to the proving key's AIRs.

    `sources()` yields the `(family, source, verkey)` triples `prove_block`
    consumes, grouped by family so the driver's family-boundary prover
    release still fires once per family.
    """

    def __init__(
        self,
        bundles: Iterable,
        key: pathlib.Path,
        gi: dict,
        *,
        publics: np.ndarray,
        proofvalues: np.ndarray,
    ) -> None:
        self.key = pathlib.Path(key)
        self.gi = gi
        self.publics = publics
        self.proofvalues = proofvalues
        self._bundles = list(bundles)
        self._air_index = _chip_air_index(gi)
        self._si: dict[str, dict] = {}
        self._keys: dict[str, Pil2Key] = {}
        self._verkeys: dict[str, np.ndarray] = {}

    def _starkinfo(self, air: str) -> dict:
        if air not in self._si:
            base = zisk_air_base(self.key, self.gi, air)
            self._si[air] = json.loads(
                pathlib.Path(f"{base}.starkinfo.json").read_text()
            )
        return self._si[air]

    def _pil2_key(self, air: str) -> Pil2Key:
        if air not in self._keys:
            self._keys[air] = zisk_pil2_key(self.key, self.gi, air)
        return self._keys[air]

    def _verkey(self, air: str) -> np.ndarray:
        if air not in self._verkeys:
            found = sorted(key_root(self.key).rglob(f"{air}.verkey.json"))
            if not found:
                raise FileNotFoundError(f"{air}: no verkey.json under {self.key}")
            self._verkeys[air] = np.array(
                json.loads(found[0].read_text()), dtype=np.uint64
            )
        return self._verkeys[air]

    def _claim(self, air: str, si: dict, air_values: np.ndarray) -> Pil2Claim:
        return Pil2Claim(
            n_bits=si["starkStruct"]["nBits"],
            n_cols=si["mapSectionsN"]["cm1"],
            publics=self.publics,
            airvalues=air_values,
            proofvalues=self.proofvalues,
            # Phase 3 `replace`s this with the block's self-derived seed;
            # phase 1 never reads it. Zeros keep the shape honest without
            # implying rw supplied a challenge.
            global_challenge=np.zeros(3, dtype=np.uint64),
        )

    def _air_values(self, air: str, si: dict, bundle) -> np.ndarray:
        """The instance's stage-1 air values, or an empty vector."""
        if not si.get("airValuesMap"):
            return np.zeros(0, dtype=np.uint64)
        if air != "Main":
            raise NotImplementedError(
                f"{air} declares airValuesMap but rw supplies stage-1 air "
                "values for Main only (riscv-witness#2360) — proving it "
                "from an rw bundle needs that vector first"
            )
        values = np.asarray(bundle.main_air_values, dtype=np.uint64).reshape(-1)
        if values.size == 0:
            raise ValueError(
                f"{air}: bundle carries no main_air_values, but the AIR "
                "declares airValuesMap (segment fired a Main trace, so the "
                "vector should be present)"
            )
        return values

    def _words(self, air: str, si: dict, trace, num_reals: int) -> np.ndarray:
        """The chip trace as canonical u64 words, shape-checked.

        The shape check is the cheap half of the rw contract; canonicality
        is the expensive half and is asserted in tests, not here — a full
        pass over the matrix is the very cost this intake exists to remove.
        """
        words = np.asarray(trace.data)
        rows, cols = 1 << si["starkStruct"]["nBits"], si["mapSectionsN"]["cm1"]
        if words.dtype != np.uint64:
            raise TypeError(f"{air}: rw trace dtype {words.dtype}, expected uint64")
        if words.shape != (rows, cols):
            raise ValueError(
                f"{air}: rw trace shape {words.shape}, expected {(rows, cols)} "
                "from the proving key's starkStruct/mapSectionsN"
            )
        if not words.flags.c_contiguous:
            # `TraceStager.stage` reinterprets the last axis with `view(F)`,
            # which silently needs a contiguous row.
            raise ValueError(f"{air}: rw trace is not row-major contiguous")
        if num_reals > rows:
            raise ValueError(f"{air}: num_reals {num_reals} exceeds {rows} rows")
        return words

    def sources(self) -> list[tuple[str, RwInstanceWitness, np.ndarray]]:
        """`(family, source, verkey)` per chip trace, grouped by family.

        Segment order within a family is bundle order, which is the
        generator's own segment order — the Main continuation chain is
        built on it (#131), so it is not free to permute.
        """
        by_family: dict[str, list[tuple[str, RwInstanceWitness, np.ndarray]]] = {}
        for seg, bundle in enumerate(self._bundles):
            # `chip_names` is the authoritative set: it lists exactly the
            # chips whose trace has rows this segment, while `traces` can
            # also carry empty entries for chips that did not fire.
            traces = {t.chip_name: t for t in bundle.traces}
            for air in bundle.chip_names:
                trace = traces.get(air)
                if trace is None:
                    raise KeyError(
                        f"rw bundle lists chip {air!r} in chip_names but has "
                        "no matching trace"
                    )
                if air not in self._air_index:
                    raise KeyError(
                        f"rw chip {air!r} is not an AIR of this proving key "
                        f"({self.key}) — chip/key mismatch"
                    )
                if air not in bundle.num_reals:
                    raise KeyError(f"rw bundle has no num_reals for chip {air!r}")
                si = self._starkinfo(air)
                witness = RwInstanceWitness(
                    # Deliberately not native's `ag0_air<i>_inst<n>`: that
                    # numbering is a pil2-runtime artifact over the whole
                    # block, and imitating it would assert an instance
                    # identity we have not established. The `_air<i>_` stem
                    # is kept so `starkinfo_for` still parses the name.
                    instance=f"rw_air{self._air_index[air]}_seg{seg}",
                    si=si,
                    hash_family=self._pil2_key(air).hash_family,
                    key=self._pil2_key(air),
                    claim=self._claim(air, si, self._air_values(air, si, bundle)),
                    words=self._words(air, si, trace, bundle.num_reals[air]),
                )
                by_family.setdefault(air, []).append(
                    (air, witness, self._verkey(air))
                )
        return [triple for family in by_family for triple in by_family[family]]
