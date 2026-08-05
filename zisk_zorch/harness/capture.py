"""Loader for a pil2-proofman ``PIL2_DUMP_DIR`` capture bundle.

A bundle is one real ``genProof``'s per-stage sections (``tools/pil2-dump``
regenerates it) plus the AIR's proving-key artifacts, in one directory.
File naming, ``.npy`` decoding, gl64 limb layout, and pil2's value-packing
conventions all live here, so the gates read protocol objects, never files.
"""

from __future__ import annotations

import json
import pathlib
from functools import cached_property

import frx.numpy as fnp
import numpy as np
from zk_dtypes import goldilocks as F
from zk_dtypes import goldilocksx3 as F3
from zk_dtypes import pfinfo
from zorch.utils.field import join_coeffs, split_coeffs

from zisk_zorch.harness.pil2 import (
    Pil2Key,
    cm_env,
    committed_column,
    const_env,
    custom_env,
    hint_value,
    scalar_env,
    values_env,
)
from zisk_zorch.harness.pil2_prover import Pil2Claim
from zisk_zorch.quotient.zerofier import inv_zerofier

P = int(pfinfo(F).modulus)

# One definition of the host-bundle convention (env var + tools/pil2-dump's
# default instance): the runnable and gates_test must resolve the SAME
# bundle or their verdicts describe different proves.
CAPTURE_ENV = "ZISK_PIL2_CAPTURE"
FIXTURE_INSTANCE = "ag0_air2_inst5"
FIXTURE_STARKINFO = "SpecifiedRanges.starkinfo.json"


def cubic(words: np.ndarray):
    """Contiguous gl64 limb triples -> a 1-D cubic device array."""
    return join_coeffs(fnp.array(words.astype(F).reshape(-1, 3)), F3)


def limbs(x) -> np.ndarray:
    """A field array flattened to its dumped form: contiguous u64 limbs."""
    return np.asarray(split_coeffs(x)).reshape(-1)


class Capture:
    """One dumped AIR instance: starkinfo/expressionsinfo shape metadata plus
    lazy, cached readers for the per-stage ``<instance>_<name>.npy`` sections."""

    def __init__(
        self, dump: pathlib.Path, instance: str, starkinfo: pathlib.Path
    ) -> None:
        self.dump, self.instance, self.starkinfo_path = dump, instance, starkinfo
        self.si = json.loads(starkinfo.read_text())
        assert self.si["nStages"] == 2, "buffer keys assume a two-stage AIR"
        assert len(self.si["boundaries"]) == 1, "only the everyRow zerofier is wired"
        ss = self.si["starkStruct"]
        self.nb, self.nbe = ss["nBits"], ss["nBitsExt"]
        self.n, self.ne = 1 << self.nb, 1 << self.nbe
        self.stride = 1 << (self.nbe - self.nb)
        self.arity = ss["merkleTreeArity"]
        self.steps = [s["nBits"] for s in ss["steps"]]
        self.cmp_map = self.si["cmPolsMap"]
        self.ev_map = self.si["evMap"]
        self.openings = self.si["openingPoints"]
        self.n_cols = self.si["mapSectionsN"]["cm1"]
        self.cm2_cols = self.si["mapSectionsN"].get("cm2", 0)
        self.n_customs = len(self.si.get("customCommits", []))
        self._sections: dict[str, np.ndarray] = {}

    # -- dumped sections ----------------------------------------------------

    def path(self, name: str) -> pathlib.Path:
        return self.dump / f"{self.instance}_{name}.npy"

    def u64(self, name: str) -> np.ndarray:
        """A dumped section's u64 words, loaded once — several gates read the
        same multi-hundred-MB sections, so uncached reloads dominate a run's
        I/O at real scale."""
        if name not in self._sections:
            arr = np.load(self.path(name))
            assert arr.dtype == np.uint64, f"{name}: expected u64 dump, got {arr.dtype}"
            self._sections[name] = arr
        return self._sections[name]

    @cached_property
    def expressionsinfo(self) -> dict:
        return json.loads(
            (
                self.starkinfo_path.parent
                / self.starkinfo_path.name.replace("starkinfo", "expressionsinfo")
            ).read_text()
        )

    @cached_property
    def cexp_code(self) -> list:
        """The composite constraint expression's SSA block."""
        return next(
            e
            for e in self.expressionsinfo["expressionsCode"]
            if e["expId"] == self.si["cExpId"]
        )["code"]

    @cached_property
    def im_col_exps(self) -> tuple | None:
        """The stage-2 LogUp chain's three SSA blocks — the ``im_col`` hint's
        numerator and denominator plus the committed ``ImPol`` — or ``None``
        for an AIR whose stage-2 carries no LogUp intermediate (a range
        check, say): no hint, no imPol column, nothing to chain."""
        ei = self.expressionsinfo
        exps = {e["expId"]: e["code"] for e in ei["expressionsCode"]}
        hints = {h["name"]: h for h in ei["hintsInfo"]}
        im_pol = next(
            (p["expId"] for p in self.cmp_map if p["stage"] == 2 and p.get("imPol")),
            None,
        )
        if "im_col" not in hints or im_pol is None:
            return None
        im = hints["im_col"]
        return (
            exps[hint_value(im, "numerator")["id"]],
            exps[hint_value(im, "denominator")["id"]],
            exps[im_pol],
        )

    @cached_property
    def const_base(self) -> np.ndarray:
        """The proving key's base-domain constant columns (pil2's own binary
        layout — the one non-npy read in the bundle)."""
        return np.fromfile(
            self.starkinfo_path.parent
            / self.starkinfo_path.name.replace(".starkinfo.json", ".const"),
            dtype=np.uint64,
        ).reshape(self.n, self.si["nConstants"])

    @cached_property
    def trace(self):
        """The settled stage-1 witness: hint-computed columns fill
        asynchronously during STEP_1, so the pre-commit dump can be
        incomplete — prefer the post-commit section."""
        name = "trace_post" if self.path("trace_post").exists() else "trace"
        words = self.u64(name)
        assert words.size == self.n * self.n_cols, "trace size mismatch"
        return fnp.array(words.astype(F).reshape(self.n, self.n_cols))

    @cached_property
    def bufs(self) -> dict:
        """Committed extended-domain sections, keyed like ``cmPolsMap`` stages
        (quotient rides as stage ``nStages + 1``)."""
        si, ne = self.si, self.ne
        bufs = {
            ("cm", 1): self.u64("cm1_ext").astype(F).reshape(ne, self.n_cols),
            ("const", 0): self.u64("const_ext").astype(F).reshape(ne, si["nConstants"]),
            ("cm", si["nStages"] + 1): self.u64("quotient_cm")
            .astype(F)
            .reshape(ne, si["mapSectionsN"]["cm" + str(si["nStages"] + 1)]),
        }
        if self.cm2_cols:
            bufs[("cm", 2)] = self.u64("cm2_ext").astype(F).reshape(ne, self.cm2_cols)
        for ci in range(self.n_customs):
            width = si["mapSectionsN"][si["customCommits"][ci]["name"] + "0"]
            bufs[("custom", ci)] = (
                self.u64(f"custom{ci}_ext").astype(F).reshape(ne, width)
            )
        return bufs

    # -- protocol objects the gates consume ----------------------------------

    def committed_column(self, entry: dict):
        """An evMap entry's committed column over the extended domain, cubic
        entries joined from their three contiguous gl64 lanes."""
        return committed_column(entry, self.cmp_map, self.bufs)

    @cached_property
    def pil2_key(self) -> Pil2Key:
        """The AIR's proving-key artifacts as the pil2-mode roles'
        configuration."""
        return Pil2Key(
            starkinfo=self.si,
            expressionsinfo=self.expressionsinfo,
            const_base=self.const_base.astype(F),
            const_ext=self.bufs[("const", 0)],
            custom_ext={ci: self.bufs[("custom", ci)] for ci in range(self.n_customs)},
        )

    def pil2_claim(self) -> Pil2Claim:
        """The dumped instance's statement — what the pil2-mode prover takes
        alongside the trace."""
        return Pil2Claim(
            n_bits=self.nb,
            n_cols=self.n_cols,
            publics=self.u64("publics"),
            airvalues=self.u64("airvalues"),
            proofvalues=self.u64("proofvalues"),
            global_challenge=self.u64("global_challenge"),
        )

    @cached_property
    def opened_columns(self) -> list:
        return [self.committed_column(e) for e in self.ev_map]

    @cached_property
    def challenges(self):
        """The prove's transcript challenges, one cubic scalar per index."""
        return cubic(self.u64("challenges"))

    def challenge(self, name: str):
        idx = next(
            i for i, c in enumerate(self.si["challengesMap"]) if c.get("name") == name
        )
        return self.challenges[idx].reshape(())

    # -- the SSA interpreter's operand environment ----------------------------

    @cached_property
    def cexp_env(self) -> dict:
        """Every operand class the composite cExp reads, over the extended
        domain: committed/constant/custom columns plus the scalar values."""
        si = self.si
        return {
            "cm": cm_env(self.cmp_map, self.bufs),
            "const": const_env(self.bufs, si["nConstants"]),
            "custom": custom_env(self.bufs, si.get("customCommits", [])),
            **self._scalar_env,
            "zi": {0: inv_zerofier(self.nb, self.nbe - self.nb)},
        }

    @cached_property
    def _scalar_env(self) -> dict:
        return scalar_env(
            self.si,
            publics=self.u64("publics"),
            airvalues=self.u64("airvalues"),
            proofvalues=self.u64("proofvalues"),
            challenges=dict(enumerate(self.challenges)),
            airgroupvalues=values_env(
                self.u64("airgroupvalues"), self.si["airgroupValuesMap"]
            ),
        )

    @cached_property
    def base_env(self) -> dict:
        """The base-domain environment the stage-2 hint expressions evaluate
        over: the settled stage-1 trace and the key's base constants, scalar
        sections shared with ``cexp_env``. ``custom``/``zi`` stay empty so a
        hint expression reaching for an extended-domain section fails loudly
        instead of silently mixing domains."""
        return {
            "cm": {i: self.trace[:, i] for i in range(self.n_cols)},
            "const": {
                i: fnp.array(self.const_base[:, i].astype(F))
                for i in range(self.si["nConstants"])
            },
            "custom": {},
            **self._scalar_env,
            "zi": {},
        }
