"""Loader for a pil2-proofman ``PIL2_DUMP_DIR`` capture bundle.

A bundle is one real ``genProof``'s per-stage sections (``tools/pil2-dump``
regenerates it) plus the AIR's proving-key artifacts, in one directory.
File naming, ``.npy`` decoding, gl64 limb layout, and pil2's value-packing
conventions all live here, so the gates read protocol objects, never files.
"""

from __future__ import annotations

import json
import pathlib
import sys
from functools import cached_property

import frx.numpy as fnp
import numpy as np
from zk_dtypes import goldilocks as F

from zisk_zorch.commit.trace_commit import unextend
from zisk_zorch.harness.pil2 import (
    MODULUS as P,
)
from zisk_zorch.harness.pil2 import (
    Pil2Key,
    cm_env,
    committed_column,
    const_env,
    cubic,
    custom_env,
    expand_scalars,
    hint_value,
    scalar_env,
    values_env,
)
from zisk_zorch.harness.pil2_prover import Pil2Claim
from zisk_zorch.quotient.zerofier import inv_zerofier

# One definition of the host-bundle convention (env var + tools/pil2-dump's
# default instance): the runnable and gates_test must resolve the SAME
# bundle or their verdicts describe different proves.
CAPTURE_ENV = "ZISK_PIL2_CAPTURE"
FIXTURE_INSTANCE = "ag0_air2_inst5"
FIXTURE_STARKINFO = "SpecifiedRanges.starkinfo.json"


def untile_gpu(flat: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """A GPU-dumped matrix section -> row-major ``(rows, cols)``.

    ``genProof_gpu`` dumps matrix buffers in the prover's storage format
    (pil2-stark ``goldilocks_trace_layout.cuh``): column-major tiles of
    256 rows x 4 cols, the last column group narrowed to ``cols % 4``.
    Row-major buffers (the FRI polynomial chain, 1-D sections) pass through
    the plain path — this is only for the committed/tree-source sections."""
    assert flat.size == rows * cols, (flat.size, rows, cols)
    assert rows % 256 == 0, rows
    out = np.empty((rows, cols), dtype=flat.dtype)
    base = 0
    for g in range((cols + 3) // 4):
        ncb = min(cols - 4 * g, 4)
        block = flat[base : base + rows * ncb].reshape(rows // 256, ncb, 256)
        out[:, 4 * g : 4 * g + ncb] = block.transpose(0, 2, 1).reshape(rows, ncb)
        base += rows * ncb
    return out


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
        # A genProof_gpu capture (detectable by its missing per-stage root
        # sections) stores matrix sections in the prover's column-major tile
        # layout and its roots inside the pinned staging `proof` buffer; a
        # CPU capture is row-major with one file per root.
        self.gpu_dump = not self.path("root1").exists()
        q_name = "cm" + str(self.si["nStages"] + 1)
        self._matrix_shapes = {
            "trace": (self.n, self.n_cols),
            "trace_post": (self.n, self.n_cols),
            "cm1_ext": (self.ne, self.n_cols),
            "cm2_base": (self.n, self.cm2_cols),
            "cm2_ext": (self.ne, self.cm2_cols),
            "q_ext": (self.ne, self.si["qDim"]),
            "quotient_cm": (self.ne, self.si["mapSectionsN"][q_name]),
            "const_ext": (self.ne, self.si["nConstants"]),
        }
        for ci, c in enumerate(self.si.get("customCommits", [])):
            self._matrix_shapes[f"custom{ci}_ext"] = (
                self.ne,
                self.si["mapSectionsN"][c["name"] + "0"],
            )

    @cached_property
    def hash_family(self) -> str:
        """The key's Poseidon family — `pilout.globalInfo.json` (at the key
        root above the starkinfo) says `"hash": "Poseidon1"` on the ziskup
        key; the example keys ship no globalInfo and are Poseidon2.

        The family decides every tree, transcript, and grind, so a wrong
        guess byte-mismatches the whole prove with nothing pointing here:
        an unrecognized value raises, and the no-globalInfo fallback (a
        starkinfo copied outside its key tree looks identical to an example
        key) announces itself on stderr."""
        for parent in self.starkinfo_path.parents:
            gi = parent / "pilout.globalInfo.json"
            if gi.exists():
                fam = json.loads(gi.read_text()).get("hash")
                if fam not in ("Poseidon1", "Poseidon2"):
                    raise ValueError(
                        f"{gi}: unknown hash family {fam!r} — expected "
                        "'Poseidon1' or 'Poseidon2'"
                    )
                return fam
        print(
            f"capture {self.instance}: no pilout.globalInfo.json above "
            f"{self.starkinfo_path}; assuming Poseidon2",
            file=sys.stderr,
        )
        return "Poseidon2"

    # -- dumped sections ----------------------------------------------------

    def path(self, name: str) -> pathlib.Path:
        return self.dump / f"{self.instance}_{name}.npy"

    def u64(self, name: str) -> np.ndarray:
        """A dumped section's u64 words, loaded once — several gates read the
        same multi-hundred-MB sections, so uncached reloads dominate a run's
        I/O at real scale. pil2's lazily-reduced arithmetic can leave +P
        residues in dumped sections, which `astype(F)` would ingest as raw
        non-canonical lanes — canonicalize on read; a canonical dump is
        unchanged."""
        if name not in self._sections:
            if self.gpu_dump and name in ("root1", "root2", "rootQ"):
                self._sections[name] = self._staged_root(name)
                return self._sections[name]
            arr = np.load(self.path(name))
            assert arr.dtype == np.uint64, f"{name}: expected u64 dump, got {arr.dtype}"
            if self.gpu_dump and name in self._matrix_shapes:
                arr = untile_gpu(arr, *self._matrix_shapes[name]).reshape(-1)
            p = np.uint64(P)
            if arr.size and int(arr.max()) >= P:
                arr = np.where(arr >= p, arr - p, arr)
            # else: already canonical — skip the full-section copy pass
            # (traces at block scale are GBs; the unconditional `where`
            # was a measurable slice of the witness-ingest tax).
            self._sections[name] = arr
        return self._sections[name]

    def _staged_root(self, name: str) -> np.ndarray:
        """Roots from the pinned staging buffer `setProof` fills: per stage
        1..nStages+1, [root (4 words)] then the arity**llv last-level digests
        when lastLevelVerification is on."""
        ss = self.si["starkStruct"]
        llv = ss.get("lastLevelVerification", 0)
        per = 4 + ((self.arity**llv) * 4 if llv else 0)
        i = {"root1": 0, "root2": 1, "rootQ": self.si["nStages"]}[name]
        staged = np.load(self.path("proof"))
        root = staged[i * per : i * per + 4]
        p = np.uint64(P)
        return np.where(root >= p, root - p, root)

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
        # `view`, not `astype`: u64() canonicalizes on read, and for
        # canonical words the F reinterpret is bit-identical — `astype` was
        # a full extra copy pass per witness (GBs at Main width, #144).
        return fnp.array(words.view(F).reshape(self.n, self.n_cols))

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

    def release(self) -> None:
        """Drop the cached sections and every derived array. The caches make
        one instance's gates cheap, but a block-scale run walks ~40 captures
        in one process — left in place they accumulate the entire dump set
        (host sections plus the device-resident trace). The next read
        reloads from disk; `pil2_key` survives so a family's reused prover
        keeps its artifacts."""
        self._sections.clear()
        for name in (
            "trace",
            "bufs",
            "opened_columns",
            "const_base",
            "base_env",
            "challenges",
            "cexp_env",
            "_scalar_env",
        ):
            self.__dict__.pop(name, None)

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
            # The dump carries customs extended-only; the stage-2 witness
            # role reads them on the base domain (Rom).
            custom_base={
                ci: unextend(self.bufs[("custom", ci)], self.stride)
                for ci in range(self.n_customs)
            },
            hash_family=self.hash_family,
        )

    def pil2_claim(self) -> Pil2Claim:
        """The dumped instance's statement — what the pil2-mode prover takes
        alongside the trace."""
        return Pil2Claim(
            n_bits=self.nb,
            n_cols=self.n_cols,
            publics=self.u64("publics"),
            # An air with no airvalues (empty ``airValuesMap``) dumps no file;
            # a missing dump for an air that HAS them stays loud.
            airvalues=(
                self.u64("airvalues")
                if self.si.get("airValuesMap")
                else np.zeros(0, dtype=np.uint64)
            ),
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
            **expand_scalars(self._scalar_env),
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
            **expand_scalars(self._scalar_env),
            "zi": {},
        }
