"""The recursion circuits' witness chain — zkin to committed stage-1 trace.

pil2's aggregation stages are STARK proves over circom-generated AIRs; the
witness is computed by the proving key's own compiled calculator (the same
``.so`` the native prover loads: ``initCircuit``/``getSizeWitness``/
``getWitness``), then ``.exec``'s add rows extend it and the sMap gather
places it into trace columns (proofman's ``get_committed_pols``).

`recursion_pil2_key` builds a `Pil2Key` straight from the proving-key
directory — no capture: base constants from the ``.const`` binary, the
extended section by the same coset LDE the prover commits with. The
constant tree's root then MUST equal the circuit's verkey, which
`prove_recursion`-side rigs assert before trusting anything else.

Key layout (ziskup): ``zisk/<group>/airs/<air>/{compressor,recursive1}``,
``zisk/<group>/recursive2`` (recursive1 shares its starkinfo), and
``zisk/vadcop_final``.
"""

from __future__ import annotations

import ctypes
import json
import pathlib

import frx
import frx.numpy as fnp
import numpy as np
from frx import Array
from zk_dtypes import goldilocks as F

from zisk_zorch.commit.trace_commit import extend
from zisk_zorch.harness.pil2 import Pil2Key

P = 0xFFFFFFFF00000001


def circuit_base(key: pathlib.Path, gi: dict, ty: str, air_idx: int) -> pathlib.Path:
    """The circuit artifact base (``<base>.{so,dat,exec,starkinfo.json}``)."""
    group0 = gi["air_groups"][0]
    airs = [a["name"] for a in gi["airs"][0]]
    root = key / "zisk" if (key / "zisk").is_dir() else key / "build"
    if ty == "compressor":
        return root / group0 / "airs" / airs[air_idx] / "compressor" / "compressor"
    if ty == "recursive1":
        return root / group0 / "airs" / airs[air_idx] / "recursive1" / "recursive1"
    if ty == "recursive2":
        return root / group0 / "recursive2" / "recursive2"
    if ty in ("vadcopfinal", "vadcop_final"):
        return root / "vadcop_final" / "vadcop_final"
    raise ValueError(f"unknown recursion circuit type {ty!r}")


def recursion_starkinfo(base: pathlib.Path) -> pathlib.Path:
    """recursive1 dirs carry const/verkey only — the shape is recursive2's."""
    si = base.with_suffix(".starkinfo.json")
    if si.exists():
        return si
    return base.parent.parent.parent.parent / "recursive2" / "recursive2.starkinfo.json"


def _add_waves(adds: np.ndarray, size_witness: int) -> list[np.ndarray]:
    """The exec adds' rows grouped by dependency depth: wave `k` reads only
    witness cells and slots settled by waves below it, so a wave evaluates as
    one vectorized batch.

    An add's operands are always earlier slots, so the depth relaxation
    converges in `max(depth) + 1` passes — five on the recursion circuits.
    Key-fixed, hence computed once per calculator."""
    n = len(adds)
    if not n:
        return []
    dep = adds[:, :2].astype(np.int64) - size_witness
    level = np.zeros(n, dtype=np.int64)
    for _ in range(n):
        nxt = np.zeros((n, 2), dtype=np.int64)
        for k in (0, 1):
            reads_add = dep[:, k] >= 0
            nxt[reads_add, k] = level[dep[reads_add, k]] + 1
        nxt = nxt.max(axis=1)
        if np.array_equal(nxt, level):
            break
        level = nxt
    else:
        raise AssertionError("exec adds do not form an acyclic chain")
    return [np.flatnonzero(level == lv) for lv in range(int(level.max()) + 1)]


@frx.jit
def _gather_rows(w: Array, smap: Array) -> Array:
    """The sMap placement gather — `committed_trace`'s device half."""
    return w[smap]


class CircomCalc:
    """One circuit's compiled witness calculator plus its exec placement data
    (proofman's ``Setup::circom_state`` ABI)."""

    def __init__(self, base: pathlib.Path):
        self.lib = ctypes.CDLL(str(base) + ".so")
        self.lib.initCircuit.restype = ctypes.c_void_p
        self.lib.initCircuit.argtypes = [ctypes.c_char_p]
        self.lib.getSizeWitness.restype = ctypes.c_uint64
        self.lib.getWitness.restype = ctypes.c_int64
        self.lib.getWitness.argtypes = [
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint64,
        ]
        self.circuit = self.lib.initCircuit((str(base) + ".dat").encode())
        assert self.circuit, f"initCircuit failed for {base}"
        self.size_witness = self.lib.getSizeWitness()
        exec_words = np.fromfile(str(base) + ".exec", dtype=np.uint64)
        self.n_adds, self.n_smap = int(exec_words[0]), int(exec_words[1])
        self.adds = exec_words[2 : 2 + self.n_adds * 4].reshape(-1, 4)
        # int64 up front: the sMap is ~n_smap x n_cols and re-converting it
        # per prove dominated committed_pols for the wide recursion circuits.
        self.smap_flat = exec_words[2 + self.n_adds * 4 :].astype(np.int64)
        self._add_waves = _add_waves(self.adds, self.size_witness)
        self._smap_dev: Array | None = None

    def witness(self, zkin: np.ndarray) -> np.ndarray:
        buf = np.zeros(self.size_witness + self.n_adds, dtype=np.uint64)
        rc = self.lib.getWitness(
            np.ascontiguousarray(zkin, dtype=np.uint64).ctypes.data_as(
                ctypes.POINTER(ctypes.c_uint64)
            ),
            self.circuit,
            buf.ctypes.data_as(ctypes.c_void_p),
            8,
        )
        assert rc == 0, f"getWitness rc={rc}"
        return buf

    def _settled(self, w: np.ndarray) -> np.ndarray:
        """The witness with its exec adds appended and every cell canonical —
        what the sMap gathers from.

        The adds run one vectorized batch per `_add_waves` level: a row may
        read an earlier row's output, but only five levels deep on the
        recursion circuits, so the Python-level multiply-add per row the
        dependency used to force (23004 rows, ~18 ms a prove on recursive2)
        collapses to five array ops.

        Canonicalizing HERE rather than on the caller's trace reduces a 41 MB
        witness instead of the 65 MB the gather repeats it into, and the
        residues are sparse (292 cells of 5 M), so it is a scan plus a
        scatter."""
        w = w.copy()
        for sel in self._add_waves:
            a = w[self.adds[sel, 0]].astype(object) * self.adds[sel, 2].astype(object)
            b = w[self.adds[sel, 1]].astype(object) * self.adds[sel, 3].astype(object)
            w[self.size_witness + sel] = ((a + b) % P).astype(np.uint64)
        # Index 0 = unmapped cell, forced zero: zeroing w[0] (w is a local
        # copy, and the adds above are already computed) makes the gather
        # itself produce the zeros — no separate mask pass over the trace.
        w[0] = 0
        residues = np.flatnonzero(w >= np.uint64(P))
        if residues.size:
            w[residues] -= np.uint64(P)
        return w

    def committed_pols(self, w: np.ndarray, n: int, n_cols: int) -> np.ndarray:
        """proofman's ``get_committed_pols``: exec adds appended to the
        witness, then the sMap gather places values into trace columns
        (index 0 = unmapped cell, forced zero). Canonical on return."""
        w = self._settled(w)
        smap = self.smap_flat.reshape(self.n_smap, n_cols)
        if n == self.n_smap:
            out = np.empty((n, n_cols), dtype=np.uint64)
            np.take(w, smap, out=out)
        else:
            out = np.zeros((n, n_cols), dtype=np.uint64)
            np.take(w, smap, out=out[: self.n_smap])
        return out

    def committed_trace(self, w: np.ndarray, n: int, n_cols: int) -> Array:
        """`committed_pols` with the placement gather on the device — the
        field-typed, canonical trace an `InnerWitness` takes.

        The gather is ~8 M scattered reads producing a 65 MB trace from a
        41 MB witness: ~17 ms of host memory latency, and the prover then
        uploads the larger of the two. Moving it behind the upload sends the
        witness instead and leaves the trace where the commit wants it. The
        sMap is key-fixed, so it is uploaded once per calculator no matter
        how many instances the circuit proves."""
        if self._smap_dev is None:
            self._smap_dev = fnp.asarray(
                self.smap_flat.reshape(self.n_smap, n_cols).astype(np.int32)
            )
        rows = _gather_rows(fnp.array(self._settled(w).view(F)), self._smap_dev)
        if n == self.n_smap:
            return rows
        return fnp.concatenate([rows, fnp.zeros((n - self.n_smap, n_cols), F)])


def recursion_pil2_key(base: pathlib.Path, hash_family: str) -> tuple[Pil2Key, dict]:
    """A `Pil2Key` from the proving-key directory alone (no capture): base
    constants from ``.const``, extended by the prover's own coset LDE."""
    si = json.loads(recursion_starkinfo(base).read_text())
    ss = si["starkStruct"]
    n = 1 << ss["nBits"]
    blowup = 1 << (ss["nBitsExt"] - ss["nBits"])
    const_base = np.fromfile(str(base) + ".const", dtype=np.uint64).reshape(
        n, si["nConstants"]
    )
    const_ext = np.asarray(
        extend(fnp.array(const_base.view(F)), blowup), dtype=np.uint64
    ).view(F)
    key = Pil2Key(
        starkinfo=si,
        expressionsinfo=json.loads(
            recursion_starkinfo(base)
            .with_name(
                recursion_starkinfo(base).name.replace(".starkinfo", ".expressionsinfo")
            )
            .read_text()
        ),
        const_base=const_base.view(F),
        const_ext=const_ext,
        custom_ext={},
        custom_base={},
        hash_family=hash_family,
    )
    return key, si
