"""Run an exported AIR's programs by manifest name — the Python twin of the
Rust bridge's artifact loader.

`Artifact` loads one ``<Air>_n<nBits>/`` directory (`export_air`), compiles
each program's StableHLO bytecode through the low-level PJRT client on
demand, and executes it with arguments bound by the manifest's input
names. Everything the Rust bridge does with the artifacts, this does with
the same bytes: the schedule replay (`replay.py`) is written against this
class so a replay/prover mismatch points at the artifacts, and a
replay/bridge mismatch at the Rust.

Arguments may be host arrays (placed on the device per the manifest's
dtype) or device arrays a previous program returned (passed through).
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

import frx
import frx.numpy as fnp
import numpy as np
from frx import Array
from frx._src.lib import xla_client
from zk_dtypes import goldilocks as F
from zk_dtypes import goldilocksx3 as F3
from zorch.utils.field import join_coeffs

_HOST_DTYPES = {
    "uint64": np.uint64,
    "uint32": np.uint32,
    "int32": np.int32,
}


class Artifact:
    """One exported AIR: its manifest and lazily compiled programs."""

    def __init__(self, path: str | pathlib.Path, device=None) -> None:
        self.path = pathlib.Path(path)
        self.manifest = json.loads((self.path / "manifest.json").read_text())
        if self.manifest.get("artifact_version") != 1:
            raise ValueError(
                f"unsupported artifact_version {self.manifest.get('artifact_version')}"
            )
        self.device = frx.devices()[0] if device is None else device
        self._executables: dict[str, Any] = {}

    # -- the manifest -------------------------------------------------------

    @property
    def programs(self) -> dict:
        return self.manifest["programs"]

    def inputs(self, name: str) -> list[dict]:
        return self.programs[name]["inputs"]

    def outputs(self, name: str) -> list[dict]:
        return self.programs[name]["outputs"]

    def has(self, name: str) -> bool:
        return name in self.programs

    # -- compile + execute ----------------------------------------------------

    def executable(self, name: str):
        """The compiled program, compiled on first use."""
        exe = self._executables.get(name)
        if exe is None:
            code = (self.path / self.programs[name]["file"]).read_bytes()
            exe = self.device.client.compile_and_load(
                code,
                executable_devices=xla_client.DeviceList((self.device,)),
                compile_options=xla_client.CompileOptions(),
            )
            self._executables[name] = exe
        return exe

    def compile_all(self) -> None:
        for name in self.programs:
            self.executable(name)

    def place(self, value, spec: dict) -> Array:
        """`value` as the device array `spec` describes. Device arrays pass
        through untouched (shape-checked); host arrays are placed by dtype:
        field words as their canonical u64 storage, cubic values as
        ``(..., 3)`` limb rows."""
        dims = tuple(spec["dims"])
        dtype = spec["dtype"]
        if isinstance(value, Array):
            if tuple(value.shape) != dims:
                raise ValueError(
                    f"{spec['name']}: expected shape {dims}, got {tuple(value.shape)}"
                )
            return value
        if dtype == "goldilocks":
            host = np.ascontiguousarray(value, dtype=np.uint64).reshape(dims).view(F)
            return frx.device_put(host, self.device)
        if dtype == "goldilocksx3":
            words = np.ascontiguousarray(value, dtype=np.uint64).reshape(*dims, 3)
            return join_coeffs(frx.device_put(words.view(F), self.device), F3)
        host = np.ascontiguousarray(value, dtype=_HOST_DTYPES[dtype]).reshape(dims)
        # uint64 words must reach the device 64 bits wide; x64 is scoped to
        # the upload so the rest of the process keeps its default widths.
        with frx.enable_x64():
            return frx.device_put(host, self.device)

    def run(self, name: str, **kwargs) -> dict[str, Array]:
        """Execute `name` with its inputs bound by manifest name; the
        outputs come back keyed by their manifest names."""
        specs = self.inputs(name)
        missing = [s["name"] for s in specs if s["name"] not in kwargs]
        if missing:
            raise KeyError(f"{name}: missing inputs {missing}")
        extra = set(kwargs) - {s["name"] for s in specs}
        if extra:
            raise KeyError(f"{name}: unknown inputs {sorted(extra)}")
        args = [self.place(kwargs[s["name"]], s) for s in specs]
        results = self.executable(name).execute(args)
        return {o["name"]: r for o, r in zip(self.outputs(name), results)}


def words(x: Array) -> np.ndarray:
    """A device field array as host canonical u64 words — cubic arrays as
    ``(..., 3)`` limb rows."""
    host = np.asarray(x)
    if host.dtype == F3:
        return host.view(np.uint64).reshape(*host.shape, 3)
    return host.view(np.uint64)


def field(x: np.ndarray) -> Array:
    """Host u64 words as a device base-field array."""
    return fnp.asarray(np.ascontiguousarray(x, dtype=np.uint64).view(F))
