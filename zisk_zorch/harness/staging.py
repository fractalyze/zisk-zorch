"""Pinned-host staging for witness uploads (#115's perf half, from #144).

`fnp.array` on a pageable numpy buffer is a synchronous pageable H2D — the
slowest way across the bus at Main width (1.28 GB/instance). Staging
through the `pinned_host` memory space splits the upload into a host copy
into DMA-able memory plus an async device transfer, so the block driver
can dispatch the next instance's upload behind the current instance's
commit kernels instead of idling at the root sync.
"""

from __future__ import annotations

import sys

import frx
import frx.numpy as fnp
import numpy as np
from frx.sharding import SingleDeviceSharding
from zk_dtypes import goldilocks as F


class TraceStager:
    """Two-hop async upload: host words -> ``pinned_host`` -> device.

    `stage` dispatches and returns immediately; callers block only where
    they consume the array. A backend without a ``pinned_host`` memory
    space falls back to the plain upload, so the driver's schedule is the
    same everywhere and only the transfer path differs."""

    def __init__(self, device=None) -> None:
        if device is None:
            # A zero-size placement resolves the ambient default device —
            # an enclosing `frx.default_device(...)` included — where
            # `frx.devices()[0]` would pin the backend's first device
            # regardless of the caller's context.
            device = next(iter(fnp.empty(0).devices()))
        self._route: tuple[SingleDeviceSharding, SingleDeviceSharding] | None = None
        try:
            kinds = {m.kind for m in device.addressable_memories()}
        except RuntimeError:  # backend predates memory spaces
            kinds = set()
        if "pinned_host" not in kinds:
            return
        pinned = SingleDeviceSharding(device, memory_kind="pinned_host")
        # Probe once: a backend that names the space but cannot place
        # buffers there should fall back now, not mid-block.
        try:
            frx.device_put(np.zeros(1, dtype=np.uint64), pinned)
        except (ValueError, RuntimeError) as e:
            # Degrading silently would hide a slow pageable block (and
            # whatever runtime breakage tripped the probe) until the next
            # re-profile against #144's baseline, so say why.
            print(
                f"TraceStager: pinned_host probe failed ({e}); "
                "falling back to pageable uploads",
                file=sys.stderr,
            )
        else:
            self._route = (pinned, SingleDeviceSharding(device, memory_kind="device"))

    def stage(self, words: np.ndarray):
        """`words` (canonical Goldilocks u64) on the device, F-typed.

        `view`, not `astype`: the `WitnessSource` contract says canonical
        words, for which the F reinterpret is bit-identical — `astype` was
        a full extra copy pass per witness (GBs at Main width, #144). The
        staged hops copy bits, never re-encode, so the result is
        bit-identical to `fnp.array(words.view(F))` by construction; the
        host buffer stays referenced until the pinned copy lands, so
        releasing the source while the upload is in flight is safe."""
        lanes = words.view(F)
        if self._route is None:
            return fnp.asarray(lanes)
        pinned, device = self._route
        return frx.device_put(frx.device_put(lanes, pinned), device)
