"""Shape-keyed interning of device arrays, releasable as a block.

Several prove-stage constants depend only on a domain shape — the extended
coset (`quotient.zerofier`), the LEv constant pack (`evals.lev`) — and cost
milliseconds to rebuild, so they are interned rather than recomputed per prove.
A plain `functools.cache` interns them for the PROCESS lifetime, which is the
one thing a block run cannot afford: `harness.block_composite` drops every
per-instance buffer as its instance finishes so that 100+ instances fit beside
a quotient working set deliberately sized to fill the card, and an interned
coset is 64 MB at `nBitsExt` 23 — residency no release path could reach.

`shape_cache` is that same interning plus a handle in the registry, so
`release_shape_caches` can drop the lot at a residency boundary, the way
`harness.pil2.release_device_sections` drops a key's uploaded sections. Shapes
recur across instances, so that boundary is a FAMILY boundary, not a
per-instance one; the entries rebuild on the next prove of that shape.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

# `Any` because what `functools.cache` hands back is a wrapper whose type is
# private to the stdlib; all the registry needs of an entry is `cache_clear`.
_CACHED: list[Any] = []


def shape_cache(fn: Callable) -> Any:
    """`functools.cache` whose entries `release_shape_caches` can drop.

    The wrapped function must be a pure function of a shape — hashable
    arguments, nothing per-prove — since a dropped entry is rebuilt on the next
    call and has to be indistinguishable from the one it replaces."""
    cached = functools.cache(fn)
    _CACHED.append(cached)
    return cached


def release_shape_caches() -> None:
    """Drop every interned shape-keyed value.

    Belongs at a residency boundary where the device buffers cost more than the
    rebuild — `block_composite`'s family boundary, beside
    `release_device_sections`. Not per prove: the rebuild is milliseconds a
    shape, but every prove of that shape would then pay it."""
    for cached in _CACHED:
        cached.cache_clear()
