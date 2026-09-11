#!/usr/bin/env python3
"""What of the proving key is in the page cache, and the two ways to set it.

proofman's `INITIALIZING_PROOFMAN` reads a fixed set of the key's files before
it sizes its GPU buffers, and pays disk for whatever is not cached. That makes
a run's init a reading of the host's page cache at the moment it started, which
is why #217 exists: #214 read init against the *arm* that ran before it and
found a step of about a second between the two groups, and the arm turned out
to be a proxy for this and not a mechanism. Anything on the host that reads a
few GiB moves the same figure; a sibling session's build server emptied the
key out of the cache between two of #217's own sweeps.

So a quoted init needs the set warm, the same way a quoted leg needs the
executable cache warm. `--warm` before a timed run is that reset;
`--evict` is the other arm of the experiment that shows it is needed, and the
census is how either is checked rather than assumed.

Three things here are easy to get wrong by hand.

**The set is not the key.** `INIT_SUFFIXES` is what that sub-phase reads: the
const pols in GPU layout, the `.exec` and `.dat` files of the recursion setups,
and the small binaries and JSON beside them. What it leaves out is most of the
key by size -- the `.const` pols and the constant trees, which the later phases
stream and which no host has the memory to hold anyway. Warming the key instead
of the set would evict the set to make room for files init does not read
there.

**mincore is a census, not a read.** It reports which of a mapping's pages are
resident without faulting any in, so taking the census does not warm what it
measures. That is the whole reason to map the file rather than read it. Python's
own `mmap` will not hand out a read-only mapping's address, so the mapping is
made through libc.

**Evicting drops clean pages only.** `POSIX_FADV_DONTNEED` is advice: a dirty
page stays, and so does one another process still has mapped. Nothing writes to
a proving key, so the cold arm is reliable on one -- but check the census it
prints rather than assuming the advice was taken, and do not reach for this on
a file something else is writing.

Usage:
  pagecache.py [--tsv|--per-file] [--warm|--evict] <label>=<path-or-glob> ...
  pagecache.py [--warm|--evict] --proofman-init <proving-key-dir>
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import glob
import os
import pathlib
import sys
import typing

PAGE = os.sysconf("SC_PAGE_SIZE")
POSIX_FADV_DONTNEED = 4
PROT_READ, MAP_SHARED = 0x1, 0x01
READ_CHUNK = 16 << 20

# The extensions `INITIALIZING_PROOFMAN` reads before it prints its buffer
# sizes. `.const` and `.consttree_gpu` are deliberately absent: they are the
# bulk of a proving key and belong to the phases after it.
INIT_SUFFIXES = ("const_gpu", "exec", "dat", "bin", "json", "so")

_libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
_libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p]
_libc.mincore.restype = ctypes.c_int
_libc.mmap.argtypes = [
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_long,
]
_libc.mmap.restype = ctypes.c_void_p
_libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
_libc.munmap.restype = ctypes.c_int
_libc.posix_fadvise.argtypes = [
    ctypes.c_int,
    ctypes.c_long,
    ctypes.c_long,
    ctypes.c_int,
]
_libc.posix_fadvise.restype = ctypes.c_int
MAP_FAILED = ctypes.c_void_p(-1).value


class Census(typing.NamedTuple):
    """Bytes of a file, or of a group of them, that are in the page cache."""

    resident: int
    total: int

    def __add__(self, other: "Census") -> "Census":
        return Census(self.resident + other.resident, self.total + other.total)

    @property
    def share(self) -> float:
        return self.resident / self.total if self.total else 0.0


def census(path: str | os.PathLike[str]) -> Census:
    """How much of `path` the kernel has cached, without faulting any of it in."""
    size = os.path.getsize(path)
    if size == 0:
        return Census(0, 0)
    fd = os.open(path, os.O_RDONLY)
    try:
        addr = _libc.mmap(None, size, PROT_READ, MAP_SHARED, fd, 0)
        if addr == MAP_FAILED:
            raise OSError(ctypes.get_errno(), f"mmap failed on {path}")
        try:
            vector = ctypes.create_string_buffer((size + PAGE - 1) // PAGE)
            if _libc.mincore(ctypes.c_void_p(addr), size, vector) != 0:
                raise OSError(ctypes.get_errno(), f"mincore failed on {path}")
            resident = sum(1 for flags in vector.raw if flags & 1)
        finally:
            _libc.munmap(ctypes.c_void_p(addr), size)
    finally:
        os.close(fd)
    # The file's last page is partial; charge the file's bytes, not the page's.
    return Census(min(resident * PAGE, size), size)


def evict(path: str | os.PathLike[str]) -> None:
    """Ask the kernel to drop `path`'s clean pages."""
    fd = os.open(path, os.O_RDONLY)
    try:
        _libc.posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def warm(path: str | os.PathLike[str]) -> None:
    """Read `path` so a run that needs it does not wait on the disk for it."""
    buffer = bytearray(READ_CHUNK)
    with open(path, "rb", buffering=0) as handle:
        while handle.readinto(buffer):
            pass


def expand(pattern: str) -> typing.Iterator[str]:
    """The files a pattern names: a directory walks, a glob globs."""
    if os.path.isdir(pattern):
        for root, _, names in os.walk(pattern):
            for name in sorted(names):
                yield os.path.join(root, name)
    else:
        yield from sorted(
            p for p in glob.glob(pattern, recursive=True) if os.path.isfile(p)
        )


def init_set(proving_key: str | os.PathLike[str]) -> list[tuple[str, str]]:
    """The labelled patterns proofman's init reads out of a proving key."""
    return [(suffix, f"{proving_key}/**/*.{suffix}") for suffix in INIT_SUFFIXES]


def apply(
    groups: list[tuple[str, str]],
    action: typing.Callable[[str], None] | None = None,
    per_file: typing.TextIO | None = None,
) -> list[tuple[str, Census]]:
    """Census every group, having first done `action` to each of its files."""
    out = []
    for label, pattern in groups:
        total = Census(0, 0)
        for path in expand(pattern):
            if action is not None:
                action(path)
            try:
                one = census(path)
            except OSError:
                continue
            if per_file is not None:
                print(f"{path}\t{one.resident}\t{one.total}", file=per_file)
            total += one
        out.append((label, total))
    return out


def _row(label: str, entry: Census) -> str:
    gib = 2**30
    return (
        f"{label:24s} {entry.resident / gib:8.3f} / {entry.total / gib:8.3f} GiB"
        f" {100 * entry.share:6.2f}%"
    )


def report(rows: list[tuple[str, Census]], tsv: bool, out: typing.TextIO) -> None:
    for label, entry in rows:
        line = (
            f"{label}\t{entry.resident}\t{entry.total}" if tsv else _row(label, entry)
        )
        print(line, file=out)
    if len(rows) > 1 and not tsv:
        print(_row("TOTAL", sum((entry for _, entry in rows), Census(0, 0))), file=out)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("group", nargs="*", help="<label>=<path-or-glob>")
    ap.add_argument(
        "--proofman-init",
        type=pathlib.Path,
        help="the set proofman's init reads out of this proving key",
    )
    ap.add_argument("--warm", action="store_true", help="read the files back in first")
    ap.add_argument(
        "--evict", action="store_true", help="drop the files from the cache first"
    )
    ap.add_argument(
        "--tsv", action="store_true", help="label, resident, total, in bytes"
    )
    ap.add_argument(
        "--per-file", action="store_true", help="the same three columns per file"
    )
    args = ap.parse_args(argv[1:])
    if args.warm and args.evict:
        ap.error("--warm and --evict are the two ends of the reset; pick one")

    groups = [
        (label, pattern) for label, _, pattern in (g.partition("=") for g in args.group)
    ]
    if args.proofman_init:
        init_groups = init_set(args.proofman_init)
        # A key whose init set matches nothing is a wrong path, not a cold one.
        # Left to the report, that prints a clean table of zeroes and exits 0,
        # and a caller warming before a timed run proceeds believing it did.
        if all(next(expand(pattern), None) is None for _, pattern in init_groups):
            ap.error(f"no proving-key files under {args.proofman_init}")
        groups += init_groups
    if not groups:
        ap.error("nothing to census: pass a <label>=<pattern> or --proofman-init")

    rows = apply(
        groups,
        evict if args.evict else warm if args.warm else None,
        sys.stdout if args.per_file else None,
    )
    if not args.per_file:
        report(rows, args.tsv, sys.stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
