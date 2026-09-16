#!/usr/bin/env python3
"""Which prover binary and PJRT plugin a run was made with, written into
`host.txt` beside the log and read back out of it by the summarizer.

`cargo-zisk-dev` reaches the bridge through a `[patch]` path into a zisk-zorch
worktree, so which one it carries is a property of the last `cargo build` and
of nothing inside the run. Without this record a run's figures cannot be
attributed to a build at all.

The prover is identified by a sha256 of its bytes, because a rebuild from
unchanged source is the same prover and mtime alone would call it different.
The plugin is identified by the path, size and mtime the bridge's executable
cache already keys an entry on (docs/bridge.md, "Compile cost"), so a plugin
those three agree on is one the cache would have reused.

One module because the format has two ends -- `run.sh` appends the lines
through the CLI here, `summarize.py` reads them through `parse` -- and two
copies of it drift.

Usage (appending to a run's host.txt):
  binaries.py --prover <path> [--plugin <path>]
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import pathlib
import shlex
import sys
import typing

PROVER, PLUGIN = "prover", "plugin"


class Binary(typing.NamedTuple):
    """One binary a run was made with, as `host.txt` records it."""

    role: str
    path: str
    size: int
    mtime: str
    sha256: str = ""

    def line(self) -> str:
        fields = [
            f"path={shlex.quote(self.path)}",
            f"size={self.size}",
            f"mtime={self.mtime}",
        ]
        if self.sha256:
            fields.append(f"sha256={self.sha256}")
        return " ".join([self.role, *fields])

    def describe(self) -> str:
        """The short form the summary header carries: enough to tell two builds
        apart, not so much that it crowds out the run's own numbers."""
        name = pathlib.PurePath(self.path).name
        # Exact bytes, not a rounded MiB: for the plugin the size is half of
        # what identifies it, and two builds a few KiB apart have to read
        # differently here or the record cannot do its job.
        what = f"sha {self.sha256[:12]}" if self.sha256 else f"{self.size} B"
        return f"{self.role} {name} {what} @{self.mtime}"


def identify(role: str, path: str, *, digest: bool) -> Binary:
    # Resolved, because a record is read from wherever the run directory ends
    # up and a relative $ZISK_BIN would name nothing there.
    binary = pathlib.Path(path).resolve()
    stat = binary.stat()
    mtime = datetime.datetime.fromtimestamp(stat.st_mtime, datetime.timezone.utc)
    sha = ""
    if digest:
        with open(binary, "rb") as f:
            sha = hashlib.file_digest(f, "sha256").hexdigest()
    return Binary(
        role, str(binary), stat.st_size, mtime.strftime("%Y-%m-%dT%H:%M:%SZ"), sha
    )


def parse(text: str) -> dict[str, Binary]:
    """The records in a `host.txt`, by role.

    The host's own `uptime` and `nvidia-smi` lines were in that file first and
    this format is a guest in it, so anything that does not read as a record --
    including a line `shlex` or `int` chokes on -- is skipped rather than
    raised on. A summary must not die on the company its records keep."""
    found = {}
    for raw in text.splitlines():
        try:
            words = shlex.split(raw)
            if not words or words[0] not in (PROVER, PLUGIN):
                continue
            fields = dict(w.split("=", 1) for w in words[1:] if "=" in w)
            found[words[0]] = Binary(
                words[0],
                fields["path"],
                int(fields["size"]),
                fields["mtime"],
                fields.get("sha256", ""),
            )
        except (KeyError, ValueError):
            continue
    return found


def describe(host_txt: pathlib.Path) -> str:
    """The identity line for a run whose `host.txt` is this path.

    A missing record is reported rather than skipped: a summary that cannot
    name the binary it describes is the state this whole module exists to make
    visible, so it must not look like a summary that simply had nothing to
    add."""
    try:
        found = parse(host_txt.read_text(errors="replace"))
    except OSError:
        found = {}
    if PROVER not in found:
        return "prover not recorded"
    parts = [found[PROVER].describe()]
    # run.sh records a plugin for its bridged arm only, so its absence says the
    # run loaded none rather than that the record is short one line.
    parts.append(found[PLUGIN].describe() if PLUGIN in found else "plugin none")
    return "  ".join(parts)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--prover", required=True, help="$ZISK_BIN: the prover that made the run"
    )
    ap.add_argument("--plugin", help="$XLA_PJRT_PLUGIN, on a bridged run")
    args = ap.parse_args(argv[1:])
    # Both records are built before either is printed, so a caller that fails
    # here leaves no half-written identity behind for a reader to trust.
    try:
        lines = [identify(PROVER, args.prover, digest=True).line()]
        if args.plugin:
            lines.append(identify(PLUGIN, args.plugin, digest=False).line())
    except OSError as err:
        ap.error(str(err))
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
