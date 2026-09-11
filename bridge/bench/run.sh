#!/bin/bash
# Prove one guest through native pil2 or the bridge and keep the log and the
# per-instance proof dumps:  run.sh <tag> <native|bridge> [ENV=value ...]
#
# Required in the environment (see docs/bridge.md "Running"):
#   ZISK_BIN  the bridged cargo-zisk        ZISK_ELF  the guest ELF
#   ZISK_PK   the proving key directory
#   ZZ_ARTIFACTS, XLA_PJRT_PLUGIN           for the bridge
# Optional: ZISK_IN (the input file, for a guest that reads one — the go
# hello-world guest does not), ZZ_RUNS (output root, default ./zz-runs),
# ZZ_MEMORY_FRACTION (default 0.45; set it to the empty string to let the
# clients allocate on demand rather than claim a share up front),
# ZISK_PROVE_FLAGS (default "-a -u": the ASM emulator, mapped memory
# unlocked; set it to the empty string for a host without the ASM
# emulator built, which is what the hello-world guest runs on),
# ZZ_WARM_KEY (default 1: read the proving-key files proofman's init reads
# into the page cache before the run — set it to 0 only to measure a cold
# init on purpose. The census into pagecache.txt happens either way; see
# docs/bridge.md "What sets a run's init is the page cache").
set -u
TAG=$1; MODE=$2; shift 2
OUT=${ZZ_RUNS:-./zz-runs}/$TAG
rm -rf "$OUT"; mkdir -p "$OUT/dumps"
if [ "$MODE" = bridge ]; then
  # `-` not `:-` for the fraction: an explicitly empty ZZ_MEMORY_FRACTION is
  # how the bridge is told to allocate on demand instead of claiming a share
  # up front, which is the only way to see a client's true working set.
  export ZZ_ARTIFACTS ZZ_CLIENTS=${ZZ_CLIENTS:-1} ZZ_MEMORY_FRACTION=${ZZ_MEMORY_FRACTION-0.45} \
         ZZ_GPU_HEADROOM_GB=${ZZ_GPU_HEADROOM_GB:-3} ZZ_LOG=${ZZ_LOG:-2}
else
  unset ZZ_ARTIFACTS
fi
export ZZ_DUMP_PROOFS="$OUT/dumps"
# proofman's init reads a fixed set of the key before it sizes its buffers, so
# a run started on a cold page cache times the disk rather than either stack --
# by seconds. Any other tenant of the host can empty that cache between two
# runs, so warm it here and keep the census beside the log.
#
# The census is taken on EVERY run, warmed or not. A cold run is precisely the
# one whose figure depends on the cache state, so it is the one that must not
# be the run with no record of it. And a census that fails takes the run with
# it: a log whose cache state could not be established is not quotable, and
# looks identical to one where it could.
WARM=(--warm); [ "${ZZ_WARM_KEY:-1}" = 0 ] && WARM=()
python3 "$(dirname "$0")/pagecache.py" "${WARM[@]}" --proofman-init "$ZISK_PK" \
    > "$OUT/pagecache.txt" || { cat "$OUT/pagecache.txt" >&2; exit 1; }
{ uptime; nvidia-smi --query-gpu=memory.used --format=csv,noheader; } > "$OUT/host.txt"
# shellcheck disable=SC2086
env "$@" /usr/bin/time -v "$ZISK_BIN" prove -e "$ZISK_ELF" ${ZISK_IN:+-i $ZISK_IN} ${ZISK_PROVE_FLAGS--a -u} \
    -k "$ZISK_PK" -g -y -o "$OUT/proof" -vv > "$OUT/run.log" 2>&1
echo "exit=$?" >> "$OUT/run.log"
grep -E 'Elapsed|exit=|<<< (INITIALIZING_PROOFMAN|CALCULATING_CONTRIBUTIONS|GENERATING_INNER_PROOFS)|verified' \
    "$OUT/run.log" | sed -E 's/^.*INFO: //; s/^\s+//'
