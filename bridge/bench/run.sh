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
# emulator built, which is what the hello-world guest runs on).
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
{ uptime; nvidia-smi --query-gpu=memory.used --format=csv,noheader; } > "$OUT/host.txt"
# shellcheck disable=SC2086
env "$@" /usr/bin/time -v "$ZISK_BIN" prove -e "$ZISK_ELF" ${ZISK_IN:+-i $ZISK_IN} ${ZISK_PROVE_FLAGS--a -u} \
    -k "$ZISK_PK" -g -y -o "$OUT/proof" -vv > "$OUT/run.log" 2>&1
echo "exit=$?" >> "$OUT/run.log"
grep -E 'Elapsed|exit=|<<< (INITIALIZING_PROOFMAN|CALCULATING_CONTRIBUTIONS|GENERATING_INNER_PROOFS)|verified' \
    "$OUT/run.log" | sed -E 's/^.*INFO: //; s/^\s+//'
