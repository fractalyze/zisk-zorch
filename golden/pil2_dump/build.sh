#!/usr/bin/env bash
# Build the stage-1 trace-commit dump harness against a pil2-proofman checkout.
#
# The harness links pil2-stark's real Goldilocks CUDA kernels (the code ZisK
# runs on GPU) and captures a byte-match reference dump for
# zisk_zorch/commit/verify_trace_commit.py. See README.md for the full recipe.
#
#   PIL2_STARK=/path/to/pil2-proofman/pil2-stark ./build.sh
#
# Requires: CUDA nvcc (tested 13.3), an sm_120 (RTX 5090) or your GPU's arch,
# and gmp. pil2-proofman must be checked out at tag v1.0.0-alpha.
set -euo pipefail

PIL2_STARK="${PIL2_STARK:?set PIL2_STARK to <pil2-proofman>/pil2-stark}"
ARCH="${CUDA_ARCH:-sm_120}"
GMP_PREFIX="${GMP_PREFIX:-/usr}"          # dir containing include/gmp.h + lib/libgmp
OUT="${OUT:-$(dirname "$0")/pil2_dump_stage1}"

G="$PIL2_STARK/src/goldilocks"
HERE="$(cd "$(dirname "$0")" && pwd)"

# Same flags pil2-stark's own goldilocks benches build with (-U STARK_POSEIDON1
# selects the Poseidon2 path). We compile the harness plus the goldilocks
# translation units it references; no google-benchmark dependency.
/usr/local/cuda/bin/nvcc -ccbin /usr/bin/g++ -allow-unsupported-compiler \
  -D__USE_CUDA__ -D__GOLDILOCKS_ENV__ -O3 -std=c++17 -arch="$ARCH" \
  -Xcompiler -O3 -Xcompiler -fopenmp -Xcompiler -fPIC -Xcompiler -mavx2 \
  -U STARK_POSEIDON1 \
  -I "$G/src" -I "$G/utils" -I "$PIL2_STARK/src" -I "$PIL2_STARK/src/utils" \
  -I "$PIL2_STARK/external/sppark" -I "$PIL2_STARK/external/sppark/ff" \
  -I "$GMP_PREFIX/include" \
  "$HERE/pil2_dump_stage1.cu" \
  "$G/src/ntt_goldilocks.cu" "$G/src/ntt_goldilocks.cpp" \
  "$G/src/poseidon2_goldilocks.cu" "$G/src/poseidon2_goldilocks.cpp" \
  "$G/src/goldilocks_base_field.cpp" "$G/src/goldilocks_tooling.cu" \
  -L "$GMP_PREFIX/lib" -lgmp -lgmpxx -lgomp \
  -o "$OUT"
echo "built $OUT"
