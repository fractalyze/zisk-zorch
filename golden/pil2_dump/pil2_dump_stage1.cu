// ---------------------------------------------------------------------------
// pil2_dump_stage1.cu -- Capture a stage-1 trace-commit byte-match reference
// from pil2-stark's real Goldilocks CUDA kernels (the code ZisK runs on GPU).
//
// Runs pil2-stark's production stage-1 commit path on a DETERMINISTIC trace:
//   splitmix64 trace (row-major)  ->  fromRowMajorToTiled  ->  NTT LDE (coset)
//     ->  Poseidon2 k-ary Merkle (linear-hash leaves)  ->  root (4 Goldilocks)
//
// The trace is drawn with the SAME splitmix64/rand_fe stream the Rust golden
// generator (golden/src/main.rs) uses, so at a golden's dims+seed the root here
// must equal that golden's committed root -- the harness's own byte-match self
// check, and the proof that this CUDA reference agrees with the `fields`-crate
// reference zisk_zorch is already byte-matched against.
//
// Output: prints the root as canonical-u64. With --dump=<dir> also writes
//   <dir>/trace.bin      raw little-endian u64, row-major N x n_cols
//   <dir>/commit.json    {n_bits, blowup_bits, n_cols, arity, seed, root[4]}
// consumed by zisk_zorch/commit/verify_trace_commit.py.
// ---------------------------------------------------------------------------
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>

// Resolved via -I flags to the pil2-stark goldilocks src/ and utils/ dirs
// (see build.sh) rather than relative paths, so this file need not live inside
// the pil2-stark tree.
#include "goldilocks_base_field.hpp"
#include "ntt_goldilocks.hpp"
#include "ntt_goldilocks.cuh"
#include "poseidon2_goldilocks.hpp"
#include "poseidon2_goldilocks.cuh"
#include "goldilocks_tooling.hpp"
#include "goldilocks_tooling.cuh"
#include "cuda_utils.hpp"

static constexpr uint64_t GOLDILOCKS_P = 0xFFFFFFFF00000001ULL;

// splitmix64 + rand_fe, byte-identical to golden/src/main.rs.
static inline uint64_t splitmix64(uint64_t &state) {
    state += 0x9E3779B97F4A7C15ULL;
    uint64_t z = state;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
static inline uint64_t rand_fe(uint64_t &state) {
    return splitmix64(state) % GOLDILOCKS_P;  // both operands < 2^64
}

// Read the Merkle root (top node = last HASH_SIZE elements of the tree buffer).
template <uint32_t W>
static void commit(uint32_t arity, uint64_t n_bits, uint64_t n_bits_ext,
                   uint64_t n_cols, const std::vector<uint64_t> &trace,
                   uint64_t root_out[HASH_SIZE]) {
    const uint64_t N = 1ULL << n_bits;
    const uint64_t N_ext = 1ULL << n_bits_ext;
    uint32_t gpu_id = 0;
    CHECKCUDAERR(cudaGetDevice((int *)&gpu_id));

    NTTGoldilocksGPU gpu_ntt(n_bits_ext, 1, &gpu_id);
    Poseidon2GoldilocksGPU<W>::initConstants(&gpu_id, 1);

    cudaStream_t stream;
    CHECKCUDAERR(cudaStreamCreate(&stream));
    TimerGPU timer(stream);

    gl64_t *d_flat, *d_src, *d_dst;
    CHECKCUDAERR(cudaMalloc((void **)&d_flat, N * n_cols * sizeof(gl64_t)));
    CHECKCUDAERR(cudaMalloc((void **)&d_src, N * n_cols * sizeof(gl64_t)));
    CHECKCUDAERR(cudaMalloc((void **)&d_dst, N_ext * n_cols * sizeof(gl64_t)));
    CHECKCUDAERR(cudaMemcpyAsync(d_flat, trace.data(), N * n_cols * sizeof(uint64_t),
                                 cudaMemcpyHostToDevice, stream));

    // Production stage-1 commit path: row-major -> tiled -> coset LDE.
    fromRowMajorToTiled(N, n_cols, d_flat, d_src, stream);
    gpu_ntt.LDE(d_dst, 0, d_src, 0, n_bits, n_bits_ext, n_cols, timer, stream);
    CHECKCUDAERR(cudaStreamSynchronize(stream));

    uint64_t tree_size = getTreeNumElements(N_ext, arity);
    Goldilocks::Element *d_tree;
    CHECKCUDAERR(cudaMalloc((void **)&d_tree, tree_size * sizeof(Goldilocks::Element)));
    Poseidon2GoldilocksGPU<W>::merkletree(arity, (uint64_t *)d_tree, (uint64_t *)d_dst,
                                          n_cols, N_ext, Layout::Tiles, stream);
    CHECKCUDAERR(cudaStreamSynchronize(stream));

    CHECKCUDAERR(cudaMemcpy(root_out, (uint64_t *)d_tree + (tree_size - HASH_SIZE),
                            HASH_SIZE * sizeof(uint64_t), cudaMemcpyDeviceToHost));

    CHECKCUDAERR(cudaFree(d_flat));
    CHECKCUDAERR(cudaFree(d_src));
    CHECKCUDAERR(cudaFree(d_dst));
    CHECKCUDAERR(cudaFree(d_tree));
    CHECKCUDAERR(cudaStreamDestroy(stream));
    NTTGoldilocksGPU::freeConstants();
}

int main(int argc, char **argv) {
    uint64_t n_bits = 3, blowup_bits = 2, n_cols = 5, arity = 4, seed = 0xF3;
    std::string dump_dir;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto val = [&](const char *k) { return std::stoull(a.substr(strlen(k))); };
        if (a.rfind("--n_bits=", 0) == 0) n_bits = val("--n_bits=");
        else if (a.rfind("--blowup_bits=", 0) == 0) blowup_bits = val("--blowup_bits=");
        else if (a.rfind("--n_cols=", 0) == 0) n_cols = val("--n_cols=");
        else if (a.rfind("--arity=", 0) == 0) arity = val("--arity=");
        else if (a.rfind("--seed=", 0) == 0) seed = std::stoull(a.substr(7), nullptr, 0);
        else if (a.rfind("--dump=", 0) == 0) dump_dir = a.substr(7);
    }
    const uint64_t n_bits_ext = n_bits + blowup_bits;
    const uint64_t N = 1ULL << n_bits;

    // Deterministic trace, row-major, same stream as the Rust golden generator.
    std::vector<uint64_t> trace(N * n_cols);
    uint64_t state = seed;
    for (auto &x : trace) x = rand_fe(state);

    uint64_t root[HASH_SIZE];
    switch (arity) {
        case 2: commit<8>(arity, n_bits, n_bits_ext, n_cols, trace, root); break;
        case 3: commit<12>(arity, n_bits, n_bits_ext, n_cols, trace, root); break;
        case 4: commit<16>(arity, n_bits, n_bits_ext, n_cols, trace, root); break;
        default: fprintf(stderr, "unsupported arity %lu (want 2/3/4)\n", arity); return 2;
    }

    printf("root = [%lu, %lu, %lu, %lu]\n", root[0], root[1], root[2], root[3]);

    if (!dump_dir.empty()) {
        std::string bin = dump_dir + "/trace.bin";
        FILE *f = fopen(bin.c_str(), "wb");
        if (!f) { perror("trace.bin"); return 3; }
        fwrite(trace.data(), sizeof(uint64_t), trace.size(), f);
        fclose(f);
        std::string js = dump_dir + "/commit.json";
        f = fopen(js.c_str(), "w");
        if (!f) { perror("commit.json"); return 3; }
        fprintf(f,
                "{\n  \"n_bits\": %lu,\n  \"blowup_bits\": %lu,\n  \"n_cols\": %lu,\n"
                "  \"arity\": %lu,\n  \"seed\": %lu,\n"
                "  \"root\": [\"%lu\", \"%lu\", \"%lu\", \"%lu\"]\n}\n",
                n_bits, blowup_bits, n_cols, arity, seed,
                root[0], root[1], root[2], root[3]);
        fclose(f);
        printf("wrote %s and %s\n", bin.c_str(), js.c_str());
    }
    return 0;
}
