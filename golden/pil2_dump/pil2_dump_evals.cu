// Standalone CUDA microbench: native pil2-stark GPU out-of-domain opening
// evaluation (evmap) stage over a synthetic Goldilocks3 (FIELD_EXTENSION=3)
// committed trace. Mirrors gsum_bench / merkle_bench / fri_bench proxies.
//
// The two kernels computeEvals_v2 and computeEvalsReduction are lifted VERBATIM
// from pil2-stark src/starkpil/starks_gpu.cu:513-614. The host launch config is
// lifted from evmap_inplace (starks_gpu.cu:616-641). The EvalInfo array is
// synthesized directly (mirroring goldilocks_tooling.cuh:118-152) instead of
// parsing a real starkinfo. TimerGPU is replaced by cudaEvent timing.
//
// Main air model (N = 2^22, N_ext = 2^23, extendBits = 1):
//   committed columns cm1 = 38 (dim-1), cm2 = 24 (dim-1), cm3 = 6 (dim-3)
//   => M = 68 evals at 1 opening point (also tried at 2 opening points => 136).
// Committed columns live in the EXTENDED buffer (2^23 rows) laid out with the
// production tile layout (getBufferOffset), subsampled by (row << 1) to the base
// domain N = 2^22 for the reduction.

#include <cstdio>
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>

#include "goldilocks_base_field.hpp"
#include "goldilocks_cubic_extension.cuh"   // gl64_t, Goldilocks3GPU, FIELD_EXTENSION
#include "poseidon2_goldilocks.hpp"
#include "poseidon2_goldilocks.cuh"          // pulls in goldilocks_trace_layout.cuh -> getBufferOffset
#include "goldilocks_tooling.hpp"

// Local copy of the EvalInfo POD (stark_info.hpp drags in nlohmann/json).
// Layout must match pil2-stark src/starkpil/stark_info.hpp:18.
struct EvalInfo
{
    uint64_t type;      // 0: cm, 1: custom, 2: fixed
    uint64_t offset;
    uint64_t stagePos;
    uint64_t stageCols;
    uint64_t dim;
    uint64_t openingPos;
    uint64_t evalPos;
};

// ================= kernels lifted VERBATIM from starks_gpu.cu:513-614 =========
__global__ void computeEvals_v2(
    uint64_t NExtended,
    uint64_t extendBits,
    uint64_t size_eval,
    uint64_t N,
    uint64_t openingsSize,
    gl64_t *d_evals,
    EvalInfo *d_evalInfo,
    gl64_t *d_cmPols,
    gl64_t *d_fixedPols,
    gl64_t *d_customComits,
    gl64_t *d_LEv,
    gl64_t *d_helper)
{

    extern __shared__ Goldilocks3GPU::Element shared_sum[];
    uint64_t evalIdx = blockIdx.x;
    uint64_t chunkIdx = blockIdx.y;

    if (evalIdx < size_eval)
    {
        EvalInfo evalInfo = d_evalInfo[evalIdx];
        gl64_t *pol;
        if (evalInfo.type == 0)
        {
            pol = d_cmPols;
        }
        else if (evalInfo.type == 1)
        {
            pol = d_customComits;
        }
        else
        {
            pol = d_fixedPols;
        }

        for (int i = 0; i < FIELD_EXTENSION; i++)
        {
            shared_sum[threadIdx.x][i]= gl64_t(uint64_t(0));
        }
        uint64_t tid = chunkIdx * blockDim.x + threadIdx.x;
        while (tid < N)
        {
            uint64_t row = (tid << extendBits);
            Goldilocks3GPU::Element LEv;
            LEv[0] = d_LEv[getBufferOffset(tid, evalInfo.openingPos * FIELD_EXTENSION, N, openingsSize * FIELD_EXTENSION)];
            LEv[1] = d_LEv[getBufferOffset(tid, evalInfo.openingPos * FIELD_EXTENSION + 1, N, openingsSize * FIELD_EXTENSION)];
            LEv[2] = d_LEv[getBufferOffset(tid, evalInfo.openingPos * FIELD_EXTENSION + 2, N, openingsSize * FIELD_EXTENSION)];
            Goldilocks3GPU::Element res;
            if (evalInfo.dim == 1)
            {
                Goldilocks3GPU::mul(res, LEv, pol[evalInfo.offset + getBufferOffset(row, evalInfo.stagePos, NExtended, evalInfo.stageCols)]);
            }
            else
            {
                Goldilocks3GPU::Element val;
                val[0] = pol[evalInfo.offset + getBufferOffset(row, evalInfo.stagePos, NExtended, evalInfo.stageCols)];
                val[1] = pol[evalInfo.offset + getBufferOffset(row, evalInfo.stagePos + 1, NExtended, evalInfo.stageCols)];
                val[2] = pol[evalInfo.offset + getBufferOffset(row, evalInfo.stagePos + 2, NExtended, evalInfo.stageCols)];
                Goldilocks3GPU::mul(res, LEv, val);
            }
            Goldilocks3GPU::add(shared_sum[threadIdx.x], shared_sum[threadIdx.x], res);
            tid += blockDim.x * gridDim.y;
        }
        __syncthreads();
        int s = (blockDim.x + 1) / 2;
        while (s > 0)
        {
            if (threadIdx.x < s)
            {
                Goldilocks3GPU::add(shared_sum[threadIdx.x], shared_sum[threadIdx.x], shared_sum[threadIdx.x + s]);
            }
            __syncthreads();
            if (s == 1)
                break;
            s = (s + 1) / 2;
        }

        __syncthreads();
        if (threadIdx.x == 0) {
            uint64_t partial_pos = evalIdx * gridDim.y + chunkIdx;
            d_helper[partial_pos * FIELD_EXTENSION] = shared_sum[0][0];
            d_helper[partial_pos * FIELD_EXTENSION + 1] = shared_sum[0][1];
            d_helper[partial_pos * FIELD_EXTENSION + 2] = shared_sum[0][2];
        }
    }
}

__global__ void computeEvalsReduction(gl64_t *d_evals, gl64_t *d_helper, EvalInfo *d_evalInfo, uint64_t size_eval, uint64_t n_eval_chunks) {
    uint64_t evalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (evalIdx < size_eval) {
        uint64_t base = evalIdx * n_eval_chunks * FIELD_EXTENSION;
        d_evals[d_evalInfo[evalIdx].evalPos * FIELD_EXTENSION] = d_helper[base + 0];
        d_evals[d_evalInfo[evalIdx].evalPos * FIELD_EXTENSION + 1] = d_helper[base + 1];
        d_evals[d_evalInfo[evalIdx].evalPos * FIELD_EXTENSION + 2] = d_helper[base + 2];
        for (int i = 1; i < n_eval_chunks; ++i) {
            d_evals[d_evalInfo[evalIdx].evalPos * FIELD_EXTENSION] += d_helper[base + i * FIELD_EXTENSION];
            d_evals[d_evalInfo[evalIdx].evalPos * FIELD_EXTENSION + 1] += d_helper[base + i * FIELD_EXTENSION + 1];
            d_evals[d_evalInfo[evalIdx].evalPos * FIELD_EXTENSION + 2] += d_helper[base + i * FIELD_EXTENSION + 2];
        }
    }
}

// ================= host driver =================
static const uint64_t GLP = 0xFFFFFFFF00000001ULL;


// args: nBits extendBits cm1 cm2 cm3 dump_dir
int main(int argc, char **argv)
{
    const uint64_t nBits = argc > 1 ? atoi(argv[1]) : 22;
    const uint64_t extendBits = argc > 2 ? atoi(argv[2]) : 1;
    const int cm1 = argc > 3 ? atoi(argv[3]) : 38;
    const int cm2 = argc > 4 ? atoi(argv[4]) : 24;
    const int cm3 = argc > 5 ? atoi(argv[5]) : 6;
    const char* dump_dir = argc > 6 ? argv[6] : nullptr;
    const uint64_t nOpeningPoints = 1;
    const uint64_t nBitsExt = nBits + extendBits;
    const uint64_t N = 1ULL << nBits, NExtended = 1ULL << nBitsExt;
    printf("# pil2 evmap dump: N=2^%lu ext=2^%lu cm=%d/%d/%d\n", nBits, nBitsExt, cm1, cm2, cm3);

    const uint64_t cols1 = cm1, cols2 = cm2, cols3 = (uint64_t)cm3 * FIELD_EXTENSION;
    const uint64_t totalCols = cols1 + cols2 + cols3;
    const uint64_t nEvals = (uint64_t)(cm1 + cm2 + cm3) * nOpeningPoints;
    const uint64_t n_eval_chunks = 16;
    const uint64_t off1 = 0, off2 = off1 + NExtended * cols1, off3 = off2 + NExtended * cols2;
    const size_t cmElems = (size_t)NExtended * totalCols;

    std::vector<EvalInfo> hInfo(nEvals);
    uint64_t e = 0;
    for (uint64_t op = 0; op < nOpeningPoints; op++) {
        for (int j = 0; j < cm1; j++) { hInfo[e] = { 0, off1, (uint64_t)j, cols1, 1, op, e }; e++; }
        for (int j = 0; j < cm2; j++) { hInfo[e] = { 0, off2, (uint64_t)j, cols2, 1, op, e }; e++; }
        for (int j = 0; j < cm3; j++) { hInfo[e] = { 0, off3, (uint64_t)(j * FIELD_EXTENSION), cols3, 3, op, e }; e++; }
    }

    gl64_t *d_cmPols, *d_LEv, *d_helper, *d_evals, *d_fixed, *d_custom;
    EvalInfo *d_info;
    size_t levElems = (size_t)N * nOpeningPoints * FIELD_EXTENSION;
    size_t helperElems = (size_t)nEvals * n_eval_chunks * FIELD_EXTENSION;
    size_t evalsElems = (size_t)nEvals * FIELD_EXTENSION;
    CHECKCUDAERR(cudaMalloc(&d_cmPols, cmElems * sizeof(gl64_t)));
    CHECKCUDAERR(cudaMalloc(&d_LEv, levElems * sizeof(gl64_t)));
    CHECKCUDAERR(cudaMalloc(&d_helper, helperElems * sizeof(gl64_t)));
    CHECKCUDAERR(cudaMalloc(&d_evals, evalsElems * sizeof(gl64_t)));
    CHECKCUDAERR(cudaMalloc(&d_info, nEvals * sizeof(EvalInfo)));
    CHECKCUDAERR(cudaMalloc(&d_fixed, sizeof(gl64_t)));
    CHECKCUDAERR(cudaMalloc(&d_custom, sizeof(gl64_t)));
    CHECKCUDAERR(cudaMemcpy(d_info, hInfo.data(), nEvals * sizeof(EvalInfo), cudaMemcpyHostToDevice));

    uint64_t seed = 0xEA1ULL ^ (nBits * 0x9E3779B97F4A7C15ULL);
    auto nextu = [&]() { seed = seed * 6364136223846793005ULL + 1442695040888963407ULL; return (seed >> 11) % GLP; };
    {
        const size_t CH = (size_t)1 << 24;
        std::vector<uint64_t> buf(CH);
        size_t done = 0;
        while (done < cmElems) {
            size_t nn = (cmElems - done < CH) ? (cmElems - done) : CH;
            for (size_t i = 0; i < nn; i++) buf[i] = nextu();
            CHECKCUDAERR(cudaMemcpy((uint64_t*)d_cmPols + done, buf.data(), nn * sizeof(uint64_t), cudaMemcpyHostToDevice));
            done += nn;
        }
        std::vector<uint64_t> lev(levElems);
        for (size_t i = 0; i < levElems; i++) lev[i] = nextu();
        CHECKCUDAERR(cudaMemcpy(d_LEv, lev.data(), levElems * sizeof(uint64_t), cudaMemcpyHostToDevice));
    }

    cudaStream_t s; CHECKCUDAERR(cudaStreamCreate(&s));
    dim3 nThreads(256);
    dim3 nBlocks(nEvals, n_eval_chunks);
    size_t shmem = nThreads.x * sizeof(Goldilocks3GPU::Element);
    dim3 nBlocks2((nEvals + nThreads.x - 1) / nThreads.x);
    computeEvals_v2<<<nBlocks, nThreads, shmem, s>>>(NExtended, extendBits, nEvals, N, nOpeningPoints,
                                                     d_evals, d_info, d_cmPols, d_fixed, d_custom, d_LEv, d_helper);
    computeEvalsReduction<<<nBlocks2, nThreads, 0, s>>>(d_evals, d_helper, d_info, nEvals, n_eval_chunks);
    CHECKCUDAERR(cudaStreamSynchronize(s));
    CHECKCUDAERR(cudaGetLastError());

    if (dump_dir) {
        std::vector<uint64_t> out(evalsElems);
        CHECKCUDAERR(cudaMemcpy(out.data(), d_evals, evalsElems * sizeof(uint64_t), cudaMemcpyDeviceToHost));
        char path[512]; snprintf(path, sizeof(path), "%s/evals.bin", dump_dir);
        FILE *f = fopen(path, "wb"); fwrite(out.data(), sizeof(uint64_t), evalsElems, f); fclose(f);
        snprintf(path, sizeof(path), "%s/evals.json", dump_dir);
        f = fopen(path, "w");
        fprintf(f, "{\n  \"n_bits\": %lu,\n  \"extend_bits\": %lu,\n  \"cm1\": %d,\n  \"cm2\": %d,\n"
                   "  \"cm3\": %d,\n  \"n_opening_points\": %lu,\n  \"lcg_seed\": \"0x%lx\"\n}\n",
                nBits, extendBits, cm1, cm2, cm3, nOpeningPoints,
                0xEA1ULL ^ (nBits * 0x9E3779B97F4A7C15ULL));
        fclose(f);
    }
    printf("done\n");
    return 0;
}
