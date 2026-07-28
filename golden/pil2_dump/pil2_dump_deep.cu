// Standalone CUDA microbench: native pil2-stark GPU DEEP FRI-polynomial stage
// (calculateFRIPolynomial -> computeFRIExpression kernel) over a synthetic
// Goldilocks3 (FIELD_EXTENSION=3) committed extended trace. Mirrors the
// gsum_bench / evmap_bench / fri_bench proxies.
//
// computeFRIExpression is lifted VERBATIM from pil2-stark
// src/starkpil/starks_gpu.cu:1271-1341. The host launch config is lifted from
// calculateFRIExpression (starks_gpu.cu:1343-1369). The grouped-by-opening
// evalsInfoFRI / evalsInfoFRISizes / d_evalInfoPerOpening tables (built at
// goldilocks_tooling.cuh:154-208 from evMap) are synthesized directly instead
// of parsing a real starkinfo. TimerGPU is replaced by cudaEvent timing.
//
// Main air model (N = 2^22, N_ext = 2^23, extendBits = 1), 1 opening point:
//   committed columns cm1 = 38 (dim-1), cm2 = 24 (dim-1), cm3 = 6 (dim-3)
//   => M = 68 columns opened at 1 opening point z.
// Committed columns live in the EXTENDED buffer (2^23 rows) laid out with the
// production tile layout (getBufferOffset). The DEEP-ALI batched quotient
//   f(x) = sum_m vf^m * (p_m(x) - p_m(xi)) / (x - xi)
// is Horner-accumulated per row; one Fp3 inverse per opening point per row.

#include <cstdio>
#include <cstdint>
#include <vector>
#include <algorithm>
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

// ============ kernel lifted VERBATIM from starks_gpu.cu:1271-1341 ============
__global__  void computeFRIExpression(uint64_t domainSize, uint64_t nOpeningPoints, gl64_t *d_fri, uint64_t* d_countsPerOpeningPos, EvalInfo **d_evalInfoPerOpening, gl64_t *d_evals, gl64_t *vf1, gl64_t *vf2, gl64_t *d_cmPols, gl64_t *d_xDivXSub, gl64_t *d_x, gl64_t *d_fixedPols, gl64_t *d_customComits, bool debug)
{
    int chunk_idx = blockIdx.x;
    uint64_t nchunks = domainSize / blockDim.x;

    extern __shared__ Goldilocks::Element shared[];

    while (chunk_idx < nchunks) {
        gl64_t *fri_pol = (gl64_t *)shared;
        gl64_t *accum = fri_pol + blockDim.x * FIELD_EXTENSION;
        gl64_t *res = accum + blockDim.x * FIELD_EXTENSION;

        uint64_t i = chunk_idx * blockDim.x;
        uint64_t r = i + threadIdx.x;
        for(uint64_t o = 0; o < nOpeningPoints; ++o) {
            for(uint64_t j = 0; j < d_countsPerOpeningPos[o]; ++j) {
                EvalInfo evalInfo = d_evalInfoPerOpening[o][j];
                gl64_t* eval = d_evals + evalInfo.evalPos * FIELD_EXTENSION;
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

                gl64_t *out = (j == 0) ? accum : res;
                if(evalInfo.dim == 1) {
                    out[threadIdx.x] = pol[evalInfo.offset + getBufferOffset(r, evalInfo.stagePos, domainSize, evalInfo.stageCols)];
                    Goldilocks3GPU::sub_13_gpu_b_const(out, out, eval);
                } else {
                    out[threadIdx.x] = pol[evalInfo.offset + getBufferOffset(r, evalInfo.stagePos, domainSize, evalInfo.stageCols)];
                    out[threadIdx.x + blockDim.x] = pol[evalInfo.offset + getBufferOffset(r, evalInfo.stagePos + 1, domainSize, evalInfo.stageCols)];
                    out[threadIdx.x + 2*blockDim.x] = pol[evalInfo.offset + getBufferOffset(r, evalInfo.stagePos + 2, domainSize, evalInfo.stageCols)];
                    Goldilocks3GPU::sub_gpu_b_const(out, out, eval);
                }
                if(j != 0) {
                    Goldilocks3GPU::mul_gpu_b_const(accum, accum, vf2);
                    Goldilocks3GPU::add_gpu_no_const(accum, accum, (gl64_t *)out);
                }
            }

            Goldilocks3GPU::sub_13_gpu_b_const(res, d_x + i, &d_xDivXSub[o * FIELD_EXTENSION]);
            Goldilocks3GPU::Element aux;
            aux[0] = res[threadIdx.x];
            aux[1] = res[blockDim.x + threadIdx.x];
            aux[2] = res[2 * blockDim.x + threadIdx.x];
            Goldilocks3GPU::inv(aux, aux);
            res[threadIdx.x] = aux[0];
            res[blockDim.x + threadIdx.x] = aux[1];
            res[2 * blockDim.x + threadIdx.x] = aux[2];

            gl64_t *out = o == 0 ? fri_pol : accum;
            Goldilocks3GPU::mul_gpu_no_const(out, accum, res);
            if(o != 0) {
                Goldilocks3GPU::mul_gpu_b_const(fri_pol, fri_pol, vf1);
                Goldilocks3GPU::add_gpu_no_const(fri_pol, fri_pol, accum);
            }
        }
        d_fri[r * FIELD_EXTENSION] = fri_pol[threadIdx.x];
        d_fri[r * FIELD_EXTENSION + 1] = fri_pol[threadIdx.x + blockDim.x];
        d_fri[r * FIELD_EXTENSION + 2] = fri_pol[threadIdx.x + 2*blockDim.x];
        chunk_idx += gridDim.x;
    }
}

// ================= host driver =================
static const uint64_t GLP = 0xFFFFFFFF00000001ULL;

static void run_case(uint64_t nOpeningPoints, int reps, uint64_t nBitsExt,
                     int cm1, int cm2, int cm3, const char* dump_dir)
{
    const uint64_t domainSize = 1ULL << nBitsExt;   // 2^23 extended domain

    // Committed-column model for the Main air (per opening point):
    //   cm1: 38 dim-1 columns
    //   cm2: 24 dim-1 columns
    //   cm3:  6 dim-3 columns (18 gl64 lanes)
    const uint64_t cols1 = cm1;                       // gl64 columns in stage cm1
    const uint64_t cols2 = cm2;                       // gl64 columns in stage cm2
    const uint64_t cols3 = (uint64_t)cm3 * FIELD_EXTENSION; // gl64 columns in stage cm3
    const uint64_t totalCols = cols1 + cols2 + cols3; // 80 gl64 columns

    const uint64_t colsPerOpening = cm1 + cm2 + cm3;  // 68 columns opened
    const uint64_t nEvals = colsPerOpening * nOpeningPoints;

    // Stage regions inside the single committed (extended) buffer.
    const uint64_t off1 = 0;
    const uint64_t off2 = off1 + domainSize * cols1;
    const uint64_t off3 = off2 + domainSize * cols2;
    const size_t cmElems = (size_t)domainSize * totalCols;

    // ---- synthesize the grouped-by-opening EvalInfo tables ----
    // evalsInfoByOpening[o] = EvalInfo[countsPerOpeningPos[o]]
    std::vector<std::vector<EvalInfo>> hInfo(nOpeningPoints);
    std::vector<uint64_t> hCounts(nOpeningPoints, 0);
    uint64_t e = 0;
    for (uint64_t op = 0; op < nOpeningPoints; op++) {
        std::vector<EvalInfo>& v = hInfo[op];
        for (int j = 0; j < cm1; j++) { v.push_back({0, off1, (uint64_t)j, cols1, 1, op, e}); e++; }
        for (int j = 0; j < cm2; j++) { v.push_back({0, off2, (uint64_t)j, cols2, 1, op, e}); e++; }
        for (int j = 0; j < cm3; j++) { v.push_back({0, off3, (uint64_t)(j * FIELD_EXTENSION), cols3, 3, op, e}); e++; }
        hCounts[op] = v.size();
    }

    // ---- device: per-opening EvalInfo arrays + array-of-pointers ----
    std::vector<EvalInfo*> dInfoPtrs(nOpeningPoints);
    for (uint64_t op = 0; op < nOpeningPoints; op++) {
        CHECKCUDAERR(cudaMalloc(&dInfoPtrs[op], hCounts[op] * sizeof(EvalInfo)));
        CHECKCUDAERR(cudaMemcpy(dInfoPtrs[op], hInfo[op].data(), hCounts[op] * sizeof(EvalInfo), cudaMemcpyHostToDevice));
    }
    EvalInfo **d_evalInfoPerOpening; uint64_t *d_counts;
    CHECKCUDAERR(cudaMalloc(&d_evalInfoPerOpening, nOpeningPoints * sizeof(EvalInfo*)));
    CHECKCUDAERR(cudaMemcpy(d_evalInfoPerOpening, dInfoPtrs.data(), nOpeningPoints * sizeof(EvalInfo*), cudaMemcpyHostToDevice));
    CHECKCUDAERR(cudaMalloc(&d_counts, nOpeningPoints * sizeof(uint64_t)));
    CHECKCUDAERR(cudaMemcpy(d_counts, hCounts.data(), nOpeningPoints * sizeof(uint64_t), cudaMemcpyHostToDevice));

    // ---- device allocations ----
    gl64_t *d_cmPols, *d_fri, *d_evals, *d_x, *d_xDivXSub, *d_vf1, *d_vf2, *d_fixed, *d_custom;
    size_t friElems   = (size_t)domainSize * FIELD_EXTENSION;
    size_t evalsElems = (size_t)nEvals * FIELD_EXTENSION;
    size_t xElems     = (size_t)domainSize;                       // d_x is dim-1
    size_t xdxsElems  = (size_t)nOpeningPoints * FIELD_EXTENSION;

    CHECKCUDAERR(cudaMalloc(&d_cmPols,   cmElems   * sizeof(gl64_t)));
    CHECKCUDAERR(cudaMalloc(&d_fri,      friElems  * sizeof(gl64_t)));
    CHECKCUDAERR(cudaMalloc(&d_evals,    evalsElems* sizeof(gl64_t)));
    CHECKCUDAERR(cudaMalloc(&d_x,        xElems    * sizeof(gl64_t)));
    CHECKCUDAERR(cudaMalloc(&d_xDivXSub, xdxsElems * sizeof(gl64_t)));
    CHECKCUDAERR(cudaMalloc(&d_vf1,      FIELD_EXTENSION * sizeof(gl64_t)));
    CHECKCUDAERR(cudaMalloc(&d_vf2,      FIELD_EXTENSION * sizeof(gl64_t)));
    CHECKCUDAERR(cudaMalloc(&d_fixed,    sizeof(gl64_t)));   // unused (no fixed cols)
    CHECKCUDAERR(cudaMalloc(&d_custom,   sizeof(gl64_t)));   // unused (no custom cols)

    // ---- fill buffers with random field elements ----
    uint64_t seed = 0xFA1ULL ^ (nOpeningPoints * 0x9E3779B97F4A7C15ULL);
    auto nextu = [&]() { seed = seed * 6364136223846793005ULL + 1442695040888963407ULL; return (seed >> 11) % GLP; };
    {
        // stream fill the committed buffer in chunks (5.4 GB)
        const size_t CH = (size_t)1 << 24;   // 16M elements per staging chunk
        std::vector<uint64_t> buf(CH);
        size_t done = 0;
        while (done < cmElems) {
            size_t n = (cmElems - done < CH) ? (cmElems - done) : CH;
            for (size_t i = 0; i < n; i++) buf[i] = nextu();
            CHECKCUDAERR(cudaMemcpy((uint64_t*)d_cmPols + done, buf.data(), n * sizeof(uint64_t), cudaMemcpyHostToDevice));
            done += n;
        }
        std::vector<uint64_t> xh(xElems);
        for (size_t i = 0; i < xElems; i++) xh[i] = nextu();
        CHECKCUDAERR(cudaMemcpy(d_x, xh.data(), xElems * sizeof(uint64_t), cudaMemcpyHostToDevice));

        std::vector<uint64_t> ev(evalsElems);
        for (size_t i = 0; i < evalsElems; i++) ev[i] = nextu();
        CHECKCUDAERR(cudaMemcpy(d_evals, ev.data(), evalsElems * sizeof(uint64_t), cudaMemcpyHostToDevice));

        std::vector<uint64_t> xd(xdxsElems);
        for (size_t i = 0; i < xdxsElems; i++) xd[i] = nextu();
        CHECKCUDAERR(cudaMemcpy(d_xDivXSub, xd.data(), xdxsElems * sizeof(uint64_t), cudaMemcpyHostToDevice));

        uint64_t vf1h[3] = {nextu(), nextu(), nextu()};
        uint64_t vf2h[3] = {nextu(), nextu(), nextu()};
        CHECKCUDAERR(cudaMemcpy(d_vf1, vf1h, 3 * sizeof(uint64_t), cudaMemcpyHostToDevice));
        CHECKCUDAERR(cudaMemcpy(d_vf2, vf2h, 3 * sizeof(uint64_t), cudaMemcpyHostToDevice));
    }

    // ---- launch config lifted from calculateFRIExpression ----
    uint32_t nthreads_ = 256;   // starkInfo.nrowsPack (GPU path, stark_info.cpp:677)
    uint32_t maxNBlocks = 512;  // starkInfo.maxNBlocks (GPU path, stark_info.cpp:678)
    uint32_t nblocks_ = std::min(maxNBlocks, (uint32_t)((domainSize + nthreads_ - 1) / nthreads_));
    size_t sharedMem = (size_t)nthreads_ * 3 * FIELD_EXTENSION * sizeof(Goldilocks::Element);
    dim3 nThreads(nthreads_);
    dim3 nBlocks(nblocks_);

    cudaStream_t s; CHECKCUDAERR(cudaStreamCreate(&s));
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);

    float best = 1e30f;
    for (int rep = -1; rep < reps; rep++) {          // rep -1 = warmup
        cudaEventRecord(a, s);
        computeFRIExpression<<<nBlocks, nThreads, sharedMem, s>>>(
            domainSize, nOpeningPoints, d_fri, d_counts, d_evalInfoPerOpening,
            d_evals, d_vf1, d_vf2, d_cmPols, d_xDivXSub, d_x, d_fixed, d_custom, false);
        cudaEventRecord(b, s);
        cudaEventSynchronize(b);
        CHECKCUDAERR(cudaGetLastError());
        float ms; cudaEventElapsedTime(&ms, a, b);
        if (rep >= 0 && ms < best) best = ms;
        if (rep == -1 && dump_dir) {
            std::vector<uint64_t> fh(friElems);
            CHECKCUDAERR(cudaMemcpy(fh.data(), d_fri, friElems * sizeof(uint64_t), cudaMemcpyDeviceToHost));
            std::string d(dump_dir);
            FILE* f = fopen((d + "/fri.bin").c_str(), "wb");
            fwrite(fh.data(), sizeof(uint64_t), friElems, f); fclose(f);
            f = fopen((d + "/deep.json").c_str(), "w");
            fprintf(f, "{\n  \"n_bits_ext\": %lu,\n  \"cm1\": %d,\n  \"cm2\": %d,\n  \"cm3\": %d,\n"
                       "  \"n_opening_points\": %lu,\n  \"lcg_seed\": \"0x%lx\"\n}\n",
                    nBitsExt, cm1, cm2, cm3, nOpeningPoints,
                    0xFA1ULL ^ (nOpeningPoints * 0x9E3779B97F4A7C15ULL));
            fclose(f);
        }
    }

    // bandwidth accounting (bytes actually streamed by the kernel, per full pass)
    double colBytes = ((double)(cm1 + cm2) * 1.0 + (double)cm3 * 3.0) * domainSize * 8.0 * nOpeningPoints; // committed reads
    double xBytes   = (double)domainSize * 8.0;                     // d_x dim-1 read
    double friBytes = (double)domainSize * 3.0 * 8.0;               // d_fri write
    double gb = (colBytes + xBytes + friBytes) / 1e9;

    printf("--- %llu opening point(s): M=%llu cols/opening, nEvals=%llu ---\n",
           (unsigned long long)nOpeningPoints, (unsigned long long)colsPerOpening, (unsigned long long)nEvals);
    printf("  committed buffer = %.2f GB (2^23 x %llu gl64)\n",
           cmElems * sizeof(gl64_t) / 1e9, (unsigned long long)totalCols);
    printf("  friexp_total_ms  = %.4f   (best of %d reps)\n", best, reps);
    printf("  traffic model: committed=%.2f GB + x=%.3f GB + fri_out=%.3f GB = %.2f GB  => %.1f GB/s effective\n",
           colBytes/1e9, xBytes/1e9, friBytes/1e9, gb, gb / (best/1e3));

    for (uint64_t op = 0; op < nOpeningPoints; op++) cudaFree(dInfoPtrs[op]);
    cudaFree(d_evalInfoPerOpening); cudaFree(d_counts);
    cudaFree(d_cmPols); cudaFree(d_fri); cudaFree(d_evals); cudaFree(d_x);
    cudaFree(d_xDivXSub); cudaFree(d_vf1); cudaFree(d_vf2); cudaFree(d_fixed); cudaFree(d_custom);
    cudaStreamDestroy(s); cudaEventDestroy(a); cudaEventDestroy(b);
}

int main(int argc, char **argv)
{
    int reps = argc > 1 ? atoi(argv[1]) : 7;
    uint64_t nb = argc > 2 ? atoi(argv[2]) : 23;
    int cm1 = argc > 3 ? atoi(argv[3]) : 38;
    int cm2 = argc > 4 ? atoi(argv[4]) : 24;
    int cm3 = argc > 5 ? atoi(argv[5]) : 6;
    const char* dump_dir = argc > 6 ? argv[6] : nullptr;
    printf("# native pil2-stark GPU DEEP FRI-polynomial (computeFRIExpression)\n");
    printf("# N_ext=2^%lu; cm1=%d d1, cm2=%d d1, cm3=%d d3 => M=%d cols opened\n\n",
           nb, cm1, cm2, cm3, cm1 + cm2 + cm3);

    run_case(1, reps, nb, cm1, cm2, cm3, dump_dir);
    return 0;
}
