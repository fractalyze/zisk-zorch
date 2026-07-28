// Standalone CUDA microbench: native pil2-stark GPU FRI stage (fold + per-layer
// Merkle) over the production ZisK schedule, on a synthetic Goldilocks3
// (FIELD_EXTENSION=3) codeword. Mirrors gsum_bench / merkle_bench proxies.
//
// The fold / intt_tinny / transposeFRI device kernels are lifted VERBATIM from
// pil2-stark src/starkpil/starks_gpu.cu (lines 643-817). The host fold setup is
// lifted from fold_inplace (starks_gpu.cu:773-800). The per-layer Merkle uses
// the SAME Poseidon2 builder buildMerkleTreeGPU dispatches to
// (Poseidon2GoldilocksGPU<16>::merkletree, arity 4), called directly to avoid
// dragging in the whole starks_api translation unit. TimerGPU is replaced by
// cudaEvent timing.
//
// Schedule (nBits): [23,20,17,14,11,8,5] -- fold by 3 bits/layer (factor-8),
// 6 fold rounds, final 5-bit poly sent in clear.

#include <cstdio>
#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

#include "goldilocks_base_field.hpp"
#include "goldilocks_cubic_extension.cuh"   // gl64_t, Goldilocks3GPU, FIELD_EXTENSION
#include "poseidon2_goldilocks.hpp"
#include "poseidon2_goldilocks.cuh"          // Poseidon2GoldilocksGPU<W>, Layout
#include "goldilocks_tooling.hpp"            // getTreeNumElements, HASH_SIZE

// CHECKCUDAERR is provided by utils/cuda_utils.cuh (checkCudaError wrapper).

// ---- constants lifted from starks_gpu.cuh / starks_gpu.cu ----
__device__ __constant__ uint64_t domain_size_inverse_[33] = {
    0x0000000000000001, 0x7fffffff80000001, 0xbfffffff40000001, 0xdfffffff20000001,
    0xefffffff10000001, 0xf7ffffff08000001, 0xfbffffff04000001, 0xfdffffff02000001,
    0xfeffffff01000001, 0xff7fffff00800001, 0xffbfffff00400001, 0xffdfffff00200001,
    0xffefffff00100001, 0xfff7ffff00080001, 0xfffbffff00040001, 0xfffdffff00020001,
    0xfffeffff00010001, 0xffff7fff00008001, 0xffffbfff00004001, 0xffffdfff00002001,
    0xffffefff00001001, 0xfffff7ff00000801, 0xfffffbff00000401, 0xfffffdff00000201,
    0xfffffeff00000101, 0xffffff7f00000081, 0xffffffbf00000041, 0xffffffdf00000021,
    0xffffffef00000011, 0xfffffff700000009, 0xfffffffb00000005, 0xfffffffd00000003,
    0xfffffffe00000002,
};

Goldilocks::Element omegas_inv_[33] = {
    Goldilocks::Element{0x1},
    Goldilocks::Element{0xffffffff00000000}, Goldilocks::Element{0xfffeffff00000001},
    Goldilocks::Element{0xfffffeff00000101}, Goldilocks::Element{0xffefffff00100001},
    Goldilocks::Element{0xfbffffff04000001}, Goldilocks::Element{0xdfffffff20000001},
    Goldilocks::Element{0x3fffbfffc0},       Goldilocks::Element{0x7f4949dce07bf05d},
    Goldilocks::Element{0x4bd6bb172e15d48c}, Goldilocks::Element{0x38bc97652b54c741},
    Goldilocks::Element{0x553a9b711648c890}, Goldilocks::Element{0x055da9bb68958caa},
    Goldilocks::Element{0xa0a62f8f0bb8e2b6}, Goldilocks::Element{0x276fd7ae450aee4b},
    Goldilocks::Element{0x7b687b64f5de658f}, Goldilocks::Element{0x7de5776cbda187e9},
    Goldilocks::Element{0xd2199b156a6f3b06}, Goldilocks::Element{0xd01c8acd8ea0e8c0},
    Goldilocks::Element{0x4f38b2439950a4cf}, Goldilocks::Element{0x5987c395dd5dfdcf},
    Goldilocks::Element{0x46cf3d56125452b1}, Goldilocks::Element{0x909c4b1a44a69ccb},
    Goldilocks::Element{0xc188678a32a54199}, Goldilocks::Element{0xf3650f9ddfcaffa8},
    Goldilocks::Element{0xe8ef0e3e40a92655}, Goldilocks::Element{0x7c8abec072bb46a6},
    Goldilocks::Element{0xe0bfc17d5c5a7a04}, Goldilocks::Element{0x4c6b8a5a0b79f23a},
    Goldilocks::Element{0x6b4d20533ce584fe}, Goldilocks::Element{0xe5cceae468a70ec2},
    Goldilocks::Element{0x8958579f296dac7a}, Goldilocks::Element{0x16d265893b5b7e85},
};

// ================= kernels lifted verbatim from starks_gpu.cu =================
__device__ void intt_tinny(gl64_t *data, uint32_t N, uint32_t logN, gl64_t *d_twiddles, uint32_t ncols)
{
    uint32_t halfN = N >> 1;
    for (uint32_t i = 0; i < N; i++) {
        uint32_t ibr = __brev(i) >> (32 - logN);
        if (ibr > i) {
            gl64_t tmp;
            for (uint32_t j = 0; j < ncols; j++) {
                tmp = data[i * ncols + j];
                data[i * ncols + j] = data[ibr * ncols + j];
                data[ibr * ncols + j] = tmp;
            }
        }
    }
    for (uint32_t i = 0; i < logN; i++) {
        for (uint32_t j = 0; j < halfN; j++) {
            for (uint32_t col = 0; col < ncols; col++) {
                uint32_t half_group_size = 1 << i;
                uint32_t group = j >> i;
                uint32_t offset = j & (half_group_size - 1);
                uint32_t index1 = (group << i + 1) + offset;
                uint32_t index2 = index1 + half_group_size;
                gl64_t factor = d_twiddles[offset * (N >> i + 1)];
                gl64_t odd_sub = gl64_t((uint64_t)data[index2 * ncols + col]) * factor;
                data[index2 * ncols + col] = gl64_t((uint64_t)data[index1 * ncols + col]) - odd_sub;
                data[index1 * ncols + col] = gl64_t((uint64_t)data[index1 * ncols + col]) + odd_sub;
            }
        }
    }
    gl64_t factor = gl64_t(domain_size_inverse_[logN]);
    for (uint32_t i = 0; i < N * ncols; i++) {
        data[i] = gl64_t((uint64_t)data[i]) * factor;
    }
}

__global__ void fold(uint64_t step, gl64_t *friPol, gl64_t *d_challenge, gl64_t *d_ppar, Goldilocks::Element omega_inv, uint64_t invShiftPow_, uint64_t invW_, uint64_t nBitsExt, uint64_t prevBits, uint64_t currentBits)
{
    extern __shared__ gl64_t s_twiddles[];
    if (threadIdx.x == 0) {
        uint64_t halfRatio = (1 << (prevBits - currentBits)) >> 1;
        s_twiddles[0] = gl64_t(uint64_t(1));
        for (uint32_t i = 1; i < halfRatio; i++) {
            s_twiddles[i] = s_twiddles[i - 1] * gl64_t(omega_inv.fe);
        }
    }
    __syncthreads();

    uint32_t polBits = prevBits;
    uint64_t sizePol = 1 << polBits;
    uint32_t foldedPolBits = currentBits;
    uint64_t sizeFoldedPol = 1 << foldedPolBits;
    uint32_t ratio = sizePol / sizeFoldedPol;

    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < sizeFoldedPol) {
        gl64_t invShift(invShiftPow_);
        gl64_t invW(invW_);
        gl64_t sinv = invShift;
        gl64_t base = invW;
        uint32_t exponent = id;
        while (exponent > 0) {
            if (exponent % 2 == 1) sinv *= base;
            base *= base;
            exponent /= 2;
        }

        gl64_t *ppar = (gl64_t *)d_ppar + id * ratio * FIELD_EXTENSION;
        for (int i = 0; i < ratio; i++) {
            int ind = i * FIELD_EXTENSION;
            for (int k = 0; k < FIELD_EXTENSION; k++) {
                ppar[ind + k] = gl64_t(friPol[(i * sizeFoldedPol + id) * FIELD_EXTENSION + k]);
            }
        }
        intt_tinny(ppar, ratio, prevBits - currentBits, s_twiddles, FIELD_EXTENSION);

        gl64_t r(1);
        for (uint64_t i = 0; i < ratio; i++) {
            Goldilocks3GPU::Element *component = (Goldilocks3GPU::Element *)&ppar[i * FIELD_EXTENSION];
            Goldilocks3GPU::mul(*component, *component, r);
            r *= sinv;
        }
        if (ratio == 0) {
            for (uint32_t i = 0; i < FIELD_EXTENSION; i++)
                friPol[id * FIELD_EXTENSION + i] = gl64_t(uint64_t(0));
        } else {
            for (uint32_t i = 0; i < FIELD_EXTENSION; i++)
                friPol[id * FIELD_EXTENSION + i] = ppar[(ratio - 1) * FIELD_EXTENSION + i];
            for (int i = ratio - 2; i >= 0; i--) {
                Goldilocks3GPU::Element aux;
                Goldilocks3GPU::mul(aux, *((Goldilocks3GPU::Element *)&friPol[id * FIELD_EXTENSION]), *((Goldilocks3GPU::Element *)&d_challenge[0]));
                Goldilocks3GPU::add(*((Goldilocks3GPU::Element *)&friPol[id * FIELD_EXTENSION]), aux, *((Goldilocks3GPU::Element *)&ppar[i * FIELD_EXTENSION]));
            }
        }
    }
}

__global__ void transposeFRI(gl64_t *d_aux, gl64_t *pol, uint64_t degree, uint64_t width)
{
    uint64_t idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t idx_y = blockIdx.y * blockDim.y + threadIdx.y;
    uint64_t height = degree / width;
    if (idx_x < width && idx_y < height) {
        uint64_t fi = idx_y * width + idx_x;
        uint64_t di = idx_x * height + idx_y;
        for (uint64_t k = 0; k < FIELD_EXTENSION; k++)
            d_aux[di * FIELD_EXTENSION + k] = pol[fi * FIELD_EXTENSION + k];
    }
}

// ================= host driver =================
static const uint64_t GLP = 0xFFFFFFFF00000001ULL;


int main(int argc, char **argv)
{
    // args: nBitsExt foldBits finalBits dump_dir   (zisk _fold_steps schedule:
    // range(nBitsExt, finalBits, -foldBits) + finalBits tail)
    uint64_t nBitsExt = argc > 1 ? atoi(argv[1]) : 23;
    int foldBits      = argc > 2 ? atoi(argv[2]) : 3;
    int finalBits     = argc > 3 ? atoi(argv[3]) : 5;
    const char* dump_dir = argc > 4 ? argv[4] : nullptr;
    const uint32_t arity = 4;
    const uint32_t W = 16;

    std::vector<int> steps;
    for (int bb = (int)nBitsExt; bb > finalBits; bb -= foldBits) steps.push_back(bb);
    steps.push_back(finalBits);
    int nsteps = (int)steps.size();
    printf("# pil2 FRI fold dump: schedule [");
    for (int i = 0; i < nsteps; i++) printf("%d%s", steps[i], i + 1 < nsteps ? "," : "");
    printf("]\n");

    uint32_t gpu_id = 0;
    cudaGetDevice((int *)&gpu_id);
    Poseidon2GoldilocksGPU<W>::initConstants(&gpu_id, 1);
    cudaStream_t s; CHECKCUDAERR(cudaStreamCreate(&s));

    // LCG order: codeword (2^nBitsExt x 3), then one cubic challenge PER FOLD
    size_t nExt = (size_t)1 << nBitsExt;
    std::vector<uint64_t> hfri(nExt * FIELD_EXTENSION);
    uint64_t seed = 0xC0FFEEULL ^ (nBitsExt * 0x9E3779B97F4A7C15ULL);
    auto nextu = [&]() { seed = seed * 6364136223846793005ULL + 1442695040888963407ULL; return (seed >> 11) % GLP; };
    for (size_t i = 0; i < hfri.size(); i++) hfri[i] = nextu();
    std::vector<uint64_t> hchal((nsteps - 1) * FIELD_EXTENSION);
    for (size_t i = 0; i < hchal.size(); i++) hchal[i] = nextu();

    gl64_t *d_friPol, *d_ppar, *d_challenge;
    CHECKCUDAERR(cudaMalloc(&d_friPol,  nExt * FIELD_EXTENSION * sizeof(gl64_t)));
    CHECKCUDAERR(cudaMalloc(&d_ppar,    nExt * FIELD_EXTENSION * sizeof(gl64_t)));
    CHECKCUDAERR(cudaMalloc(&d_challenge, FIELD_EXTENSION * sizeof(gl64_t)));
    CHECKCUDAERR(cudaMemcpy(d_friPol, hfri.data(), hfri.size() * sizeof(uint64_t), cudaMemcpyHostToDevice));

    for (int step = 1; step < nsteps; step++) {
        int prevBits = steps[step - 1];
        int currentBits = steps[step];
        CHECKCUDAERR(cudaMemcpy(d_challenge, &hchal[(step - 1) * FIELD_EXTENSION],
                                FIELD_EXTENSION * sizeof(uint64_t), cudaMemcpyHostToDevice));
        uint32_t ratio = 1u << (prevBits - currentBits);
        uint64_t halfRatio = ratio >> 1;
        uint64_t sizeFoldedPol = 1ULL << currentBits;
        Goldilocks::Element omega_inv = omegas_inv_[prevBits - currentBits];
        Goldilocks::Element invShiftPow = Goldilocks::inv(Goldilocks::shift());
        for (uint32_t j = 0; j < (uint32_t)(nBitsExt - prevBits); j++) Goldilocks::square(invShiftPow, invShiftPow);
        Goldilocks::Element invW = Goldilocks::inv(Goldilocks::w(prevBits));
        dim3 nThreads(256);
        dim3 nBlocks((sizeFoldedPol + nThreads.x - 1) / nThreads.x);
        size_t sharedMem = halfRatio * sizeof(gl64_t);
        fold<<<nBlocks, nThreads, sharedMem, s>>>(step, d_friPol, d_challenge, d_ppar,
            omega_inv, invShiftPow.fe, invW.fe, nBitsExt, prevBits, currentBits);
        CHECKCUDAERR(cudaStreamSynchronize(s));
        CHECKCUDAERR(cudaGetLastError());
        if (dump_dir) {
            size_t elems = sizeFoldedPol * FIELD_EXTENSION;
            std::vector<uint64_t> out(elems);
            CHECKCUDAERR(cudaMemcpy(out.data(), d_friPol, elems * sizeof(uint64_t), cudaMemcpyDeviceToHost));
            char path[512]; snprintf(path, sizeof(path), "%s/layer%d.bin", dump_dir, step);
            FILE *f = fopen(path, "wb"); fwrite(out.data(), sizeof(uint64_t), elems, f); fclose(f);
            printf("  layer %d (nBits=%d) dumped\n", step, currentBits);
        }
    }
    if (dump_dir) {
        char path[512]; snprintf(path, sizeof(path), "%s/fri.json", dump_dir);
        FILE *f = fopen(path, "w");
        fprintf(f, "{\n  \"n_bits_ext\": %lu,\n  \"fold_bits\": %d,\n  \"final_bits\": %d,\n"
                   "  \"steps\": [", nBitsExt, foldBits, finalBits);
        for (int i = 0; i < nsteps; i++) fprintf(f, "%d%s", steps[i], i + 1 < nsteps ? ", " : "");
        fprintf(f, "],\n  \"lcg_seed\": \"0x%lx\"\n}\n", 0xC0FFEEULL ^ (nBitsExt * 0x9E3779B97F4A7C15ULL));
        fclose(f);
    }
    printf("done\n");
    return 0;
}
