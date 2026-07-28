// Native pil2 GPU baseline for the LogUp grand-sum stage.
// Pairs pil2-stark's ACTUAL scan (accOperationGPU / prescan, lifted verbatim
// from src/starkpil/hints.cu — the work-efficient Blelloch scan the CUDA prover
// runs in calculateWitnessSTD_gpu) with a per-element cubic inverse+fold
// (num * den^{-1}, summed over I) built from Goldilocks3GPU — the same
// multiplyHintFieldsGPU math. Times both cost centers on the same GPU as the
// zisk-zorch gsum.py run, so the comparison is apples-to-apples.

#include <cstdio>
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>
#include "goldilocks_cubic_extension.cuh"

#define LOG_NUM_BANKS 5
#define CONFLICT_FREE_OFFSET(n) 0
#define CHECKCUDAERR(call) do { cudaError_t e=(call); if(e!=cudaSuccess){ \
  printf("CUDA err %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); } } while(0)

#include "scan_lifted.cuh"   // scan_sum_1/3, scan_prod_1/3, prescan, prescan_correction, accOperationGPU

// local[r] = sum_i num[r,i] * inverse(den[r,i])   — per-row, I inversions each.
__global__ void inv_fold(gl64_t* local, const gl64_t* num, const gl64_t* den, uint32_t N, uint32_t I) {
    uint32_t r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= N) return;
    Goldilocks3GPU::Element acc; acc[0]=gl64_t(uint64_t(0)); acc[1]=acc[0]; acc[2]=acc[0];
    for (uint32_t i = 0; i < I; i++) {
        uint64_t base = (uint64_t)(r*I + i) * 3;
        Goldilocks3GPU::Element d, dinv, prod, n;
        d[0]=den[base]; d[1]=den[base+1]; d[2]=den[base+2];
        n[0]=num[base]; n[1]=num[base+1]; n[2]=num[base+2];
        Goldilocks3GPU::inv(dinv, d);
        Goldilocks3GPU::mul(prod, n, dinv);
        Goldilocks3GPU::add(acc, acc, prod);
    }
    local[r*3]=acc[0]; local[r*3+1]=acc[1]; local[r*3+2]=acc[2];
}


// args: nBits I dump_dir
int main(int argc, char** argv){
    const uint64_t P = 0xFFFFFFFF00000001ULL;
    int nBits = argc>1 ? atoi(argv[1]) : 22;
    uint32_t I = argc>2 ? atoi(argv[2]) : 8;
    const char* dump_dir = argc>3 ? argv[3] : nullptr;
    uint32_t N = 1u << nBits;
    printf("# pil2 gsum dump: N=2^%d I=%u\n", nBits, I);
    cudaStream_t s; cudaStreamCreate(&s);
    uint64_t seed = 0xC0FFEEULL ^ ((uint64_t)nBits * 0x9E3779B97F4A7C15ULL);
    // NOTE the +1: the bench draws in [1, p] to keep denominators nonzero.
    auto nextu = [&](){ seed=seed*6364136223846793005ULL+1442695040888963407ULL; return (seed>>11)%P + 1; };
    size_t nElem=(size_t)N*I;
    std::vector<uint64_t> hnum(nElem*3), hden(nElem*3);
    for(size_t i=0;i<nElem*3;i++){ hnum[i]=nextu(); hden[i]=nextu(); }
    gl64_t *d_num,*d_den,*d_local,*d_helper;
    cudaMalloc(&d_num,nElem*3*sizeof(gl64_t)); cudaMalloc(&d_den,nElem*3*sizeof(gl64_t));
    cudaMalloc(&d_local,(size_t)N*3*sizeof(gl64_t));
    cudaMalloc(&d_helper,(size_t)(N/128+4096)*3*sizeof(gl64_t));
    cudaMemcpy(d_num,hnum.data(),nElem*3*sizeof(uint64_t),cudaMemcpyHostToDevice);
    cudaMemcpy(d_den,hden.data(),nElem*3*sizeof(uint64_t),cudaMemcpyHostToDevice);
    uint32_t t=256,b=(N+t-1)/t;
    inv_fold<<<b,t,0,s>>>(d_local,d_num,d_den,N,I);
    cudaStreamSynchronize(s);
    std::vector<uint64_t> out((size_t)N*3);
    if (dump_dir) {
        cudaMemcpy(out.data(),d_local,(size_t)N*3*sizeof(uint64_t),cudaMemcpyDeviceToHost);
        char path[512]; snprintf(path,sizeof(path),"%s/local.bin",dump_dir);
        FILE*f=fopen(path,"wb"); fwrite(out.data(),8,out.size(),f); fclose(f);
    }
    accOperationGPU(d_local,N,true,3,d_helper,s);
    cudaStreamSynchronize(s);
    CHECKCUDAERR(cudaGetLastError());
    if (dump_dir) {
        cudaMemcpy(out.data(),d_local,(size_t)N*3*sizeof(uint64_t),cudaMemcpyDeviceToHost);
        char path[512]; snprintf(path,sizeof(path),"%s/gsum.bin",dump_dir);
        FILE*f=fopen(path,"wb"); fwrite(out.data(),8,out.size(),f); fclose(f);
        snprintf(path,sizeof(path),"%s/gsum.json",dump_dir);
        f=fopen(path,"w");
        fprintf(f,"{\n  \"n_bits\": %d,\n  \"interactions\": %u,\n  \"lcg_seed\": \"0x%lx\"\n}\n",
                nBits, I, 0xC0FFEEULL ^ ((uint64_t)nBits * 0x9E3779B97F4A7C15ULL));
        fclose(f);
    }
    printf("done\n");
    return 0;
}
