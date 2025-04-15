#include <cuda_bf16.h>

static constexpr int MLP_BM = 128;
static constexpr int NUM_THREADS = 512;
static constexpr int SMEM_TILE_WIDTH = MLP_BM + 8; 
static constexpr int SHARED_MEM_SIZE = NUM_THREADS * SMEM_TILE_WIDTH * sizeof(__nv_bfloat16);

template <typename T>
__global__ 
__launch_bounds__(NUM_THREADS, 1)
void scatter_add_kernel(
    const T* __restrict__ input_packed,           // [batch, M*MLP_BM, F] input
    T* __restrict__ output_unpacked_colmajor,       // [batch, M*MLP_BM, F] output
    const int32_t* __restrict__ sp_inds,            // [batch, M, F]
    const int32_t* __restrict__ sp_counts,          // [batch, M]
    int M,                                          // e.g. 30
    int F                                          // e.g. 12288
);