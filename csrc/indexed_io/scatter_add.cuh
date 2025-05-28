#pragma once
#include <cuda_bf16.h>

/**
 * The scatter_add kernel is invoked by two entrypoints: 
 * (1) torch.ops.chipmunk.csp_scatter_add
 * (2) torch.ops.chipmunk.csp_mlp_mm2_and_scatter_add
 * 
 * Therefore, we define a common interface for the two entrypoints to call the same underlying kernel
 * to avoid code duplication.
 */

static constexpr int MLP_BM = 128;
static constexpr int NUM_THREADS = 512; // autotuned across [128, 256, 512, 768, 1024]
// reduce bank conflicts 8x by adding 4 bfloat16's of padding to each row while staying in line with the 128x128 tile size of the mlp_mm2 kernel
static constexpr int SMEM_TILE_WIDTH = MLP_BM + 8; 
static constexpr int SHARED_MEM_SIZE = NUM_THREADS * SMEM_TILE_WIDTH * sizeof(__nv_bfloat16);

template <typename T>
__global__  __launch_bounds__(NUM_THREADS, 1) 
void scatter_add_kernel(
    const T* __restrict__ input_packed,        
    T* __restrict__ output_unpacked_colmajor,    
    const int32_t* __restrict__ sp_inds,
    const int32_t* __restrict__ sp_counts,          
    int M,
    int F
);
