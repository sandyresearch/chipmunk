#include <torch/extension.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cub/cub.cuh>
#include <curand_kernel.h>

using bf16 = at::BFloat16;
using fp16 = at::Half;
using fp32 = float;

/*
  Suppose these shapes (for batch dimension B, row-group dimension M=30, repetition R=8, feature dimension F=12288):
  
  sp_counts:       [B, M]
  sp_inds:         [B, M, F]         (int32)
  bmfc1:           [B, M*R, F]       (bf16)   # input tensor
  bm_mid_cache:    [B, M*R, F]       (bf16)   # output tensor that receives scattered data
  
  Overall logic (pseudo-code):
    For each row r in [0, M*R):
      - Find how many valid indices exist = sp_counts[ b, (r // R) ]
      - For each valid index c in [0, valid_count):
          col_idx = sp_inds[b, (r // R), c]
          bm_mid_cache[b, r, col_idx] = bmfc1[b, r, col_idx]
    where b indexes the batch.

  We assign one threadblock per row in the [M*R] dimension (per batch). 
  threadIdx.x loops over valid indices up to sp_counts[row].
*/

// CUDA kernel: each block handles one row-within-batch (M*R per batch).
template <typename T>
__global__ void copy_indices_kernel(
    const T* __restrict__ input_bmfc1,       // [batch, M*R, F] input
    T* __restrict__ output_bm_mid_cache,     // [batch, M*R, F] output
    const int32_t* __restrict__ sp_inds,     // [batch, M, F]
    const int32_t* __restrict__ sp_counts,   // [batch, M]
    const int   batch_size,                  // B
    const int   M,                           // 30
    const int   R,                           // 8  (the "replication" or expanded row dimension)
    const int   F)                           // 12288
{
    // Global row index across batch as well
    //   total_rows = batch_size * (M*R).
    int global_row = blockIdx.x; 
    if (global_row >= batch_size * M * R) {
        return;
    }

    // Compute which batch this row belongs to and which "row-within-batch" it is
    const int b          = global_row / (M * R);           // which batch
    const int row_in_mr  = global_row % (M * R);           // row index within [0, M*R)
    const int base_m     = row_in_mr / R;                  // original row in [0, M)
    // (row_in_mr % R) is which replicate, if you need it.

    __shared__ int valid_count_smem;
    // How many valid indices exist in this row (row_in_mr corresponds to base_m)
    if (threadIdx.x == 0) {
        valid_count_smem = sp_counts[b * M + base_m];
    }
    __syncthreads();
    const int valid_count = valid_count_smem;

    // Each thread loops over the valid portion of sp_inds
    for (int c = threadIdx.x; c < valid_count; c += blockDim.x) {
        // sp_inds[b, base_m, c]
        const int col_idx = sp_inds[b * (M * F) + base_m * F + c];
        // Address in input & output: [b, row_in_mr, col_idx]
        const int offset = b * (M * R * F)
                              + row_in_mr * F
                              + col_idx;
        
        // Copy from input_bmfc1 to output_bm_mid_cache
        output_bm_mid_cache[offset] = input_bmfc1[offset];
    }
}

namespace chipmunk {
// A simple launcher in C++ (PyTorch extension style) for demonstration:
void copy_indices(
    at::Tensor bmfc1,         // [B, M*R, F], bf16
    at::Tensor bm_mid_cache,  // [B, M*R, F], bf16
    at::Tensor sp_inds,       // [B, M, F],   int32
    at::Tensor sp_counts      // [B, M],      int32
)
{
    // Check all tensors are on CUDA
    TORCH_CHECK(bmfc1.is_cuda(), "bmfc1 must be a CUDA tensor");
    TORCH_CHECK(bm_mid_cache.is_cuda(), "bm_mid_cache must be a CUDA tensor");
    TORCH_CHECK(sp_inds.is_cuda(), "sp_inds must be a CUDA tensor");
    TORCH_CHECK(sp_counts.is_cuda(), "sp_counts must be a CUDA tensor");

    // Check sp_inds and sp_counts are int32
    TORCH_CHECK(sp_inds.scalar_type() == at::ScalarType::Int, "sp_inds must be int32");
    TORCH_CHECK(sp_counts.scalar_type() == at::ScalarType::Int, "sp_counts must be int32");

    // Check shapes
    const int B = bmfc1.size(0);
    const int M = sp_inds.size(1);
    const int F = sp_inds.size(2);
    const int R = bm_mid_cache.size(1) / M;

    const int total_rows = B * M * R;
    dim3 grid(total_rows);
    dim3 block(256);

    // Cast pointer types for the kernel
    const int32_t* inds_ptr   = sp_inds.data_ptr<int32_t>();
    const int32_t* counts_ptr = sp_counts.data_ptr<int32_t>();

    // Launch the kernel
    if (bmfc1.scalar_type() == at::ScalarType::BFloat16) {
        const bf16*  in_ptr   = bmfc1.data_ptr<bf16>();
        bf16*        out_ptr  = bm_mid_cache.data_ptr<bf16>();
        copy_indices_kernel<bf16><<<grid, block>>>(
            in_ptr,
            out_ptr,
            inds_ptr,
            counts_ptr,
            B, M, R, F
        );
    } else if (bmfc1.scalar_type() == at::ScalarType::Float) {
        const fp32*  in_ptr   = bmfc1.data_ptr<fp32>();
        fp32*        out_ptr  = bm_mid_cache.data_ptr<fp32>();
        copy_indices_kernel<fp32><<<grid, block>>>(
            in_ptr,
            out_ptr,
            inds_ptr,
            counts_ptr,
            B, M, R, F
        );
    } else if (bmfc1.scalar_type() == at::ScalarType::Half) {
        const fp16*  in_ptr   = bmfc1.data_ptr<fp16>();
        fp16*        out_ptr  = bm_mid_cache.data_ptr<fp16>();
        copy_indices_kernel<fp16><<<grid, block>>>(
            in_ptr,
            out_ptr,
            inds_ptr,
            counts_ptr,
            B, M, R, F
        );
    } else {
        TORCH_CHECK(false, "Unsupported tensor type");
    }

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("copy_indices_kernel launch failed: %s\n", cudaGetErrorString(err));
    }
    // Typically you'd also synchronize if needed, etc.
}

}