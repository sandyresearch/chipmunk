#include <torch/extension.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cub/cub.cuh>
#include <c10/cuda/CUDAStream.h>
#include <curand_kernel.h>
#include "kittens.cuh"
#include "../common/all.cuh"
#include "scatter_add.cuh"

using namespace kittens;

template <typename T>
__global__ 
__launch_bounds__(NUM_THREADS, 1)
void scatter_add_kernel(
    const T* __restrict__ input_packed,             // [batch, M*MLP_BM, F] input
    T* __restrict__ output_unpacked_colmajor,       // [batch, M*MLP_BM, F] output
    const int32_t* __restrict__ sp_inds,            // [batch, M, F]
    const int32_t* __restrict__ sp_counts,          // [batch, M]
    int M,                                          // e.g. 30
    int F                                           // e.g. 12288
)
{
    // Each block handles one “row” (i.e. one M*MLP_BM slice) within the batch.
    int global_row = blockIdx.x;
    // int b     = global_row / M;  // which batch element
    static constexpr int b = 0;
    static constexpr int batch_size = 1;
    int cur_m = global_row;  // which row in [0, M)
    int valid_count = sp_counts[b * M + cur_m];
    int valid_count_next = 0;

    // Process rows in a grid–stride loop.
    for (; global_row < batch_size * M; global_row += gridDim.x) {
        // b     = global_row / M;           // which batch
        cur_m = global_row % M;           // row index within [0, M)
        bool is_last_iter = (global_row >= (batch_size*M) - gridDim.x);
        extern __shared__ T global_smem[];
        
        if (!is_last_iter) {
            int global_row_next = global_row + gridDim.x;
            int b_next = global_row_next / M;
            int cur_m_next = global_row_next % M;
            valid_count_next = sp_counts[b_next * M + cur_m_next];
        }

        for (int c = threadIdx.x; c < valid_count; c += blockDim.x) {
            const int col_idx = sp_inds[b * (M * F) + cur_m * F + c];
            
            kittens::tma::store_async_read_wait();

            #pragma unroll
            for (int cur_r = 0; cur_r < MLP_BM; cur_r++) {
                const int packed_offset = b * (M * MLP_BM * F) + (cur_m*MLP_BM + cur_r) * F + c;
                global_smem[threadIdx.x*SMEM_TILE_WIDTH + cur_r] = input_packed[packed_offset];
            }

            const int unpacked_offset_col_major = b * (F * M * MLP_BM) + col_idx * (M * MLP_BM) + (cur_m * MLP_BM);
            auto *output = &output_unpacked_colmajor[unpacked_offset_col_major];
            chipmunk::store_add_async(output, &global_smem[threadIdx.x*SMEM_TILE_WIDTH], MLP_BM);
        }
        __syncwarp();

        // Prepare for next row.
        valid_count = valid_count_next;
    }
}

namespace chipmunk {
// A simple launcher in C++ (PyTorch extension style) for demonstration:
void csp_scatter_add(
    at::Tensor packed,
    at::Tensor unpacked_colmajor,
    at::Tensor sp_inds,       // [B, M, F],   int32
    at::Tensor sp_counts,     // [B, M],      int32
    int64_t num_sms
)
{
    // Check all tensors are on CUDA
    TORCH_CHECK(packed.is_cuda(), "packed must be a CUDA tensor");
    TORCH_CHECK(packed.is_contiguous(), "packed must be contiguous");
    TORCH_CHECK(packed.dim() == 3, "packed must be a 3D tensor");
    TORCH_CHECK(unpacked_colmajor.is_cuda(), "unpacked_colmajor must be a CUDA tensor");
    TORCH_CHECK(unpacked_colmajor.is_contiguous(), "unpacked_colmajor must be contiguous");
    TORCH_CHECK(unpacked_colmajor.dim() == 3, "unpacked_colmajor must be a 3D tensor");
    TORCH_CHECK(sp_inds.is_cuda(), "sp_inds must be a CUDA tensor");
    TORCH_CHECK(sp_inds.is_contiguous(), "sp_inds must be contiguous");
    TORCH_CHECK(sp_inds.dim() == 3, "sp_inds must be a 3D tensor");
    TORCH_CHECK(sp_counts.is_cuda(), "sp_counts must be a CUDA tensor");
    TORCH_CHECK(sp_counts.is_contiguous(), "sp_counts must be contiguous");
    TORCH_CHECK(sp_counts.dim() == 2, "sp_counts must be a 2D tensor");

    // Check packed and unpacked_colmajor are bf16
    TORCH_CHECK(packed.scalar_type() == at::ScalarType::BFloat16, "packed must be bfloat16");
    TORCH_CHECK(unpacked_colmajor.scalar_type() == at::ScalarType::BFloat16, "unpacked_colmajor must be bfloat16");

    // Check sp_inds and sp_counts are int32
    TORCH_CHECK(sp_inds.scalar_type() == at::ScalarType::Int, "sp_inds must be int32");
    TORCH_CHECK(sp_counts.scalar_type() == at::ScalarType::Int, "sp_counts must be int32");

    // Check shapes
    const int B = packed.size(0);
    const int M = sp_inds.size(1);
    const int F = sp_inds.size(2);

    TORCH_CHECK(B == sp_inds.size(0), "B must match");
    TORCH_CHECK(B == 1, "B must be 1");
    TORCH_CHECK(B == sp_counts.size(0), "B must match");
    TORCH_CHECK(B == unpacked_colmajor.size(0), "B must match");

    TORCH_CHECK(M*MLP_BM == packed.size(1), "packed must have M*MLP_BM rows");
    TORCH_CHECK(M*MLP_BM == unpacked_colmajor.size(2), "unpacked_colmajor must have M*MLP_BM rows");
    TORCH_CHECK(M == sp_counts.size(1), "M must match");
    TORCH_CHECK(M == sp_inds.size(1), "M must match");

    TORCH_CHECK(F == packed.size(2), "packed must have F columns");
    TORCH_CHECK(F == unpacked_colmajor.size(1), "unpacked_colmajor must have F columns");

    // Launch configuration.
    dim3 grid(num_sms);
    dim3 block(NUM_THREADS);

    // Cast pointer types for the kernel
    const bf16*  in_ptr   = reinterpret_cast<bf16*>(packed.data_ptr<c10::BFloat16>());
    bf16*        out_ptr  = reinterpret_cast<bf16*>(unpacked_colmajor.data_ptr<c10::BFloat16>());
    const int32_t* inds_ptr   = sp_inds.data_ptr<int32_t>();
    const int32_t* counts_ptr = sp_counts.data_ptr<int32_t>();

    cudaFuncSetAttribute(scatter_add_kernel<bf16>, cudaFuncAttributeMaxDynamicSharedMemorySize, SHARED_MEM_SIZE);

    // Get the current stream. This will ensure that the kernel is launched on the stream active
    // in the PyTorch context (or the one you want to use).
    cudaStream_t current_stream = c10::cuda::getCurrentCUDAStream();

    // Launch the kernel on the retrieved stream.
    scatter_add_kernel<bf16><<<grid, block, SHARED_MEM_SIZE, current_stream>>>(
        in_ptr,
        out_ptr,
        inds_ptr,
        counts_ptr,
        sp_inds.size(1), // M
        sp_inds.size(2) // F
    );

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("scatter_add_kernel launch failed: %s\n", cudaGetErrorString(err));
    }
}
}