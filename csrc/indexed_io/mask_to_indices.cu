#include <torch/extension.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cub/cub.cuh>
#include <curand_kernel.h>

constexpr int NUM_THREADS = 32;

// Kernel for boolean mask → (indices, counts).
// dims: [b, h, m, n]
//
// Each warp handles one row out of the b*h*m rows.
//   1. Scan          : Thread 0 scans 0, 32, 64, ...
//   2. Sum           : Prefix sum nonzero counts across the warp.
//   3. Write indices : Thread 0 writes indices[row_idx, 0:prefix_sum[0]]
//                      Thread 1 writes indices[row_idx, prefix_sum[0]:prefix_sum[1]]
//                      ...
//   4. Write counts  : Only thread 0 writes counts[row_idx] = prefix_sum[31]
__global__ void mask_to_indices_kernel(
    const bool* __restrict__ mask,   // [b, h, m, n]
    int* __restrict__ indices,       // [b, h, m, n]
    int* __restrict__ counts,        // [b, h, m]
    const int multiple_of,
    const int b,
    const int h,
    const int m,
    const int n,
    const int pad_n)
{
    // Identify which [b,h,m] row we are in
    const int b_idx = blockIdx.x;
    const int h_idx = blockIdx.y;
    const int m_idx = blockIdx.z;
    const int row_idx        = (b_idx * h + h_idx) * m + m_idx;
    const int indices_offset = row_idx * n;
    const int output_offset  = row_idx * pad_n;
    const int counts_offset  = row_idx;

    const int tid         = threadIdx.x;
    const int num_threads = blockDim.x;

    // Use CUB block-scan storage
    __shared__ cub::BlockScan<int, NUM_THREADS>::TempStorage temp_storage;

    // 1) Each thread scans columns -> local_count
    int local_count = 0;
    for (int col = tid; col < n; col += num_threads) {
        // optional: indices[indices_offset + col] = -1;
        if (mask[indices_offset + col]) {
            local_count++;
        }
    }

    // 2) BlockScan (exclusive or inclusive, you pick) to get offsets
    int offset = 0;    // the prefix for this thread
    int total  = 0;    // final sum for entire block
    cub::BlockScan<int, NUM_THREADS>(temp_storage).ExclusiveSum(local_count, offset, total);
    int padded_total = ((total + multiple_of - 1) / multiple_of) * multiple_of;

    // 3) Now each thread can write columns
    for (int col = tid; col < n; col += num_threads) {
        if (mask[indices_offset + col]) {
            indices[output_offset + offset] = col;
            offset++;
        }
    }

    // 4) Handle padding and write counts
    int padding = padded_total - total;
    offset = total;
    if (tid == 0) {
        // Thread 0 gets the padding so no atomics
        if (padding > 0) {
            for (int col = 0; col < n; col++) {
                if (!mask[indices_offset + col]) {
                    indices[output_offset + offset] = col;
                    offset++;
                    if (offset == padded_total) {
                        break;
                    }
                }
            }
        }
        counts[counts_offset] = padded_total;
    }
}

// C++ entry point
namespace chipmunk {
std::vector<at::Tensor> mask_to_indices(
    at::Tensor mask,      // [b, h, m, n], bool
    int64_t multiple_of_,
    int64_t pad_to_multiple_of_ = 192
)
{
    TORCH_CHECK(mask.dim() == 4, "mask must be 4-dimensional [b, h, m, n]");
    TORCH_CHECK(mask.scalar_type() == at::kBool, "mask must be bool type");

    int multiple_of = static_cast<int>(multiple_of_);
    int pad_to_multiple_of = static_cast<int>(pad_to_multiple_of_);
    const int b = mask.size(0);
    const int h = mask.size(1);
    const int m = mask.size(2);
    const int n = mask.size(3);
    const int pad_n = ((mask.size(3) + pad_to_multiple_of - 1) / pad_to_multiple_of) * pad_to_multiple_of;

    at::Tensor indices = torch::empty(
        {
            static_cast<const uint>(b), 
            static_cast<const uint>(h), 
            static_cast<const uint>(m), 
            static_cast<const uint>(pad_n)
        },
        torch::TensorOptions().dtype(torch::kInt32).device(mask.device()));
    at::Tensor counts = torch::empty(
        {
            static_cast<const uint>(b), 
            static_cast<const uint>(h), 
            static_cast<const uint>(m)
        },
        torch::TensorOptions().dtype(torch::kInt32).device(mask.device()));


    const bool* mask_ptr    = mask.data_ptr<bool>();
    int* indices_ptr        = indices.data_ptr<int>();
    int* counts_ptr         = counts.data_ptr<int>();

    dim3 grid(b, h, m);      // One block per row
    dim3 block(NUM_THREADS); // One warp per row  (NUM_THREADS=32)

    // Kernel launch
    mask_to_indices_kernel<<<grid, block>>>(
        mask_ptr,
        indices_ptr,
        counts_ptr,
        multiple_of,
        b, h, m, n, pad_n
    );
    return {indices, counts};
}
}

#ifdef TEST_MODULE
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mask_to_indices", &mask_to_indices, "Boolean mask to indices and counts (CUDA)",
          py::arg("mask"),
          py::arg("multiple_of"),
          py::arg("pad_to_multiple_of"));
}
#endif
