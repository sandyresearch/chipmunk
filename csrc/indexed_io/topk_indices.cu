#include <torch/extension.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cub/cub.cuh>
#include <curand_kernel.h>

#define NUM_THREADS 1024

constexpr int SAMPLE_SIZE = 1024;
constexpr int NUM_ELEMENTS_PER_THREAD = SAMPLE_SIZE / NUM_THREADS;
using bf16 = at::BFloat16;
using fp16 = at::Half;
using fp32 = float;

struct CustomLess {
  template <typename DataType>
  __device__ bool operator()(const DataType &lhs, const DataType &rhs) {
    return lhs < rhs;
  }
};

// CUDA kernel: each block handles one row within a given batch.
template <typename T>
__global__ void topk_indices_kernel(
    const T* __restrict__ activation,   // [batch, rows, cols] input activations
    int* __restrict__ indices,            // [batch, rows, cols] output indices (pre-filled with -1)
    int* __restrict__ counts,             // [batch, rows] output count for each row
    const int rows,
    const int cols,
    const float quantile,
    const int multiple_of,
    const float random_amount)
{
    // Determine batch and row from block indices.
    const int row = blockIdx.x;
    const int batch = blockIdx.y;

    // Advance the pointers to the current batch and row.
    activation += batch * rows * cols + row * cols;
    indices    += batch * rows * cols + row * cols;
    counts     += batch * rows + row;

    const int tid = threadIdx.x;
    curandState state;
    unsigned long long seed = blockIdx.x * blockDim.x + blockIdx.y * gridDim.x * blockDim.x + threadIdx.x;
    seed += reinterpret_cast<int const*>(activation)[0];
    curand_init(seed, 0, 0, &state);

    if (quantile == 0) {
        // Edge case - include all indices.
        for (int col = tid; col < cols; col += NUM_THREADS) {
            indices[col] = col;
        }
        if (tid == 0) {
            counts[0] = cols;
        }
        return;
    } else if (quantile == 1) {
        // Edge case - include no indices.
        for (int col = tid; col < cols; col += NUM_THREADS) {
            indices[col] = -1;
        }
        if (tid == 0) {
            counts[0] = 0;
        }
        return;
    }

    // Define types for block-level sorting and loading (unchanged).
    using BlockMergeSort = cub::BlockMergeSort<T, NUM_THREADS, NUM_ELEMENTS_PER_THREAD>;
    using BlockLoad      = cub::BlockLoad<T, NUM_THREADS, NUM_ELEMENTS_PER_THREAD, cub::BLOCK_LOAD_STRIPED>;

    // Union to reuse shared memory.
    typedef union {
        typename BlockMergeSort::TempStorage temp_storage_shuffle;
        typename BlockLoad::TempStorage      temp_storage_load;
    } SharedStorage;

    __shared__ SharedStorage smem;
    // Shared variable for cutoff threshold (set by the thread at the quantile position).
    __shared__ T smem_cutoff_threshold;
    // Shared counter for how many valid values have been found for this row.
    __shared__ unsigned int count;

    // Shared memory for the padding phase.
    __shared__ int base_count_shared;
    __shared__ int remainder_shared;

    T thread_keys[NUM_ELEMENTS_PER_THREAD];

    // Load data in a striped manner and sort.
    BlockLoad(smem.temp_storage_load).Load(activation, thread_keys);
    BlockMergeSort(smem.temp_storage_shuffle).Sort(thread_keys, CustomLess());

    // Use one thread (at the quantile position) to set the cutoff threshold.
    if (tid == (int)(NUM_THREADS * quantile)) {
        smem_cutoff_threshold = thread_keys[0];
        count = 0;
    }
    __syncthreads();

    // Each thread keeps track of its last invalid (non-qualifying) index.
    int last_invalid_value = -1;
    // Load the cutoff threshold into a register.
    const float cutoff_threshold = smem_cutoff_threshold;

    // Loop over columns in a striped access pattern.
    for (int col = tid; col < cols; col += NUM_THREADS) {
        bool val = (activation[col] >= cutoff_threshold) || (curand_uniform(&state) < random_amount);
        if (val) {
            // Atomically get the next position and write the valid index.
            unsigned int pos = atomicInc(&count, cols);
            indices[pos] = col;
        } else {
            // Save candidate invalid index in case we need to pad.
            last_invalid_value = col;
        }
    }
    __syncthreads();

    // ------------ Padding Phase ------------
    // "count" is the number of valid indices stored.
    // Ensure that "count" is a multiple of multiple_of by computing required padding.
    if (tid == 0) {
        int mod = count % multiple_of;
        int r = (mod == 0) ? 0 : (multiple_of - mod);
        base_count_shared = count;
        remainder_shared = r;
        counts[0] = count + r;
    }
    __syncthreads();

    // Threads with a candidate last_invalid_value attempt to fill the padding gap.
    if (last_invalid_value != -1 && remainder_shared > 0) {
        // Each such thread tries to add one more index.
        int pos = atomicAdd(&count, 1);
        indices[pos] = last_invalid_value;
    }
}

// C++ API for launching the kernel. This function is callable from Python.
namespace chipmunk {
void topk_indices(
    at::Tensor activation,   // [batch, rows, cols]
    at::Tensor indices,            // [batch, rows, cols]
    at::Tensor counts,             // [batch, rows]
    double sparsity_amount_,
    int64_t multiple_of_,
    double random_amount_)
{
    float sparsity_amount = static_cast<float>(sparsity_amount_);
    int multiple_of = static_cast<int>(multiple_of_);
    float random_amount = static_cast<float>(random_amount_);
    // Check tensor dimensions.
    TORCH_CHECK(activation.dim() == 3, "activation must be 3-dimensional [batch, rows, cols]");
    TORCH_CHECK(indices.dim() == 3, "indices must be 3-dimensional [batch, rows, cols]");
    TORCH_CHECK(counts.dim() == 2, "counts must be 2-dimensional [batch, rows]");
    TORCH_CHECK(0 <= sparsity_amount && sparsity_amount <= 1, "sparsity_amount must be between 0 and 1");

    const auto batch = activation.size(0);
    const auto rows  = activation.size(1);
    const auto cols  = activation.size(2);

    int* indices_ptr = indices.data_ptr<int>();
    int* counts_ptr  = counts.data_ptr<int>();

    // Launch a 2D grid: one block per row per batch.
    const dim3 grid(rows, batch);
    const dim3 block(NUM_THREADS);

    if (activation.scalar_type() == at::kBFloat16) {
        using T = bf16;
        const T* activation_ptr = reinterpret_cast<const T*>(activation.data_ptr<c10::BFloat16>());
        topk_indices_kernel<T><<<grid, block>>>(
            activation_ptr,
            indices_ptr,
            counts_ptr,
            rows,
            cols,
            sparsity_amount,
            multiple_of,
            random_amount
        );
    }
    else if (activation.scalar_type() == at::kHalf) {
        using T = fp16;
        const T* activation_ptr = reinterpret_cast<const T*>(activation.data_ptr<c10::Half>());
        topk_indices_kernel<T><<<grid, block>>>(
            activation_ptr,
            indices_ptr,
            counts_ptr,
            rows,
            cols,
            sparsity_amount,
            multiple_of,
            random_amount
        );
    }
    else if (activation.scalar_type() == at::kFloat) {
        using T = fp32;
        const T* activation_ptr = reinterpret_cast<const T*>(activation.data_ptr<float>());
        topk_indices_kernel<T><<<grid, block>>>(
            activation_ptr,
            indices_ptr,
            counts_ptr,
            rows,
            cols,
            sparsity_amount,
            multiple_of,
            random_amount
        );
    }
    else {
        TORCH_CHECK(false, "Unsupported dtype for activation tensor");
    }
}
}
