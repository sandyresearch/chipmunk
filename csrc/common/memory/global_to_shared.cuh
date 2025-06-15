#pragma once
#include "kittens.cuh"

namespace chipmunk {

using namespace kittens;

/**
 * @brief The most performance-critical piece of code in Chipmunk - performs a column-major load from global to shared memory,
 * gathering certain columns from sparse indices.
 * 
 * @param dst_ptr: A 64x256 matrix in shared memory that will store the result
 * @param src: The source pointer (expected to be in global memory)
 * @param iter: The k block group (in units of 64 rows) along the B matrix that we're tiling over
 * @param idx: The k block group (in units of 64 rows) along the B matrix that we're tiling over
 * @param indices: An array of 16 indices in registers that this thread is responsible for loading
 */
template<ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void load_async_gather(ST &dst_ptr, const GL &src, int iter, const COORD &idx, int *indices) {
    using T = typename ST::dtype;

    constexpr int N_THREADS = 128;
    constexpr int axis = 2;

    const int row_stride = src.template stride<axis>();
    constexpr int elem_per_memcpy = sizeof(float4)/sizeof(typename ST::dtype);
    constexpr int memcpy_per_row = ST::cols / elem_per_memcpy;
    constexpr int total_calls = (ST::height*ST::width * kittens::TILE_ROW_DIM<T>*kittens::TILE_COL_DIM<T> + N_THREADS*elem_per_memcpy-1) / (N_THREADS*elem_per_memcpy); // round up

    coord<> unit_coord = idx.template unit_coord<axis, 3>();
    typename GL::dtype *src_ptr = (typename GL::dtype*)&src[unit_coord];
    uint32_t dst_ptrs = __cvta_generic_to_shared(&dst_ptr.data[0]);
    int laneid = threadIdx.x % N_THREADS;

    uint32_t swizzle_offset = dst_ptr.idx(static_cast<uint32_t>(0), {laneid / memcpy_per_row, (laneid*elem_per_memcpy) % ST::cols});

    #pragma unroll
    for(int i = 0; i < total_calls; i++) {
        int load_idx_cur  = (i) * N_THREADS + laneid; 
        int load_idx_next = (i+1) * N_THREADS + laneid;
        int row_next = load_idx_next / memcpy_per_row;

        int shared_row = load_idx_cur / memcpy_per_row;
        int col = (load_idx_cur*elem_per_memcpy) % ST::cols;

        asm volatile(
            "cp.async.cg.shared.global [%0], [%1], 16;\n"
            :: "r"(dst_ptrs + swizzle_offset), "l"(&src_ptr[indices[i]*row_stride + col])
            // :: "r"(dst_ptrs + swizzle_offset), "l"(&indices[i][col])
            : "memory"
        );

        // Compiler can't infer swizzle_offset as a compile-time constant, so we need to add 2048 to it
        // for each call to cp.async.cg.shared.global.
        swizzle_offset += 2048;
    }
}

/**
 * @brief Commit the async group so that it can be `cp.async.wait_group<N>`'d on later.
 */
__device__ inline static void cp_async_fence() {
    asm volatile("cp.async.commit_group;\n");
}


/**
 * @brief When all previous non-committed cp.async operations performed by this thread are complete,
 * this thread will trip a transaction on the semaphore.
 * 
 * @param sem: The semaphore to trip
 */
__device__ inline static void cp_async_semaphore(semaphore &sem) {
    asm volatile(
        "cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n" 
        :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&sem)))
        : "memory"
    );
}

/**
 * @brief Large asynchronous bulk load operation using the TMA. We use this to load the sparse indices
 * from global memory into shared memory.
 * 
 * @param dst: The destination pointer (expected to be in shared memory)
 * @param src: The source pointer (expected to be in global memory)
 * @param num_bytes: The number of bytes to load
 * @param bar: The semaphore to trip once the load is complete
 */
__device__ inline static void load_async_bytes(void *dst, const void *src, int num_bytes, semaphore &bar) {
    uint32_t s_dst_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
    uint32_t s_bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&bar));
    uint64_t g_src_ptr = static_cast<uint64_t>(__cvta_generic_to_global(src));
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
        :: "r"(s_dst_ptr), "l"(g_src_ptr), "r"(num_bytes), "r"(s_bar_ptr)
        : "memory"
    );
}

/**
 * @brief Bulk store add operation.
 *
 * This operation performs an atomic add operation on the input value (expected to be in shared memory)
 * and stores the result in the destination pointer (expected to be in global memory).
 * 
 * It uses the TMA's reduce operation to perform the atomic add. This function is used in the scatter-add kernel
 * (see csrc/indexed_io/scatter_add.cu) to accumulate into the sparse activations unpacked tensor (column major).
 * 
 * @param dst: The destination pointer (expected to be in global memory)
 * @param src: The source pointer (expected to be in shared memory)
 * @param num_elements: The number of elements to store
 */

__device__ inline static void store_add_async(bf16 *dst, const bf16 *src, int num_elements) {
    uint32_t s_src_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(src));;
     int32_t num_bytes = num_elements * sizeof(bf16);
    asm volatile ("fence.proxy.async.shared::cta;\n" ::: "memory");
    asm volatile(
        "cp.reduce.async.bulk.global.shared::cta.bulk_group.add.noftz.bf16 [%0], [%1], %2;\n"
        :: "l"(dst), "r"(s_src_ptr), "r"(num_bytes)
        : "memory"
    );
}

/**
 * @brief Create a 4D tensor map using non-contiguous strides for all dimensions except for the last dimension.
 * The last dimension must be contiguous; this is a limitation inherent to the TMA hardware.
 * 
 * This function makes it possible to take non-contiguous q, k, and v matrices in for attention kernels.
 * 
 * @param stride1: The stride for the batch dimension (tensor.stride(0))
 * @param stride2: The stride for the depth dimension (tensor.stride(1))
 * @param stride3: The stride for the row dimension (tensor.stride(2))
 */

template<ducks::st::all ST, int axis>
__host__ static inline void create_tensor_map_with_strides(CUtensorMap *tma_map, const typename ST::dtype *src, int batch, int depth, int rows, int cols, int stride1, int stride2, int stride3) {
    using dtype = typename ST::dtype;
    static_assert(axis==0 || axis==1 || axis==2, "axis must be 0, 1, or 2");
    
    constexpr uint32_t  tma_dim = 5; // Always use all 5D
    void *global_addr = (void*)(src);

    constexpr CUtensorMapDataType     tma_format      = (
        std::is_same_v<dtype, bf16>  ? CU_TENSOR_MAP_DATA_TYPE_BFLOAT16 :
        std::is_same_v<dtype, half>  ? CU_TENSOR_MAP_DATA_TYPE_FLOAT16 :
        std::is_same_v<dtype, float> ? CU_TENSOR_MAP_DATA_TYPE_FLOAT32 :
        std::is_same_v<dtype, fp8e4m3> ? CU_TENSOR_MAP_DATA_TYPE_UINT8 :
        std::is_same_v<dtype, fp8e5m2> ? CU_TENSOR_MAP_DATA_TYPE_UINT8 :
        CUtensorMapDataType(-1)
    );
    constexpr CUtensorMapInterleave   tma_interleave  = CU_TENSOR_MAP_INTERLEAVE_NONE;
    constexpr CUtensorMapL2promotion  tma_l2Promotion = CU_TENSOR_MAP_L2_PROMOTION_NONE;
    constexpr CUtensorMapFloatOOBfill tma_oobFill     = CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;
    constexpr CUtensorMapSwizzle      tma_swizzle     = (
        ST::swizzle_bytes == 32  ? CU_TENSOR_MAP_SWIZZLE_32B  :
        ST::swizzle_bytes == 64  ? CU_TENSOR_MAP_SWIZZLE_64B  :
        ST::swizzle_bytes == 128 ? CU_TENSOR_MAP_SWIZZLE_128B : 
        CU_TENSOR_MAP_SWIZZLE_NONE
    );

    uint64_t gmem_shape [5] = {0, 0, 0, 0, 0};
    uint64_t gmem_stride[4] = {0, 0, 0, 0};
    uint32_t smem_shape [5] = {0, 0, 0, 0, 0};
    uint32_t smem_stride[5] = {1, 1, 1, 1, 1};

    constexpr uint64_t shared_tile_height = ST::rows; 
    constexpr uint64_t shared_tile_width  = ST::cols;

    constexpr int swizzle_elements = ST::swizzle_bytes / sizeof(dtype);

    static_assert(axis == 2, "axis must be 2");
    gmem_shape[0] = swizzle_elements;
    gmem_shape[1] = (uint64_t)rows;
    gmem_shape[2] = (uint64_t)(cols+swizzle_elements-1) / swizzle_elements; // round up, note this can potentially screw up out of bounds access handling :/
    gmem_shape[3] = (uint64_t)depth;
    gmem_shape[4] = (uint64_t)batch;

    gmem_stride[0] = (uint64_t)stride3 * sizeof(dtype);
    gmem_stride[1] = ST::swizzle_bytes;
    gmem_stride[2] = (uint64_t)stride2 * sizeof(dtype);
    gmem_stride[3] = (uint64_t)stride1 * sizeof(dtype);
    smem_shape[0] = swizzle_elements;
    smem_shape[1] = shared_tile_height;
    smem_shape[2] = shared_tile_width / swizzle_elements;
    smem_shape[3] = 1;
    smem_shape[4] = 1;

    // ensure that the global address is always 16-byte aligned 
    assert((reinterpret_cast<uint64_t>(global_addr) & 0b1111) == 0);

    assert(gmem_stride[0] % 16 == 0); // gmem_stride[0] elements must be a multiple of 16B
    assert(gmem_stride[1] % 16 == 0); // gmem_stride[1] elements must be a multiple of 16B
    assert(gmem_stride[2] % 16 == 0); // gmem_stride[2] elements must be a multiple of 16B
    assert(gmem_stride[3] % 16 == 0); // gmem_stride[2] elements must be a multiple of 16B

    assert(smem_shape[0] <= 256); // smem_shape[0] elements must be <= 256
    assert(smem_shape[1] <= 256); // smem_shape[1] elements must be <= 256
    assert(smem_shape[2] <= 256); // smem_shape[2] elements must be <= 256

    assert((smem_shape[0]*sizeof(dtype)) % 16 == 0); // if wgmma_interleave is none, then smem_shape[0] * sizeof(dtype) must be a multiple of 16B

    assert(smem_stride[0] <= 8); // smem_stride[0] must be less <= 8
    assert(smem_stride[1] <= 8); // smem_stride[1] must be less <= 8
    assert(smem_stride[2] <= 8); // smem_stride[2] must be less <= 8
    assert(smem_stride[3] <= 8); // smem_stride[3] must be less <= 8
    assert(smem_stride[4] <= 8); // smem_stride[3] must be less <= 8

    assert(smem_stride[0] == 1); // smem_stride[0] is ignored when wgmma_interleave is none

    if constexpr (tma_interleave == CU_TENSOR_MAP_INTERLEAVE_NONE && tma_swizzle != CU_TENSOR_MAP_SWIZZLE_NONE) {
        assert(smem_shape[0] * sizeof(dtype) <= ST::swizzle_bytes);
    }

    const uint64_t *gmem_shape_ptr = &gmem_shape[0];
    const uint64_t *gmem_stride_ptr = &gmem_stride[0]; 
    const uint32_t *smem_shape_ptr = &smem_shape[0];
    const uint32_t *smem_stride_ptr = &smem_stride[0];

    CUresult result = cuTensorMapEncodeTiled(
        tma_map,
        tma_format,
        tma_dim,
        global_addr,
        gmem_shape_ptr,
        gmem_stride_ptr, 
        smem_shape_ptr,
        smem_stride_ptr,
        tma_interleave,
        tma_swizzle,
        tma_l2Promotion,
        tma_oobFill);

    const char *error_string;
    CUresult res = cuGetErrorString(result, &error_string);
    if (result != CUDA_SUCCESS) {
        std::cerr << "Error in strided tile TMA descriptor creation: " << error_string << std::endl;
    }
}


/**
* @brief Creates a tensor map for the given source vector.
*
* This function creates a tensor map (CUtensorMap) for the specified source shared vector type. The tensor map
* is used to describe the shape and layout of the tensor in memory. The function sets up the tensor
* map based on the provided source tensor pointer and the layout specified by the SV template parameter.
*
* @tparam SV The source tensor type, which must be TMA-compatible.
* @tparam num_vectors The number of vectors present in global memory.
* @param tma_map Pointer to the CUtensorMap object to be initialized.
* @param src Pointer to the source tensor data in global memory.
*/
template<ducks::sv::all SV, int axis>
__host__ static inline void create_tensor_map_with_strides(CUtensorMap *tma_map, const typename SV::dtype *src, int batch, int depth, int rows, int cols, int stride1, int stride2, int stride3) {
    using dtype = typename SV::dtype;
    static_assert(axis == -1, "for vector TMA, row axis must be -1 as it's unused");
    static_assert(SV::length <= 256 || (SV::length*sizeof(dtype)) % 128 == 0);
    // There is technically a way around ^ that involves instantiating two separate TMA descriptors, one of size 256
    // and the other of size %256, but this is a fairly mild restriction and the other approach is a real PITA and incurs other costs.
    
    constexpr uint32_t  tma_dim     = 4;
    void               *global_addr = (void*)(src);

    constexpr CUtensorMapDataType     tma_format      = (
        std::is_same_v<dtype, bf16>  ? CU_TENSOR_MAP_DATA_TYPE_BFLOAT16 :
        std::is_same_v<dtype, half>  ? CU_TENSOR_MAP_DATA_TYPE_FLOAT16 :
        std::is_same_v<dtype, float> ? CU_TENSOR_MAP_DATA_TYPE_FLOAT32 :
        std::is_same_v<dtype, fp8e4m3> ? CU_TENSOR_MAP_DATA_TYPE_UINT8 :
        std::is_same_v<dtype, fp8e5m2> ? CU_TENSOR_MAP_DATA_TYPE_UINT8 :
        CUtensorMapDataType(-1)
    );
    constexpr CUtensorMapInterleave   tma_interleave  = CU_TENSOR_MAP_INTERLEAVE_NONE;
    constexpr CUtensorMapL2promotion  tma_l2Promotion = CU_TENSOR_MAP_L2_PROMOTION_NONE;
    constexpr CUtensorMapFloatOOBfill tma_oobFill     = CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;
    constexpr CUtensorMapSwizzle      swizzle         = CU_TENSOR_MAP_SWIZZLE_NONE;

    constexpr uint64_t dim1 = tma::detail::sv_tma_dim1<SV>; // inner dim
    // constexpr uint64_t dim2 = sv_tma_dim2<SV>; outer dim, not used here.

    uint64_t gmem_shape [4] = {(uint64_t)cols, (uint64_t)rows, (uint64_t)depth, (uint64_t)batch};
    uint64_t gmem_stride[3] = {(uint64_t)cols*sizeof(dtype), (uint64_t)cols*rows*sizeof(dtype), (uint64_t)cols*rows*depth*sizeof(dtype)};
    uint32_t smem_shape [4] = {(uint32_t)dim1, 1, 1, 1};
    uint32_t smem_stride[4] = {1, 1, 1, 1};

    gmem_stride[0] = (uint64_t)stride3 * sizeof(dtype);
    gmem_stride[1] = (uint64_t)stride2 * sizeof(dtype);
    gmem_stride[2] = (uint64_t)stride1 * sizeof(dtype);

    // ensure that the global address is always 16-byte aligned 
    assert((reinterpret_cast<uint64_t>(global_addr) & 0b1111) == 0);

    assert(smem_shape[0] <= 256); // smem_shape[0] elements must be <= 256.

    const uint64_t *gmem_shape_ptr = &gmem_shape[0];
    const uint64_t *gmem_stride_ptr = &gmem_stride[0]; 
    const uint32_t *smem_shape_ptr = &smem_shape[0];
    const uint32_t *smem_stride_ptr = &smem_stride[0];

    CUresult result = cuTensorMapEncodeTiled(
        tma_map,
        tma_format,
        tma_dim,
        global_addr,
        gmem_shape_ptr,
        gmem_stride_ptr, 
        smem_shape_ptr,
        smem_stride_ptr,
        tma_interleave,
        swizzle,
        tma_l2Promotion,
        tma_oobFill
    );

    const char *error_string;
    CUresult res = cuGetErrorString(result, &error_string);
    if (result != CUDA_SUCCESS) {
        std::cerr << "Error in strided vector TMA descriptor creation: " << error_string << std::endl;
    }
};


/**
 * @brief Loads data from global memory into a shared memory tile, without assuming that the global tile is contiguous.
 * Strides for b, d, and r are passed in as a 3-tuple.
 *
 * @tparam ST The type of the shared tile.
 * @param[out] dst The destination shared memory tile.
 * @param[in] src The source global memory array.
 * @param[in] idx The coordinate of the tile in the global memory array.
 * @param[in] strides The strides for b, d, and r.
 */
template<bool test_for_oob, ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void load_strided(ST &dst, const GL &src, const COORD &idx, int3 strides) {
    using T = typename ST::dtype;
    const int row_stride = strides.z;
    constexpr int N_THREADS = 128;
    constexpr int axis = 2;
    constexpr int elem_per_memcpy = sizeof(float4)/sizeof(typename ST::dtype);
    constexpr int memcpy_per_row = dst.cols / elem_per_memcpy;
    constexpr int total_calls = (dst.height*dst.width * kittens::TILE_ROW_DIM<T>*kittens::TILE_COL_DIM<T> + N_THREADS*elem_per_memcpy-1) / (N_THREADS*elem_per_memcpy); // round up
    constexpr int total_rows = dst.height*dst.width;

    typename GL::dtype *src_ptr = (typename GL::dtype*)&src[
        idx.b*strides.x + 
        idx.d*strides.y + 
        idx.r*strides.z*ST::rows + 
        idx.c*1*ST::cols
    ];
    uint32_t dst_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&dst.data[0]));
    int laneid = threadIdx.x % N_THREADS;

    #pragma unroll
    for(int i = 0; i < total_calls; i++) {

        int load_idx = i * N_THREADS + laneid;
        
        int row = load_idx / memcpy_per_row;
        int col = (load_idx*elem_per_memcpy) % dst.cols;
        float4 tmp;
        if constexpr (test_for_oob) {
            if (row > src.rows) {
                // Don't continue here - return - since `row` increases monotonically with `i`
                return;
            }
        }
        move<float4>::ldg(tmp, (float4*)&src_ptr[row*row_stride + col]);
        move<float4>::sts(dst.idx(dst_ptr, {row, col}), tmp);
    }
}
}