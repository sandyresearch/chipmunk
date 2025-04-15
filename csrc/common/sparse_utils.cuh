namespace chipmunk {

using namespace kittens;

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
        std::cerr << "Error in tile TMA descriptor creation: " << error_string << std::endl;
    }
}

__device__ static inline float fast_tanh(float x) {
    float y;
    asm volatile ("tanh.approx.f32 %0, %1; " : "=f"(y) : "f"(x));
    return y;
}

__device__ static inline float gelu_elementwise(float f) {
    return f * 0.5f * (1.0f + fast_tanh(f * 0.79788456f * (1 + f * f *0.044715f)));
}

template<ducks::rt::all RT>
__device__ static inline void gelu(RT &tile) {
    #pragma unroll
    for (int tile_y = 0; tile_y < RT::height; tile_y++) {
        #pragma unroll
        for (int tile_x = 0; tile_x < RT::width; tile_x++) {
            #pragma unroll
            for (int i = 0; i < std::remove_reference_t<decltype(tile.tiles[0][0])>::packed_per_thread; i++) {
                tile.tiles[tile_y][tile_x].data[i].x = gelu_elementwise(tile.tiles[tile_y][tile_x].data[i].x);
                tile.tiles[tile_y][tile_x].data[i].y = gelu_elementwise(tile.tiles[tile_y][tile_x].data[i].y);
            }
        }
    }
}

template<bool add, ducks::rt::all RT>
__device__ static inline void load_bias(RT &accum, bf16 *bias) {
    int lane = kittens::laneid(); // lane within the warp
    /*
    * To understand these layouts see https://docs.nvidia.com/cuda/parallel-thread-execution/#matrix-fragments-for-wgmma-mma-async-m64nnk16
    * Each thread owns 16 of these bias values.
    */
    int col_idx = lane%4;
    bf16_2 biases[8] {
        {bias[0*8 + col_idx*2 + 0], bias[0*8 + col_idx*2 + 1]},
        {bias[1*8 + col_idx*2 + 0], bias[1*8 + col_idx*2 + 1]},
        {bias[2*8 + col_idx*2 + 0], bias[2*8 + col_idx*2 + 1]},
        {bias[3*8 + col_idx*2 + 0], bias[3*8 + col_idx*2 + 1]},
        {bias[4*8 + col_idx*2 + 0], bias[4*8 + col_idx*2 + 1]},
        {bias[5*8 + col_idx*2 + 0], bias[5*8 + col_idx*2 + 1]},
        {bias[6*8 + col_idx*2 + 0], bias[6*8 + col_idx*2 + 1]},
        {bias[7*8 + col_idx*2 + 0], bias[7*8 + col_idx*2 + 1]},
    };
    #pragma unroll
    for (int tile_y = 0; tile_y < RT::height; tile_y++) {
        #pragma unroll
        for (int tile_x = 0; tile_x < RT::width; tile_x++) {
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                // we don't use +=; we can directly use = because this is called at the start of the mainloop!
                if constexpr (add) {
                    accum.tiles[tile_y][tile_x].data[j].x += __bfloat162float(biases[tile_x*2+j/2].x);
                    accum.tiles[tile_y][tile_x].data[j].y += __bfloat162float(biases[tile_x*2+j/2].y);
                } else {
                    accum.tiles[tile_y][tile_x].data[j].x = __bfloat162float(biases[tile_x*2+j/2].x);
                    accum.tiles[tile_y][tile_x].data[j].y = __bfloat162float(biases[tile_x*2+j/2].y);
                }
            }
        }
    }
}


template<int N_BLOCK, ducks::rt::all RT, ducks::st::all ST>
__device__ static inline void sub_transposed(RT *accum, ST *pa_cache) {
    static_assert(RT::width == ST::width, "accum and pa_cache must have the same width");
    static_assert(RT::height*warpgroup::GROUP_WARPS == ST::height, "accum and pa_cache must have the same height at a warpgroup level");

    int lane = kittens::laneid();
    int workerid = warpgroup::warpid();

    #pragma unroll
    for (int tile_y = 0; tile_y < RT::height; tile_y++) {
        #pragma unroll
        for (int tile_x = 0; tile_x < RT::width; tile_x++) {
            int col_idx = tile_x*16 + (lane%4)*2;
            int row_idx = (workerid+tile_y)*16 + (lane/4);
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                int col_offset = (i/2)*8+col_idx;
                int row_offset = (i%2)*8+row_idx;
                uint32_t swizzle_offset_1 = pa_cache[0].idx(static_cast<uint32_t>(0), {col_offset+0, row_offset}) / sizeof(bf16);
                uint32_t swizzle_offset_2 = pa_cache[0].idx(static_cast<uint32_t>(0), {col_offset+1, row_offset}) / sizeof(bf16);
                for (int n = 0; n < N_BLOCK; n++) {
                    bf16 &v1_s = pa_cache[n][swizzle_offset_1];
                    bf16 &v2_s = pa_cache[n][swizzle_offset_2];
                    // GLOBAL MEMORY PA_CACHE - for debugging only
                    // bool should_print = blockIdx.x == 0 && threadIdx.x == 1 && tile_x == 0 && tile_y == 0 && n == 0 && i == 0;
                    // bf16 v1_g = args.globals.pa_cache[{ args.globals.pa_cache.rows-1-((args.common.coord.y+n)*64+col_offset+0), args.common.coord.x*64+row_offset }];
                    // bf16 v2_g = args.globals.pa_cache[{ args.globals.pa_cache.rows-1-((args.common.coord.y+n)*64+col_offset+1), args.common.coord.x*64+row_offset }];
                    // if (should_print) {
                    //     bool correct_v1 = v1_g == v1_s;
                    //     bool correct_v2 = v2_g == v2_s;
                    //     printf("args.iter=%d, v1_s: %f, v1_s_addr: %p, v1_g: %f, v1_g_addr: %p, correct: %d\n", args.task_iter, float(v1_s), &v1_s, float(v1_g), &v1_g, correct_v1);
                    //     printf("args.iter=%d, v2_s: %f, v2_s_addr: %p, v2_g: %f, v2_g_addr: %p, correct: %d\n", args.task_iter, float(v2_s), &v2_s, float(v2_g), &v2_g, correct_v2);
                    // }
                    auto &tile = accum[n].tiles[tile_y][tile_x];
                    tile.data[i].x -= __bfloat162float(v1_s);
                    tile.data[i].y -= __bfloat162float(v2_s);
                }
            }
        }
    }
}

template<int axis = 2, ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>, int N_THREADS=128>
__device__ static inline void load_async_gather(ST &dst_ptr, const GL &src, int iter, const COORD &idx, int *indices) {
    using T = typename ST::dtype;

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

        // global to shared: use cp.async
        asm volatile(
            "cp.async.cg.shared.global [%0], [%1], 16;\n"
            :: "r"(dst_ptrs + swizzle_offset), "l"(&src_ptr[indices[i]*row_stride + col])
            // :: "r"(dst_ptrs + swizzle_offset), "l"(&indices[i][col])
            : "memory"
        );

        swizzle_offset += 2048;
    }
}

__device__ inline static void cp_async_fence() {
    asm volatile("cp.async.commit_group;\n");
}

__device__ inline static void cp_async_semaphore(semaphore &sem) {
    asm volatile(
        "cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n" 
        :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&sem)))
        : "memory"
    );
}

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

}