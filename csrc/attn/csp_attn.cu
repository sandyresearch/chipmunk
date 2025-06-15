#include "kittens.cuh"
#include "prototype.cuh"
#include <cuda_pipeline.h>
#include "../common/all.cuh"

using namespace kittens;
using namespace kittens::prototype;
using namespace kittens::prototype::lcf;

/* PARAMETERIZATION: 2 CONSUMERS */
// static constexpr int NUM_CONSUMER_WARPGROUPS = 2;
// static constexpr int NUM_CONSUMER_REGISTERS = 184;
// static constexpr int NUM_PRODUCER_REGISTERS = 136;
// static constexpr int KV_TILE_ROWS = 176;
// static constexpr bool CACHE_SWIZZLE_OFFSETS = true;

/* PARAMETERIZATION: 3 CONSUMERS */
static constexpr int NUM_CONSUMER_WARPGROUPS = 3;
static constexpr int NUM_CONSUMER_REGISTERS = 152;
static constexpr int NUM_PRODUCER_REGISTERS = 56;
static constexpr int KV_TILE_ROWS = 112;
static constexpr int INDICES_LOAD_INTERVAL = 64;
static constexpr bool CACHE_SWIZZLE_OFFSETS = true;

// static constexpr int MAX_SUPPORTED_ATTN_N = 4032;
static constexpr bool ENABLE_DYNAMIC_INDICES = true;
static constexpr bool LOAD_INDICES_FROM_TMA = true;

template<int D, int NUM_WORKERS> struct attn_fwd_layout {
    using qo_tile   = st_bf<64, D>;
    using kv_tile   = st_bf<D==64?192:KV_TILE_ROWS, D>;
    using qo_global = kittens::gl<bf16, -1, -1, -1, D, qo_tile>;
    using kv_global = kittens::gl<bf16, -1, -1, -1, D, kv_tile>;
    static constexpr int kv_load_iters = kv_tile::rows / 8;

    struct globals { 
        qo_global O, Q; 
        kv_global K, V; 
        int *indices, *indices_counts;
        int3 q_stride, k_stride, v_stride;
        int3 indices_stride;
    };
    struct input_block    { kv_tile k, v; };
    // struct scratch_block  { qo_tile q[NUM_WORKERS]; int indices[INPUT_PIPE_STAGES][KV_TILE_ROWS]; semaphore indices_bar[INPUT_PIPE_STAGES]; };
    struct scratch_block  { qo_tile q[NUM_WORKERS]; int indices[INDICES_LOAD_INTERVAL * KV_TILE_ROWS]; semaphore indices_bar; };
    struct common_state   { int batch, head, seq; };
    struct producer_state {
        uint32_t swizzle_offsets[kv_load_iters];
        uint32_t indices_phase = 0;
    };
    struct consumer_state {
        rt_fl<16, qo_tile::cols> o_reg;
        col_vec<rt_fl<16, kv_tile::rows>> max_vec, norm_vec;

        col_vec<rt_fl<16, kv_tile::rows>> max_vec_last_scaled, max_vec_scaled;
        rt_fl<16, kv_tile::rows> att_block;
        rt_bf<16, kv_tile::rows> att_block_mma;
    };
};


template<int D, int O_SCALE> struct attn_fwd_template {
    static constexpr int NUM_CONSUMER_WARPS = NUM_CONSUMER_WARPGROUPS * 4, NUM_WORKERS = NUM_CONSUMER_WARPS/4, INPUT_PIPE_STAGES = 2;
    using layout = attn_fwd_layout<D, NUM_WORKERS>;
    static constexpr int PRODUCER_BARRIER_ARRIVALS = 128;

    __device__ static inline int get_indices_count(const layout::qo_global &Q, int batch, int head, int seq, int *indices_counts) {
        // we index into the indices_counts array for the current batch and head. todo: fix indices offsets for multiple batches and heads!
        // batch can be 1 over the maximally allowed because get_indices_count can be called for OOB threadblocks in the grid
        // we force-reset it to 0 as a hack to avoid cuda illegal memory access errors
        batch = 0;
        int H = Q.depth; 
        int N_groups = (Q.rows +  (layout::qo_tile::rows * NUM_CONSUMER_WARPGROUPS) - 1) / (layout::qo_tile::rows * NUM_CONSUMER_WARPGROUPS); // the total number of indices groups per head
        int offset = (batch * H * N_groups) + (head * N_groups) + (seq);
        return indices_counts[offset];
    }

    __device__ static inline void common_setup(common_setup_args<layout> args) {
        int task_id = gridDim.x*args.task_iter + blockIdx.x;
        int seq_q = (args.globals.Q.rows + NUM_WORKERS*layout::qo_tile::rows - 1)/(NUM_WORKERS*layout::qo_tile::rows);
        args.common.batch = task_id / (seq_q*args.globals.K.depth); task_id -= args.common.batch * seq_q * args.globals.K.depth;
        args.common.head  = task_id / seq_q;                        task_id -= args.common.head  * seq_q;
        args.common.seq   = task_id;
        int num_iters = get_indices_count(args.globals.Q, args.common.batch, args.common.head, args.common.seq, args.globals.indices_counts);
        // we assume num_iters % layout::kv_tile::rows == 0 - our indices kernel will make sure of this!
        // otherwise we would need to do num_iters = (num_iters + layout::kv_tile::rows - 1) / layout::kv_tile::rows;
        num_iters /= layout::kv_tile::rows;

        args.num_iters = args.common.batch < args.globals.Q.batch ? num_iters : -1;
    }

    struct producer {
        __device__ static inline void load_indices(int *s_indices, int *g_indices, int indices_count, int iter, int3 indices_coord, int3 indices_stride, const layout::qo_global &Q, const layout::kv_global &K, semaphore &bar) {
            // local indices coordinates
            int batch = indices_coord.x;
            int head = indices_coord.y;
            int seq = indices_coord.z;
            // global indices shape
            int H = Q.depth;
            int N_groups = (Q.rows + layout::qo_tile::rows * NUM_CONSUMER_WARPGROUPS - 1) / (layout::qo_tile::rows * NUM_CONSUMER_WARPGROUPS); // the total number of indices groups per head
            int N_queries = Q.rows;
            // this indexing calcluation is the SAME as the one in the get_indices_count, but the stride between elements is now `N_keys`
            g_indices += (batch * indices_stride.x)         + (head * indices_stride.y)     + (seq * indices_stride.z) + iter * layout::kv_tile::rows;
         // g_indices += (batch * H * N_groups * N_queries) + (head * N_groups * N_queries) + (seq * N_queries)        + iter * layout::kv_tile::rows;

            if (ENABLE_DYNAMIC_INDICES) {
                if (LOAD_INDICES_FROM_TMA) {
                    uint32_t s_indices_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(s_indices));
                    uint32_t s_bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&bar));
                    uint64_t g_indices_ptr = static_cast<uint64_t>(__cvta_generic_to_global(g_indices));
                    // int num_bytes_to_copy = layout::kv_tile::rows * sizeof(int);
                    int num_bytes_to_copy = min(INDICES_LOAD_INTERVAL * layout::kv_tile::rows, indices_count - iter * layout::kv_tile::rows) * sizeof(int);

                    tma::expect_bytes(bar, num_bytes_to_copy);

                    asm volatile(
                        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
                        :: "r"(s_indices_ptr), "l"(g_indices_ptr), "r"(num_bytes_to_copy), "r"(s_bar_ptr)
                        : "memory"
                    );
                } else {
                    // for debugging only - load data but without TMA
                    for (int i = 0; i < INDICES_LOAD_INTERVAL * layout::kv_tile::rows; i++) {
                        s_indices[i] = g_indices[i];
                    }
                    arrive(bar);
                }
            } else {
                // for debugging only - arange indices
                for (int i = 0; i < INDICES_LOAD_INTERVAL * layout::kv_tile::rows; i++) {
                    s_indices[i] = iter * layout::kv_tile::rows + i;
                }
                arrive(bar);
            }
        }

        template<int axis = 2, typename layout, ducks::st::all ST, int N_THREADS=128>
        __device__ static inline void cache_swizzle_offsets(uint32_t *swizzle_offsets) {
            using T = typename ST::dtype;
            constexpr int elem_per_memcpy = sizeof(float4)/sizeof(typename ST::dtype);
            constexpr int memcpy_per_row = ST::cols / elem_per_memcpy;
            constexpr int total_calls = (ST::height*ST::width * kittens::TILE_ROW_DIM<T>*kittens::TILE_COL_DIM<T> + N_THREADS*elem_per_memcpy-1) / (N_THREADS*elem_per_memcpy); // round up
            static_assert(total_calls == layout::kv_load_iters, "total_calls must be equal to layout::kv_load_iters");

            int laneid = threadIdx.x % N_THREADS;

            #pragma unroll
            for(int i = 0; i < total_calls; i++) {

                int load_idx = i * N_THREADS + laneid;                
                int r = load_idx / memcpy_per_row;
                int c = (load_idx*elem_per_memcpy) % ST::cols;

                static constexpr int swizzle_repeat = ST::swizzle_bytes * 8;
                static constexpr int subtile_cols   = ST::swizzle_bytes / sizeof(T);
                const int outer_idx = c/subtile_cols;
                const uint32_t addr = sizeof(T)*(outer_idx*ST::rows*subtile_cols + r*subtile_cols + c%subtile_cols);
                const uint32_t swizzle = ((addr % swizzle_repeat) >> 7) << 4;
                swizzle_offsets[i] = addr ^ swizzle;
            }
        }

        template<int axis = 2, bool CACHE_SWIZZLE_OFFSETS = true, ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>, int N_THREADS=128>
        __device__ static inline void load_async_gather(ST &dst, ST &dst2, const GL &src, const GL &src2, int3 stride1, int3 stride2, int2 idx, int iter, uint32_t *swizzle_offsets, int *s_indices) {
            using T = typename ST::dtype;
            constexpr int elem_per_memcpy = sizeof(float4)/sizeof(typename ST::dtype);
            constexpr int memcpy_per_row = dst.cols / elem_per_memcpy;
            constexpr int total_calls = (dst.height*dst.width * kittens::TILE_ROW_DIM<T>*kittens::TILE_COL_DIM<T> + N_THREADS*elem_per_memcpy-1) / (N_THREADS*elem_per_memcpy); // round up

            // coord<> unit_coord = idx.template unit_coord<axis, 3>();
            typename GL::dtype *src_ptr =  (typename GL::dtype*)&src[0]  + stride1.x * idx.x + stride1.y * idx.y;
            typename GL::dtype *src2_ptr = (typename GL::dtype*)&src2[0] + stride2.x * idx.x + stride2.y * idx.y;
            uint32_t dst_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&dst.data[0]));
            uint32_t dst2_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&dst2.data[0]));

            // int *indices = s_indices + iter * layout::kv_tile::rows; // offset by the indices in the current batch
            // int *indices = s_indices;
            int *indices = s_indices + (iter & (INDICES_LOAD_INTERVAL - 1)) * layout::kv_tile::rows;

            int laneid = threadIdx.x % N_THREADS;
            // double buffer the indices to hide smem load latency
            //                             v computed from the formula in the inner loop below
            int row_double_buf[2] {indices[0*N_THREADS + laneid / memcpy_per_row], -1};
            
            #pragma unroll
            for(int i = 0; i < total_calls; i++) {
                int load_idx_cur  = (i)   * N_THREADS + laneid;
                int load_idx_next = (i+1) * N_THREADS + laneid;
                int global_row = row_double_buf[i%2];
                int row = load_idx_cur / memcpy_per_row;
                int col = (load_idx_cur*elem_per_memcpy) % dst.cols;
                uint32_t swizzle_offset;
                row_double_buf[(i+1)%2] = indices[(load_idx_next) / memcpy_per_row]; // put the next index in the double buffer

                if constexpr (CACHE_SWIZZLE_OFFSETS) {
                    swizzle_offset = swizzle_offsets[i];
                } else {
                    swizzle_offset = dst.idx((uint32_t) 0, {row, col});
                }

                asm volatile(
                    "cp.async.cg.shared.global [%0], [%1], 16;\n"
                    :: "r"(dst_ptr + swizzle_offset), "l"(&src_ptr[global_row*stride1.z + col])
                    : "memory"
                );

                asm volatile(
                    "cp.async.cg.shared.global [%0], [%1], 16;\n"
                    :: "r"(dst2_ptr + swizzle_offset), "l"(&src2_ptr[global_row*stride2.z + col])
                    : "memory"
                );
            }
        }

        __device__ static inline void setup(producer_setup_args<layout> args) {
            warpgroup::decrease_registers<NUM_PRODUCER_REGISTERS>();

            if (warpgroup::laneid() == 0) {
                if (args.task_iter == 0) {
                    init_semaphore(args.scratch.indices_bar, 1, 0);
                }
                load_indices(args.scratch.indices, args.globals.indices, args.num_iters*KV_TILE_ROWS, 0, {args.common.batch, args.common.head, args.common.seq}, args.globals.indices_stride, args.globals.Q, args.globals.K, args.scratch.indices_bar);
            }
            warpgroup::sync(0); // broadcast the semaphore initialization to all threads

            if (CACHE_SWIZZLE_OFFSETS) {
                cache_swizzle_offsets<2, layout, layout::kv_tile>(args.state.swizzle_offsets);
            }
        }
        __device__ static inline void load(producer_load_args<layout> args) {
            // (1) Wait for previous stage's indices load to finish.
            if ((args.iter & (INDICES_LOAD_INTERVAL - 1)) == 0) {
                kittens::wait(args.scratch.indices_bar, args.state.indices_phase);
                args.state.indices_phase ^= 1;
            }

            // (2) Load K and V.
            load_async_gather<2, CACHE_SWIZZLE_OFFSETS>(args.input.k, args.input.v, args.globals.K, args.globals.V, args.globals.k_stride, args.globals.v_stride, {args.common.batch, args.common.head}, args.iter, args.state.swizzle_offsets, &(args.scratch.indices[0]));

            // (3) Kick off indices load for next stage.
            if (((args.iter + 1) & (INDICES_LOAD_INTERVAL - 1)) == 0 && args.iter + 1 < args.num_iters) {
                if (warpgroup::laneid() == 0) {
                    load_indices(args.scratch.indices, args.globals.indices, args.num_iters*KV_TILE_ROWS, args.iter + 1, {args.common.batch, args.common.head, args.common.seq}, args.globals.indices_stride,  args.globals.Q, args.globals.K, args.scratch.indices_bar);
                }
            }

            // (4) Wait for K and V.
            asm volatile(
                "cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n" 
                :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&args.inputs_arrived)))
                : "memory"
            );
        }
    };
    struct consumer {
        __device__ static inline void setup(consumer_setup_args<layout> args) {
            warpgroup::increase_registers<NUM_CONSUMER_REGISTERS>();
            if((args.common.seq*NUM_WORKERS + warpgroup::groupid())*layout::qo_tile::rows < args.globals.Q.rows) // out of bounds?
                chipmunk::load_strided</*test_for_oob=*/true>(args.scratch.q[warpgroup::groupid()], args.globals.Q,
                                {args.common.batch, args.common.head, args.common.seq*NUM_WORKERS+warpgroup::groupid(), 0}, args.globals.q_stride);
            zero(args.state.o_reg);
            zero(args.state.norm_vec);
            neg_infty(args.state.max_vec);
            warpgroup::sync(warpgroup::groupid());
        }
        __device__ static inline void compute(consumer_compute_args<layout> args) {
            constexpr float TEMPERATURE_SCALE = (D == 128) ? 0.08838834764f*1.44269504089f : 0.125f*1.44269504089f;
            // A = Q @ K.T
            warpgroup::mm_ABt(args.state.att_block, args.scratch.q[warpgroup::groupid()], args.input.k);
            mul(args.state.max_vec_last_scaled, args.state.max_vec, TEMPERATURE_SCALE);
            // copy(args.state.max_vec_last_scaled, args.state.max_vec_scaled);
            warpgroup::mma_async_wait();
            // softmax
            right_fill(args.state.att_block, args.state.att_block, args.globals.K.rows - args.iter*layout::kv_tile::rows, base_types::constants<float>::neg_infty());
            row_max(args.state.max_vec, args.state.att_block, args.state.max_vec); // accumulate onto the max_vec
            mul(args.state.max_vec_scaled, args.state.max_vec, TEMPERATURE_SCALE);
            mul(args.state.att_block, args.state.att_block, TEMPERATURE_SCALE);
            // row_max(args.state.max_vec_scaled, args.state.att_block, args.state.max_vec_scaled); // accumulate onto the max_vec_scaled
            sub_row(args.state.att_block, args.state.att_block, args.state.max_vec_scaled);
            exp2(args.state.att_block, args.state.att_block);
            sub(args.state.max_vec_last_scaled, args.state.max_vec_last_scaled, args.state.max_vec_scaled);
            exp2(args.state.max_vec_last_scaled, args.state.max_vec_last_scaled);
            mul(args.state.norm_vec, args.state.norm_vec, args.state.max_vec_last_scaled);
            row_sum(args.state.norm_vec, args.state.att_block, args.state.norm_vec); // accumulate onto the norm_vec
            mul_row(args.state.o_reg, args.state.o_reg, args.state.max_vec_last_scaled); // normalize o_reg before mma
            copy(args.state.att_block_mma, args.state.att_block); // convert to bf16 for mma
            // O += A @ V
            warpgroup::mma_AB(args.state.o_reg, args.state.att_block_mma, args.input.v);
            warpgroup::mma_async_wait();
            if(laneid() == 0) arrive(args.inputs_finished); // done!
        }
        __device__ static inline void finish(consumer_finish_args<layout> args) {
            if((args.common.seq*NUM_WORKERS+warpgroup::groupid())*64 < args.globals.Q.rows) { // out of bounds?
                div_row(args.state.o_reg, args.state.o_reg, args.state.norm_vec);
                auto &o_smem = reinterpret_cast<typename layout::qo_tile&>(args.scratch.q[warpgroup::groupid()]);
                if constexpr (O_SCALE != 1) {
                    mul(args.state.o_reg, args.state.o_reg, static_cast<float>(O_SCALE));
                }
                warpgroup::store(o_smem, args.state.o_reg);
                warpgroup::sync(warpgroup::groupid());
                if(warpgroup::warpid() == 0)
                    tma::store_add_async(args.globals.O, o_smem, {args.common.batch, args.common.head, args.common.seq*NUM_WORKERS+warpgroup::groupid(), 0});
                tma::store_async_read_wait();
            }
            __syncwarp();
            if(laneid() == 0) arrive(args.finish_finished); // done!
        }
    };
};

#ifdef TORCH_COMPILE
#include "pyutils/torch_helpers.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <iostream>

namespace chipmunk {
void csp_attn(at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor o, at::Tensor indices, at::Tensor indices_counts, int64_t o_scale)
{
    using ker_template = attn_fwd_template<128, 1>;

    auto batch    = q.size(0);
    auto seq_len  = q.size(2); 
    auto kseq_len  = k.size(2); 
    auto head_dim = q.size(3); 
    auto qo_heads = q.size(1);
    auto kv_heads = k.size(1);
    auto num_indices_groups = (seq_len+(ker_template::NUM_WORKERS * ker_template::layout::qo_tile::rows)-1) / (ker_template::NUM_WORKERS * ker_template::layout::qo_tile::rows);

    TORCH_CHECK(o_scale == 1 || o_scale == -1, "o_scale must be 1 or -1");
    TORCH_CHECK(indices_counts.is_contiguous(), "Indices counts must be contiguous");

    TORCH_CHECK(indices_counts.dim() == 3, "Indices counts must be a 3D tensor");
    TORCH_CHECK(indices.dim() == 4, "Indices must be a 4D tensor");
    // Make sure that indices and indices_counts are ints
    TORCH_CHECK(indices.dtype() == torch::kInt32, "Indices must be a 32-bit integer tensor");
    TORCH_CHECK(indices_counts.dtype() == torch::kInt32, "Indices counts must be a 32-bit integer tensor");

    // check to see that these dimensions match for all inputs
    TORCH_CHECK(q.size(0) == batch, "Q batch dimension - idx 0 - must match for all inputs");
    TORCH_CHECK(k.size(0) == batch, "K batch dimension - idx 0 - must match for all inputs");
    TORCH_CHECK(v.size(0) == batch, "V batch dimension - idx 0 - must match for all inputs");
    TORCH_CHECK(o.size(0) == batch, "O batch dimension - idx 0 - must match for all inputs");
    TORCH_CHECK(indices.size(0) == batch, "Indices batch dimension - idx 0 - must match for all inputs");
    TORCH_CHECK(indices_counts.size(0) == batch, "Indices counts batch dimension - idx 0 - must match for all inputs");

    TORCH_CHECK(q.size(2) == seq_len, "Q sequence length dimension - idx 2 - must match for all inputs");
    TORCH_CHECK(k.size(2) == kseq_len, "K sequence length dimension - idx 2 - must match for all inputs");
    TORCH_CHECK(v.size(2) == kseq_len, "V sequence length dimension - idx 2 - must match for all inputs");
    TORCH_CHECK(o.size(2) == seq_len, "O sequence length dimension - idx 2 - must match for all inputs");
    TORCH_CHECK(indices.size(2) == num_indices_groups, "Indices query group dimension - idx 2 - must match for all inputs");
    TORCH_CHECK(indices_counts.size(2) == num_indices_groups, "Indices counts query group dimension - idx 2 - must match for all inputs");

    TORCH_CHECK(q.size(3) == head_dim, "Q head dimension - idx 3 - must match for all non-vector inputs");
    TORCH_CHECK(k.size(3) == head_dim, "K head dimension - idx 3 - must match for all non-vector inputs");
    TORCH_CHECK(v.size(3) == head_dim, "V head dimension - idx 3 - must match for all non-vector inputs");
    TORCH_CHECK(o.size(3) == head_dim, "O head dimension - idx 3 - must match for all non-vector inputs");

    TORCH_CHECK(indices.size(3) == seq_len, "Indices sequence length dimension - idx 3 - must match for all inputs");
    // cp.async.bulk.tensor requires 16-byte alignment of gmem operand - indices must be a multiple of 4
    TORCH_CHECK(indices.stride(2) * sizeof(int) % 16 == 0, "Indices stride must divide by 16 bytes (4 int32s) evenly. Either make indices non-contiguous or use a sequence length that's a multiple of 4.");

    TORCH_CHECK(qo_heads == kv_heads, "QO heads must be equal to KV heads");
    TORCH_CHECK(q.size(1) == qo_heads, "QO head dimension - idx 1 - must match for all inputs");
    TORCH_CHECK(k.size(1) == kv_heads, "KV head dimension - idx 1 - must match for all inputs");
    TORCH_CHECK(v.size(1) == kv_heads, "KV head dimension - idx 1 - must match for all inputs"); 
    TORCH_CHECK(o.size(1) == qo_heads, "O head dimension - idx 1 - must match for all inputs");
    TORCH_CHECK(indices.size(1) == qo_heads, "Indices QO head dimension - idx 1 - must match for all inputs");
    TORCH_CHECK(indices_counts.size(1) == qo_heads, "Indices counts QO head dimension - idx 1 - must match for all inputs");

    auto hr = qo_heads / kv_heads;

    c10::BFloat16* q_ptr = q.data_ptr<c10::BFloat16>();
    c10::BFloat16* k_ptr = k.data_ptr<c10::BFloat16>();
    c10::BFloat16* v_ptr = v.data_ptr<c10::BFloat16>();
    int* d_indices = indices.data_ptr<int>();
    int* d_indices_counts = indices_counts.data_ptr<int>();

    bf16*  d_q = reinterpret_cast<bf16*>(q_ptr);
    bf16*  d_k = reinterpret_cast<bf16*>(k_ptr);
    bf16*  d_v = reinterpret_cast<bf16*>(v_ptr);
    
    bf16*  o_ptr = reinterpret_cast<bf16*>(o.data_ptr<c10::BFloat16>());
    bf16*  d_o   = reinterpret_cast<bf16*>(o_ptr);
    // TORCH_CHECK(seq_len % (ker_template::NUM_WORKERS*ker_template::layout::qo_tile::rows) == 0, "sequence length must be divisible by num_workers * qo_tile::rows");
    if (head_dim != 128) {
        throw std::runtime_error("Head dimension must be 128");
    }

    ker_template::layout::qo_global Qg(d_q, static_cast<unsigned int>(batch), static_cast<unsigned int>(qo_heads), static_cast<unsigned int>(seq_len), nullptr);
    ker_template::layout::kv_global Kg(d_k, static_cast<unsigned int>(batch), static_cast<unsigned int>(kv_heads), static_cast<unsigned int>(kseq_len), nullptr);
    ker_template::layout::kv_global Vg(d_v, static_cast<unsigned int>(batch), static_cast<unsigned int>(kv_heads), static_cast<unsigned int>(kseq_len), nullptr);
    ker_template::layout::qo_global Og(d_o, static_cast<unsigned int>(batch), static_cast<unsigned int>(qo_heads), static_cast<unsigned int>(seq_len), nullptr);

    chipmunk::create_tensor_map_with_strides<ker_template::layout::qo_tile, 2>(&Qg.tma_descs.tma_desc, d_q, batch, qo_heads, seq_len, head_dim, q.stride(0), q.stride(1), q.stride(2));
    chipmunk::create_tensor_map_with_strides<ker_template::layout::kv_tile, 2>(&Kg.tma_descs.tma_desc, d_k, batch, kv_heads, kseq_len, head_dim, k.stride(0), k.stride(1), k.stride(2));
    chipmunk::create_tensor_map_with_strides<ker_template::layout::kv_tile, 2>(&Vg.tma_descs.tma_desc, d_v, batch, kv_heads, kseq_len, head_dim, v.stride(0), v.stride(1), v.stride(2));
    chipmunk::create_tensor_map_with_strides<ker_template::layout::qo_tile, 2>(&Og.tma_descs.tma_desc, d_o, batch, qo_heads, seq_len, head_dim, o.stride(0), o.stride(1), o.stride(2));
    
    ker_template::layout::globals globals {
        Og, Qg, Kg, Vg, d_indices, d_indices_counts,
        {q.stride(0), q.stride(1), q.stride(2)},
        {k.stride(0), k.stride(1), k.stride(2)},
        {v.stride(0), v.stride(1), v.stride(2)},
        {indices.stride(0), indices.stride(1), indices.stride(2)}
    };

    auto mem_size = kittens::MAX_SHARED_MEMORY - 2000;
    dim3 grid(132, 1, 1);
    constexpr int BLOCK_SIZE = prototype::detail::NUM_THREADS_v<ker_template>;

    if (o_scale == -1) {
        cudaFuncSetAttribute(
            prototype::lcf::kernel<attn_fwd_template<128, -1>>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            mem_size
        );
        prototype::lcf::kernel<attn_fwd_template<128, -1>><<<grid, BLOCK_SIZE, mem_size>>>(globals);
    } else {
        cudaFuncSetAttribute(
            prototype::lcf::kernel<attn_fwd_template<128, 1>>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            mem_size
        );
        prototype::lcf::kernel<attn_fwd_template<128, 1>><<<grid, BLOCK_SIZE, mem_size>>>(globals);
    }

    CHECK_CUDA_ERROR(cudaGetLastError());
}
}

#else
#include "h100_lcf_harness.impl"
#endif
