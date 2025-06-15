#include "kittens.cuh"
#include <cooperative_groups.h>
#include <iostream>
#include "../common/all.cuh"


constexpr int CONSUMER_WARPGROUPS = (3); 
constexpr int CONSUMER_WARPS = CONSUMER_WARPGROUPS*kittens::WARPGROUP_WARPS;
constexpr int CONSUMER_REGISTERS  = (160);
// constexpr int CONSUMER_WARPGROUPS = (2); 
// constexpr int CONSUMER_REGISTERS  = (192);
constexpr int PRODUCER_WARPGROUPS = (1); 
constexpr int NUM_WARPGROUPS      = (CONSUMER_WARPGROUPS+PRODUCER_WARPGROUPS); 
constexpr int NUM_WORKERS         = (NUM_WARPGROUPS*kittens::WARPGROUP_WARPS); 
constexpr int COLSUM_STORE_INTERVAL = 16;

constexpr bool FUSE_REDUCE = true;
constexpr int CS_SMEM_SIZE = FUSE_REDUCE ? COLSUM_STORE_INTERVAL : CONSUMER_WARPGROUPS*4*COLSUM_STORE_INTERVAL;

using namespace kittens;
namespace cg = cooperative_groups;

template<int D> struct cs_attend_ker_tile_dims {};
template<> struct cs_attend_ker_tile_dims<64> {
    constexpr static int tile_width = (64);
    constexpr static int qo_height  = (4*16);
    constexpr static int kv_height  = (8*16);
    constexpr static int stages     = (4); 
};
template<> struct cs_attend_ker_tile_dims<128> {
    constexpr static int tile_width = (128);
    constexpr static int qo_height  = (4*16);
    constexpr static int kv_height  = (8*16);
    constexpr static int stages     = (2); 
};

template<int D> struct cs_globals {
    using q_tile    =         st_bf<cs_attend_ker_tile_dims<D>::qo_height, cs_attend_ker_tile_dims<D>::tile_width>;
    using k_tile    =         st_bf<cs_attend_ker_tile_dims<D>::kv_height, cs_attend_ker_tile_dims<D>::tile_width>;
    using v_tile    =         st_bf<cs_attend_ker_tile_dims<D>::kv_height, cs_attend_ker_tile_dims<D>::tile_width>;
    using p_col_vec = col_vec<st_fl<cs_attend_ker_tile_dims<D>::qo_height, cs_attend_ker_tile_dims<D>::tile_width>>;
    using l_col_vec = col_vec<st_fl<cs_attend_ker_tile_dims<D>::qo_height, cs_attend_ker_tile_dims<D>::tile_width>>;
    using o_tile    =         st_bf<cs_attend_ker_tile_dims<D>::qo_height, cs_attend_ker_tile_dims<D>::tile_width>;

    using q_gl = gl<bf16,  -1, -1, -1, -1, q_tile>;
    using k_gl = gl<bf16,  -1, -1, -1, -1, k_tile>;
    using v_gl = gl<bf16,  -1, -1, -1, -1, v_tile>;
    using o_gl = gl<bf16,  -1, -1, -1, -1, o_tile>;
    using p_gl = gl<float, -1, -1, -1, -1, p_col_vec>;
    using l_gl = gl<float, -1, -1, -1, -1, l_col_vec>;
    using c_gl = gl<bf16, -1, -1, -1, -1>;

    q_gl q;
    k_gl k;
    v_gl v;
    p_gl p;
    l_gl l;
    o_gl o;
    c_gl c;

    const int kN; 
    const int hr;
};

__device__ static inline void store_colsum(bf16 *g_indices, bf16 *s_indices, int num_bytes_to_copy) {
    uint64_t g_indices_ptr = static_cast<uint64_t>(__cvta_generic_to_global(g_indices));
    uint32_t s_indices_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(s_indices));

    asm volatile ("fence.proxy.async.shared::cta;\n" ::: "memory");
    asm volatile(
        "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;\n"
        :: "l"(g_indices_ptr), "r"(s_indices_ptr), "r"(num_bytes_to_copy)
        : "memory"
    );
    tma::store_commit_group();
}


template<int D, bool is_causal>
__global__  __launch_bounds__((NUM_WORKERS)*kittens::WARP_THREADS, 1)
void cs_attend_ker(const __grid_constant__ cs_globals<D> g) {
    extern __shared__ int __shm[]; 
    tma_swizzle_allocator al((int*)&__shm[0]);
    int warpid = kittens::warpid(), warpgroupid = warpid/kittens::WARPGROUP_WARPS;

    using K = cs_attend_ker_tile_dims<D>;

    using q_tile    =         st_bf<K::qo_height, K::tile_width>;
    using k_tile    =         st_bf<K::kv_height, K::tile_width>;
    using v_tile    =         st_bf<K::kv_height, K::tile_width>;
    using p_col_vec = col_vec<st_fl<K::qo_height, K::tile_width>>;
    using l_col_vec = col_vec<st_fl<K::qo_height, K::tile_width>>;
    using m_col_vec = col_vec<st_fl<K::qo_height, K::tile_width>>;
    using c_row_vec = row_vec<st_bf<K::qo_height, K::kv_height>>;
    using o_tile    =         st_bf<K::qo_height, K::tile_width>;
    
    q_tile    (&q_smem)[CONSUMER_WARPGROUPS] = al.allocate<q_tile, CONSUMER_WARPGROUPS>();
    k_tile    (&k_smem)[K::stages]           = al.allocate<k_tile, K::stages          >();
    v_tile    (&v_smem)[K::stages]           = al.allocate<v_tile, K::stages          >();
    p_col_vec (&p_smem)[CONSUMER_WARPGROUPS] = al.allocate<p_col_vec, CONSUMER_WARPGROUPS>();
    l_col_vec (&l_smem)[CONSUMER_WARPGROUPS] = al.allocate<l_col_vec, CONSUMER_WARPGROUPS>();
    c_row_vec (&c_smem)[CS_SMEM_SIZE] = al.allocate<c_row_vec, CS_SMEM_SIZE>();
    auto      (*o_smem)                      = reinterpret_cast<o_tile(*)>(q_smem);
    
    int kv_blocks   = (g.kN + K::kv_height - 1) / (K::kv_height);
    int kv_head_idx = blockIdx.y / g.hr;
    int seq_idx     = blockIdx.x * CONSUMER_WARPGROUPS; 
    int seq_group   = blockIdx.x;

    __shared__ kittens::semaphore qsmem_semaphore, psmem_semaphore, k_smem_arrived[K::stages], v_smem_arrived[K::stages], compute_done[K::stages];
    if (threadIdx.x == 0) { 
        init_semaphore(qsmem_semaphore, 0, 1); 
        init_semaphore(psmem_semaphore, 0, 1);
        for(int j = 0; j < K::stages; j++) {
            init_semaphore(k_smem_arrived[j], 0, 1); 
            init_semaphore(v_smem_arrived[j], 0, 1); 
            init_semaphore(compute_done[j], CONSUMER_WARPGROUPS, 0); 
        }

        tma::expect_bytes(qsmem_semaphore, sizeof(q_smem));

        for (int wg = 0; wg < CONSUMER_WARPGROUPS; wg++) {
            coord<q_tile> q_tile_idx = {blockIdx.z, blockIdx.y, (seq_idx) + wg, 0};
            tma::load_async(q_smem[wg], g.q, q_tile_idx, qsmem_semaphore);
        }

        tma::expect_bytes(psmem_semaphore, sizeof(p_smem));

        for (int wg = 0; wg < CONSUMER_WARPGROUPS; wg++) {
            coord<p_col_vec> p_tile_idx = {blockIdx.z, blockIdx.y, 0, (seq_idx) + wg};
            tma::load_async(p_smem[wg], g.p, p_tile_idx, psmem_semaphore);
        }

        for (int j = 0; j < K::stages - 1; j++) {
            coord<k_tile> kv_tile_idx = {blockIdx.z, kv_head_idx, j, 0};
            tma::expect_bytes(k_smem_arrived[j], sizeof(k_tile));
            tma::load_async(k_smem[j], g.k, kv_tile_idx, k_smem_arrived[j]);
            tma::expect_bytes(v_smem_arrived[j], sizeof(v_tile));
            tma::load_async(v_smem[j], g.v, kv_tile_idx, v_smem_arrived[j]);
        }
    }
    if (FUSE_REDUCE) {
        zero(c_smem[1]);
        zero(c_smem[2]);
        zero(c_smem[3]);
    }
    __syncthreads(); 

    int pipe_idx = K::stages - 1; 
    
    if(warpgroupid == NUM_WARPGROUPS-1) {
        warpgroup::decrease_registers<32>();      
        
        int kv_iters; 
        if constexpr (is_causal) {
            kv_iters = (seq_idx * (K::qo_height/kittens::TILE_ROW_DIM<bf16>)) - 1 + (CONSUMER_WARPGROUPS * (K::qo_height/kittens::TILE_ROW_DIM<bf16>)); 
            kv_iters = ((kv_iters / (K::kv_height/kittens::TILE_ROW_DIM<bf16>)) == 0) ? (0) : ((kv_iters / (K::kv_height/kittens::TILE_ROW_DIM<bf16>)) - 1);
        }
        else { kv_iters = kv_blocks-2; }

        if(warpid == NUM_WORKERS-4) {
            for (auto kv_idx = pipe_idx - 1; kv_idx <= kv_iters; kv_idx++) {
                coord<k_tile> kv_tile_idx = {blockIdx.z, kv_head_idx, kv_idx + 1, 0};
                tma::expect_bytes(k_smem_arrived[(kv_idx+1)%K::stages], sizeof(k_tile));
                tma::load_async(k_smem[(kv_idx+1)%K::stages], g.k, kv_tile_idx, k_smem_arrived[(kv_idx+1)%K::stages]);
                tma::expect_bytes(v_smem_arrived[(kv_idx+1)%K::stages], sizeof(v_tile));
                tma::load_async(v_smem[(kv_idx+1)%K::stages], g.v, kv_tile_idx, v_smem_arrived[(kv_idx+1)%K::stages]);

                if (FUSE_REDUCE) {
                    if (kv_idx > 3) {
                        // store from 2 iterations ago (input pipeline depth) to avoid consumer sync
                        bf16* dst = &g.c[{blockIdx.z, blockIdx.y, seq_group, (kv_idx - 4) * K::kv_height}];
                        bf16* src;
                        src = &c_smem[(kv_idx - 3)%COLSUM_STORE_INTERVAL][0];
                        store_colsum(dst, src, K::kv_height * sizeof(bf16));
                    }
                }

                kittens::wait(compute_done[(kv_idx)%K::stages], (kv_idx/K::stages)%2);
            }
        }
    }
    else {
        warpgroup::increase_registers<CONSUMER_REGISTERS>();

        rt_fl<16, K::kv_height>  att_block;
        rt_bf<16, K::kv_height>  att_block_mma;
        rt_fl<16, K::tile_width> o_reg;
        
        col_vec<rt_fl<16, K::kv_height>> max_vec, norm_vec, max_vec_last_scaled, max_vec_scaled, prev_norm_vec;
        col_vec<rt_bf<16, K::kv_height>> max_vec_last_scaled_bf, max_vec_scaled_bf, prev_norm_vec_bf, prev_max_vec_bf;
        row_vec<rt_bf<16, K::kv_height>> c_reg;

        neg_infty(max_vec);
        zero(norm_vec);
        zero(o_reg);
        zero(c_reg);

        int kv_iters; 
        if constexpr (is_causal) {
            kv_iters = (seq_idx * 4) - 1 + (CONSUMER_WARPGROUPS * 4);
            kv_iters = (kv_iters/8);
        }
        else { kv_iters = kv_blocks - 1; }

        kittens::wait(qsmem_semaphore, 0);
        kittens::wait(psmem_semaphore, 0);
        warpgroup::load(prev_norm_vec, p_smem[warpgroupid]);

        for (auto kv_idx = 0; kv_idx <= kv_iters; kv_idx++) {
        
            kittens::wait(k_smem_arrived[(kv_idx)%K::stages], (kv_idx/K::stages)%2);
            warpgroup::mm_ABt(att_block, q_smem[warpgroupid], k_smem[(kv_idx)%K::stages]);
            if (FUSE_REDUCE && warpgroupid == 0) {
                zero(c_smem[(kv_idx + 4) % COLSUM_STORE_INTERVAL]);
            }
            
            copy(max_vec_last_scaled, max_vec);
            if constexpr (D == 64) { mul(max_vec_last_scaled, max_vec_last_scaled, 1.44269504089f*0.125f); }
            else                   { mul(max_vec_last_scaled, max_vec_last_scaled, 1.44269504089f*0.08838834764f); }
            
            warpgroup::mma_async_wait();
            right_fill(att_block, att_block, g.k.rows - kv_idx*K::kv_height, base_types::constants<float>::neg_infty());

            if constexpr (is_causal) {
                const int q_blk = (seq_idx * (K::qo_height/kittens::TILE_ROW_DIM<bf16>)) + warpid; 
                      int k_blk = (kv_idx * (K::kv_height/kittens::TILE_ROW_DIM<bf16>)); 

                #pragma unroll
                for(int _ = 0; k_blk == (kv_iters-1)*(K::kv_height/kittens::TILE_ROW_DIM<bf16>) || k_blk == (kv_iters)*(K::kv_height/kittens::TILE_ROW_DIM<bf16>); k_blk+=10000) {
                    #pragma unroll
                    for (auto j = 0; j < (K::kv_height/kittens::TILE_ROW_DIM<bf16>); j++) {
                        auto k_idx = k_blk + j;
                        auto &attn_subtile = reinterpret_cast<rt_fl<16, 16>&>(att_block.tiles[0][j]);

                        if      (k_idx >  q_blk) { neg_infty  (attn_subtile); }
                        else if (k_idx == q_blk) { make_causal(attn_subtile, attn_subtile, kittens::base_types::constants<float>::neg_infty()); }
                        __syncwarp();
                    }
                }
            }

            row_max(max_vec, att_block, max_vec);
            
            if constexpr (D == 64) { 
                mul(att_block, att_block,    1.44269504089f*0.125f); 
                mul(max_vec_scaled, max_vec, 1.44269504089f*0.125f);
            }
            else                   { 
                mul(att_block, att_block,    1.44269504089f*0.08838834764f); 
                mul(max_vec_scaled, max_vec, 1.44269504089f*0.08838834764f);
            }

            sub_row(att_block, att_block, max_vec_scaled);
            exp2(att_block, att_block);
            sub(max_vec_last_scaled, max_vec_last_scaled, max_vec_scaled);
            exp2(max_vec_last_scaled,       max_vec_last_scaled);
            mul(norm_vec,            norm_vec,     max_vec_last_scaled);
            row_sum(norm_vec,  att_block, norm_vec);
            copy(att_block_mma, att_block); 
            mul_row(o_reg, o_reg, max_vec_last_scaled); 

            kittens::wait(v_smem_arrived[(kv_idx)%K::stages], (kv_idx/K::stages)%2); 

            warpgroup::mma_AB(o_reg, att_block_mma, v_smem[(kv_idx)%K::stages]);

            // colsum //
            // e^(x - max_vec_scaled) * e^(max_vec_scaled) * 1 / (e^(prev_m) * prev_l)
            exp2(max_vec_last_scaled, max_vec_scaled);
            mul(max_vec_last_scaled, max_vec_last_scaled, prev_norm_vec);
            copy(max_vec_scaled_bf, max_vec_last_scaled);
            mul_row(att_block_mma, att_block_mma, max_vec_scaled_bf);

            col_sum(c_reg, att_block_mma);
            if constexpr (FUSE_REDUCE) {
                chipmunk::store_add(c_smem[(kv_idx + 1) % COLSUM_STORE_INTERVAL], c_reg);
            }
            else {
                store(c_smem[(kv_idx%COLSUM_STORE_INTERVAL) * CONSUMER_WARPS + warpid], c_reg);
            }

            if (FUSE_REDUCE) {
                // if (kv_idx > 1) {
                //     // store from 2 iterations ago (input pipeline depth) to avoid consumer sync
                //     bf16* dst = &g.c[{blockIdx.z, blockIdx.y, seq_group, (kv_idx - 2) * K::kv_height}];
                //     bf16* src;
                //     src = &c_smem[(kv_idx - 1)%COLSUM_STORE_INTERVAL][0];
                //     if (warpid == 0) {
                //         // store_colsum(dst, src, K::kv_height * sizeof(bf16));
                //     }
                // }
            }
            else {
                int idx = seq_idx * 4 + warpid;
                bf16* dst = &g.c[{blockIdx.z, blockIdx.y, idx, kv_idx * K::kv_height}];
                bf16* src;
                src = &c_smem[(kv_idx%COLSUM_STORE_INTERVAL) * CONSUMER_WARPS + warpid][0];
                store_colsum(dst, src, K::kv_height * sizeof(bf16));
            }
            ////////////

            warpgroup::mma_async_wait();
            if(warpgroup::laneid() == 0) arrive(compute_done[(kv_idx)%K::stages], 1);
        }

        div_row(o_reg, o_reg, norm_vec);
        warpgroup::store(o_smem[warpgroupid], o_reg); 
        warpgroup::sync(warpgroupid+4);

        if (warpid % 4 == 0) {
            coord<o_tile> o_tile_idx = {blockIdx.z, blockIdx.y, (seq_idx) + warpgroupid, 0};
            tma::store_async(g.o, o_smem[warpgroupid], o_tile_idx);
        }

        // store as single constant
        exp2(max_vec_scaled, max_vec_scaled);
        mul(max_vec_scaled, max_vec_scaled, norm_vec);
        unary_op<chipmunk::base_ops::rcp>(max_vec_scaled, max_vec_scaled);
        warpgroup::store(l_smem[warpgroupid], max_vec_scaled);
        warpgroup::sync(warpgroupid+4);

        if (FUSE_REDUCE) {
            group<12>::sync(14);
            for (int i = 4; i >= 0; i--) {
                bf16* dst = &g.c[{blockIdx.z, blockIdx.y, seq_group, (kv_iters - i) * K::kv_height}];
                bf16* src;
                // store from last 4 iterations (input pipeline depth * 2)
                src = &c_smem[(kv_iters - i + 1) % COLSUM_STORE_INTERVAL][0];
                if (warpid == 0) {
                    store_colsum(dst, src, K::kv_height * sizeof(bf16));
                }
            }
        }

        if (warpid % 4 == 0) {
            coord<l_col_vec> tile_idx = {blockIdx.z, blockIdx.y, 0, (seq_idx) + warpgroupid};
            tma::store_async(g.l, l_smem[warpgroupid], tile_idx);
        }
        tma::store_async_wait();
    }
}


#ifdef TORCH_COMPILE

#include "pyutils/torch_helpers.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <iostream>

namespace chipmunk {
std::vector<at::Tensor> 
dense_colsum_attn(at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor p)
{
    // CHECK_INPUT(q);
    // CHECK_INPUT(k);
    // CHECK_INPUT(v);
    // CHECK_INPUT(p);

    auto batch    = q.size(0);
    auto seq_len  = q.size(2); 
    auto kseq_len = k.size(2);
    auto head_dim = q.size(3); 
    auto is_causal = false; 
    auto qo_heads = q.size(1);
    auto kv_heads = k.size(1);

    // check to see that these dimensions match for all inputs
    TORCH_CHECK(q.size(0) == batch, "Q batch dimension - idx 0 - must match for all inputs");
    TORCH_CHECK(k.size(0) == batch, "K batch dimension - idx 0 - must match for all inputs");
    TORCH_CHECK(v.size(0) == batch, "V batch dimension - idx 0 - must match for all inputs");
    TORCH_CHECK(p.size(0) == batch, "P batch dimension - idx 0 - must match for all inputs");

    TORCH_CHECK(q.size(2) == seq_len, "Q sequence length dimension - idx 2 - must match for all inputs");
    TORCH_CHECK(p.size(2) == seq_len, "P sequence length dimension - idx 2 - must match for all inputs");
    TORCH_CHECK(k.size(2) == kseq_len, "K sequence length dimension - idx 2 - must match for all inputs");
    TORCH_CHECK(v.size(2) == kseq_len, "V sequence length dimension - idx 2 - must match for all inputs");

    TORCH_CHECK(q.size(3) == head_dim, "Q head dimension - idx 3 - must match for all non-vector inputs");
    TORCH_CHECK(k.size(3) == head_dim, "K head dimension - idx 3 - must match for all non-vector inputs");
    TORCH_CHECK(v.size(3) == head_dim, "V head dimension - idx 3 - must match for all non-vector inputs");
    TORCH_CHECK(p.size(3) == 1, "P unsqueezed dimension - idx 3 - must be 1");

    TORCH_CHECK(qo_heads >= kv_heads, "QO heads must be greater than or equal to KV heads");
    TORCH_CHECK(qo_heads % kv_heads == 0, "QO heads must be divisible by KV heads");
    TORCH_CHECK(q.size(1) == qo_heads, "QO head dimension - idx 1 - must match for all inputs");
    TORCH_CHECK(k.size(1) == kv_heads, "KV head dimension - idx 1 - must match for all inputs");
    TORCH_CHECK(v.size(1) == kv_heads, "KV head dimension - idx 1 - must match for all inputs");  
    TORCH_CHECK(p.size(1) == kv_heads, "P head dimension - idx 1 - must match for all inputs");

    TORCH_CHECK(p.stride(1) % (16/sizeof(float)) == 0, "P must have a stride multiple of 4 for TMA alignment requirements.");

    auto hr = qo_heads / kv_heads;
    auto seq_downsample = FUSE_REDUCE ? 192 : 16;
    auto qg = (seq_len+seq_downsample-1) / seq_downsample;

    c10::BFloat16* q_ptr = q.data_ptr<c10::BFloat16>();
    c10::BFloat16* k_ptr = k.data_ptr<c10::BFloat16>();
    c10::BFloat16* v_ptr = v.data_ptr<c10::BFloat16>();
    float* p_ptr = p.data_ptr<float>();

    bf16*  d_q = reinterpret_cast<bf16*>(q_ptr);
    bf16*  d_k = reinterpret_cast<bf16*>(k_ptr);
    bf16*  d_v = reinterpret_cast<bf16*>(v_ptr);
    float* d_p = reinterpret_cast<float*>(p_ptr);

    // all strides need to be a multiple of 16b
    // technically we could pad to 4xfp32 and 8xbf16 but that's too much work, instead let's assume smallest size
    constexpr int SEQUENCE_STRIDE_PADDING = 16 / sizeof(bf16); 
    const uint seq_len_padded = static_cast<uint>((seq_len + SEQUENCE_STRIDE_PADDING - 1) / SEQUENCE_STRIDE_PADDING * SEQUENCE_STRIDE_PADDING);

    // for the returned outputs
    at::Tensor o     = torch::empty({static_cast<const uint>(batch), 
                                        static_cast<const uint>(qo_heads), 
                                        static_cast<const uint>(seq_len), 
                                        static_cast<const uint>(head_dim)}, v.options());
    
    at::Tensor cs = torch::empty_strided(
        {static_cast<const uint>(batch), static_cast<const uint>(qo_heads), static_cast<const uint>(qg), static_cast<const uint>(seq_len)}, 
        {static_cast<const uint>(qo_heads*qg*seq_len_padded), static_cast<const uint>(qg*seq_len_padded), static_cast<const uint>(seq_len_padded), static_cast<const uint>(1)}, 
        v.options()
    );

    at::Tensor l_vec = torch::empty_strided(
        {static_cast<const uint>(batch), static_cast<const uint>(qo_heads), static_cast<const uint>(seq_len), static_cast<const uint>(1)}, 
        {static_cast<const uint>(qo_heads*seq_len_padded), static_cast<const uint>(seq_len_padded), static_cast<const uint>(1), static_cast<const uint>(0)}, 
        torch::TensorOptions().dtype(torch::kFloat).device(q.device())
    );
        
    bf16*  o_ptr = reinterpret_cast<bf16*>(o.data_ptr<c10::BFloat16>());
    bf16*  d_o   = reinterpret_cast<bf16*>(o_ptr);

    bf16*  cs_ptr = reinterpret_cast<bf16*>(cs.data_ptr<c10::BFloat16>());
    bf16*  d_cs   = reinterpret_cast<bf16*>(cs_ptr);

    float* l_ptr = reinterpret_cast<float*>(l_vec.data_ptr<float>());
    float* d_l   = reinterpret_cast<float*>(l_ptr);

    auto stream = at::cuda::getCurrentCUDAStream().stream(); 
    if (head_dim != 128) {
        throw std::runtime_error("Head dimension must be 128");
    }
    
    using q_tile    =         st_bf<cs_attend_ker_tile_dims<128>::qo_height, cs_attend_ker_tile_dims<128>::tile_width>;
    using k_tile    =         st_bf<cs_attend_ker_tile_dims<128>::kv_height, cs_attend_ker_tile_dims<128>::tile_width>;
    using v_tile    =         st_bf<cs_attend_ker_tile_dims<128>::kv_height, cs_attend_ker_tile_dims<128>::tile_width>;
    using p_col_vec = col_vec<st_fl<cs_attend_ker_tile_dims<128>::qo_height, cs_attend_ker_tile_dims<128>::tile_width>>;
    using l_col_vec = col_vec<st_fl<cs_attend_ker_tile_dims<128>::qo_height, cs_attend_ker_tile_dims<128>::tile_width>>;
    using c_row_vec = row_vec<st_bf<cs_attend_ker_tile_dims<128>::qo_height, cs_attend_ker_tile_dims<128>::tile_width>>;
    using o_tile    =         st_bf<cs_attend_ker_tile_dims<128>::qo_height, cs_attend_ker_tile_dims<128>::tile_width>;

    using q_global = gl<bf16,  -1, -1, -1, -1, q_tile>;
    using k_global = gl<bf16,  -1, -1, -1, -1, k_tile>;
    using v_global = gl<bf16,  -1, -1, -1, -1, v_tile>;
    using p_global = gl<float, -1, -1, -1, -1, p_col_vec>;
    using l_global = gl<float, -1, -1, -1, -1, l_col_vec>;
    using c_global = gl<bf16,  -1, -1, -1, -1>;
    using o_global = gl<bf16,  -1, -1, -1, -1, o_tile>;

    using globals      = cs_globals<128>;

    q_global qg_arg{d_q, static_cast<unsigned int>(batch), static_cast<unsigned int>(qo_heads), static_cast<unsigned int>(seq_len), 128U};
    k_global kg_arg{d_k, static_cast<unsigned int>(batch), static_cast<unsigned int>(kv_heads), static_cast<unsigned int>(kseq_len), 128U};
    v_global vg_arg{d_v, static_cast<unsigned int>(batch), static_cast<unsigned int>(kv_heads), static_cast<unsigned int>(kseq_len), 128U};
    p_global pg_arg{d_p, static_cast<unsigned int>(batch), static_cast<unsigned int>(qo_heads), 1U,   static_cast<unsigned int>(seq_len_padded)};
    l_global lg_arg{d_l, static_cast<unsigned int>(batch), static_cast<unsigned int>(qo_heads), 1U,   static_cast<unsigned int>(seq_len_padded)};
    c_global cg_arg{d_cs, static_cast<unsigned int>(batch), static_cast<unsigned int>(qo_heads), static_cast<unsigned int>(qg), static_cast<unsigned int>(seq_len_padded)};
    o_global og_arg{d_o, static_cast<unsigned int>(batch), static_cast<unsigned int>(qo_heads), static_cast<unsigned int>(seq_len), 128U};

    chipmunk::create_tensor_map_with_strides<q_tile, 2>(&qg_arg.tma_descs.tma_desc, d_q, batch, qo_heads, seq_len, head_dim, q.stride(0), q.stride(1), q.stride(2));
    chipmunk::create_tensor_map_with_strides<k_tile, 2>(&kg_arg.tma_descs.tma_desc, d_k, batch, kv_heads, kseq_len, head_dim, k.stride(0), k.stride(1), k.stride(2));
    chipmunk::create_tensor_map_with_strides<v_tile, 2>(&vg_arg.tma_descs.tma_desc, d_v, batch, kv_heads, kseq_len, head_dim, v.stride(0), v.stride(1), v.stride(2));

    chipmunk::create_tensor_map_with_strides<p_col_vec, -1>(&pg_arg.tma_descs.tma_desc, d_p, batch, qo_heads, 1U, seq_len, p.stride(0), p.stride(1), p.stride(3));
    chipmunk::create_tensor_map_with_strides<l_col_vec, -1>(&lg_arg.tma_descs.tma_desc, d_l, batch, qo_heads, 1U, seq_len, l_vec.stride(0), l_vec.stride(1), l_vec.stride(3));


    globals g{qg_arg, kg_arg, vg_arg, pg_arg, lg_arg, og_arg, cg_arg, static_cast<int>(kseq_len), static_cast<int>(hr)};

    auto mem_size = kittens::MAX_SHARED_MEMORY;
    auto threads  = NUM_WORKERS * kittens::WARP_THREADS;

    // TORCH_CHECK(seq_len % (CONSUMER_WARPGROUPS*kittens::TILE_DIM*4) == 0, "sequence length must be divisible by 192");
    dim3 grid((seq_len+(CONSUMER_WARPGROUPS*kittens::TILE_ROW_DIM<bf16>*4)-1)/(CONSUMER_WARPGROUPS*kittens::TILE_ROW_DIM<bf16>*4), qo_heads, batch);

    if (is_causal) {
        cudaFuncSetAttribute(
            cs_attend_ker<128, true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            mem_size
        );

        cs_attend_ker<128, true><<<grid, (32*NUM_WORKERS), mem_size, stream>>>(g);
    }
    else {
        cudaFuncSetAttribute(
            cs_attend_ker<128, false>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            mem_size
        );

        cs_attend_ker<128, false><<<grid, (32*NUM_WORKERS), mem_size, stream>>>(g);
    }



    CHECK_CUDA_ERROR(cudaGetLastError());

    return {o, cs, l_vec};
}
}

#endif