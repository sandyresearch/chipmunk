#include "kittens.cuh"
#include "prototype.cuh"
#include "common/common.cuh"
#include "common/templates.cuh"
#include "../common/all.cuh"
#ifdef TORCH_COMPILE
#include "pyutils/torch_helpers.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <iostream>
#endif

using namespace kittens;
using namespace kittens::prototype;
using namespace kittens::prototype::lcf;

static constexpr int PRODUCER_REGISTERS = 72;
static constexpr int CONSUMER_REGISTERS = 216;

static constexpr bool USE_PERSISTENT_GRID = true;
static constexpr bool USE_TMA_LOAD_B = false;
static constexpr bool USE_TMA_LOAD_INDICES = true;
static constexpr bool ENABLE_DYNAMIC_INDICES = true;

static constexpr bool ENABLE_GELU = true;
static constexpr bool ENABLE_BIAS = true;
static constexpr bool ENABLE_PA_CACHE = true;


static constexpr int OUT_OF_INDICES_BOUND = -2;

namespace kittens {

namespace prototype {
namespace lcf {

template<typename lcft> // load-compute-store-finish template
__global__ __launch_bounds__(detail::NUM_THREADS_v<lcft>, detail::NUM_BLOCKS_v<lcft>)
void matmul_kernel(const __grid_constant__ typename lcft::layout::globals globals) {
    static_assert(kernel_template<lcft>, "lcf kernel template parameter does not satisfy concept requirements");
    using L              = typename lcft::layout;
    using CKL            = complete_kittens_layout<L>; // complete the layout by filling in the optional types with empty
    using common_state   = typename CKL::common_state_t;
    using producer_state = typename CKL::producer_state_t;
    using consumer_state = typename CKL::consumer_state_t;
    using input_block    = typename CKL::input_block_t;
    using scratch_block  = typename CKL::scratch_block_t;
    using finish_block   = typename CKL::finish_block_t;
    using input_alloc_block   = typename CKL::input_alloc_block_t;
    using scratch_alloc_block = typename CKL::scratch_alloc_block_t;
    constexpr int MAX_SHARED_MEMORY = detail::MAX_SHARED_MEMORY_v<lcft>;
    constexpr int INPUT_PIPE_STAGES = detail::INPUT_PIPE_STAGES_v<lcft>;
    static_assert(INPUT_PIPE_STAGES >= 1 && INPUT_PIPE_STAGES <= 16, "Invalid number of input pipe stages");
    static_assert(
        INPUT_PIPE_STAGES*sizeof(input_alloc_block) + sizeof(scratch_alloc_block)
        <= MAX_SHARED_MEMORY-1024, "Shared memory usage exceeds limits"
    );
    constexpr int NUM_CONSUMER_WARPS = detail::NUM_CONSUMER_WARPS_v<lcft>;
    constexpr int NUM_PRODUCER_WARPS = detail::NUM_PRODUCER_WARPS_v<lcft>;
    
    using everyone = group<detail::NUM_WARPS_v<lcft>>;
    
    extern __shared__ int __shm[];
    shared_allocator alloc(&__shm[0]); // allocate shared memory
    scratch_alloc_block (&scratch_smem)                     = alloc.allocate<scratch_alloc_block>();
    input_alloc_block   (&input_smem)  [INPUT_PIPE_STAGES]  = alloc.allocate<input_alloc_block,  INPUT_PIPE_STAGES>();

    // figure out where we're going to put the finish block
    constexpr int FINISH_BLOCK_OFFSET = (MAX_SHARED_MEMORY-1024)/detail::NUM_BLOCKS_v<lcft> - sizeof(finish_block);
    static_assert(FINISH_BLOCK_OFFSET >= 0, "Finish block is too large for shared memory.");
    constexpr int NON_FINISH_BLOCK_SPACE = FINISH_BLOCK_OFFSET - 1024 - sizeof(scratch_alloc_block); // including the losses from alignment
    constexpr int SAFE_STAGES_BETWEEN_BLOCKS = NON_FINISH_BLOCK_SPACE/sizeof(input_alloc_block)<INPUT_PIPE_STAGES?NON_FINISH_BLOCK_SPACE/sizeof(input_alloc_block):INPUT_PIPE_STAGES;
    finish_block  (*finish_smem) = reinterpret_cast<finish_block*>((((uint64_t)&__shm[0] + FINISH_BLOCK_OFFSET)/1024)*1024); // alignment

    // Initialize semaphores. This is constant for all two-stage producer-consumer kernels.
    __shared__ kittens::semaphore inputs_arrived[INPUT_PIPE_STAGES], inputs_finished[INPUT_PIPE_STAGES];
    __shared__ kittens::semaphore finish_finished;
    __shared__ kittens::semaphore bias_bar;
    uint32_t semaphore_bitfield = 0xFFFF0000; // ***_finished phase bits start as 1s, ***_arrived phase bits start as 0s
    common_state common;

    if(warpid() >= NUM_CONSUMER_WARPS) { // code path for producer warps
        using producers = group<NUM_PRODUCER_WARPS>;
        if (warpid() == NUM_CONSUMER_WARPS) { // a single warp (in fact a single thread) does these.
            for(int i = 0; i < INPUT_PIPE_STAGES; i++) {
                init_semaphore(inputs_arrived[i], detail::PRODUCER_BARRIER_ARRIVALS_v<lcft>, 0); // needs to wait on each producer warp
                init_semaphore(inputs_finished[i], detail::CONSUMER_BARRIER_ARRIVALS_v<lcft>, 0); // needs to wait on one thread from each consumer warp
            }
            init_semaphore(finish_finished, detail::CONSUMER_BARRIER_ARRIVALS_v<lcft>, 0); // consumer warps must say they are done with the finish block
            init_semaphore(bias_bar, 128, 0);
        }
        everyone::sync(15); // all warps must arrive here, confirming semaphore initialization is visible to all threads.
        producer_state p_state;
        for(int task_iter = 0; true; task_iter++) {
            int num_iters = -1;
            common_setup_args<L> unif{common, task_iter, num_iters, globals, *scratch_smem};
            lcft::common_setup(unif, bias_bar);
            if (num_iters == OUT_OF_INDICES_BOUND) {
                // make sure the consumer has finished their finish stage before we skip multiple tasks
                everyone::sync(15);
                kittens::wait(finish_finished, (task_iter%2)^1);
                continue;
            };
            if (num_iters <= 0) return; // no work to do
            int input_ring = 0; // tracking which input block is being loaded
            int load_iter;
            lcft::producer::setup({p_state, unif}, bias_bar);
            for(load_iter = 0; load_iter < SAFE_STAGES_BETWEEN_BLOCKS && load_iter<num_iters; load_iter++) { // fill the pipeline
                kittens::wait(inputs_finished[input_ring], get_phasebit<1>(semaphore_bitfield, input_ring));
                update_phasebit<1>(semaphore_bitfield, input_ring);
                lcft::producer::load({p_state, *input_smem[input_ring], inputs_arrived[input_ring], load_iter, unif});
                input_ring=ring_advance<INPUT_PIPE_STAGES>(input_ring);
            }
            kittens::wait(finish_finished, (task_iter%2)^1); // wait for consumer to finish their finish stage before we can do the rest.
            for(; load_iter<num_iters; load_iter++) { // fill the pipeline
                kittens::wait(inputs_finished[input_ring], get_phasebit<1>(semaphore_bitfield, input_ring));
                update_phasebit<1>(semaphore_bitfield, input_ring);
                lcft::producer::load({p_state, *input_smem[input_ring], inputs_arrived[input_ring], load_iter, unif});
                input_ring=ring_advance<INPUT_PIPE_STAGES>(input_ring);
            }
            producers::sync(13); // producer warps must finish before consumer warps can proceed
        } // task iter loop
    } // producer warpgroup
    else { // code path for consumer warps
        using consumers = group<NUM_CONSUMER_WARPS>;
        everyone::sync(15); // all warps must arrive here, confirming semaphore initialization is visible to all threads.
        consumer_state c_state;
        for(int task_iter = 0; true; task_iter++) {
            int num_iters = -1;
            common_setup_args<L> unif{common, task_iter, num_iters, globals, *scratch_smem};
            lcft::common_setup(unif, bias_bar);
            if (num_iters == OUT_OF_INDICES_BOUND) {
                // finish early - we don't have any work to do for this task so move on to the next one
                everyone::sync(15);
                if (laneid() == 0) arrive(finish_finished);
                continue;
            }
            if (num_iters <= 0) return; // no work to do
            int input_ring = 0; // tracking which input block is being loaded
            lcft::consumer::setup({c_state, unif}, bias_bar);
#ifdef CONSUMER_UNROLL
            #pragma unroll CONSUMER_UNROLL_VALUE
#endif
            for(int it = 0; it < num_iters; it++) {
                kittens::wait(inputs_arrived[input_ring], get_phasebit<0>(semaphore_bitfield, input_ring)); // wait for memory to arrive, phase changes at half the rate of the ring
                update_phasebit<0>(semaphore_bitfield, input_ring);
                lcft::consumer::compute({c_state, *input_smem[input_ring], inputs_finished[input_ring], it, unif});
                input_ring=ring_advance<INPUT_PIPE_STAGES>(input_ring);
            } // work loop
            consumers::sync(14); // cannot overwrite finish block until all consumer warps are done.
            lcft::consumer::finish({c_state, *finish_smem, finish_finished, unif});
            consumers::sync(14); // cannot overwrite finish block until all consumer warps are done.
        } // task iter loop
    } // consumer warpgroup
}

} // namespace lcf
} // namespace prototype
} // namespace kittens



template<int M_BLOCK, int N_BLOCK>
struct matmul_layout {
    using  base_tile      = st_bf<64, 64>;
    using  tall_tile      = st_bf<64*N_BLOCK, 64>;
    using  bias_vec       = sv_bf<N_BLOCK*64>;
    static constexpr int NUM_INDICES = base_tile::cols * N_BLOCK;
    static constexpr int b_load_iters = base_tile::rows / 16;

    using  global_layout_ac   = gl<bf16, 1, 1, -1, -1, base_tile>;
    using  global_layout_b    = gl<bf16, 1, 1, -1, -1, tall_tile>;
    using  global_layout_bias = gl<bf16, 1, 1, 1, -1,  bias_vec >;
    struct globals        { 
        global_layout_ac A;
        global_layout_b B;
        global_layout_ac C; 
        global_layout_ac pa_cache;
        global_layout_bias bias;
        int *indices; 
        int *indices_counts;
    };
    struct input_block    { base_tile a[M_BLOCK]; tall_tile b; };
    struct scratch_block  { 
        base_tile pa_cache[M_BLOCK][N_BLOCK];
        // we need to double buffer the bias and indices because consumer::finish happens at the same time as producer::setup
        bias_vec bias[2];
        int indices[2][NUM_INDICES]; 
        semaphore indices_bar; 
    };
    struct finish_block   { base_tile c[M_BLOCK][N_BLOCK]; };
    struct common_state   { int2 coord; };
    struct producer_state {
        int indices_reg[16];
    };
    struct consumer_state { 
        rt_fl<16, 64> accum[N_BLOCK];
    };
};

template<int _M_BLOCK=2, int _N_BLOCK=4, int _SUPER_M=12>
struct matmul_template {
    static constexpr int M_BLOCK = _M_BLOCK, N_BLOCK = _N_BLOCK, SUPER_M = _SUPER_M;
    using layout    = matmul_layout<M_BLOCK, N_BLOCK>;
    using wide_tile = st_bf<64, 64*N_BLOCK>;
    // 1 from the A matrix (loaded by a single thread in the TMA), and 128 from the B matrix (loaded by cp.async, the mbarrier is tripped for every thread in the producer warpgroup)
    static constexpr int PRODUCER_BARRIER_ARRIVALS = 1+(USE_TMA_LOAD_B ? 0 : 128); 
    static constexpr int NUM_CONSUMER_WARPS=M_BLOCK*4, INPUT_PIPE_STAGES=3;

    __host__ static inline dim3 grid(int M, int N, int K) {
        return dim3(USE_PERSISTENT_GRID ? 132 : M*N/(M_BLOCK*N_BLOCK*layout::base_tile::num_elements));
    }

      // ThunderKittens template functions
    __device__ static inline void common_setup(common_setup_args<layout> args, semaphore &bias_bar) {
        int Rblocks = args.globals.C.rows / (M_BLOCK*64), Cblocks = args.globals.C.cols / (N_BLOCK*64);
        int super_rows = (Rblocks/SUPER_M)*SUPER_M,
            final_rows = Rblocks - super_rows,
            super_repeat = SUPER_M*Cblocks;
        int task_id = args.task_iter*gridDim.x + blockIdx.x;
        if (task_id < super_rows * Cblocks) // 32*16 = 512
            args.common.coord = { SUPER_M*(task_id/super_repeat) + task_id%SUPER_M, (task_id%super_repeat)/SUPER_M }; 
        else if (task_id < Rblocks*Cblocks) { // 512
            int remainder_id = task_id - super_rows*Cblocks;
            args.common.coord = { super_rows + (remainder_id%final_rows), remainder_id/final_rows };
        }
        else { // Id is too high, no more work to do
            args.num_iters = -1;
            return;
        }
        args.num_iters = args.globals.A.cols/64;
        int id = warpgroup::groupid() == NUM_CONSUMER_WARPS/4 ? 0 : warpgroup::groupid(); // producer sets as 0
        int indices_group_idx = args.common.coord.x;
        int indices_count = ENABLE_DYNAMIC_INDICES ? args.globals.indices_counts[indices_group_idx]/(N_BLOCK*64) : 999999999;
        if (args.common.coord.y >= indices_count) {
            if constexpr (USE_PERSISTENT_GRID) {
                // no more work to do - we've reached the end of the indices
                args.num_iters = OUT_OF_INDICES_BOUND;
                if (warpgroup::groupid() == NUM_CONSUMER_WARPS/4) {
                    arrive(bias_bar);
                }
            } else {
                args.num_iters = -1;
            }
        } else {
            args.common.coord = { args.common.coord.x*M_BLOCK + id, args.common.coord.y*N_BLOCK };
        }
    }

    struct producer {
        template<typename layout>
        __device__ static inline void load_indices(const producer_setup_args<layout>& args) {
            if (kittens::laneid() == 0) {
                int *s_indices = args.scratch.indices[args.task_iter%2];

                int N = layout::NUM_INDICES;

                int indices_row = args.common.coord.x / M_BLOCK;
                int indices_col_block = args.common.coord.y / N_BLOCK;

                int indices_row_offset = indices_row * args.globals.B.rows; // indices_row * N (remember - B is col major!)
                int indices_col_offset = indices_col_block * layout::NUM_INDICES;
                
                if (ENABLE_DYNAMIC_INDICES) {
                    int *g_indices = args.globals.indices+indices_row_offset+indices_col_offset;
                    if (USE_TMA_LOAD_INDICES) {
                        int num_bytes_to_copy = N * sizeof(int);
                        tma::expect_bytes(args.scratch.indices_bar, num_bytes_to_copy);
                        chipmunk::load_async_bytes(s_indices, g_indices, num_bytes_to_copy, args.scratch.indices_bar);
                    } else {
                        for (int i = 0; i < N; i++) s_indices[i] = g_indices[i];
                        arrive(args.scratch.indices_bar);
                    }
                } else {
                    for (int i = 0; i < N; i++) s_indices[i] = indices_col_offset + i;
                    arrive(args.scratch.indices_bar);
                }
            }
        }

        __device__ static inline void setup(producer_setup_args<layout> args, semaphore &bias_bar) {
            warpgroup::decrease_registers<PRODUCER_REGISTERS>(); // decrease registers for producers
            if (ENABLE_DYNAMIC_INDICES) {
                if (warpgroup::warpid() == 0) {
                    init_semaphore(args.scratch.indices_bar, 1, 0);
                    load_indices(args);
                }
                warpgroup::sync(0);
                kittens::wait(args.scratch.indices_bar, 0);
                if (warpgroup::warpid() == 0) {
                    // the grid is persistent! we need to invalidate the semaphore for the next task we might receive.
                    invalidate_semaphore(args.scratch.indices_bar);
                }
            }
            
            if (ENABLE_BIAS) {
                auto &bias    = args.scratch.bias   [args.task_iter%2];
                auto *indices = args.scratch.indices[args.task_iter%2];
                // we can't use coalesced memory loads because the indices are not contiguous
                for (int i = warpgroup::laneid(); i < bias.length; i += warpgroup::GROUP_THREADS) {
                    bias[i] = args.globals.bias[indices[i]];
                }
                arrive(bias_bar);
            }

            int laneid = warpgroup::laneid() / 8; // 0 to 128 becomes 0 to 16
            // typename GL::dtype *src_ptr = (typename GL::dtype*)&src[{0, args.common.coord.x+m}];
            for (int i = 0; i < 16; i++) {
                int index = args.scratch.indices[args.task_iter%2][laneid + i*16];
                args.state.indices_reg[i] = index;
            }
        }
        __device__ static inline void load(producer_load_args<layout> args) {
            if (warpgroup::warpid() == 0) {

                if (USE_TMA_LOAD_B) { tma::expect(args.inputs_arrived, args.input);   }
                else                { tma::expect(args.inputs_arrived, args.input.a); }
                
                for(int i = 0; i < M_BLOCK; i++) {
                    tma::load_async(args.input.a[i], args.globals.A, {args.common.coord.x+i, args.iter}, args.inputs_arrived);
                }
                if (USE_TMA_LOAD_B) {
                    tma::load_async(args.input.b, args.globals.B, {args.common.coord.y / N_BLOCK, args.iter}, args.inputs_arrived);
                }
            }
        
            if (!USE_TMA_LOAD_B) {
                chipmunk::load_async_gather(args.input.b, args.globals.B, args.iter, {0, args.iter}, args.state.indices_reg);

                // when we are done loading the tile, trip the semaphore
                chipmunk::cp_async_semaphore(args.inputs_arrived);
            }
            // place this in the load() function - NOT setup - which is called while the consumer is still setting up
            if (ENABLE_PA_CACHE && args.iter == INPUT_PIPE_STAGES) {
                for (int m = 0; m < M_BLOCK; m++) {
                    auto &pa_cache_as_tall_tile = reinterpret_cast<st_bf<256, 64>&>(args.scratch.pa_cache[m][0]);
                    chipmunk::load_async_gather(pa_cache_as_tall_tile, args.globals.pa_cache, args.common.coord.x+m, {0, args.common.coord.x+m}, args.state.indices_reg);
                }

                // note - we never actually call cp.async.wait on this - we just rely on the fact that INPUT_PIPE_STAGES << k/64 so the memory is (usually) always visible!
                chipmunk::cp_async_fence();
            }
        }
    };
    struct consumer {
        __device__ static inline void setup(consumer_setup_args<layout> args, semaphore &bias_bar) {
            warpgroup::increase_registers<CONSUMER_REGISTERS>(); // increase registers for consumers
            if (ENABLE_BIAS) {
                kittens::wait(bias_bar, args.task_iter%2);
                for (int n = 0; n < N_BLOCK; n++) chipmunk::load_bias<false>(args.state.accum[n], &args.scratch.bias[args.task_iter%2][n*64]);
            } else {
                for (int n = 0; n < N_BLOCK; n++) zero(args.state.accum[n]);
            }
        }
        __device__ static inline void compute(consumer_compute_args<layout> args) {
            warpgroup::mma_ABt(
                reinterpret_cast<rt_fl<16, 64*N_BLOCK>&>(args.state.accum[0]),
                args.input.a[warpgroup::groupid()],
                args.input.b
            );
            warpgroup::mma_async_wait();
            if(laneid() == 0) arrive(args.inputs_finished);
        }


        __device__ static inline void finish(consumer_finish_args<layout> args) {
            if (ENABLE_GELU) for (int n = 0; n < N_BLOCK; n++) {
                unary_map<chipmunk::base_ops::gelu>(args.state.accum[n], args.state.accum[n]); // GeLU
            }
            // for (int n = 0; n < N_BLOCK; n++) {
            //     warpgroup::store(args.finish.c[warpgroup::groupid()][n], args.state.accum[n]);
            // }
            // load_async_gather<N_BLOCK>(args.scratch.pa_cache[warpgroup::groupid()], args.globals.pa_cache, {0, args.common.coord.x}, args.scratch.indices[args.task_iter%2]);
            if (ENABLE_PA_CACHE) {
                chipmunk::sub_transposed<N_BLOCK>(args.state.accum, args.scratch.pa_cache[warpgroup::groupid()]); // Cache diff
            }
            for (int n = 0; n < N_BLOCK; n++) {
                warpgroup::store(args.finish.c[warpgroup::groupid()][n], args.state.accum[n]);
            }
            warpgroup::sync(warpgroup::groupid()+4);
            if (warpgroup::warpid() == 0) {
                for (int i = 0; i < N_BLOCK; i++) {
                    tma::store_async(args.globals.C, args.finish.c[warpgroup::groupid()][i], {args.common.coord.x, args.common.coord.y+i});
                }
                tma::store_async_read_wait(); // wait that store is finished before reusing finish memory
            }
            if (!ENABLE_BIAS) for (int n = 0; n < N_BLOCK; n++) {
                zero(args.state.accum[n]);
            }
            if (laneid() == 0) arrive(args.finish_finished);
        }
    };
};


constexpr bool NCU = false;
#include <iostream>
#include <random>
#include <cuda_bf16.h>
#include <omp.h>

inline float gelu(float x) {
    if (!ENABLE_GELU) return x;

    static constexpr float sqrt_2_over_pi = 0.7978845608028654f;
    static constexpr float coef = 0.044715f;
    float cdf = 0.5f * (1.0f + tanhf((sqrt_2_over_pi * (x + coef * x * x * x))));
    return x * cdf;
}

void cpu_gemm(float* a, float* b, float* c, float *pa_cache, float* bias, int M, int N, int K) {
    #pragma omp parallel for collapse(2) // otherwise the CPU version takes for everrrrrr
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += a[i * K + k] * b[(N-1-j) * K + k];
            }
            if (ENABLE_BIAS)     sum += bias[N-1-j];
            if (ENABLE_GELU)     sum  = gelu(sum);
            if (ENABLE_PA_CACHE) sum -= pa_cache[(N-1-j) * M + i];;
            c[i * N + j] = sum;
        }
    }
}

template<typename mmt>
void inner_run(bf16 *d_A, bf16 *d_B, bf16 *d_C, bf16 *d_bias, bf16 *d_pa_cache, int *d_indices, int *d_indices_counts, size_t M, size_t N, size_t K, dim3 grid, dim3 block) {
    using global_layout_ac = typename mmt::layout::global_layout_ac;
    using global_layout_b = typename mmt::layout::global_layout_b;
    using global_layout_bias = typename mmt::layout::global_layout_bias;
    using globals  = typename mmt::layout::globals;
    global_layout_ac Ag{d_A, nullptr, nullptr, M, K};
    global_layout_b Bg{d_B, nullptr, nullptr, N, K}; // col major!
    global_layout_ac Cg{d_C, nullptr, nullptr, M, N};
    global_layout_ac pa_cacheg{d_pa_cache, nullptr, nullptr, N, M}; // col major!
    global_layout_bias Biasg{d_bias, nullptr, nullptr, nullptr, N};
    globals G{Ag, Bg, Cg, pa_cacheg, Biasg, d_indices, d_indices_counts};
    prototype::lcf::matmul_kernel<mmt><<<grid, block, MAX_SHARED_MEMORY-1024>>>(G);
    // dim3 new_grid(M / (M_BLOCK*64), N / (N_BLOCK*64));
    // matmul_kernel_no_ws<mmt><<<new_grid, NUM_CONSUMER_WARPGROUPS*128, MAX_SHARED_MEMORY-1024>>>(G);
}

template<typename mmt>
int run_benchmark(size_t M, size_t N, size_t K) {
    cudaError_t cudaStatus;

    std::cout << "--------------------  M=" << M << " N=" << N << " K=" << K << "  --------------------\n";
    std::cout << "Block size: " << mmt::M_BLOCK*64 << "x" << mmt::N_BLOCK*64 << "\n";

    // Allocate host memory
    float *h_A = new float[M * K];
    float *h_B = new float[N * K]; // col major!
    float *h_C = new float[M * N];
    float *h_pa_cache = new float[N * M]; // col major!
    float *h_bias = new float[N];
    float *h_C_ref = new float[M * N];

    int num_indices_blocks = M / 128;
    int *h_indices = new int[num_indices_blocks * N];
    int *h_indices_counts = new int[num_indices_blocks];

    for (int i = 0; i < num_indices_blocks; i++) {
        for (int j = 0; j < N; j++) {
            h_indices[i * N + j] = N-1-j;
        }
        h_indices_counts[i] = N;
    }

    std::cout << "Allocated host memory" << std::endl;

    // Initialize random number generator
    std::random_device rd;
    std::mt19937 gen(42);
    std::uniform_real_distribution<> dis(-0.5, 0.5);

    // Initialize matrices with random values
    for (int i = 0; i < M * K; ++i) h_A[i] = dis(gen);
    for (int i = 0; i < K * N; ++i) h_B[i] = dis(gen);
    for (int i = 0; i < N; ++i) h_bias[i] = dis(gen);
    for (int i = 0; i < M * N; ++i) h_pa_cache[i] = dis(gen);

    std::cout << "Initialized matrices" << std::endl;

    // Perform CPU matrix multiplication for reference
    if(true) cpu_gemm(h_A, h_B, h_C_ref, h_pa_cache, h_bias, M, N, K);

    std::cout << "Performed CPU matrix multiplication" << std::endl;

    // Allocate device memory
    __nv_bfloat16 *d_A, *d_B, *d_C, *d_bias; __nv_bfloat16 *d_pa_cache;
    cudaMalloc(&d_A, M*K*sizeof(__nv_bfloat16));
    cudaMalloc(&d_B, K*N*sizeof(__nv_bfloat16));
    cudaMalloc(&d_C, M*N*sizeof(__nv_bfloat16));
    cudaMalloc(&d_bias, N*sizeof(__nv_bfloat16));
    cudaMalloc(&d_pa_cache, N*M*sizeof(float));
    int *d_indices, *d_indices_counts;
    cudaMalloc(&d_indices, num_indices_blocks * N * sizeof(int));
    cudaMalloc(&d_indices_counts, num_indices_blocks * sizeof(int));

    cudaMemcpy(d_indices, h_indices, num_indices_blocks * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_indices_counts, h_indices_counts, num_indices_blocks * sizeof(int), cudaMemcpyHostToDevice);

    // Check for CUDA errors
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(cudaStatus) << std::endl;
        // Optionally, you might want to exit the program or handle the error in some way
        return -1;
    }

    std::cout << "Allocated device memory" << std::endl;

    // Convert to __nv_bfloat16 and copy to device
    __nv_bfloat16 *h_A_bf16 = new __nv_bfloat16[M * K];
    __nv_bfloat16 *h_B_bf16 = new __nv_bfloat16[K * N];
    __nv_bfloat16 *h_bias_bf16 = new __nv_bfloat16[N];
    __nv_bfloat16 *h_pa_cache_bf16 = new __nv_bfloat16[M * N];
    for (int i = 0; i < M * K; ++i) h_A_bf16[i] = __float2bfloat16(h_A[i]);
    for (int i = 0; i < K * N; ++i) h_B_bf16[i] = __float2bfloat16(h_B[i]);
    for (int i = 0; i < N; ++i) h_bias_bf16[i] = __float2bfloat16(h_bias[i]);
    for (int i = 0; i < M * N; ++i) h_pa_cache_bf16[i] = __float2bfloat16(h_pa_cache[i]);
    cudaMemcpy(d_A, h_A_bf16, M*K*2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B_bf16, K*N*2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, h_bias_bf16, N*2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_pa_cache, h_pa_cache_bf16, M*N*sizeof(d_pa_cache[0]), cudaMemcpyHostToDevice);

    std::cout << "Copied matrices to device" << std::endl;

    unsigned long mem_size = MAX_SHARED_MEMORY - 1024;
    cudaFuncSetAttribute(prototype::lcf::matmul_kernel<mmt>, cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    // cudaFuncSetAttribute(matmul_kernel_no_ws<mmt>, cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size);

    // Launch kernel
    dim3 grid(mmt::grid(M, N, K));
    dim3 block(kittens::prototype::detail::NUM_THREADS_v<mmt>);
    // std::cout << "Launching warmup kernel with grid (" << grid.x << ", " << grid.y << "), block (" << block.x << ")\n";
    for(int i = 0; i < (NCU ? 0 : 2); i++) { // warmup
        inner_run<mmt>(d_A, d_B, d_C, d_bias, d_pa_cache, d_indices, d_indices_counts, M, N, K, grid, block);
    }

    // Start timing
    cudaDeviceSynchronize();
    // std::cout << "Launching kernel with grid (" << grid.x << ", " << grid.y << "), block (" << block.x << ")\n";
    auto start = std::chrono::high_resolution_clock::now();

    constexpr int ITERS = (NCU ? 1 : 10);
    for(int i = 0; i < ITERS; i++) {
        inner_run<mmt>(d_A, d_B, d_C, d_bias, d_pa_cache, d_indices, d_indices_counts, M, N, K, grid, block);
        if (i % 1000 == 0) {
            std::cout << "Iteration " << i << std::endl;
            cudaDeviceSynchronize();
        }
    }
    cudaDeviceSynchronize();

    // End timing
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate duration
    std::chrono::duration<double> diff = end - start;
    double useconds = diff.count() * 1e6 / ITERS;

    // Calculate TFLOPs
    double flops = double(2.0) * M * N * K; // 2 FLOPs per multiply-add
    double tflops = (flops / useconds) / 1e6;

    std::cout << "Avg Kernel execution time: " << useconds << " us\n";
    std::cout << "Achieved performance: " << tflops << " TFLOPs\n";
    
    // Check for CUDA errors
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(cudaStatus) << std::endl;
        // Optionally, you might want to exit the program or handle the error in some way
        return -1;
    }

    // Copy result back to host
    __nv_bfloat16 *h_C_bf16 = new __nv_bfloat16[M * N];
    cudaMemcpy(h_C_bf16, d_C, M*N*2, cudaMemcpyDeviceToHost);

    std::cout << "Copied result back to host" << std::endl;

    // Convert result back to float for comparison
    for (int i = 0; i < M * N; ++i) h_C[i] = __bfloat162float(h_C_bf16[i]);

    std::cout << "Converted result back to float" << std::endl;

    // Check result
    float max_error = 0.0f;
    int error_count = 0;
    for (int col = 0; col < N; col++) {
        for (int row = 0; row < M; row++) {
            float error = std::abs(h_C[row * N + col] - h_C_ref[row * N + col]);
            if(error > 0.1) { // large because of bf16 vs fp32 numerics
                if(error_count < 20) std::cout << "Error at row " << row << " col " << col << ": " << h_C[row * N + col] << " (computed) != " << h_C_ref[row * N + col] << " (ref)" << std::endl;
                else if(error_count == 21) std::cout << "Too many errors to show them all.\n";
                error_count++;
            }
            max_error = std::max(max_error, error);
        }
    }

    std::cout << "Max error: " << max_error << std::endl;
    std::cout << "Error count: " << error_count << std::endl;

    // Clean up
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] h_C_ref;
    delete[] h_A_bf16;
    delete[] h_B_bf16;
    delete[] h_C_bf16;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}


#ifdef TORCH_COMPILE
namespace chipmunk {
void csp_mlp_mm1(at::Tensor a, at::Tensor b_colmajor, at::Tensor c, at::Tensor bias, at::Tensor pa_cache_colmajor, at::Tensor indices, at::Tensor indices_counts)
{
    using ker_template = matmul_template<2, 4, 8>;
    CHECK_INPUT(a);
    CHECK_INPUT(b_colmajor);
    CHECK_INPUT(c);
    CHECK_INPUT(bias);
    CHECK_INPUT(pa_cache_colmajor);
    CHECK_INPUT(indices);
    CHECK_INPUT(indices_counts);

    auto M = a.size(0);
    auto K = a.size(1);
    auto N = b_colmajor.size(0);
    auto K_ = b_colmajor.size(1);
    if (K != K_) {
        throw std::runtime_error("K must match for all inputs");
    }
    auto num_indices_groups = M / (ker_template::M_BLOCK*64);

    TORCH_CHECK(a.dim() == 2, "a must be a 2D tensor");
    TORCH_CHECK(a.dtype() == torch::kBFloat16, "a must be a bfloat16 tensor");

    TORCH_CHECK(b_colmajor.dim() == 2, "b_colmajor must be a 2D tensor");
    TORCH_CHECK(b_colmajor.dtype() == torch::kBFloat16, "b_colmajor must be a bfloat16 tensor");

    TORCH_CHECK(c.dim() == 2, "c must be a 2D tensor");
    TORCH_CHECK(c.dtype() == torch::kBFloat16, "c must be a bfloat16 tensor");
    TORCH_CHECK(c.size(0) == M, "c must match the number of rows in a");
    TORCH_CHECK(c.size(1) == N, "c must match the number of columns in b_colmajor");

    TORCH_CHECK(indices_counts.dim() == 1, "Indices counts must be a 1D tensor");
    TORCH_CHECK(indices_counts.dtype() == torch::kInt32, "Indices counts must be a 32-bit integer tensor");
    TORCH_CHECK(indices_counts.size(0) == num_indices_groups, "Indices counts must match the number of indices groups");
    
    TORCH_CHECK(indices.dim() == 2, "Indices must be a 2D tensor");
    TORCH_CHECK(indices.dtype() == torch::kInt32, "Indices must be a 32-bit integer tensor");
    TORCH_CHECK(indices.size(0) == num_indices_groups, "Indices must match the number of indices groups");
    TORCH_CHECK(indices.size(1) == N, "Indices must be a 2D tensor with N columns");
    
    TORCH_CHECK(bias.dim() == 1, "Bias must be a 1D tensor");
    TORCH_CHECK(bias.dtype() == torch::kBFloat16, "Bias must be a bfloat16 tensor");
    TORCH_CHECK(bias.size(0) == N, "Bias must match the number of columns in b_colmajor");

    TORCH_CHECK(pa_cache_colmajor.dim() == 2, "pa_cache_colmajor must be a 2D tensor");
    TORCH_CHECK(pa_cache_colmajor.dtype() == torch::kBFloat16, "pa_cache_colmajor must be a bfloat16 tensor");
    TORCH_CHECK(pa_cache_colmajor.size(0) == N, "pa_cache_colmajor must match the number of columns in b_colmajor");
    TORCH_CHECK(pa_cache_colmajor.size(1) == M, "pa_cache_colmajor must match the number of rows in a");

    int* d_indices = indices.data_ptr<int>();
    int* d_indices_counts = indices_counts.data_ptr<int>();
    bf16*  d_a        = reinterpret_cast<bf16*>(a.data_ptr<c10::BFloat16>());
    bf16*  d_b        = reinterpret_cast<bf16*>(b_colmajor.data_ptr<c10::BFloat16>());
    bf16*  d_bias     = reinterpret_cast<bf16*>(bias.data_ptr<c10::BFloat16>());
    bf16*  d_pa_cache = reinterpret_cast<bf16*>(pa_cache_colmajor.data_ptr<c10::BFloat16>());
    bf16*  d_c        = reinterpret_cast<bf16*>(c.data_ptr<c10::BFloat16>());

    using globals = ker_template::layout::globals;
    using gl_layout_ac = ker_template::layout::global_layout_ac;
    using gl_layout_b = ker_template::layout::global_layout_b;
    using gl_layout_bias = ker_template::layout::global_layout_bias;

    gl_layout_ac Ag{d_a, nullptr, nullptr, M, K};
    gl_layout_b Bg{d_b, nullptr, nullptr, N, K}; // col major!
    gl_layout_ac Cg{d_c, nullptr, nullptr, M, N};
    gl_layout_ac pa_cacheg{d_pa_cache, nullptr, nullptr, N, M}; // col major!
    gl_layout_bias Biasg{d_bias, nullptr, nullptr, nullptr, N};
    globals G{Ag, Bg, Cg, pa_cacheg, Biasg, d_indices, d_indices_counts};

    auto mem_size = kittens::MAX_SHARED_MEMORY - 1024;
    dim3 grid(ker_template::grid(M, N, K));
    constexpr int BLOCK_SIZE = prototype::detail::NUM_THREADS_v<ker_template>;

    auto kernel_fn = matmul_kernel<ker_template>;

    cudaFuncSetAttribute(kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    kernel_fn<<<grid, BLOCK_SIZE, mem_size>>>(G);
}
}
#else

int main() {
    int M = 3840;
    int N = 12288;
    int K = 3072;
    // int M = 4096, N = 4096, K = 4096;
    run_benchmark<matmul_template<2,4,8>>(M, N, K);
    return 0;
}

#endif
