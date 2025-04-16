#pragma once
#include "kittens.cuh"

namespace chipmunk {

using namespace kittens;

/**
 * @brief Perform A -= B.T, where A is an accumulator matrix (in registers) and B is a post-activation cache matrix (in shared memory).
 * 
 * This is used in the MLP matmul 1 kernel to subtract the transposed post-activation cache from the accumulator.
 * 
 * @param accum: The accumulator matrix (in registers)
 * @param pa_cache: The post-activation cache matrix (in shared memory)
 */
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
                    auto &tile = accum[n].tiles[tile_y][tile_x];
                    tile.data[i].x -= __bfloat162float(v1_s);
                    tile.data[i].y -= __bfloat162float(v2_s);
                }
            }
        }
    }
}

/**
 * @brief Load a 1D bias vector (in registers) into a 2D accumulator.
 * 
 * It is expected that the number of elements in bias is equivalent to the number of columns in the accumulator.
 * To understand these layouts see https://docs.nvidia.com/cuda/parallel-thread-execution/#matrix-fragments-for-wgmma-mma-async-m64nnk16
 * Under standard matrix sizes, each thread owns 16 of these bias values.
 * 
 * @param add: Whether to add the bias to the accumulator or to overwrite it
 */
template<bool add, ducks::rt::all RT>
__device__ static inline void load_bias(RT &accum, bf16 *bias) {
    int lane = kittens::laneid(); // lane within the warp
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

}