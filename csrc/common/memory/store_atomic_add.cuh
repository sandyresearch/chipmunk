#pragma once

#include <type_traits>

#include "kittens.cuh"
#include "common/common.cuh"
#include "types/types.cuh"
#include "../elementwise/atomic_add.cuh"

namespace chipmunk {

using namespace kittens;

template<ducks::sv::all SV, ducks::rv::all RV>
__device__ inline static void store_add(SV &dst, const RV &src) {
    using T2 = RV::dtype;
    using U = SV::dtype;
    using U2 = base_types::packing<U>::packed_type;
    using T = base_types::packing<T2>::unpacked_type;

    static_assert(dst.length == src.length);
    
    int laneid = ::kittens::laneid();
    U* dst_ptr = &dst.data[0];

    __syncwarp();
    // static_assert(std::is_same_v<typename RV::layout, align_l>);
    if constexpr (std::is_same_v<typename RV::layout, align_l>) {
        // #pragma unroll
        for(auto w = 0; w < (src.outer_dim+3)/4; w++) {
            int idx = w*64 + (laneid/4)*8 + 2*(laneid%4);
            int o_dim = w*4 + (laneid/4) / 2;
            int i_dim = (laneid/4) % 2;
            // this should be a maximally coalesced store. I hope!
            if(idx < src.outer_dim*16) {
                U2 tmp = base_types::convertor<U2, T2>::convert(src[o_dim][i_dim]);

                chipmunk::atomic_add<U2>::apply(reinterpret_cast<U2*>(dst_ptr + idx), tmp);
            }
        }
    }
    else if constexpr (std::is_same_v<typename RV::layout, ortho_l>) {
        // really hoping https://stackoverflow.com/questions/15029765/is-coalescing-triggered-for-accessing-memory-in-reverse-order is still true
        // otherwise there will be some pain :/
        #pragma unroll
        for(auto w = 0; w < (src.outer_dim+1)/2; w++) {
            int idx = w*32 + (laneid%4)*8 + (laneid/4);
            int o_dim = w*2 + (laneid%4) / 2;
            // this should be a maximally coalesced load.
            if(idx < src.outer_dim*16) {
                U tmp;
                if(laneid%2==0) tmp = base_types::convertor<U, T>::convert(src[o_dim][0].x);
                else tmp = base_types::convertor<U, T>::convert(src[o_dim][0].y);
                chipmunk::atomic_add<U>::apply(reinterpret_cast<U*>(dst_ptr + sizeof(typename SV::dtype)*idx), tmp);
            }
        }
    }
    else if constexpr (std::is_same_v<typename RV::layout, naive_l>) {
        #pragma unroll
        for(auto w = 0; w < src.outer_dim; w++) {
            if(w < src.outer_dim-1 || src.length%32 == 0 || laneid<16) {
                U tmp = base_types::convertor<U, T>::convert(src[w][0]);
                chipmunk::atomic_add<U>::apply(reinterpret_cast<U*>(dst_ptr + sizeof(typename SV::dtype)*(w*32 + laneid)), tmp);
            }
        }
    }
}

}