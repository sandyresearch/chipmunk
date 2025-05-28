#pragma once
#include "common/base_types.cuh"

namespace chipmunk {

/**
 * @brief Atomic add operation.
 *
 * This operation performs an atomic add operation on the input value (expected to be in registers)
 * and stores the result in the destination pointer (expected to be in shared memory).
 * 
 * It works for packed and unpacked types. Currently implemented for:
 * - bf16
 * - fp32
 * 
 * This assists in the implementation of the column-sum attention kernel (see csrc/attn/dense_colsum_attn.cu), where
 * we need to reduce across warps into shared memory.
 */

template<typename T>
struct atomic_add {
    __device__ static inline void apply(T* dst, const T& src) {
        static_assert(sizeof(T) == 0, "atomic_add is not implemented for this type");
    }
};

template<> struct atomic_add<float> {
    __device__ static inline void apply(float* dst, const float& src) {
        atomicAdd(dst, src);
    }
};
template<> struct atomic_add<bf16> {
    __device__ static inline void apply(bf16* dst, const bf16& src) {
        atomicAdd(dst, src);
    }
};
template<> struct atomic_add<bf16_2> {
    __device__ static inline void apply(bf16_2* dst, const bf16_2& src) {
        atomicAdd(reinterpret_cast<bf16*>(dst)+0, src.x);
        atomicAdd(reinterpret_cast<bf16*>(dst)+1, src.y);
    }
};

template<> struct atomic_add<float2> {
    __device__ static inline void apply(float2* dst, const float2& src) {
        atomicAdd(reinterpret_cast<float*>(dst)+0, src.x);
        atomicAdd(reinterpret_cast<float*>(dst)+1, src.y);
    }
};

template<> struct atomic_add<float4> {
    __device__ static inline void apply(float4* dst, const float4& src) {
        atomicAdd(reinterpret_cast<float*>(dst)+0, src.x);
        atomicAdd(reinterpret_cast<float*>(dst)+1, src.y);
        atomicAdd(reinterpret_cast<float*>(dst)+2, src.z);
        atomicAdd(reinterpret_cast<float*>(dst)+3, src.w);
    }
};


}