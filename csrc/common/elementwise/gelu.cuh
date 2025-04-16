#pragma once
#include "kittens.cuh"

namespace chipmunk {
namespace base_ops {
namespace detail{
__device__ static inline float fast_tanh(float x) {
    float y;
    asm volatile ("tanh.approx.f32 %0, %1; " : "=f"(y) : "f"(x));
    return y;
}
}
/**
 * @brief GeLU operation.
 *
 * This operation calculates the GeLU of the input value using the fast tanh approximation.
 *
 * @tparam T The data type of the input and output values.
 * @param x[in] The input value.
 * @return The result of GeLU function applied to the input.
 */

struct gelu {
    template<typename T> static __device__ inline T op(const T &f) {
        return static_cast<T>(gelu::op<float>(static_cast<float>(f)));
    }
};

template<> __device__ inline float gelu::op<float>(const float &f) {
    return f * 0.5f * (1.0f + detail::fast_tanh(f * 0.79788456f * (1 + f * f *0.044715f)));
}

template<> __device__ inline float2 gelu::op<float2>(const float2 &f) {
    return float2{ gelu::op(f.x), gelu::op(f.y) };
}

}
}
