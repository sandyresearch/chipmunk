#pragma once
#include "kittens.cuh"
#include "common/base_types.cuh"

namespace chipmunk {
namespace base_ops {

/**
 * @brief Reciprocal operation.
 *
 * This operation calculates the reciprocal of the input value.
 *
 * @tparam T The data type of the input and output values.
 * @param x[in] The input value.
 * @return The result of ReLU function applied to the input.
 */
struct rcp {
    template<typename T> static __device__ inline T op(const T &x) { return 1.f / x; }
};
template<> __device__ inline float  rcp::op<float> (const float &x ) { return __frcp_rn(x);                                  }
template<> __device__ inline float2 rcp::op<float2>(const float2 &x) { return float2{__frcp_rn(x.x), __frcp_rn(x.y)};         }
template<> __device__ inline bf16   rcp::op<bf16>  (const bf16 &x  ) { return hrcp(x);    }
template<> __device__ inline bf16_2 rcp::op<bf16_2>(const bf16_2 &x) { return h2rcp(x);    }
template<> __device__ inline half   rcp::op<half>  (const half &x  ) { return hrcp(x);    }
template<> __device__ inline half_2 rcp::op<half_2>(const half_2 &x) { return h2rcp(x);    }

}
}