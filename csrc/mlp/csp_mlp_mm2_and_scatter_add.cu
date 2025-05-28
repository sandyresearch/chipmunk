#include <torch/extension.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cub/cub.cuh>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <curand_kernel.h>
#include "kittens.cuh"
#include "../common/all.cuh"
#include "../indexed_io/scatter_add.cuh"

#define checkCudaErrors(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error in file '" << __FILE__ << "' line " << __LINE__ \
                  << ": " << err << " - " << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define checkCuErrors(call) do { \
    CUresult err = call; \
    if (err != CUDA_SUCCESS) { \
        const char* errStr; \
        cuGetErrorString(err, &errStr); \
        std::cerr << "CUDA driver error in file '" << __FILE__ << "' line " << __LINE__ \
                  << ": " << err << " - " << errStr << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define checkAnyCudaErrors() do { \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error in file '" << __FILE__ << "' line " << __LINE__ \
                  << ": " << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)

using bf16 = __nv_bfloat16;

inline static void run_op_matmul_2(
    const bf16* in_ptr,
    bf16* out_ptr,
    const int32_t* inds_ptr,
    const int32_t* counts_ptr,
    int M_scatter_add,
    int F,
    CUfunction matmul_kernel,
    const bf16* d_A,
    const bf16* d_B,
    bf16* d_C,
    int M,
    int N,
    int K,
    int num_ctas_scatter_add,
    cudaStream_t stream1,
    cudaStream_t stream2
) {
    scatter_add_kernel<bf16><<<num_ctas_scatter_add, NUM_THREADS, SHARED_MEM_SIZE, stream1>>>(
        in_ptr,
        out_ptr,
        inds_ptr,
        counts_ptr,
        M_scatter_add, // M
        F
    );
    checkAnyCudaErrors();
    // Define grid/block configuration (converted to driver API launch dimensions)
    int matmul_shared_mem_size = 231424;
    int gridX = 132 - num_ctas_scatter_add, gridY = 1, gridZ = 1;
    int blockX = 256, blockY = 1, blockZ = 1;
    void *kernelArgs[] = {
        &d_A, &d_B, &d_C,
        &inds_ptr, &counts_ptr,
        &M, &N, &K,
        &K, &N, &N
    };

    checkCuErrors(cuFuncSetAttribute(matmul_kernel, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, matmul_shared_mem_size));
    checkCuErrors(cuLaunchKernel(matmul_kernel,
        gridX, gridY, gridZ,
        blockX, blockY, blockZ,
        matmul_shared_mem_size, stream2, // shared memory and stream
        kernelArgs, 0)
    );
    checkAnyCudaErrors();
}

namespace chipmunk {

void csp_mlp_mm2_and_scatter_add(
    at::Tensor packed,
    at::Tensor unpacked_colmajor,
    at::Tensor sp_inds,       // [B, M, F],   int32
    at::Tensor sp_counts,     // [B, M],      int32
    at::Tensor mma_a,
    at::Tensor mma_b,
    at::Tensor mma_c,
    int64_t num_ctas_scatter_add,
    int64_t matmul_kernel_ptr
)
{
    // Check all tensors are on CUDA
    TORCH_CHECK(packed.is_cuda(), "packed must be a CUDA tensor");
    TORCH_CHECK(packed.is_contiguous(), "packed must be contiguous");
    TORCH_CHECK(packed.dim() == 3, "packed must be a 3D tensor");
    TORCH_CHECK(unpacked_colmajor.is_cuda(), "unpacked_colmajor must be a CUDA tensor");
    TORCH_CHECK(unpacked_colmajor.is_contiguous(), "unpacked_colmajor must be contiguous");
    TORCH_CHECK(unpacked_colmajor.dim() == 3, "unpacked_colmajor must be a 3D tensor");
    TORCH_CHECK(sp_inds.is_cuda(), "sp_inds must be a CUDA tensor");
    TORCH_CHECK(sp_inds.is_contiguous(), "sp_inds must be contiguous");
    TORCH_CHECK(sp_inds.dim() == 3, "sp_inds must be a 3D tensor");
    TORCH_CHECK(sp_counts.is_cuda(), "sp_counts must be a CUDA tensor");
    TORCH_CHECK(sp_counts.is_contiguous(), "sp_counts must be contiguous");
    TORCH_CHECK(sp_counts.dim() == 2, "sp_counts must be a 2D tensor");

    // Check packed and unpacked_colmajor are bf16
    TORCH_CHECK(packed.scalar_type() == at::ScalarType::BFloat16, "packed must be bfloat16");
    TORCH_CHECK(unpacked_colmajor.scalar_type() == at::ScalarType::BFloat16, "unpacked_colmajor must be bfloat16");

    // Check sp_inds and sp_counts are int32
    TORCH_CHECK(sp_inds.scalar_type() == at::ScalarType::Int, "sp_inds must be int32");
    TORCH_CHECK(sp_counts.scalar_type() == at::ScalarType::Int, "sp_counts must be int32");

    // Check shapes
    int B = packed.size(0);
    int M_scatter_add = sp_inds.size(1);
    int F_scatter_add = sp_inds.size(2);

    TORCH_CHECK(B == sp_inds.size(0), "B must match");
    TORCH_CHECK(B == sp_counts.size(0), "B must match");
    TORCH_CHECK(B == unpacked_colmajor.size(0), "B must match");

    TORCH_CHECK(M_scatter_add*MLP_BM == packed.size(1), "packed must have M_scatter_add*MLP_BM rows");
    TORCH_CHECK(M_scatter_add*MLP_BM == unpacked_colmajor.size(2), "unpacked_colmajor must have M_scatter_add*MLP_BM rows");
    TORCH_CHECK(M_scatter_add == sp_counts.size(1), "M_scatter_add must match");
    TORCH_CHECK(M_scatter_add == sp_inds.size(1), "M_scatter_add must match");

    TORCH_CHECK(F_scatter_add == packed.size(2), "packed must have F_scatter_add columns");
    TORCH_CHECK(F_scatter_add == unpacked_colmajor.size(1), "unpacked_colmajor must have F_scatter_add columns");

    // Cast pointer types for the kernel
    bf16*  in_ptr   = reinterpret_cast<bf16*>(packed.data_ptr<c10::BFloat16>());
    bf16*        out_ptr  = reinterpret_cast<bf16*>(unpacked_colmajor.data_ptr<c10::BFloat16>());
    int32_t* inds_ptr   = sp_inds.data_ptr<int32_t>();
    int32_t* counts_ptr = sp_counts.data_ptr<int32_t>();

    checkAnyCudaErrors();

    bf16* mma_a_ptr = reinterpret_cast<bf16*>(mma_a.data_ptr<c10::BFloat16>());
    bf16* mma_b_ptr = reinterpret_cast<bf16*>(mma_b.data_ptr<c10::BFloat16>());
    bf16* mma_c_ptr = reinterpret_cast<bf16*>(mma_c.data_ptr<c10::BFloat16>());

    cudaStream_t default_stream = c10::cuda::getCurrentCUDAStream();
    static bool initialized = false;
    static cudaGraphExec_t exec;
    static cudaGraphNode_t node_scatter_add = nullptr, node_matmul = nullptr;
    static CUDA_KERNEL_NODE_PARAMS params_scatter_add, params_matmul;

    cudaFuncSetAttribute(scatter_add_kernel<bf16>, cudaFuncAttributeMaxDynamicSharedMemorySize, SHARED_MEM_SIZE);
    int M = mma_a.size(1);
    int N = mma_b.size(2);
    int K = mma_a.size(2);

    CUfunction matmul_kernel = reinterpret_cast<CUfunction>(matmul_kernel_ptr);
    
    int which_half = 1;

    if (false) {
        run_op_matmul_2(in_ptr, out_ptr, inds_ptr, counts_ptr, M_scatter_add, F_scatter_add, matmul_kernel, mma_a_ptr, mma_b_ptr, mma_c_ptr, M, N, K, num_ctas_scatter_add, default_stream, default_stream);
        cudaDeviceSynchronize();
        checkAnyCudaErrors();
        // return;
    }

    if (!initialized) {
        initialized = true;

        cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
        cudaStream_t stream1, stream2;
        cudaEvent_t startEvent, stopEvent;
        cudaGraph_t graph;

        checkCudaErrors(cudaGraphCreate(&graph, 0));
        checkCudaErrors(cudaStreamCreate(&stream1));
        checkCudaErrors(cudaStreamCreateWithPriority(&stream2, cudaStreamDefault, 4));
        checkCudaErrors(cudaEventCreate(&startEvent));
        checkCudaErrors(cudaEventCreate(&stopEvent));
        checkCudaErrors(cudaStreamBeginCapture(stream1, cudaStreamCaptureModeGlobal));
        checkCudaErrors(cudaEventRecord(startEvent, stream1));
        checkCudaErrors(cudaStreamWaitEvent(stream2, startEvent, 0));

        run_op_matmul_2(in_ptr, out_ptr, inds_ptr, counts_ptr, M_scatter_add, F_scatter_add, matmul_kernel, mma_a_ptr, mma_b_ptr, mma_c_ptr, M, N, K, num_ctas_scatter_add, stream1, stream2);


        checkAnyCudaErrors();
        checkCudaErrors(cudaEventRecord(stopEvent, stream2));
        checkCudaErrors(cudaStreamWaitEvent(stream1, stopEvent, 0));
        checkCudaErrors(cudaStreamEndCapture(stream1, &graph));
        
        checkCuErrors(cuCtxSynchronize());
        
        size_t numNodes = 0;
        checkCudaErrors(cudaGraphGetNodes(graph, NULL, &numNodes));

        std::vector<cudaGraphNode_t> nodes(numNodes);
        checkCudaErrors(cudaGraphGetNodes(graph, nodes.data(), &numNodes));

        checkAnyCudaErrors();
        for (int i = 0; i < numNodes; i++) {
            CUgraphNode cu_n = nodes[i];
            CUgraphNodeType type;
            checkCuErrors(cuGraphNodeGetType(cu_n, &type));
            // std::cout << "Node " << i << " is of type 0x" << std::hex << type << std::dec << std::endl;
            if (type == 0) {
                CUDA_KERNEL_NODE_PARAMS params;
                checkCuErrors(cuGraphKernelNodeGetParams(cu_n, &params));
                // std::cout << "    { .func = " << params.func << ", .gridDim.x = " << params.gridDimX << ", .gridDim.y = " << params.gridDimY << ", .gridDim.z = " << params.gridDimZ << ", .blockDim.x = " << params.blockDimX << ", .blockDim.y = " << params.blockDimY << ", .blockDim.z = " << params.blockDimZ << ", .sharedMemBytes = " << params.sharedMemBytes << " }" << std::endl;
                union cudaKernelNodeAttrValue attr;
                if (node_scatter_add == nullptr) {
                    node_scatter_add = cu_n;
                    params_scatter_add = params;
                    attr.priority = -4;
                } else if (node_matmul == nullptr) {
                    node_matmul = cu_n;
                    params_matmul = params;
                    attr.priority = 4;
                }
                checkCudaErrors(cudaGraphKernelNodeSetAttribute(cu_n, cudaLaunchAttributePriority, &attr));
            }
        }
        checkCudaErrors(cudaGraphInstantiate(&exec, graph, NULL, NULL, 0));
    }

    void *matmul_kernel_args[] = {
        &mma_a_ptr, &mma_b_ptr, &mma_c_ptr,
        &inds_ptr, &counts_ptr,
        &M, &N, &K,
        &K, &N, &N
    };
    params_matmul.kernelParams = matmul_kernel_args;
    checkCuErrors(cuGraphExecKernelNodeSetParams(exec, node_matmul, &params_matmul));

    void *scatter_add_kernel_args[] = {
        &in_ptr, &out_ptr,
        &inds_ptr, &counts_ptr,
        &M_scatter_add, &F_scatter_add
    };
    params_scatter_add.kernelParams = scatter_add_kernel_args;
    checkCuErrors(cuGraphExecKernelNodeSetParams(exec, node_scatter_add, &params_scatter_add));
    checkCudaErrors(cudaGraphLaunch(exec, default_stream));
    // cudaDeviceSynchronize();
    checkAnyCudaErrors();
}

}
