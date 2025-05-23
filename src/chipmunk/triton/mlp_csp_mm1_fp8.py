import triton.language as tl
from triton.language.extra import libdevice
import torch
from triton import cdiv
import triton

BLOCK_SIZE_M = 128
BLOCK_SIZE_N = 128
BLOCK_SIZE_K = 128

GROUP_SIZE_M = 8

@triton.jit
def tanh(x):
    # Tanh is just a scaled sigmoid
    return 2 * tl.sigmoid(2 * x) - 1

# from xformers impl.
@triton.jit
def gelu(x):
    """
    GeLU_ activation - Gaussian error linear unit

    .. _GeLU: https://arxiv.org/pdf/1606.08415.pdf
    """
    return 0.5 * x * (1 + tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))


def get_cuda_autotune_config():
    return [
        triton.Config({}, num_stages=stage, num_warps=warps)
        for stage in [3, 4, 5, 6]
        for warps in [4, 8]
    ]


@triton.autotune(configs=get_cuda_autotune_config(), key=['M', 'N', 'K'])
@triton.jit
def matmul_kernel_one_fp8(
        # Pointers to matrices
        a_ptr, b_ptr, fc1b_ptr, c_ptr,
        sparsity_indices_ptr, sparsity_indices_counts,
        sparse_act_unpacked_ptr,
        # Matrix dimensions
        M, N, K,
        # The stride variables represent how much to increase the ptr by when moving by 1
        # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
        # by to get the element one row down (A has M rows).
        stride_am, stride_ak,  #
        stride_bk, stride_bn,  #
        stride_cm, stride_cn,  #
        stride_unpacked_m, stride_unpacked_n, #
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
        scale_a,
        scale_b,
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    assert a_ptr.dtype.element_ty == b_ptr.dtype.element_ty, "a and b must have the same dtype"

    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m


    sparsity_indices_count = tl.load(sparsity_indices_counts + pid_m)
    
    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetic` section for details
    if pid_n * BLOCK_SIZE_N >= sparsity_indices_count:
        return
    
    # the % M and % N help to handle the case where the matrix is not a multiple of the block size
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = tl.load(sparsity_indices_ptr + (pid_m * N) + (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    
    unpacked_offset = (offs_am[:, None] * stride_unpacked_m + offs_bn[None, :] * stride_unpacked_n)
    
    # pacache = tl.load(sparse_act_unpacked_ptr + unpacked_offset)
    # pacache = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    # acc = tl.load(sparse_act_unpacked_ptr + unpacked_offset).to(tl.float32)
    # acc *= -1
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # We accumulate along the K dimension.
        acc = tl.dot(a, b, acc)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    acc = acc * tl.load(scale_a)
    acc = acc * tl.load(scale_b)

    bias = tl.load(fc1b_ptr + offs_bn)[None, :]
    # gelu
    acc = gelu(acc + bias)
    # acc = acc + bias

    acc = acc.to(tl.bfloat16)

    # post-gelu subtraction of prev_sparse_act_unpacked
    acc2 = acc - tl.load(sparse_act_unpacked_ptr + unpacked_offset)
    # Write back the block of the output matrix C with masks.
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M)

    tl.store(sparse_act_unpacked_ptr + unpacked_offset, acc, mask=c_mask)
    tl.store(c_ptrs, acc2, mask=c_mask)

def csp_mlp_mm1_fp8(a, b, fc1b, sparsity_indices, sparsity_indices_counts, sparse_act_unpacked_inout, sparse_act_packed_out, scale_a, scale_b):
    # Check constraints.
    M, K = a.shape
    N, K = b.shape
    
    num_m_blocks = cdiv(M, BLOCK_SIZE_M)
    num_n_blocks = cdiv(N, BLOCK_SIZE_N)

    assert b.shape[1] % BLOCK_SIZE_N == 0, "B must evenly divide BLOCK_SIZE_N"

    grid = lambda META: (num_m_blocks * num_n_blocks, )
    compiled = matmul_kernel_one_fp8[grid](
        a, b, fc1b, sparse_act_packed_out,  #
        sparsity_indices, sparsity_indices_counts,  #
        sparse_act_unpacked_inout, #
        M, N, K,  #
        a.stride(0), a.stride(1),  #
        b.stride(1), b.stride(0),  #
        sparse_act_packed_out.stride(0), sparse_act_packed_out.stride(1),
        sparse_act_unpacked_inout.stride(1), sparse_act_unpacked_inout.stride(0),
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M, scale_a, scale_b, #
    )
