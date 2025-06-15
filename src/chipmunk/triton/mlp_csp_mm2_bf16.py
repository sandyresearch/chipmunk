import triton.language as tl
import torch
import triton

BLOCK_SIZE_M = 128
BLOCK_SIZE_N = 256
BLOCK_SIZE_K = 64
GROUP_SIZE_M = 8

def get_cuda_autotune_config():
    return [
        triton.Config({}, num_stages=3, num_warps=8),
    ]

@triton.jit
def _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS):
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n

@triton.autotune(configs=get_cuda_autotune_config(), key=['M', 'N', 'K'])
@triton.jit
def csp_mlp_mm2_kernel(
        # Pointers to matrices
        a_ptr, b_ptr, c_ptr, #
        sparsity_indices_ptr, sparsity_indices_counts, #
        # Matrix dimensions
        M, N, K,
        # The stride variables represent how much to increase the ptr by when moving by 1
        # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
        # by to get the element one row down (A has M rows).
        stride_am, stride_ak,  #
        stride_bk, stride_bn,  #
        stride_cm, stride_cn,  #
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #
        GROUP_SIZE_M: tl.constexpr
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    start_pid = tl.program_id(axis=0)
    num_sms = tl.num_programs(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_tiles = num_pid_m * num_pid_n
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    for tile_id in tl.range(start_pid, num_tiles, num_sms):
        pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, num_sms)

        sparsity_indices_count = tl.load(sparsity_indices_counts + pid_m)

        # ----------------------------------------------------------
        # Create pointers for the first blocks of A and B.
        # We will advance this pointer as we move in the K direction
        # and accumulate
        # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
        # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
        # See above `Pointer Arithmetic` section for details

        offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        c_ptrs = c_ptr + (offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn)
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        # accumulator = tl.load(c_ptrs).to(tl.float32)

        # offs_k = tl.load(sparsity_indices_block_ptr)
        offs_ak = tl.arange(0, BLOCK_SIZE_K)
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak)
        # b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
        sparsity_indices_block_ptrs = sparsity_indices_ptr + (pid_m * K) + tl.arange(0, BLOCK_SIZE_K)
        offs_bk = tl.load(sparsity_indices_block_ptrs)
        # -----------------------------------------------------------
        # Iterate to compute a block of the C matrix.
        # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
        # of fp32 values for higher accuracy.
        # `accumulator` will be converted back to fp16 after the loop.
        
        k_iters = tl.cdiv(sparsity_indices_count, BLOCK_SIZE_K)
        for k in range(0, k_iters):
            # offs_bk = tl.arange(0, BLOCK_SIZE_K) + k * BLOCK_SIZE_K
            b_ptrs = b_ptr + (offs_bk[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
            sparsity_indices_block_ptrs += BLOCK_SIZE_K
            offs_bk = tl.load(sparsity_indices_block_ptrs)
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
            # We accumulate along the K dimension.
            accumulator = tl.dot(a, b, accumulator)
            a_ptrs += BLOCK_SIZE_K * stride_ak
            # Advance the ptrs to the next K block.
        # You can fuse arbitrary activation functions here
        # while the accumulator is still in FP32!
        c = accumulator.to(tl.bfloat16)
        c += tl.load(c_ptrs)

        # -----------------------------------------------------------
        # Write back the block of the output matrix C with masks.
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, c, mask=c_mask)


def csp_mlp_mm2_bf16(a, b, sparsity_indices, sparsity_indices_counts, output, num_sms=torch.cuda.get_device_properties(0).multi_processor_count):
    # Check constraints.
    M, K = a.shape
    K, N = b.shape
    
    assert b.shape[1] % BLOCK_SIZE_N == 0, "B must evenly divide BLOCK_SIZE_N"

    c = output
    # 1D launch kernel where each block gets its own program.
    return csp_mlp_mm2_kernel[(num_sms, )](
        a, b, c,  #
        sparsity_indices, sparsity_indices_counts,  #
        M, N, K,  #
        a.stride(0), a.stride(1),  #
        b.stride(0), b.stride(1),  #
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M, #
    )

csp_mlp_mm2_bf16_function_ptr = csp_mlp_mm2_bf16(
    torch.randn((256, 256), dtype=torch.bfloat16, device='cuda'), 
    torch.randn((256, 256), dtype=torch.bfloat16, device='cuda'), 
    torch.arange(0, 256, 1, device='cuda', dtype=torch.int32).repeat(2, 1).contiguous(), 
    torch.full((2,), 256, device='cuda', dtype=torch.int32), 
    output=torch.empty((256, 256), device='cuda', dtype=torch.bfloat16),
    num_sms=1
).function

__all__ = ['csp_mlp_mm2_bf16', 'csp_mlp_mm2_bf16_function_ptr']