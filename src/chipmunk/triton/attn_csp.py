import torch
import math
import triton
import triton.language as tl
from chipmunk.util import get_kernel_config_attn
from einops import rearrange

DEVICE = 'cuda'

cdiv = lambda a, b: (a + b - 1) // b
configs = [
    triton.Config({'BLOCK_M': BM, 'BLOCK_N': BN}, num_stages=s, num_warps=w) \
    for BM in [64]\
    for BN in [64]\
    for s in [3]\
    for w in [4]\
]
def keep(conf):
    BLOCK_M = conf.kwargs["BLOCK_M"]
    BLOCK_N = conf.kwargs["BLOCK_N"]
    if BLOCK_M * BLOCK_N < 128 * 128 and conf.num_warps == 8:
        return False
    return True

@triton.jit
def _sparse_attn_fwd_inner(acc, l_i, m_i, q,  #
                    K_block_ptr_orig, V_block_ptr_orig,  #
                    start_m, qk_scale,  #
                    indices_group,
                    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,  #
                    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,  #
                    N_CTX: tl.constexpr, fp8_v: tl.constexpr, #
                    stride_k_seqlen, stride_v_seqlen,  #
                    sparsity_indices_ptr, sparsity_counts_ptr, #
                    should_mask_kv: tl.constexpr,
                    ):
    sparsity_count = tl.load(sparsity_counts_ptr + indices_group)
    # sparsity_count = tl.load(sparsity_counts_ptr + start_m)
    # sparsity_count = 0
    sparsity_offsets = tl.arange(0, BLOCK_N)
    sparsity_indices_ptr += indices_group * N_CTX + sparsity_offsets
    # sparsity_indices_ptr += start_m * N_CTX + sparsity_offsets
    n_iters = tl.cdiv(sparsity_count, BLOCK_N)
    cur_iter = 0
    # loop over k, v and update accumulator
    for start_n in range(0, sparsity_count, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        sparsity_indices = tl.load(sparsity_indices_ptr)
        # sparsity_indices = tl.arange(0, BLOCK_N) + start_n
        # sparsity_indices = tl.zeros_like(sparsity_indices)
        
        tl.device_assert(tl.max(sparsity_indices) < N_CTX)
        tl.device_assert(tl.min(sparsity_indices) >= 0)

        K_block_ptr = K_block_ptr_orig + (sparsity_indices[None, :]) * stride_k_seqlen
        V_block_ptr = V_block_ptr_orig + (sparsity_indices[:, None]) * stride_v_seqlen
        # Commented out lines are for when we use random sparsity counts, in production it's always a multiple of BLOCK_N = 64
        # K_block_ptr = K_block_ptr_orig + (sparsity_indices[None, :] % N_CTX) * stride_k_seqlen
        # V_block_ptr = V_block_ptr_orig + (sparsity_indices[:, None] % N_CTX) * stride_v_seqlen
        # is_valid_mask = sparsity_offsets < sparsity_count - start_n # shape (BLOCK_N,)
        k = tl.load(K_block_ptr)
        qk = tl.dot(q, k)
        m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
        # qk = qk * qk_scale + tl.where(is_valid_mask[None, :], 0, -1.0e6)
        # qk -= m_ij[:, None]
        qk = qk * qk_scale - m_ij[:, None] # use fused multiply add!
        if should_mask_kv:
            qk = tl.where(start_n + offs_n[None, :] < N_CTX, qk, -1.0e6)
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        # -- update output accumulator --
        acc = acc * alpha[:, None]
        # update acc
        v = tl.load(V_block_ptr)
        p = p.to(tl.bfloat16)
        acc = tl.dot(p, v, acc)
        m_i = m_ij
        sparsity_indices_ptr += BLOCK_N
        cur_iter += 1

    return acc, l_i, m_i

@triton.autotune(list(filter(keep, configs)), key=["N_CTX", "HEAD_DIM"])
@triton.jit
def _sparse_attn_fwd(Q, K, V, sm_scale, M, L, Out, Out_accum, Out_scale: tl.constexpr, #
              sparsity_indices, sparsity_counts, #
              stride_qz, stride_qh, stride_qm, stride_qk,  #
              stride_kz, stride_kh, stride_kn, stride_kk,  #
              stride_vz, stride_vh, stride_vk, stride_vn,  #
              stride_oz, stride_oh, stride_om, stride_on,  #
              stride_spiz, stride_spih,  #
              stride_spcz, stride_spch,  #
              Z, H, N_CTX,  #
              HEAD_DIM: tl.constexpr,  #
              BLOCK_M: tl.constexpr,  #
              BLOCK_N: tl.constexpr,  #
              STAGE: tl.constexpr,
              should_mask_kv: tl.constexpr,  #
              num_qg_per_indices_group: tl.constexpr,
              ):
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    qo_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
    k_offset = off_z.to(tl.int64) * stride_kz + off_h.to(tl.int64) * stride_kh
    v_offset = off_z.to(tl.int64) * stride_vz + off_h.to(tl.int64) * stride_vh

    spi_offset = off_z.to(tl.int64) * stride_spiz + off_h.to(tl.int64) * stride_spih
    spi_ptr = sparsity_indices + spi_offset
    spc_offset = off_z.to(tl.int64) * stride_spcz + off_h.to(tl.int64) * stride_spch
    spc_ptr = sparsity_counts + spc_offset

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_headsize = tl.arange(0, HEAD_DIM)

    indices_group = start_m // num_qg_per_indices_group

    # block pointers
    Q_block_ptr = (
        Q
        + qo_offset
        + offs_m[:, None] * stride_qm
        + offs_headsize[None, :] * stride_qk
    )
    K_block_ptr = (
        K
        + k_offset
        + (offs_n[None, :] // BLOCK_N) * stride_kn
        + offs_headsize[:, None] * stride_kk
    )
    V_block_ptr = (
        V
        + v_offset
        + (offs_n[:, None] // BLOCK_N) * stride_vk
        + offs_headsize[None, :] * stride_vn
    )
    O_block_ptr = (
        Out
        + qo_offset
        + offs_m[:, None] * stride_om
        + offs_headsize[None, :] * stride_on
    )
    O_accum_block_ptr = (
        Out_accum
        + qo_offset
        + offs_m[:, None] * stride_om
        + offs_headsize[None, :] * stride_on
    )
    # initialize offsets
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    # load scales
    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/log(2)
    # load q: it will stay in SRAM throughout
    qo_mask = (offs_m < N_CTX)[:, None]
    q = tl.load(Q_block_ptr, mask=qo_mask)
    # stage 1: off-band
    # For causal = True, STAGE = 3 and _sparse_attn_fwd_inner gets 1 as its STAGE
    # For causal = False, STAGE = 1, and _sparse_attn_fwd_inner gets 3 as its STAGE
    acc, l_i, m_i = _sparse_attn_fwd_inner(acc, l_i, m_i, q, K_block_ptr, V_block_ptr,  #
                                    start_m, qk_scale,  #
                                    indices_group,
                                    BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                    4 - STAGE, offs_m, offs_n, N_CTX, V.dtype.element_ty == tl.float8e5,  #
                                    stride_kn, stride_vk, #
                                    spi_ptr, spc_ptr, #
                                    should_mask_kv,
                                    )
    # epilogue
    # m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    m_ptrs = M + off_hz * N_CTX + offs_m
    l_ptrs = L + off_hz * N_CTX + offs_m
    tl.store(m_ptrs, m_i, mask=offs_m < N_CTX)
    tl.store(l_ptrs, l_i, mask=offs_m < N_CTX)
    acc *= Out_scale # will get optimized out when Out_scale is 1.0 since it's tl.constexpr
    acc += tl.load(O_accum_block_ptr, mask=qo_mask)
    tl.store(O_block_ptr, acc.to(Out.type.element_ty), mask=qo_mask)

class _sparse_attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, sparsity_indices, sparsity_counts, O_scale = 1.0):
        o_accum = torch.zeros_like(q)
        sm_scale = 1.0 / math.sqrt(q.shape[-1])
        # shape constraints
        HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
        # when v is in float8_e5m2 it is transposed.
        HEAD_DIM_V = v.shape[-1]
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}
        o = torch.empty_like(q)
        stage = 1
        should_mask_kv = q.shape[-2] % 64 != 0
        extra_kern_args = {}

        bm = get_kernel_config_attn()['bm']
        assert bm % 64 == 0, "BM must be a multiple of 64"
        num_qg_per_indices_group = bm // 64

        grid = lambda args: (triton.cdiv(q.shape[2], args["BLOCK_M"]), q.shape[0] * q.shape[1], 1)
        M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        L = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        _sparse_attn_fwd[grid](
            q, k, v, sm_scale, M, L, o, o_accum, O_scale,  #
            sparsity_indices, sparsity_counts, #
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),  #
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),  #
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),  #
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),  #
            sparsity_indices.stride(0), sparsity_indices.stride(1), #
            sparsity_counts.stride(0), sparsity_counts.stride(1), #
            q.shape[0], q.shape[1],  #
            N_CTX=q.shape[2],  #
            HEAD_DIM=HEAD_DIM_K,  #
            STAGE=stage,  #
            should_mask_kv=should_mask_kv,
            num_qg_per_indices_group=num_qg_per_indices_group,
            **extra_kern_args)

        return o, (M.unsqueeze(-1), L.unsqueeze(-1))

csp_attn = _sparse_attention.apply


def main():
    import pickle
    import chipmunk
    q, k, v, inds, counts = pickle.load(open('tensors.pkl', 'rb'))
    inds = inds[:, :, :, :q.shape[2]].contiguous()
    if not torch.all(counts % 64 == 0):
        breakpoint()
    for b in range(inds.shape[0]):
        for h in range(0, inds.shape[1]):
            for m in range(0, inds.shape[2]):
                # inds[b,h,m,:] = torch.arange(inds.shape[3]-1, -1, -1)
                # inds[b,h,m,:] = torch.arange(0, inds.shape[3])
                relevant_indices = inds[b,h,m,:counts[b,h,m]]
                if not torch.all((relevant_indices >= 0) & (relevant_indices < q.shape[2])):
                    breakpoint()
                pass
    print('beginning kernel...')
    o = chipmunk.ops.csp_attn(q, k, v, inds, counts)
    print(o.shape)

if __name__ == '__main__':
    main()