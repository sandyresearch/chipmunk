import triton
import triton.language as tl
import torch
import math
import torch.nn.functional as F

DEVICE = 'cuda'


@triton.jit
def _attn_fwd_inner(acc, l_i, m_i, q,  #
                    K_block_ptr, V_block_ptr,  #
                    start_m, qk_scale,  #
                    seqlen,
                    stride_vk, stride_kn,
                    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,  #
                    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,  #
                    N_CTX: tl.constexpr, fp8_v: tl.constexpr, should_mask_kv: tl.constexpr):
    # loop over k, v and update accumulator
    for start_n in range(0, N_CTX, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        k = tl.load(K_block_ptr)
        # qk = tl.dot(q, k) + tl.where(start_n + offs_n[None, :] < seqlen, 0, -1.0e6)
        qk = tl.dot(q, k)
        # qk = tl.where(start_n + offs_n[None, :] < 4592, qk, -1.0e6)
        m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
        qk = qk * qk_scale - m_ij[:, None]
        # qk = qk * qk_scale - m_ij[:, None] + tl.where(start_n + offs_n[None, :] < 4592, 0, -1.0e6)
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
        # v = tl.where(start_n + offs_n[:, None] < 4592, v, 0).to(tl.bfloat16)
        if fp8_v:
            p = p.to(tl.float8e5)
        else:
            p = p.to(tl.bfloat16)
        acc = tl.dot(p, v, acc)
        # update m_i and l_i
        m_i = m_ij
        # V_block_ptr += BLOCK_N * stride_vk
        # K_block_ptr += BLOCK_N * stride_kn
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
    return acc, l_i, m_i


# We don't run auto-tuning every time to keep the tutorial fast. Keeping
# the code below and commenting out the equivalent parameters is convenient for
# re-tuning.
configs = [
    triton.Config({'BLOCK_M': BM, 'BLOCK_N': BN}, num_stages=s, num_warps=w) \
    # for BM in [64, 128]\
    # for BN in [32, 64]\
    for BM in [64]\
    for BN in [64]\
    for s in ([3, 4, 7])\
    for w in [4, 8]\
]


def keep(conf):
    BLOCK_M = conf.kwargs["BLOCK_M"]
    BLOCK_N = conf.kwargs["BLOCK_N"]
    if BLOCK_M * BLOCK_N < 128 * 128 and conf.num_warps == 8:
        return False
    return True


@triton.autotune(list(filter(keep, configs)), key=["N_CTX", "HEAD_DIM"])
@triton.jit
def _attn_fwd(Q, K, V, sm_scale, M, L, Out, seqlen,  #
              stride_qz, stride_qh, stride_qm, stride_qk,  #
              stride_kz, stride_kh, stride_kn, stride_kk,  #
              stride_vz, stride_vh, stride_vk, stride_vn,  #
              stride_oz, stride_oh, stride_om, stride_on,  #
              Z, H, N_CTX,  #
              HEAD_DIM: tl.constexpr,  #
              BLOCK_M: tl.constexpr,  #
              BLOCK_N: tl.constexpr,  #
              STAGE: tl.constexpr,
              should_mask_kv: tl.constexpr  #
              ):
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    qo_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
    k_offset = off_z.to(tl.int64) * stride_kz + off_h.to(tl.int64) * stride_kh
    v_offset = off_z.to(tl.int64) * stride_vz + off_h.to(tl.int64) * stride_vh
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_headsize = tl.arange(0, HEAD_DIM)
    
    Q_block_ptr = (
        Q
        + qo_offset
        + offs_m[:, None] * stride_qm
        + offs_headsize[None, :] * stride_qk
    )
    v_order: tl.constexpr = (0, 1) if V.dtype.element_ty == tl.float8e5 else (1, 0)
    # V_block_ptr = (
    #     V
    #     + v_offset
    #     + offs_n[:, None] * stride_vk
    #     + offs_headsize[None, :] * stride_vn
    # )
    
    V_block_ptr = tl.make_block_ptr(
        base=V + v_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, HEAD_DIM),
        order=v_order,
    )

    # K_block_ptr = (
    #     K
    #     + k_offset
    #     + (offs_n[None, :] // BLOCK_N) * stride_kn
    #     + offs_headsize[:, None] * stride_kk
    # )
    K_block_ptr = tl.make_block_ptr(
        base=K + k_offset,
        shape=(HEAD_DIM, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_N),
        order=(0, 1),
    )
    O_ptrs = Out + qo_offset + (start_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]) * stride_om + tl.arange(0, HEAD_DIM)[None, :] * stride_on

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    qk_scale = sm_scale
    qk_scale *= 1.44269504 
    qo_mask = (offs_m < N_CTX)[:, None]
    q = tl.load(Q_block_ptr, mask=qo_mask)
    acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, K_block_ptr, V_block_ptr,  #
                                    start_m, qk_scale, seqlen,  #
                                    stride_vk, stride_kn,
                                    BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                    4 - STAGE, offs_m, offs_n, N_CTX, V.dtype.element_ty == tl.float8e5, should_mask_kv  #
                                    )
    acc = acc / l_i[:, None]
    m_ptrs = M + off_hz * N_CTX + offs_m
    l_ptrs = L + off_hz * N_CTX + offs_m
    tl.store(m_ptrs, m_i, mask=offs_m < seqlen)
    tl.store(l_ptrs, l_i, mask=offs_m < seqlen)
    tl.store(O_ptrs, acc.to(Out.type.element_ty), mask=qo_mask)

class _attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v):
        # shape constraints
        HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
        # when v is in float8_e5m2 it is transposed.
        HEAD_DIM_V = v.shape[-1]
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}
        o = torch.empty_like(q)
        sm_scale = 1/math.sqrt(HEAD_DIM_K)
        stage = 1
        should_mask_kv = q.shape[-2] % 64 != 0
        extra_kern_args = {}
        grid = lambda args: (triton.cdiv(q.shape[2], args["BLOCK_M"]), q.shape[0] * q.shape[1], 1)
        M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        L = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        seqlen = q.shape[2]
        _attn_fwd[grid](
            q, k, v, sm_scale, M, L, o, seqlen,  #
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),  #
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),  #
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),  #
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),  #
            q.shape[0], q.shape[1],  #
            N_CTX=q.shape[2],  #
            HEAD_DIM=HEAD_DIM_K,  #
            STAGE=stage,  #
            should_mask_kv=should_mask_kv,
            **extra_kern_args)

        return o, (M.unsqueeze(-1), L.unsqueeze(-1))


dense_attn = _attention.apply

def main():
    """
    Test on an arbitrary sequence length that % 64 != 0.
    """
    torch.set_default_device('cuda')
    torch.set_default_dtype(torch.bfloat16)

    qkv_shape = (1, 24, 2385, 128)
    q = torch.randn(qkv_shape)
    k = torch.randn(qkv_shape)
    v = torch.randn(qkv_shape)
    o, (M, L) = dense_attn(q, k, v)
    o_ref = F.scaled_dot_product_attention(q, k, v)
    print(o.shape, o_ref.shape)
    print(torch.allclose(o, o_ref, atol=1e-1, rtol=1e-1))

if __name__ == '__main__':
    main()
