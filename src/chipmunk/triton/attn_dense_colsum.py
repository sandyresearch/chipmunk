import torch
import math
import triton
import triton.language as tl

from torch.nn import functional as F

DEVICE = 'cuda'

cdiv = lambda a, b: (a + b - 1) // b

# We don't run auto-tuning every time to keep the tutorial fast. Keeping
# the code below and commenting out the equivalent parameters is convenient for
# re-tuning.
configs = [
    triton.Config({'BLOCK_M': BM, 'BLOCK_N': BN}, num_stages=s, num_warps=w) \
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

@triton.jit
def _full_attn_fwd_inner(acc, l_i, m_i, q,  #
                    prev_maxes_final_ptrs, #
                    prev_normalization_final_ptrs, #
                    blocksums_ptrs,
                    softmax_stride_b, softmax_stride_h, softmax_stride_n, #
                    blocksums_stride_b, blocksums_stride_h, blocksums_stride_m, blocksums_stride_n,
                    K_block_ptr, V_block_ptr,  #
                    start_m, qk_scale, seqlen,  #
                    H, #
                    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,  #
                    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,  #
                    N_CTX: tl.constexpr, fp8_v: tl.constexpr,
                    ):
    # non-causal full attention
    lo, hi = 0, N_CTX
    K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))
    off_hb = tl.program_id(1)
    off_b = off_hb // H
    off_h = off_hb % H
    softmax_data_offset = off_b.to(tl.int64) * softmax_stride_b + off_h.to(tl.int64) * softmax_stride_h + (start_m * BLOCK_M + tl.arange(0, BLOCK_M)) * softmax_stride_n
    # blocksums_ptrs += off_b.to(tl.int64) * blocksums_stride_b + off_h.to(tl.int64) * blocksums_stride_h + start_m * blocksums_stride_m + tl.arange(0, BLOCK_N) * blocksums_stride_n
    bsp = blocksums_ptrs + off_b.to(tl.int64) * blocksums_stride_b + off_h.to(tl.int64) * blocksums_stride_h + start_m * blocksums_stride_m + tl.arange(0, BLOCK_N) * blocksums_stride_n
    
    # previous m and l values
    prev_maxes_final = tl.load(prev_maxes_final_ptrs + softmax_data_offset)
    prev_normalization_final = tl.load(prev_normalization_final_ptrs + softmax_data_offset, mask=(offs_m < seqlen), other=1.0e6)

    # blocksums = tl.zeros([BLOCK_M], dtype=tl.float32)
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        k = tl.load(K_block_ptr)
        # q_dot_k = tl.dot(q, k)
        q_dot_k = tl.dot(q, k)
        # q_dot_k = tl.where(start_n + offs_n[None, :] < 4592, q_dot_k, -1.0e6)
        qk = q_dot_k
        m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
        # qk = qk * qk_scale - m_ij[:, None]
        qk = qk * qk_scale - m_ij[:, None]
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
        p = p.to(tl.bfloat16)
        acc = tl.dot(p, v, acc)
        m_i = m_ij

        # ---------------- PREVIOUS PHASE OF SOFTMAX -------------------
        qk_prev = q_dot_k * qk_scale - prev_maxes_final[:, None]
        p_prev = tl.math.exp2(qk_prev)
        p_prev = (p_prev / prev_normalization_final[:, None])
        # p_prev = tl.where(offs_m[:, None] < seqlen, p_prev, 0)
        blocksums = tl.sum(p_prev, 0)
        tl.store(bsp, blocksums, mask=(start_n + offs_n) < seqlen)

        # ----------------- UPDATE POINTERS -----------------
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        bsp += BLOCK_N * blocksums_stride_n
    
    return acc, l_i, m_i

@triton.autotune(list(filter(keep, configs)), key=["N_CTX", "HEAD_DIM"])
@triton.jit
def _full_attn_fwd(Q, K, V, sm_scale, M, L, Out, seqlen,  #
              prev_maxes_ptr, #
              prev_normalization_final_ptrs, #
              blocksums_ptrs, #
              softmax_stride_b, softmax_stride_h, softmax_stride_n, #
              blocksums_stride_b, blocksums_stride_h, blocksums_stride_m, blocksums_stride_n, #
              stride_qz, stride_qh, stride_qm, stride_qk,  #
              stride_kz, stride_kh, stride_kn, stride_kk,  #
              stride_vz, stride_vh, stride_vk, stride_vn,  #
              stride_oz, stride_oh, stride_om, stride_on,  #
              Z, H, N_CTX,  #
              HEAD_DIM: tl.constexpr,  #
              BLOCK_M: tl.constexpr,  #
              BLOCK_N: tl.constexpr,  #
              STAGE: tl.constexpr,  #
              ):
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh

    # block pointers
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    v_order: tl.constexpr = (0, 1) if V.dtype.element_ty == tl.float8e5 else (1, 0)
    V_block_ptr = tl.make_block_ptr(
        base=V + qvk_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, HEAD_DIM),
        order=v_order,
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + qvk_offset,
        shape=(HEAD_DIM, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_N),
        order=(0, 1),
    )
    # O_block_ptr = tl.make_block_ptr(
    #     base=Out + qvk_offset,
    #     shape=(N_CTX, HEAD_DIM),
    #     strides=(stride_om, stride_on),
    #     offsets=(start_m * BLOCK_M, 0),
    #     block_shape=(BLOCK_M, HEAD_DIM),
    #     order=(1, 0),
    # )
    offs_o = (start_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]) * stride_om + tl.arange(0, HEAD_DIM)[None, :] * stride_on
    O_ptrs = Out + qvk_offset + offs_o

    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    # load scales
    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/log(2)
    # load q: it will stay in SRAM throughout
    q = tl.load(Q_block_ptr)
    # stage 1: off-band
    # For causal = True, STAGE = 3 and _attn_fwd_inner gets 1 as its STAGE
    # For causal = False, STAGE = 1, and _attn_fwd_inner gets 3 as its STAGE
    acc, l_i, m_i = _full_attn_fwd_inner(
        acc, l_i, m_i, q, 
        prev_maxes_ptr,  
        prev_normalization_final_ptrs, 
        blocksums_ptrs,
        softmax_stride_b, softmax_stride_h, softmax_stride_n,
        blocksums_stride_b, blocksums_stride_h, blocksums_stride_m, blocksums_stride_n,
        K_block_ptr, V_block_ptr,  #
        start_m, qk_scale, seqlen,  #
        H, #
        BLOCK_M, HEAD_DIM, BLOCK_N,  #
        4 - STAGE, offs_m, offs_n, N_CTX, V.dtype.element_ty == tl.float8e5,  #
    )
    # epilogue
    # m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    m_ptrs = M + off_hz * N_CTX + offs_m
    l_ptrs = L + off_hz * N_CTX + offs_m
    tl.store(m_ptrs, m_i, mask=offs_m < seqlen)
    tl.store(l_ptrs, l_i, mask=offs_m < seqlen)
    # tl.store(O_block_ptr, acc.to(Out.type.element_ty))
    tl.store(O_ptrs, acc.to(Out.type.element_ty), mask=offs_m[:, None] < seqlen)
    # tl.store(O_ptrs, acc.to(Out.type.element_ty))

class _full_attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, prev_lse):
        prev_maxes, prev_normalization = prev_lse
        # shape constraints
        HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
        # when v is in float8_e5m2 it is transposed.
        HEAD_DIM_V = v.shape[-1]
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}
        sm_scale = 1/math.sqrt(HEAD_DIM_K)
        stage = 1

        # mb = q.shape[2] // 128 if q.shape[2] % 128 == 0 else q.shape[2] // 128 + 1
        mb = triton.cdiv(q.shape[2], 64)
        # print(f'mb: {mb}')
        # print(f'q.shape[2] // 128: {q.shape[2] // 128}')
        # print(f'triton cdiv: {triton.cdiv(q.shape[2], 128)}')

        grid = lambda args: (triton.cdiv(q.shape[2], args["BLOCK_M"]), q.shape[0] * q.shape[1], 1)
        # grid = lambda args: (mb, q.shape[0] * q.shape[1], 1)

        # print(f'grid: {grid({})}')
        
        M = torch.zeros((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        L = torch.zeros_like(M, dtype=torch.float32)
        o = torch.empty_like(q)
        
        blocksums = torch.zeros((q.shape[0], q.shape[1], mb, q.shape[2]), device=q.device, dtype=torch.float32)
        # print(f'blocksums: {blocksums.shape}')
        seqlen = q.shape[2]
        # seqlen = 4592
        _full_attn_fwd[grid](
            q, k, v, sm_scale, M, L, o, seqlen,  #
            prev_maxes, #
            prev_normalization, #
            blocksums,
            prev_maxes.stride(0), prev_maxes.stride(1), prev_maxes.stride(2), #
            blocksums.stride(0), blocksums.stride(1), blocksums.stride(2), blocksums.stride(3), #
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),  #
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),  #
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),  #
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),  #
            q.shape[0], q.shape[1],  #
            N_CTX=q.shape[2],  #
            HEAD_DIM=HEAD_DIM_K,  #
            STAGE=stage
        )

        return o, blocksums, (M, L)

dense_colsum_attn = _full_attention.apply
