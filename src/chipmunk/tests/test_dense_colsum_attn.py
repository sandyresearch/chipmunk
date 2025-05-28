import math
import chipmunk
import torch
from einops import rearrange

torch.set_default_device('cuda')
torch.set_default_dtype(torch.bfloat16)

for seqlen in range(512, 2048, 64):
    b, h, n, d = 1, 24, seqlen, 128
    n_groups   = (n + 192 - 1) // 192

    q = torch.randn(b, h, n, d)
    k = torch.randn(b, h, n, d)
    v = torch.randn(b, h, n, d)

    indices = torch.arange(n, dtype=torch.int32).repeat((b, h, n_groups, 1)).contiguous()
    counts  = torch.full((b, h, n_groups), n, dtype=torch.int32)

    # ─── Chipmunk kernels ────────────────────────────────────────────────
    o_dense,        lse = chipmunk.ops.dense_attn(q, k, v)
    o_dense_cs, cs, _   = chipmunk.ops.dense_colsum_attn(q, k, v, lse)

    # ─── PyTorch reference output ───────────────────────────────────────
    o_ref = torch.nn.functional.scaled_dot_product_attention(q, k, v)

    # ─── Reference LSE / nv computation ─────────────────────────────────
    softmax_scale = 1.0 / math.sqrt(d)
    logits = (q.float() @ k.float().transpose(-2, -1)) * softmax_scale
    m      = logits.max(dim=-1, keepdim=True).values
    logits -= m
    p      = logits.exp()
    l_ref  = p.sum(dim=-1, keepdim=True)           # ← “lse”
    pad_size = (192 - (p.size(2) % 192)) % 192
    p_padded = torch.nn.functional.pad(p, (0, 0, 0, pad_size))
    cs_ref = rearrange(p_padded.to(q.dtype), 'b h (m r) n -> b h m r n', r=192).sum(dim=-2).contiguous()

    # --------------------------------------------------------------------

    # diff metrics
    o_max_diff  = (o_dense_cs - o_ref).abs().max()
    o_mean_diff = (o_dense_cs - o_ref).abs().mean()
    cs_mean_diff = (cs - cs_ref).abs().mean()
    cs_max_diff     = (cs  - cs_ref.to(cs.dtype)).abs().max()

    print(
        f"seqlen {seqlen:4d} | "
        f"o_max_diff {o_max_diff:.2e}  o_mean_diff {o_mean_diff:.2e} (o_mean_abs {o_dense_cs.abs().mean():.2e}) | "
        f"cs_max_diff {cs_max_diff:.2e} cs_mean_diff {cs_mean_diff:.2e} (cs_mean_abs {cs.abs().mean():.2e})"
    )