import chipmunk
import torch

torch.set_default_device('cuda')
torch.set_default_dtype(torch.bfloat16)

for seqlen in range(512, 2048, 64):
    b, h, n, d = 1, 24, seqlen, 128
    n_groups = (n + 192 - 1) // 192

    q = torch.randn(b, h, n, d)
    k = torch.randn(b, h, n, d)
    v = torch.randn(b, h, n, d)

    o = torch.zeros(b, h, n, d)

    indices = torch.arange(n, dtype=torch.int32).repeat((b, h, n_groups, 1)).contiguous()
    counts = torch.full((b, h, n_groups), n, dtype=torch.int32)

    chipmunk.ops.csp_attn(q, k, v, o, indices, counts, 1)

    o_ref = torch.nn.functional.scaled_dot_product_attention(q, k, v)

    o_max_diff = (o - o_ref).abs().max()
    o_mean_diff = (o - o_ref).abs().mean()
    
    print(f"seqlen: {seqlen} (% 192 = {seqlen % 192}, % 112 = {seqlen % 112}), o_max_diff: {o_max_diff:.2f}, o_mean_diff: {o_mean_diff:.2f}")
