import chipmunk
import torch
import triton

torch.set_default_device('cuda')
torch.set_default_dtype(torch.bfloat16)

for seqlen in range(4096, 4096+128, 128):
    for is_contiguous in [False, True]:
        b, h, n, d = 1, 24, seqlen, 128
        n_groups = (n + 192 - 1) // 192

        def make_tensor(shape, is_contiguous, fill_value=None):
            if is_contiguous:
                new_vec = torch.randn(shape)
            else:
                new_shape = (shape[2], shape[0], shape[1], shape[3])
                new_vec = torch.randn(*new_shape)
                new_vec = new_vec.permute(1, 2, 0, 3)
            if fill_value is not None:
                new_vec.fill_(fill_value)
            return new_vec
                

        q = make_tensor((b, h, n, d), is_contiguous)
        k = make_tensor((b, h, n, d), is_contiguous)
        v = make_tensor((b, h, n, d), is_contiguous)

        o, _ = torch.ops.chipmunk.dense_attn(q, k, v)

        o_ref = torch.nn.functional.scaled_dot_product_attention(q, k, v)

        o_max_diff = (o - o_ref).abs().max()
        o_mean_diff = (o - o_ref).abs().mean()
        
        print(f"is_contig={is_contiguous}, seqlen: {seqlen} (% 192 = {seqlen % 192}, % 128 = {seqlen % 112}), o_max_diff: {o_max_diff:.2f}, o_mean_diff: {o_mean_diff:.2f}")

        time_ours = triton.testing.do_bench(lambda: torch.ops.chipmunk.dense_attn(q, k, v), warmup=100, rep=1000)
        time_ref = triton.testing.do_bench(lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v), warmup=100, rep=1000)
        print(f"time_ours: {time_ours:.3f}, time_ref: {time_ref:.3f}")
