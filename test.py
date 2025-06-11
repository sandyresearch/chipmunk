import chipmunk
import torch

torch.set_default_device('cuda')
torch.set_default_dtype(torch.bfloat16)
torch.backends.cuda.enable_cudnn_sdp(False)

group_size = 192
kv_tile_size = 112
qkv_shape = (1, 24, 112*8, 128)
b, h, n, d = qkv_shape
n_groups = (n + group_size - 1) // group_size
n_kv_padded = (n + 4 - 1) // 4 * 4

# Input tensors
k = torch.randn(qkv_shape)
v = torch.randn(qkv_shape)
o = torch.zeros(qkv_shape)
q = torch.randn(qkv_shape)

# Expanded indices: (b, h, n_groups, n)
indices = torch.arange(n_kv_padded, dtype=torch.int32).expand(b, h, n_groups, n_kv_padded).contiguous()[:,:,:,:n]

# Counts: (b, h, n_groups) filled with n
counts = torch.full((b, h, n_groups), n, dtype=torch.int32, device='cuda')

# Call the kernel
torch.ops.chipmunk.csp_attn(q, k, v, o, indices, counts, 1)

o_ref = torch.nn.functional.scaled_dot_product_attention(q,k,v)

assert torch.allclose(o,o_ref,rtol=1e-1,atol=1e-1), breakpoint()