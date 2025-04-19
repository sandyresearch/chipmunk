import torch
from torch import nn
import torch.nn.functional as F
import chipmunk_tk_kernels
from einops import rearrange
import triton

from .chipmunk import bitpack, bitunpack
from .ops import torch_tk_attn, torch_colsum_attn_fwd, torch_csp_attn_fwd, torch_mask_to_indices
from .config import GLOBAL_CONFIG

from chipmunk.util.storage import AttnStorage

import torch._dynamo
# torch._dynamo.config.cache_size_limit = 1 << 32
# torch._dynamo.config.accumulated_cache_size_limit = 1 << 32

def test_tk_attn():
    b, h, n, d = 1, 24, 4224, 128
    q = torch.randn(b, h, n, d, device='cuda', dtype=torch.bfloat16)
    k = torch.randn(b, h, n, d, device='cuda', dtype=torch.bfloat16)
    v = torch.randn(b, h, n, d, device='cuda', dtype=torch.bfloat16)
    o, l = tk_attn(q, k, v, return_l=True)

    o += 1

    perf = lambda ms: (4 * b * h * n * n * d) * 1e-12 / (ms * 1e-3)
    ms = triton.testing.do_bench(lambda: tk_attn(q, k, v, return_l=True))
    print(f'tk: {perf(ms)} TFLOPS')

    ms = triton.testing.do_bench(lambda: F.scaled_dot_product_attention(q, k, v))
    print(f'spda: {perf(ms)} TFLOPS')

def tk_attn_forward(q, k, v, inference_step=None):
    return tk_attn(q, k, v, return_l=False)

def tk_attn(q, k, v, return_l=False):
    # return
    """
    Compute variable length attention in ThunderKittens.
    """
    pm = 192
    if q.shape[-2] % pm == 0:
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        o, l = torch_tk_attn(q, k, v)
        if return_l:
            return o, l
        else:
            return o

    # pad
    n = q.shape[-2]
    padded_n = ((n + pm - 1) // pm) * pm
    qp = torch.empty(q.shape[:-2] + (padded_n, q.shape[-1]), dtype=q.dtype, device=q.device)
    qp[..., :n, :] = q

    # contiguous
    qp = qp.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    # compute
    o, l = torch_tk_attn(qp, k, v)

    # unpad
    o = o[..., :n, :].contiguous()
    # leave l padded to pass back in
    l[..., n:, :] = 0

    if return_l:
        return o, l
    else:
        return o

def colsum_tk_attention(q, k, v, p, fuse_reduce=True):
        """
        Compute variable length attention in ThunderKittens.
        """
        wq = 16   # queries per warp
        pm = 192  # queries per producer (16 * 4 * 3)
        if q.shape[-2] % pm == 0:
            o, cs, l = torch_colsum_attn_fwd(q, k, v, p)
            if not fuse_reduce:
                cs = rearrange(cs, 'b h (m r) n -> b h m r n', r=pm//wq).sum(dim=-2)
            return o, cs, l

        # pad
        n = q.shape[-2]
        padded_n = ((n + pm - 1) // pm) * pm
        qp = torch.empty(q.shape[:-2] + (padded_n, q.shape[-1]), dtype=q.dtype, device=q.device)
        qp[..., :n, :] = q

        # contiguous
        qp = qp.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        p = p.contiguous()

        assert p.shape[-2] == padded_n

        # compute
        o, cs, l = torch_colsum_attn_fwd(qp, k, v, p)

        # unpad
        o = o[..., :n, :].contiguous()
        # leave l padded to pass back in
        l[..., n:, :] = 0
        if fuse_reduce:
            # use k since on single gpu q is padded to 119056
            kseq = k.shape[-2]
            kgroups = (kseq + pm - 1) // pm
            cs = cs[..., :kgroups, :kseq]
        else:
            cs = rearrange(cs, 'b h (m r) n -> b h m r n', r=pm//wq).sum(dim=-2)[..., :n]
        return o, cs, l

def csp_tk_attn(q, k, v, indices, indices_counts):
    if q.shape[-2] % 192 == 0:
        return torch_csp_attn_fwd(q, k, v, indices, indices_counts)

    # pad
    n = q.shape[-2]
    padded_n = ((n + 192 - 1) // 192) * 192
    qp = torch.zeros(q.shape[:-2] + (padded_n, q.shape[-1]), dtype=q.dtype, device=q.device)
    qp[..., :n, :] = q

    # pad indices to multiple of 16
    indicesp = indices

    # contiguous
    qp = qp.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    indicesp = indicesp.contiguous()

    o = torch_csp_attn_fwd(qp, k, v, indicesp, indices_counts)

    # unpad
    o = o[..., :n, :].contiguous()
    return o

class SparseDiffAttention(nn.Module):
    def __init__(self, layer_num, o_cache_stream=None, o_cache_gpu_to_cpu_event=None, o_cache_cpu_to_gpu_event=None, o_cache_shape=None, start_step=0, full_every=100, start_layer=0):
        super().__init__()
        self.layer_num = layer_num
        self.storage = AttnStorage(layer_num, init_names=['indices', 'out_cache'])

        self.start_step = start_step
        self.full_every = full_every
        self.start_layer = start_layer

        self.sparse_attn_query_groups = None
        self.topk = None
        self.prev_l = None
        self.seqlen = None

    @torch.compile(dynamic=False)
    def random_and_topk(self, cs):
        mask = torch.randint(0, 100, cs.shape, device=cs.device, dtype=torch.uint8) == 0
        mask.scatter_(-1, cs.topk(k=self.topk, dim=-1).indices, True)
        return mask

    # @torch.compile(dynamic=False)
    def forward(self, q, k, v, inference_step):
        """
        Compute variable length attention in ThunderKittens.

        Indices and indices counts are already rounded up to ceil(n / 192).
        """
        self.seqlen = q.shape[-2]

        if (
            inference_step < self.start_step
            # or inference_step % self.full_every == 0
            or inference_step in set([0, 1, 10, 25, 40])
        ):
            if inference_step > 0 and self.topk > 0:
                o, cs, l = colsum_tk_attention(q, k, v, self.prev_l)
                # 1% randomness
                mask = self.random_and_topk(cs)
                del cs
                lr_mask = bitunpack(self.packed_lr_mask, self.mask_shape)
                # don't go over seqlen
                mask *= self.sparse_attn_query_groups
                mask = mask | lr_mask
                del lr_mask
                self.prev_l = l

                self.storage.set_indices(bitpack(mask)[0])

                # cache full - sparse
                indices, indices_counts = torch_mask_to_indices(mask, 128, pad_to_multiple_of=192)
                del mask
                spo = csp_tk_attn(q, k, v, indices, indices_counts)
                self.storage.set_out_cache(o - spo)

                del spo
                del indices
                del indices_counts
                # torch.cuda.empty_cache()
                
                return o
            else:
                o, l = tk_attn(q, k, v, return_l=True)
                self.prev_l = l
                return o
        

        packed_mask = self.storage.get_indices()
        mask = bitunpack(packed_mask, self.mask_shape)
        del packed_mask
        indices, indices_counts = torch_mask_to_indices(mask, 128, pad_to_multiple_of=192)

        o = csp_tk_attn(q, k, v, indices, indices_counts)
        o += self.storage.get_out_cache()

        del mask
        del indices
        del indices_counts
        # torch.cuda.empty_cache()

        if inference_step == GLOBAL_CONFIG['steps'] - 1:
            del self.packed_mask
            del self.packed_lr_mask
            self.prev_l = None
            if hasattr(self, 'o_cache_gpu'):
                del self.o_cache_gpu

            # if self.layer_num == 37:
            #     print(f'attn sparsity: {self.attn_sparsity / self.attn_sparsity_count}')
                
        return o
