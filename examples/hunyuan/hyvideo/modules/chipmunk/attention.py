import torch
from torch import nn
import torch.nn.functional as F
import chipmunk_tk_kernels
from einops import rearrange
import triton

from .chipmunk import bitpack, bitunpack
from .ops import torch_tk_attn, torch_colsum_attn_fwd, torch_csp_attn_fwd, torch_mask_to_indices
from .config import GLOBAL_CONFIG

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

class TKAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, return_l=False):
        return
        """
        Compute variable length attention in ThunderKittens.
        """
        pm = 192
        if q.shape[-2] % pm == 0:
            return torch.ops.chipmunk_tk_kernels.attn_fwd(q, k, v)

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
        o, l = torch.ops.chipmunk_tk_kernels.attn_fwd(qp, k, v)

        # unpad
        o = o[..., :n, :].contiguous()
        # leave l padded to pass back in
        l[..., n:, :] = 0

        if return_l:
            return o, l
        else:
            return o

class SparseAttention(nn.Module):
    def __init__(self, layer_num, start_step=0, full_every=100, start_layer=0):
        super().__init__()
        self.layer_num = layer_num
        self.indices_fn = None

        self.start_step = start_step
        self.full_every = full_every
        self.start_layer = start_layer

        self.inference_step = None

        self.attn_sparsity = 0
        self.attn_sparsity_count = 0

        self.topk = None
        self.prev_l = None
        # [b, h, qg] mask
        self.full_attn_query_groups = None

    def forward(self, q, k, v):
        # return
        """
        Compute variable length attention in ThunderKittens.

        Indices and indices counts are already rounded up to ceil(n / 192).
        """
        if self.inference_step == 0 and self.lr_mask is None:
            self.lr_mask = self.indices_fn()
            self.mask = self.lr_mask

        print(f'inference_step: {self.inference_step}, layer_num: {self.layer_num}')

        if (
            self.inference_step < self.start_step
            or self.layer_num < self.start_layer
            or self.inference_step % self.full_every == 0
        ):
            # TODO: colsum + topk + set mask
            if self.inference_step > 0 and self.topk > 0:
                o, cs, l = colsum_tk_attention(q, k, v, self.prev_l)
                self.mask = torch.zeros_like(self.lr_mask)
                # self.mask = torch.randike(self.lr_mask) < 0.01
                self.mask.scatter_(-1, cs.topk(k=self.topk, dim=-1).indices, True)
                # don't go over seqlen
                self.mask *= ((self.lr_mask.sum(dim=-1, keepdim=True) + self.topk) < q.shape[-2])
                self.mask = self.mask | self.lr_mask
                self.prev_l = l
                return o
            else:
                o, l = tk_attn(q, k, v, return_l=True)
                self.prev_l = l
                return o
        

        self.attn_sparsity += (self.mask.sum() / self.mask.numel())
        self.attn_sparsity_count += 1

        # indices, indices_counts = masktoinds(self.mask, multiple=128)
        # indices, indices_counts = torch.ops.chipmunk_tk_kernels.mask_to_indices(self.mask, 128, pad_to_multiple_of=192)
        indices, indices_counts = torch_mask_to_indices(self.mask, 128, pad_to_multiple_of=192)

        if q.shape[-2] % 192 == 0:
            return torch.ops.chipmunk_tk_kernels.csp_attn_fwd(q, k, v, indices, indices_counts)

        # pad
        n = q.shape[-2]
        padded_n = ((n + 192 - 1) // 192) * 192
        # print(f'padded_n: {padded_n}')
        # qp = F.pad(q, (0, 0, 0, padded_n - n))
        qp = torch.zeros(q.shape[:-2] + (padded_n, q.shape[-1]), dtype=q.dtype, device=q.device)
        qp[..., :n, :] = q
        # kp = F.pad(k, (0, 0, 0, padded_n - n))
        # vp = F.pad(v, (0, 0, 0, padded_n - n))
        # print(f'qp.shape: {qp.shape}')

        # pad indices to multiple of 16
        indicesp = torch.empty(indices.shape[:-1] + (padded_n,), dtype=indices.dtype, device=indices.device)
        indicesp[..., :n] = indices

        # compute
        # raise Exception(f'q: {q.shape}, k: {k.shape}, v: {v.shape}, inds: {indices.shape}, counts: {indices_counts.shape}, max indices: {indices.max(dim=-1)}, counts: {indices_counts}')

        # contiguous
        qp = qp.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        indicesp = indicesp.contiguous()

        # torch.save(qp, 'csp-qp.pt')
        # torch.save(q, 'csp-q.pt')
        # torch.save(k, 'csp-k.pt')
        # torch.save(v, 'csp-v.pt')
        # torch.save(indices, 'csp-indices.pt')
        # torch.save(indices_counts, 'csp-indices_counts.pt')
        # print(f'qp shape: {qp.shape}')
        # print(f'k shape: {k.shape}')
        # print(f'v shape: {v.shape}')
        # print(f'indicesp shape: {indicesp.shape}')
        # print(f'indices_counts shape: {indices_counts.shape}')
        # o = torch.ops.chipmunk_tk_kernels.csp_attn_fwd(qp, k, v, indicesp, indices_counts)
        o = torch_csp_attn_fwd(qp, k, v, indicesp, indices_counts)
        # print(f'o.shape before unpad: {o.shape}')

        if self.inference_step == 49:
            del self.mask
            del self.lr_mask

            if self.layer_num == 37:
                print(f'attn sparsity: {self.attn_sparsity / self.attn_sparsity_count}')

        # unpad
        o = o[..., :n, :].contiguous()
        return o

class SparseDiffAttention(nn.Module):
    def __init__(self, layer_num, o_cache_stream=None, o_cache_gpu_to_cpu_event=None, o_cache_cpu_to_gpu_event=None, o_cache_shape=None, start_step=0, full_every=100, start_layer=0):
        super().__init__()
        self.layer_num = layer_num
        self.o_cache_stream = o_cache_stream
        self.o_cache_gpu_to_cpu_event = o_cache_gpu_to_cpu_event
        self.o_cache_cpu_to_gpu_event = o_cache_cpu_to_gpu_event

        self.indices_fn = None

        self.start_step = start_step
        self.full_every = full_every
        self.start_layer = start_layer

        self.inference_step = None

        self.attn_sparsity = 0
        self.attn_sparsity_count = 0

        # [b, h, qg] mask
        self.sparse_attn_query_groups = None
        self.topk = None
        self.prev_l = None

        self.seqlen = None

        ### O-Cache Offloading ###
        #
        # We have
        #   (1) a preallocated pinned cpu tensor                     (not shared by layers),
        #   (2) a preallocated double buffer of n-major gpu tensors  (shared by layers),
        #   (3) a preallocated double buffer of b-major gpu tensors  (shared by layers).
        #
        self.o_cache_cpu = None
        if o_cache_stream is not None:
            # preallocate pinned cpu tensors for full pcie bandwidth!
            self.o_cache_cpu = torch.empty(
                # [n, b, h, d]
                o_cache_shape,
                device='cpu',
                dtype=torch.bfloat16,
                pin_memory=True,
            )
        self.o_cache_gpu_nmaj = None
        self.o_cache_gpu_bmaj = None
        ##########################

    # executes during forward pass of layer n - 1
    def load_o_cache(self):
        if self.o_cache_stream is None or self.prev_l is None:
            # no offloading
            self.o_cache_cpu_to_gpu_event.record()
            return

        # cpu -> gpu in separate stream
        with torch.cuda.stream(self.o_cache_stream):
            n = self.seqlen

            # n in first dim for contiguous copy with variable seqlen
            self.o_cache_gpu_nmaj[self.layer_num % 2].copy_(self.o_cache_cpu[:n], non_blocking=True)
            self.o_cache_gpu_bmaj[self.layer_num % 2].copy_(self.o_cache_gpu_nmaj[self.layer_num % 2].view_as(self.o_cache_gpu_bmaj[self.layer_num % 2]), non_blocking=False)
            self.o_cache_gpu = self.o_cache_gpu_bmaj[self.layer_num % 2]

            self.o_cache_cpu_to_gpu_event.record()


    # executes during forward pass of layer n
    def update_o_cache(self, o_cache_gpu):
        if self.o_cache_stream is None:
            # no offloading
            self.o_cache_gpu = o_cache_gpu
            self.o_cache_gpu_to_cpu_event.record()
        else:
            # gpu -> cpu in separate stream
            with torch.cuda.stream(self.o_cache_stream):
                # need to wait for default stream to finish producing o_cache_gpu
                torch.cuda.current_stream().wait_stream(torch.cuda.default_stream())
                n = self.seqlen
                # n in first dim for contiguous copy with variable seqlen
                self.o_cache_gpu_nmaj[self.layer_num % 2].copy_(o_cache_gpu.view_as(self.o_cache_gpu_nmaj[self.layer_num % 2]))
                self.o_cache_cpu[:n].copy_(self.o_cache_gpu_nmaj[self.layer_num % 2], non_blocking=True)

                self.o_cache_gpu_to_cpu_event.record()

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
            or inference_step % self.full_every == 0
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

                # self.attn_sparsity += (mask.sum() / mask.numel())
                # self.attn_sparsity_count += 1

                self.packed_mask, _ = bitpack(mask)

                # cache full - sparse
                indices, indices_counts = torch_mask_to_indices(mask, 128, pad_to_multiple_of=192)
                del mask
                spo = csp_tk_attn(q, k, v, indices, indices_counts)
                self.update_o_cache(o - spo)

                del spo
                del indices
                del indices_counts
                # torch.cuda.empty_cache()
                
                return o
            else:
                o, l = tk_attn(q, k, v, return_l=True)
                self.prev_l = l
                # dummy event for next wait
                if self.o_cache_stream is not None:
                    self.o_cache_gpu_to_cpu_event.record()
                return o
        


        self.packed_mask = self.packed_mask.to(q.device)
        mask = bitunpack(self.packed_mask, self.mask_shape)
        indices, indices_counts = torch_mask_to_indices(mask, 128, pad_to_multiple_of=192)

        o = csp_tk_attn(q, k, v, indices, indices_counts)
        o += self.o_cache_gpu
        # dummy event for next wait
        if self.o_cache_stream is not None:
            self.o_cache_gpu_to_cpu_event.record()

        # self.attn_sparsity += (mask.sum() / mask.numel())
        # self.attn_sparsity_count += 1
        # if self.layer_num == 37:
        #     print(f'attn sparsity: {self.attn_sparsity / self.attn_sparsity_count}')
        # raise Exception(f'attn sparsity: {self.attn_sparsity / self.attn_sparsity_count}')

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
