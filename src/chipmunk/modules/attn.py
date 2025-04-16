import torch
from torch import Tensor
from ..util.layer_counter import LayerCounter
from ..util.config import GLOBAL_CONFIG
from einops import rearrange
import chipmunk.ops
import torch.nn.functional as F
import triton

other_stream = torch.cuda.Stream()

class SparseDiffAttn:
    def __init__(self, layer_counter: LayerCounter):
        self.layer_counter = layer_counter

    def _fast_bsd_attention_192(self, q: Tensor, k: Tensor, v: Tensor, inference_step: int, do_full_step: bool) -> Tensor:
        attn_config = GLOBAL_CONFIG['sparsity']['attention']
        bm = 192
        layer = self.layer_num
        sm_scale = 1 / (q.shape[-1] ** 0.5)
        if layer <= 1: # in pctx.full-layers
            o = F.scaled_dot_product_attention(q, k, v)
            return o
        self.default_stream.wait_stream(other_stream)

        if do_full_step:
            if inference_step == 0:
                o, l_ours = chipmunk.ops.dense_attn(q, k, v)
                l_ours[..., k.shape[-2]:, :] = 0
                self.f_softmax_norm_l = l_ours
                return o
            elif inference_step == 1:
                prev_l = self.f_softmax_norm_l
                o, bs, l = chipmunk.ops.dense_colsum_attn(q, k, v, prev_l)
                tk = int(256 * round((attn_config['top_keys'] * q.shape[-2]) / 256))
                # Fill padding values with -100000 to ensure they aren't selected by topk
                kseq = k.shape[-2]
                kgroups = (kseq + bm - 1) // bm
                bs = bs[..., :kgroups, :kseq]
                inds = torch.topk(bs, k=tk, dim=-1).indices
                counts = torch.full((q.shape[0], q.shape[1], triton.cdiv(q.shape[-2], bm)), tk, device=q.device, dtype=torch.int32)
                inds = torch.cat([inds, torch.zeros((*counts.shape, q.shape[-2] - tk), device=q.device, dtype=torch.int32)], dim=-1).to(torch.int32)
                self.f_attn_inds = inds
                self.f_attn_counts = counts
            else:
                # o, m, l = base_attention(q, k, v, False, sm_scale)
                o, l_ours = chipmunk.ops.dense_attn(q, k, v)

            inds = self.f_attn_inds
            counts = self.f_attn_counts
            self.f_o_cache = o.clone()
            chipmunk.ops.csp_attn(q, k, v, self.f_o_cache, inds, counts, -1)
            self.f_o_cache_ref = self.f_o_cache.clone()
            return o

        inds = self.f_attn_inds
        counts = self.f_attn_counts
        o = self.f_o_cache
        chipmunk.ops.csp_attn(q, k, v, o, inds, counts, 1)
        
        return o

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        inference_step, layer, submodule = self.layer_counter.increment()
        do_full_step = self.layer_counter.should_do_full_attn_step()
        
        if q.shape[-2] % 192 == 0 or self.layer_num <= 1:
            o = self._fast_bsd_attention_192(q, k, v, inference_step, do_full_step)
            o = rearrange(o, "B H L D -> B L (H D)")
            return o
        n = q.shape[-2]
        pm = 192

        padded_n = ((n + pm - 1) // pm) * pm
        qp = torch.zeros(q.shape[:-2] + (padded_n, q.shape[-1]), dtype=q.dtype, device=q.device)
        qp[..., :n, :] = q
        o = self._fast_bsd_attention_192(qp, k, v, inference_step, do_full_step)
        o = o[..., :n, :]
        o = rearrange(o, "B H L D -> B L (H D)")
        return o
    