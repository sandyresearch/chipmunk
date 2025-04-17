import torch
from torch import Tensor
from torch.nn import functional as F
from ..util.config import GLOBAL_CONFIG
import chipmunk.ops
from chipmunk.util.storage import AttnStorage
from einops import rearrange
import triton
from ..util.layer_counter import LayerCounter

class SparseDiffAttn:
    def __init__(self, layer_num: int, layer_counter: LayerCounter):
        self.layer_num = layer_num
        self.layer_counter = layer_counter
        self.storage = AttnStorage(layer_num)

    def fast_attention_qpadded(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        inference_step: int,
        do_full_step: bool,
    ) -> Tensor:
        attn_config = GLOBAL_CONFIG['sparsity']['attention']
        bm = attn_config['mbm']
        assert bm == 192, "The kernel was written for BM=192. You may need to change the kernel."
        layer = self.layer_num
        multiple_of = attn_config['counts_multiple_of']

        if layer <= 1:
            return F.scaled_dot_product_attention(q, k, v)

        # ─────────── FULL STEP ───────────
        if do_full_step:
            if inference_step == 0:
                o, lse = chipmunk.ops.dense_attn(q, k, v)
                lse[..., k.shape[-2]:, :] = 0
                self.storage.set_lse_constants(lse)
                return o

            elif inference_step == 1:
                prev_lse = self.storage.get_lse_constants()
                o, bs, _ = chipmunk.ops.dense_colsum_attn(q, k, v, prev_lse)

                tk = int(multiple_of * round((attn_config['top_keys'] * q.shape[-2]) / multiple_of))
                kseq = k.shape[-2]
                kgroups = (kseq + bm - 1) // bm
                bs = bs[..., :kgroups, :kseq]

                inds = torch.topk(bs, k=tk, dim=-1).indices
                counts = torch.full(
                    (q.shape[0], q.shape[1], triton.cdiv(q.shape[-2], bm)),
                    tk,
                    device=q.device,
                    dtype=torch.int32,
                )
                pad = torch.empty((*counts.shape, q.shape[-2] - tk),
                                  device=q.device,
                                  dtype=torch.int32)
                inds = torch.cat([inds, pad], dim=-1).to(torch.int32)

                self.storage.set_indices(inds)
                self.storage.set_counts(counts)

            else:
                o, _ = chipmunk.ops.dense_attn(q, k, v)

            inds   = self.storage.get_indices()
            counts = self.storage.get_counts()

            o_cache = o.clone()
            chipmunk.ops.csp_attn(q, k, v, o_cache, inds, counts, -1)
            self.storage.set_out_cache(o_cache)
            return o

        # ─────────── SPARSE STEP ───────────
        inds   = self.storage.get_indices()
        counts = self.storage.get_counts()
        o      = self.storage.get_out_cache()
        chipmunk.ops.csp_attn(q, k, v, o, inds, counts, 1)
        return o
    
    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        inference_step, layer, submodule = self.layer_counter.increment()
        do_full_step = self.layer_counter.should_do_full_attn_step()
        bm = GLOBAL_CONFIG['attn']['mbm']
        
        if q.shape[-2] % bm == 0 or self.layer_num <= 1:
            o = self.fast_attention_qpadded(q, k, v, inference_step, do_full_step)
            o = rearrange(o, "B H L D -> B L (H D)")
            return o
        n = q.shape[-2]

        padded_n = ((n + bm - 1) // bm) * bm
        qp = torch.zeros(q.shape[:-2] + (padded_n, q.shape[-1]), dtype=q.dtype, device=q.device)
        qp[..., :n, :] = q
        o = self.fast_attention_qpadded(qp, k, v, inference_step, do_full_step)
        o = o[..., :n, :]
        o = rearrange(o, "B H L D -> B L (H D)")
        return o
