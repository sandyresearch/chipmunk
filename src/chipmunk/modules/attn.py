import torch
from torch import Tensor
from torch.nn import functional as F
import chipmunk.ops
from chipmunk.util import AttnStorage, LayerCounter, GLOBAL_CONFIG
from einops import rearrange
import triton

class SparseDiffAttn:
    def __init__(self, layer_num: int, layer_counter: LayerCounter):
        self.layer_num = layer_num
        self.layer_counter = layer_counter
        self.storage = AttnStorage(layer_num)

    def _fast_attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        inference_step: int,
        do_full_step: bool,
    ) -> Tensor:
        attn_config = GLOBAL_CONFIG['attn']
        bm = attn_config['mbm']
        assert bm == 192, "The kernel was written for BM=192. You may need to change the kernel."
        layer = self.layer_num
        multiple_of = attn_config['counts_multiple_of']
        
        if layer < attn_config['first_n_dense_layers']:
            o, lse = chipmunk.ops.dense_attn(q, k, v)
            return o

        # ─────────── FULL STEP ───────────
        if inference_step == 0:
            o, lse = chipmunk.ops.dense_attn(q, k, v)
            self.storage.set_lse_constants(lse)
            return o

        elif inference_step == 1 or do_full_step:
            if inference_step == 1:
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
            chipmunk.ops.csp_attn(q.contiguous(), k, v, o_cache, inds, counts, -1)
            self.storage.set_out_cache(o_cache)
            return o

        # ─────────── SPARSE STEP ───────────
        inds   = self.storage.get_indices()
        counts = self.storage.get_counts()
        o      = self.storage.get_out_cache()
        if not self.storage.out_cache.is_offload_enabled:
            # Our kernel will write to o in place, so we need to clone it if it's not offloaded
            o = o.clone()
        chipmunk.ops.csp_attn(q.contiguous(), k, v, o, inds, counts, 1)
        return o
    
    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        if not GLOBAL_CONFIG['attn']['is_enabled']:
            return F.scaled_dot_product_attention(q, k, v)

        inference_step, layer, submodule = self.layer_counter.increment()
        do_full_step = self.layer_counter.should_do_full_attn_step()
        
        o = self._fast_attention(q, k, v, inference_step, do_full_step)
        return o

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)