from typing import Tuple
import torch
from torch import Tensor
from torch.nn import functional as F
from chipmunk.util import GLOBAL_CONFIG
from chipmunk.ops.voxel import get_local_indices_with_text
import chipmunk.ops
from chipmunk.util import AttnStorage, LayerCounter
from chipmunk.ops import bitpack
import triton

# Initialized based on sequence shape
singleton_static_mask = None
singleton_video_query_groups = None

class SparseDiffAttn:
    def __init__(self, layer_num: int, layer_counter: LayerCounter):
        self.layer_num = layer_num
        self.layer_counter = layer_counter
        self.storage = AttnStorage(layer_num, init_names=['indices', 'out_cache'])

    def initialize_static_mask(self, seq_shape: Tuple, txt_len: int, local_heads_num: int, device: torch.device):
        if len(seq_shape) == 2:
            raise NotImplementedError("Not yet implemented for 2D sequences")

        tt, th, tw = seq_shape

        attn_config = GLOBAL_CONFIG['attn']
        rk = attn_config['random_keys']
        topk = attn_config['top_keys']
        lv = attn_config['local_voxels']
        topk = int(topk * (tt * th * tw))

        mask, _, _ = get_local_indices_with_text(
            vid_shape=(tt, th, tw),
            txt_len=txt_len,
            voxel_shape=(4, 6, 8),
            local_shape=(lv, lv, lv),
            rk=rk,
            device=device
        )

        mask = mask[None, None, :, :].expand(1, local_heads_num, -1, -1).contiguous()
        sparse_attn_query_groups = ((mask.sum(dim=-1, keepdim=True) + topk) < (tt * th * tw + txt_len))

        # Update singletons
        global singleton_static_mask
        global singleton_video_query_groups
        singleton_static_mask = mask
        singleton_video_query_groups = sparse_attn_query_groups

    @torch.compile(dynamic=False)
    def random_and_topk(self, cs, topk):
        mask = torch.randint(0, 100, cs.shape, device=cs.device, dtype=torch.uint8) == 0
        mask.scatter_(-1, cs.topk(k=topk, dim=-1).indices, True)

        qg = cs.shape[-2]
        n = cs.shape[-1]
        mask = (mask * singleton_video_query_groups[..., :qg, :n]) | singleton_static_mask[..., :qg, :n]

        return mask

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
        multiple_of = attn_config['counts_multiple_of'] if not attn_config['pad_qkv_before_kernel'] else 128
        do_padding = attn_config['pad_qkv_before_kernel']

        if layer < attn_config['first_n_dense_layers']:
            o, _ = chipmunk.ops.dense_attn(q, k, v)
            return o

        # ─────────── FULL STEP ───────────
        if do_full_step:
            if inference_step == 0:
                if do_padding:
                    o, lse = chipmunk.ops.dense_attn(q, k, v)
                else:
                    o, lse = torch.ops.chipmunk.dense_attn(q, k, v)
                lse[..., k.shape[-2]:, :] = 0
                self.storage.set_lse_constants(lse)

                return o

            elif inference_step == 1 or attn_config['recompute_mask']:
                prev_lse = self.storage.get_lse_constants()
                if do_padding:
                    o, bs, lse = chipmunk.ops.dense_colsum_attn(q, k, v, prev_lse)
                else:
                    o, bs, lse = torch.ops.chipmunk.dense_colsum_attn(q, k, v, prev_lse)
                # zero out the lse constants for the padded tokens
                lse[..., k.shape[-2]:, :] = 0
                self.storage.set_lse_constants(lse)

                tk = int(multiple_of * round((attn_config['top_keys'] * k.shape[-2]) / multiple_of))
                
                if attn_config['should_compress_indices']:
                    mask = self.random_and_topk(bs, tk)
                    packed, mask_shape = bitpack(mask)
                    self.mask_shape = mask_shape
                    self.storage.set_indices(packed)
                else:
                    kseq = k.shape[-2]
                    kgroups = (kseq + bm - 1) // bm
                    bs = bs[..., :kgroups, :kseq]
                    inds = torch.topk(bs, k=tk, dim=-1).indices
                    counts = torch.full((q.shape[0], q.shape[1], triton.cdiv(q.shape[-2], bm)), tk, device=q.device, dtype=torch.int32)
                    pad = torch.empty((*counts.shape, q.shape[-2] - tk), device=q.device, dtype=torch.int32)
                    inds = torch.cat([inds, pad], dim=-1).to(torch.int32)

                    self.storage.set_indices(inds)
                    self.storage.set_counts(counts)

            else:
                o, _ = chipmunk.ops.dense_attn(q, k, v)

            if attn_config['should_compress_indices']:
                mask         = self.storage.get_indices()
                inds, counts = chipmunk.ops.mask_to_indices(mask, multiple_of, bm)
            else:
                inds   = self.storage.get_indices()
                counts = self.storage.get_counts()

            if do_padding:
                o_cache = o - chipmunk.ops.csp_attn(q, k, v, inds, counts)
            else:
                o_cache = o.clone()
                torch.ops.chipmunk.csp_attn(q.contiguous(), k, v, o_cache, inds, counts, -1)
            self.storage.set_out_cache(o_cache)
            return o

        # ─────────── SPARSE STEP ───────────
        inds   = self.storage.get_indices()
        counts = self.storage.get_counts()
        o      = self.storage.get_out_cache()
        if do_padding:
            o = o + chipmunk.ops.csp_attn(q, k, v, inds, counts)
        else:
            if not self.storage.out_cache.is_offload_enabled:
                # Our kernel will write to o in place, so we need to clone it if it's not offloaded
                o = o.clone()
            torch.ops.chipmunk.csp_attn(q.contiguous(), k, v, o, inds, counts, 1)
        return o
    
    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        do_full_step = self.layer_counter.should_do_full_attn_step()
        if not GLOBAL_CONFIG['attn']['is_enabled']:
            return F.scaled_dot_product_attention(q, k, v)

        inference_step, layer, submodule = self.layer_counter.increment()

        return self._fast_attention(q, k, v, inference_step, do_full_step)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)