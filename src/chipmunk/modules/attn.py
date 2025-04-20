import torch
from torch import Tensor
from torch.nn import functional as F
from torch import nn
from chipmunk.util import GLOBAL_CONFIG
import chipmunk.ops
from chipmunk.util.storage import AttnStorage, LayerCounter
from chipmunk.util.bitpack import bitpack, bitunpack
from einops import rearrange
import triton

singleton_video_query_groups = torch.ones(1, 24, 621, 119056, device=torch.device('cuda'), dtype=torch.bool)
singleton_static_mask = torch.zeros(1, 24, 621, 119056, device=torch.device('cuda'), dtype=torch.bool)

singleton_video_query_groups[..., -4:, :] = False
singleton_static_mask[..., -4:, 1024:] = True

class SparseDiffAttn(nn.Module):
    def __init__(self, layer_num: int, layer_counter: LayerCounter):
        super().__init__()
        self.layer_num = layer_num
        self.layer_counter = layer_counter
        self.storage = AttnStorage(layer_num, init_names=['indices', 'out_cache'])

    @torch.compile(dynamic=False)
    def random_and_topk(self, cs, topk):
        mask = torch.randint(0, 100, cs.shape, device=cs.device, dtype=torch.uint8) == 0
        mask.scatter_(-1, cs.topk(k=topk, dim=-1).indices, True)

        qg = cs.shape[-2]
        n = cs.shape[-1]
        mask = (mask * singleton_video_query_groups[..., :qg, :n]) | singleton_static_mask[..., :qg, :n]

        return mask

    def fast_attention_qpadded(
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
        # print(f'step {inference_step} do_full_step: {do_full_step} layer: {layer}')
        if do_full_step:
            if inference_step == 0:
                o, lse = chipmunk.ops.dense_attn(q, k, v)
                # zero out the lse constants for the padded tokens
                lse[..., k.shape[-2]:, :] = 0
                self.storage.set_lse_constants(lse)

                return o

            # elif inference_step == 1:
            else:
                prev_lse = self.storage.get_lse_constants()
                o, bs, lse = chipmunk.ops.dense_colsum_attn(q, k, v, prev_lse)
                # zero out the lse constants for the padded tokens
                lse[..., k.shape[-2]:, :] = 0
                self.storage.set_lse_constants(lse)

                tk = int(multiple_of * round((attn_config['top_keys'] * k.shape[-2]) / multiple_of))
                # kseq = k.shape[-2]
                # kgroups = (kseq + bm - 1) // bm
                # bs = bs[..., :kgroups, :kseq]

                # inds = torch.topk(bs, k=tk, dim=-1).indices
                # counts = torch.full(
                #     (q.shape[0], q.shape[1], triton.cdiv(q.shape[-2], bm)),
                #     tk,
                #     device=q.device,
                #     dtype=torch.int32,
                # )
                # pad = torch.empty((*counts.shape, q.shape[-2] - tk),
                #                   device=q.device,
                #                   dtype=torch.int32)
                # inds = torch.cat([inds, pad], dim=-1).to(torch.int32)
                mask = self.random_and_topk(bs, tk)
                # Full attention from text
                # mask[..., -2:, :] = True
                # mask[..., -2:, :1024] = False

                packed, mask_shape = bitpack(mask)
                self.mask_shape = mask_shape

                self.storage.set_indices(packed)
                # self.storage.set_counts(counts)

                inds, counts = chipmunk.ops.mask_to_indices(mask, multiple_of, bm)

            # else:
            #     o, _ = chipmunk.ops.dense_attn(q, k, v)

            # inds   = self.storage.get_indices()
            # counts = self.storage.get_counts()

                o_cache = o.clone()
                # chipmunk.ops.csp_attn(q, k, v, o_cache, inds, counts, -1)
                sp = chipmunk.ops.csp_attn(q, k, v, inds, counts)
                self.storage.set_out_cache(o_cache - sp)
                return o

        # ─────────── SPARSE STEP ───────────
        # inds   = self.storage.get_indices()
        # counts = self.storage.get_counts()
        # print(f'layer {layer} sparse step getting indices')
        packed = self.storage.get_indices()
        mask = bitunpack(packed, self.mask_shape)
        inds, counts = chipmunk.ops.mask_to_indices(mask, multiple_of, bm)
        o      = self.storage.get_out_cache()

        # if not self.storage.out_cache.is_offload_enabled:
        #     # csp_attn will write to o in place, so we need to clone it if it's not offloaded
        #     o = o.clone()

        # bandaid till bindings for new kernels work
        # chipmunk.ops.csp_attn(q, k, v, o, inds, counts, 1)
        sp = chipmunk.ops.csp_attn(q, k, v, inds, counts)
        return sp + o
    
    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        do_full_step = self.layer_counter.should_do_full_attn_step()
        inference_step, layer, submodule = self.layer_counter.increment()
        # print(f'inference_step: {inference_step}, layer: {layer}, submodule: {submodule}')
        if inference_step == GLOBAL_CONFIG['steps'] - 1 and layer == self.layer_counter.num_layers - 1 and submodule == self.layer_counter.num_submodules_per_layer - 1:
            # print(f'resetting layer counter at step')
            self.layer_counter.reset()
        bm = GLOBAL_CONFIG['attn']['mbm']
        return self.fast_attention_qpadded(q, k, v, inference_step, do_full_step)
        
        if q.shape[-2] % bm == 0:
            # Our kernels are happy!
            o = self.fast_attention_qpadded(q, k, v, inference_step, do_full_step)
            return o
        else:
            # Pad queries and outputs to the nearest multiple of bm
            # Our kernels are not happy with non-192 seqlens for query dim :(
            n = q.shape[-2]
            padded_n = ((n + bm - 1) // bm) * bm
            qp = torch.zeros(q.shape[:-2] + (padded_n, q.shape[-1]), dtype=q.dtype, device=q.device)
            qp[..., :n, :] = q
            o = self.fast_attention_qpadded(qp, k, v, inference_step, do_full_step)
            o = o[..., :n, :]
            return o

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)