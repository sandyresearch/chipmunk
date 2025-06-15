from typing import Tuple
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from chipmunk.util import GLOBAL_CONFIG, get_kernel_config_attn
from chipmunk.ops.voxel import get_local_indices_with_text
import chipmunk.ops
from chipmunk.util import AttnStorage, LayerCounter
from chipmunk.ops import bitpack, bitunpack
import triton
import pickle

# Initialized based on sequence shape
singleton_static_mask = None
singleton_video_query_groups = None

class SparseDiffAttn(nn.Module):
    def __init__(self, layer_num: int, layer_counter: LayerCounter):
        super().__init__()
        self.layer_num = layer_num
        self.layer_counter = layer_counter
        self.storage = AttnStorage(layer_num, init_names=['indices', 'out_cache'])
        self.mask_shape = [None] * GLOBAL_CONFIG['num_model_invocations_per_inference_step']

    def initialize_1d_static_mask(self, q: Tensor):
        attn_config = GLOBAL_CONFIG['attn']
        kernel_config = get_kernel_config_attn()
        bm = kernel_config['bm']
        topk = attn_config['top_keys']
        lw1d = attn_config['local_1d_window']

        b, h, n, d = q.shape
        qg = n // bm

        # make mask a bit bigger to account for unbatched CFG
        mask = torch.zeros((qg + 3, n + 512), device=q.device, dtype=torch.bool)

        # Apply local 1D window
        if lw1d > 0:
            window_size = int(lw1d * n)
            # Each query group (dim=0, a chunk of 192 queries) in [qg, n] attends to a local 1D window
            query_groups = n // bm
            
            for qg in range(query_groups):
                # Calculate the center position for this query group
                center_pos = qg * bm + bm // 2
                
                # Define the window boundaries (ensuring we don't go out of bounds)
                window_start = max(0, center_pos - window_size // 2)
                window_end = min(n, center_pos + window_size // 2)
                
                # For the current query group, allow attention to tokens within the window
                mask[qg, window_start:window_end] = True

        mask = mask[None, None, :, :].expand(1, h, -1, -1).contiguous()
        sparse_attn_query_groups = ((mask.sum(dim=-1, keepdim=True) + topk) < (n))

        # Update singletons
        global singleton_static_mask
        global singleton_video_query_groups
        singleton_static_mask = mask
        singleton_video_query_groups = sparse_attn_query_groups

    def initialize_static_mask(self, seq_shape: Tuple, txt_len: int, local_heads_num: int, device: torch.device):
        if len(seq_shape) == 2:
            raise NotImplementedError("Not yet implemented for 2D sequences")

        tt, th, tw = seq_shape

        attn_config = GLOBAL_CONFIG['attn']
        kernel_config = get_kernel_config_attn()
        bm = kernel_config['bm']
        multiple_of = kernel_config['counts_multiple_of']
        rk = attn_config['random_keys']
        topk = attn_config['top_keys']
        lv = attn_config['local_voxels']
        lw1d = attn_config['local_1d_window']
        topk = int(topk * (tt * th * tw))
        assert bm == 192 or bm == 64, f'bm must be 192 or 64, got {bm}'
        # Apply local 3D window
        mask, _, _ = get_local_indices_with_text(
            vid_shape=(tt, th, tw),
            txt_len=txt_len,
            voxel_shape=(4, 6, 8) if bm == 192 else (4, 4, 4),
            local_shape=(lv, lv, lv),
            rk=rk,
            device=device,
            kv_tile_size=multiple_of
        )

        # Apply local 1D window
        if lw1d > 0:
            window_size = int(lw1d * (tt * th * tw))
            # Each query group (dim=0, a chunk of 192 queries) in [qg, n] attends to a local 1D window
            total_seq_len = tt * th * tw + txt_len
            query_groups = (tt * th * tw) // bm
            
            for qg in range(query_groups):
                # Calculate the center position for this query group
                center_pos = qg * bm + bm // 2
                
                # Define the window boundaries (ensuring we don't go out of bounds)
                window_start = max(0, center_pos - window_size // 2)
                window_end = min(tt * th * tw, center_pos + window_size // 2)
                
                # For the current query group, allow attention to tokens within the window
                mask[qg, window_start:window_end] = True
                # mask[0, 0, qg, tt * th * tw:total_seq_len] = True  # Always attend to text tokens

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
        if singleton_video_query_groups is not None:
            mask = mask * singleton_video_query_groups[..., :qg, :n]
        if singleton_static_mask is not None:
            mask = mask | singleton_static_mask[..., :qg, :n]

        return mask

    @torch.compiler.disable
    def _fast_attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        inference_step: int,
        do_full_step: bool,
    ) -> Tensor:
        attn_config = GLOBAL_CONFIG['attn']
        attn_kernel_config = get_kernel_config_attn()   
        bm = attn_kernel_config['bm']
        layer = self.layer_num

        multiple_of = attn_kernel_config['counts_multiple_of']
        indices_pad_to = attn_kernel_config['indices_pad_to']
        provider = attn_config['provider']

        if layer < attn_config['first_n_dense_layers']:
            o, _ = chipmunk.ops.dense_attn(q, k, v)
            return o

        # ─────────── FULL STEP ───────────
        if do_full_step:
            if inference_step == 0:
                o, lse = chipmunk.ops.dense_attn(q, k, v)         
                self.storage.set_lse_constants(lse)
                return o

            elif inference_step == 1 or attn_config['recompute_mask']:
                prev_lse = self.storage.get_lse_constants()
                o, bs, lse = chipmunk.ops.dense_colsum_attn(q, k, v, prev_lse)
                indices_count = int(multiple_of * round((attn_config['top_keys'] * k.shape[-2]) / multiple_of))
                
                if attn_config['should_compress_indices']:
                    mask = self.random_and_topk(bs, indices_count) if indices_count > 0 else singleton_static_mask[..., :bs.shape[-2], :bs.shape[-1]]
                    packed, mask_shape = bitpack(mask)
                    self.mask_shape[self.layer_counter.cur_model_invocation_per_step] = mask_shape
                    self.storage.set_indices(packed)
                    inds, counts = chipmunk.ops.mask_to_indices(mask, multiple_of, indices_pad_to)
                    inds = inds[:,:,:,:k.shape[-2]]
                
                else:
                    inds = torch.topk(bs, k=indices_count, dim=-1).indices
                    counts = torch.full((q.shape[0], q.shape[1], triton.cdiv(q.shape[-2], bm)), indices_count, device=q.device, dtype=torch.int32)
                    # Pad the stride, but not the shape, of indices so that the TMA stride gets aligned to 16 bytes
                    padding_amount = (q.shape[-2] - indices_count + indices_pad_to - 1) // indices_pad_to * indices_pad_to
                    inds = torch.cat([inds, torch.empty((*counts.shape, padding_amount), device=q.device, dtype=torch.int32)], dim=-1).to(torch.int32)
                    inds = inds[:,:,:,:k.shape[-2]]
                    self.storage.set_indices(inds)
                    self.storage.set_counts(counts)
            else:
                o, _ = chipmunk.ops.dense_attn(q, k, v)

            if not attn_config['recompute_mask']:
                if attn_config['should_compress_indices']:
                    packed         = self.storage.get_indices()
                    mask           = bitunpack(packed, self.mask_shape[self.layer_counter.cur_model_invocation_per_step])
                    inds, counts   = chipmunk.ops.mask_to_indices(mask, multiple_of, indices_pad_to)
                    inds = inds[:,:,:,:k.shape[-2]]
                else:
                    inds   = self.storage.get_indices()
                    counts = self.storage.get_counts()
            
            if provider == 'cuda': o_cache = o.clone()
            else:                  o_cache = o
            
            o_cache = chipmunk.ops.csp_attn(q, k, v, inds, counts, o_cache, -1)
            
            self.storage.set_out_cache(o_cache)
            return o

        # ─────────── SPARSE STEP ───────────
        if attn_config['should_compress_indices']:
            packed         = self.storage.get_indices()
            mask           = bitunpack(packed, self.mask_shape[self.layer_counter.cur_model_invocation_per_step])
            inds, counts   = chipmunk.ops.mask_to_indices(mask, multiple_of, indices_pad_to)
            inds = inds[:,:,:,:k.shape[-2]]
        else:
            inds   = self.storage.get_indices()
            counts = self.storage.get_counts()
        
        o = self.storage.get_out_cache()
        if provider == 'cuda': o = o.clone()
        o = chipmunk.ops.csp_attn(q, k, v, inds, counts, o, 1)
        
        return o
    
    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        if not GLOBAL_CONFIG['attn']['is_enabled']:
            out = F.scaled_dot_product_attention(q, k, v)
            self.layer_counter.increment()
            return out

        if singleton_static_mask is None:
            self.initialize_1d_static_mask(q)

        do_full_step = self.layer_counter.should_do_full_attn_step()
        inference_step = self.layer_counter.cur_inference_step
        out = self._fast_attention(q, k, v, inference_step, do_full_step)
        self.layer_counter.increment()
        self.storage.complete_cur_layer()
        return out


    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
