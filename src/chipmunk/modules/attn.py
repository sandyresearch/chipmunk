from typing import Tuple
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from chipmunk.util import GLOBAL_CONFIG, update_global_config
from chipmunk.ops.voxel import get_local_indices_with_text
import chipmunk.ops
from chipmunk.util import AttnStorage, LayerCounter
from chipmunk.cache.token_cache import TokenCache
from chipmunk.ops import bitpack, bitunpack, compute_coverage_mask
import triton

# Initialized based on sequence shape
singleton_static_mask = None
singleton_video_query_groups = None

class SparseDiffAttn(nn.Module):
    def __init__(self, layer_num: int, layer_counter: LayerCounter, token_cache: TokenCache | None = None):
        super().__init__()
        self.layer_num = layer_num
        self.layer_counter = layer_counter
        self.storage = AttnStorage(layer_num, init_names=['indices', 'out_cache'])
        self.mask_shape = [None] * GLOBAL_CONFIG['num_model_invocations_per_inference_step']
        self.token_cache = token_cache

        # debug
        self.attn_sparsity = 0
        self.attn_sparsity_count = 0

    def initialize_static_mask(self, seq_shape: Tuple, txt_len: int, local_heads_num: int, device: torch.device):
        if len(seq_shape) == 2:
            raise NotImplementedError("Not yet implemented for 2D sequences")

        tt, th, tw = seq_shape
        update_global_config({'non_text_seqlen': tt * th * tw})

        attn_config = GLOBAL_CONFIG['attn']
        rk = attn_config['random_keys']
        topk = attn_config['top_keys']
        lv = attn_config['local_voxels']
        lw1d = attn_config['local_1d_window']
        topk = int(topk * (tt * th * tw))

        # Apply local 3D window
        full_attn_from_3d_tail = GLOBAL_CONFIG['attn']['full_attn_from_3d_tail']
        full_attn_to_3d_tail = GLOBAL_CONFIG['attn']['full_attn_to_3d_tail']
        mask, _, _ = get_local_indices_with_text(
            vid_shape=(tt, th, tw),
            txt_len=txt_len,
            voxel_shape=(4, 6, 8),
            local_shape=(lv, lv, lv),
            rk=rk,
            device=device,
            full_tail_from_attn=full_attn_from_3d_tail,
            full_tail_to_attn=full_attn_to_3d_tail,
        )

        if lv == 0 and GLOBAL_CONFIG['model_name'] == 'flux':
            mask.fill_(0)

        # Apply local 1D window
        if lw1d > 0:
            window_size = int(lw1d * (tt * th * tw))
            # Each query group (dim=0, a chunk of 192 queries) in [qg, n] attends to a local 1D window
            total_seq_len = tt * th * tw + txt_len
            query_groups = (tt * th * tw) // 192  # Assuming 192 queries per group
            
            for qg in range(query_groups):
                # Calculate the center position for this query group
                center_pos = qg * 192 + 192 // 2
                
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

    def random_and_topk(self, cs, topk):
        mask = torch.randint(0, 100, cs.shape, device=cs.device, dtype=torch.uint8) == 0
        mask.scatter_(-1, cs.topk(k=topk, dim=-1).indices, True)

        qg = cs.shape[-2]
        n = cs.shape[-1]

        if GLOBAL_CONFIG['attn']['coverage'] is not None:
            mask = compute_coverage_mask(cs, GLOBAL_CONFIG['attn']['coverage'], mask)

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

        # coord = (self.layer_counter.cur_inference_step, self.layer_counter.cur_model_invocation_per_step, self.layer_counter.cur_layer, self.layer_counter.cur_layer_submodule)
        # bkpt_coords = {(1, 0, 2, 0), (2, 0, 2, 0), (1, 1, 2, 0), (2, 1, 2, 0)}
        # if coord in bkpt_coords:
        #     print('q.shape', q.shape, 'k.shape', k.shape, 'v.shape', v.shape)
        #     breakpoint()

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

                # Score tokens as a function of column sums
                if self.token_cache: self.token_cache.score(bs)

                tk = int(multiple_of * round((attn_config['top_keys'] * k.shape[-2]) / multiple_of))
                
                if attn_config['should_compress_indices']:
                    mask = self.random_and_topk(bs, tk) if tk > 0 else singleton_static_mask[..., :bs.shape[-2], :bs.shape[-1]]
                    packed, mask_shape = bitpack(mask)
                    self.mask_shape[self.layer_counter.cur_model_invocation_per_step] = mask_shape
                    self.storage.set_indices(packed)
                    inds, counts = chipmunk.ops.mask_to_indices(mask, multiple_of, bm)
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

            if not attn_config['recompute_mask']:
                if attn_config['should_compress_indices']:
                    packed         = self.storage.get_indices()
                    mask           = bitunpack(packed, self.mask_shape[self.layer_counter.cur_model_invocation_per_step])
                    inds, counts   = chipmunk.ops.mask_to_indices(mask, multiple_of, bm)
                else:
                    inds   = self.storage.get_indices()
                    counts = self.storage.get_counts()

            if attn_config['delta_cache']:
                if do_padding:
                    o_cache = o - chipmunk.ops.csp_attn(q, k, v, inds, counts)
                else:
                    o_cache = o.clone()
                    inds = inds[:, :, :, :q.shape[-2]].contiguous()
                    torch.ops.chipmunk.csp_attn(q, k, v, o_cache, inds, counts, -1)
                    self.storage.set_out_cache(o_cache)

            if attn_config['debug']:
                sparsity = torch.sum(counts) / inds.numel()
                print(f'step {inference_step:02} layer {self.layer_num:02} sparsity: {sparsity:.4f}')

            if attn_config['should_compress_indices']:
                del inds, counts, mask

            return o

        # ─────────── SPARSE STEP ───────────
        if attn_config['should_compress_indices']:
            packed         = self.storage.get_indices()
            mask           = bitunpack(packed, self.mask_shape[self.layer_counter.cur_model_invocation_per_step])
            inds, counts   = chipmunk.ops.mask_to_indices(mask, multiple_of, bm)
        else:
            inds   = self.storage.get_indices()
            counts = self.storage.get_counts()
        
        if do_padding:
            if attn_config['delta_cache']:
                o = self.storage.get_out_cache()
                o = o + chipmunk.ops.csp_attn(q, k, v, inds, counts)
            else:
                o = chipmunk.ops.csp_attn(q, k, v, inds, counts)
        else:
            if attn_config['delta_cache']:
                o = self.storage.get_out_cache()
                if not self.storage.out_cache.is_offload_enabled:
                    # Our kernel will write to o in place, so we need to clone it if it's not offloaded
                    o = o.clone()
            else:
                o = torch.zeros_like(q)
            inds = inds[:, :, :, :q.shape[-2]].contiguous()
            torch.ops.chipmunk.csp_attn(q, k, v, o, inds, counts, 1)

        if GLOBAL_CONFIG['attn']['debug']:
            sparsity = torch.sum(counts) / inds.numel()
            self.attn_sparsity += sparsity
            self.attn_sparsity_count += 1
            self.layer_counter.attn_sparsity += sparsity
            self.layer_counter.attn_sparsity_count += 1

        if attn_config['should_compress_indices']:
            del inds, counts, mask

        return o
    
    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        do_full_step = self.layer_counter.should_do_full_attn_step()
        inference_step, layer, submodule = self.layer_counter.increment()
        if GLOBAL_CONFIG['offline_search']['is_enabled']:
            if inference_step == 0 and layer == 0:
                print(f'=== RUNNING OFFLINE SEARCH ===')
            print(f'inference_step: {inference_step}, do_full_step: {do_full_step}')

        if not GLOBAL_CONFIG['attn']['is_enabled']:
            if GLOBAL_CONFIG['token_cache']['is_enabled']:
                assert GLOBAL_CONFIG['attn']['delta_cache'] == True, "Delta cache must be enabled for token cache"
                if inference_step == 0:
                    if layer == 0:
                        print(f'Step {inference_step}: Computing Attention')
                    o, l = chipmunk.ops.dense_attn(q, k, v)
                    l[..., k.shape[-2]:, :] = 0
                    self.storage.set_lse_constants(l)
                    self.storage.set_out_cache(o)
                # ToCa recomputes attention every 3 steps
                elif inference_step == 1 or inference_step % 3 == 0:
                    if layer == 0:
                        print(f'Step {inference_step}: Computing Attention')
                    l = self.storage.get_lse_constants()
                    o, cs, _ = chipmunk.ops.dense_colsum_attn(q, k, v, l)
                    if self.token_cache: self.token_cache.score(cs)
                    self.storage.set_out_cache(o)
                else:
                    if layer == 0:
                        print(f'Step {inference_step}: Skipping Attention')
                    o = self.storage.get_out_cache()
                return o
            else:
                if q.shape[-2] < 10000:
                    # n < 10000 is the threshold for using F.scaled_dot_product_attention
                    return F.scaled_dot_product_attention(q, k, v)
                else:
                    return chipmunk.ops.dense_attn(q, k, v)[0]

        o = self._fast_attention(q, k, v, inference_step, do_full_step)

        if GLOBAL_CONFIG['attn']['debug'] and inference_step == GLOBAL_CONFIG['steps'] - 1:
            if self.attn_sparsity_count > 0:
                print(f'layer {self.layer_num} attn sparsity: {self.attn_sparsity / self.attn_sparsity_count}')
                print(f'avg attn sparsity: {self.layer_counter.attn_sparsity / self.layer_counter.attn_sparsity_count}')

        return o

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)