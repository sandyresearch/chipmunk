import torch
from ..util.layer_counter import LayerCounter
from ..util.config import GLOBAL_CONFIG
from einops import rearrange
import chipmunk.ops

def block_mean(x: torch.Tensor, mbm: int):
    return rearrange(x, 'b (mb mbm) c -> b mb mbm c', mbm=mbm).mean(dim=2)

class SparseDiffMlp:
    def __init__(self, layer_counter: LayerCounter, fc1: torch.nn.Linear, activation: torch.nn.Module, fc2: torch.nn.Linear):
        # Store them in lists so that PyTorch doesn't count them as children modules - we don't want to count the weights
        # as children modules, only hold references to them!
        self.fc1 = [fc1]
        self.fc2 = [fc2]
        self.fc2w_T = [fc2.weight.data.transpose(0, 1).contiguous()]
        self.layer_counter = layer_counter
        self.activation = activation
        pass

    def forward(self, x: torch.Tensor):
        fc1 = self.fc1[0]
        fc2 = self.fc2[0]

        do_full_step = self.layer_counter.should_do_full_mlp_step()
        inference_step, layer, submodule = self.layer_counter.increment()

        if layer <= 1:
            return fc2(self.activation(fc1(x)))
        
        mlp_config = GLOBAL_CONFIG['mlp']
        MBM = mlp_config['mbm']
        BM = mlp_config['bm']
        
        multiple_of = 256
        sparsity_amount = 1 - mlp_config['top_keys']
        
        if do_full_step:
            mid = fc1(x)
            pa = self.activation(mid)
            out = fc2(pa)
            self.pa_cache_colmajor = pa.transpose(-1, -2).contiguous()
            self.out_cache = out
            self.bm_mid_cache = block_mean(mid, MBM)
            return out
        
        if not (inference_step % mlp_config['block_mask_cache'] != 0 and hasattr(self, 'sp_inds') and inference_step >= 10):
            bmfc1 = fc1(block_mean(x, MBM))
            r = BM // MBM
            mdiff = (bmfc1 - self.bm_mid_cache).abs()
            mdiff = rearrange(mdiff, 'b (mb r) f -> b r mb f', r=r).sum(dim=1)

            if not hasattr(self, 'sp_inds'):
                self.sp_inds = torch.empty_like(mdiff, dtype=torch.int32, device=x.device)
            if not hasattr(self, 'sp_counts'):
                self.sp_counts = torch.empty((mdiff.size(0), mdiff.size(1)), dtype=torch.int32, device=x.device)
            chipmunk.ops.topk_indices(mdiff, self.sp_inds, self.sp_counts, sparsity_amount, multiple_of, mlp_config['random_keys'])
            chipmunk.ops.copy_indices(bmfc1, self.bm_mid_cache, self.sp_inds, self.sp_counts)

        assert x.ndim == 3, "x must be 3D - (B, N, C)"
        assert x.shape[0] == 1, "x must be 1D - (N, C) - batch size must be 1"

        x = x[0]
        B, N, C = x.shape

        assert N % MBM == 0, "N must be a multiple of MBM"
        assert C % BM == 0, "C must be a multiple of BM"
        assert B == 1, "B must be 1"

        chipmunk.ops.mlp(
            x=x[0], 
            fc1w=fc1.weight.data,
            fc1b=fc1.bias.data, 
            fc2w_T=self.fc2w_T,
            indices=self.sp_inds[0], 
            counts=self.sp_counts[0], 
            sparse_act_unpacked_T=self.pa_cache_colmajor[0], 
            cached_out=self.out_cache[0], 
            num_sms_scatter_add=self.num_sms_scatter_add
        )

