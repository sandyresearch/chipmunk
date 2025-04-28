import torch
import torch.nn as nn
from ..util.layer_counter import LayerCounter
from ..util.config import GLOBAL_CONFIG
from einops import rearrange
import chipmunk.ops
from chipmunk.cache.token_cache import TokenCache
from chipmunk.util import MlpStorage

def block_mean(x: torch.Tensor, mbm: int):
    return rearrange(x, 'b (mb mbm) c -> b mb mbm c', mbm=mbm).mean(dim=2)

class SparseDiffMlp(nn.Module):
    def __init__(
        self,
        layer_num: int,
        layer_counter: LayerCounter,
        fc1: torch.nn.Linear,
        activation: torch.nn.Module,
        fc2: torch.nn.Linear,
        heuristic_sms_scatter_add: int = 6,
        token_cache: TokenCache | None = None
    ):
        super().__init__()
        self.fc1           = [fc1]
        self.fc2           = [fc2]
        self.fc2w_T        = [fc2.weight.data.transpose(0, 1).contiguous()]
        self.layer_counter = layer_counter
        self.activation    = activation
        self.storage       = MlpStorage(layer_num, init_names=['out_cache'])
        self.token_cache   = token_cache
    
        self.num_sms_scatter_add = heuristic_sms_scatter_add

    def forward(self, x: torch.Tensor):
        fc1, fc2 = self.fc1[0], self.fc2[0]

        do_full  = self.layer_counter.should_do_full_mlp_step()
        inference_step, layer, submodule = self.layer_counter.increment()

        if not GLOBAL_CONFIG['mlp']['is_enabled']:
            if GLOBAL_CONFIG['token_cache']['is_enabled']:
                # ToCa recomputes MLP every 3 steps
                if inference_step < 2 or inference_step % 3 == 0:
                    o = fc2(self.activation(fc1(x)))
                    self.storage.set_out_cache(o)
                # ToCa sparse step
                else:
                    gathered = self.token_cache.gather(x)
                    o = fc2(self.activation(fc1(gathered)))
                    o = self.token_cache.scatter(o, self.storage.get_out_cache())
                    self.storage.set_out_cache(o)
                return o

            return fc2(self.activation(fc1(x)))

        assert x.ndim == 3 and x.shape[0] == 1, "x must be (1, N, C)"

        mlp_cfg              = GLOBAL_CONFIG['mlp']
        MBM, BM              = mlp_cfg['mbm'], mlp_cfg['bm']
        sparsity             = 1 - mlp_cfg['top_keys']
        multiple_of          = mlp_cfg['counts_multiple_of']
        first_n_dense_layers = mlp_cfg['first_n_dense_layers']
        
        if layer < first_n_dense_layers:
            return fc2(self.activation(fc1(x)))

        # ─────────── FULL STEP ───────────
        if do_full:
            mid = fc1(x)
            pa  = self.activation(mid)
            out = fc2(pa)

            self.storage.set_sparse_act_T(pa.transpose(-1, -2).contiguous())
            self.storage.set_out_cache(out)
            self.storage.set_blockmean_mid_cache(block_mean(mid, MBM))
            return out

        # ─── decide whether to recompute indices ───
        cached_inds = self.storage.get_indices()
        recompute   = not (
            inference_step % mlp_cfg['block_mask_cache'] != 0
            and cached_inds is not None
            and inference_step >= 10
        )

        if recompute:
            bmfc1   = fc1(block_mean(x, MBM))
            r       = BM // MBM
            mdiff   = (bmfc1 - self.storage.get_blockmean_mid_cache()).abs()
            mdiff   = rearrange(mdiff, 'b (mb r) f -> b r mb f', r=r).sum(dim=1)

            inds   = torch.empty_like(mdiff, dtype=torch.int32, device=x.device)
            counts = torch.empty(
                (mdiff.size(0), mdiff.size(1)), dtype=torch.int32, device=x.device
            )
            chipmunk.ops.topk_indices(
                mdiff, inds, counts, sparsity, multiple_of, mlp_cfg['random_keys']
            )
            chipmunk.ops.copy_indices(bmfc1,
                                      self.storage.get_blockmean_mid_cache(),
                                      inds,
                                      counts)

            self.storage.set_indices(inds)
            self.storage.set_counts(counts)

        # Select batch 0
        indices      = self.storage.get_indices()[0]
        counts       = self.storage.get_counts()[0]
        out_cache    = self.storage.get_out_cache()[0]
        sparse_act_T = self.storage.get_sparse_act_T()[0]

        if fc1.weight.dtype == torch.float8_e4m3fn:
            x = self.fc1.quantize_input(x)
            mm1_scale_a = fc1.input_scale_reciprocal
            mm1_scale_b = fc1.scale_reciprocal
        else:
            mm1_scale_a = None
            mm1_scale_b = None

        chipmunk.ops.mlp(
            x=x[0],
            fc1w=fc1.weight.data,
            fc1b=fc1.bias.data,
            fc2w_T=self.fc2w_T[0],
            indices=indices,
            counts=counts,
            sparse_act_T=sparse_act_T,
            cached_out=out_cache,
            num_sms_scatter_add=self.num_sms_scatter_add,
            mm1_scale_a=mm1_scale_a,
            mm1_scale_b=mm1_scale_b,
        )

        out_cache = out_cache.unsqueeze(0)
        self.storage.set_out_cache(out_cache)
        return out_cache
    
    # def __call__(self, *args, **kwargs):
    #     return self.forward(*args, **kwargs)