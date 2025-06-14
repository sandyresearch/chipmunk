import torch
from torch import Tensor
from .offloaded_tensor import MaybeOffloadedTensor
from chipmunk.util import GLOBAL_CONFIG

class MlpStorage:
    def __init__(self, layer_num: int):
        self.layer_num = layer_num

        self.sparse_act_T = None
        self.out_cache = None
        self.indices = None
        self.counts = None
        self.blockmean_mid_cache = None
    
    def complete_cur_layer(self):
        if self.blockmean_mid_cache is not None:
            self.blockmean_mid_cache.complete_cur_layer()
        if self.out_cache is not None:
            self.out_cache.complete_cur_layer()
        if self.indices is not None:
            self.indices.complete_cur_layer()
        if self.counts is not None:
            self.counts.complete_cur_layer()

    def get_blockmean_mid_cache(self):
        if self.blockmean_mid_cache is None:
            return None
        return self.blockmean_mid_cache.get_loaded_value()

    def set_blockmean_mid_cache(self, blockmean_mid_cache: Tensor):
        if self.blockmean_mid_cache is None:
            self.blockmean_mid_cache = MaybeOffloadedTensor('mlp.blockmean_mid_cache', self.layer_num, blockmean_mid_cache.dtype, blockmean_mid_cache.device)
        self.blockmean_mid_cache.offload(blockmean_mid_cache)

    def get_sparse_act_T(self):
        if self.sparse_act_T is None:
            return None
        return self.sparse_act_T.get_loaded_value()

    def set_sparse_act_T(self, sparse_act_T: Tensor):
        if self.sparse_act_T is None:
            self.sparse_act_T = MaybeOffloadedTensor('mlp.sparse_act_T', self.layer_num, sparse_act_T.dtype, sparse_act_T.device)
        self.sparse_act_T.offload(sparse_act_T)

    def get_out_cache(self):
        if self.out_cache is None:
            return None
        return self.out_cache.get_loaded_value()

    def set_out_cache(self, out_cache: Tensor):
        if self.out_cache is None:
            self.out_cache = MaybeOffloadedTensor('mlp.out_cache', self.layer_num, out_cache.dtype, out_cache.device)
        self.out_cache.offload(out_cache)

    def get_indices(self):
        if self.indices is None:
            return None
        return self.indices.get_loaded_value()

    def set_indices(self, indices: Tensor):
        if self.indices is None:
            self.indices = MaybeOffloadedTensor('mlp.indices', self.layer_num, indices.dtype, indices.device)
        self.indices.offload(indices)

    def get_counts(self):
        if self.counts is None:
            return None
        return self.counts.get_loaded_value()

    def set_counts(self, counts: Tensor):
        if self.counts is None:
            self.counts = MaybeOffloadedTensor('mlp.counts', self.layer_num, counts.dtype, counts.device)
        self.counts.offload(counts)

    def load_async(self):
        if self.sparse_act_T is not None:
            self.sparse_act_T.load_async()
        if self.out_cache is not None:
            self.out_cache.load_async()
        if self.indices is not None:
            self.indices.load_async()
        if self.counts is not None:
            self.counts.load_async()
    
    def load_async_wait(self):
        if self.sparse_act_T is not None:
            self.sparse_act_T.load_async_wait()
        if self.out_cache is not None:
            self.out_cache.load_async_wait()
        if self.indices is not None:
            self.indices.load_async_wait()
        if self.counts is not None:
            self.counts.load_async_wait()

class AttnStorage:
    def __init__(self, layer_num: int, init_names: list[str]=[]):
        self.layer_num = layer_num

        self.indices = None
        self.counts = None
        self.out_cache = None
        self.lse_constants = None

        if GLOBAL_CONFIG['offloading']['global_disable_offloading']:
            return

        # for name in init_names:
        if 'out_cache' in init_names:
            self.out_cache = MaybeOffloadedTensor(
                f'attn.out_cache',
                self.layer_num, torch.bfloat16,
                torch.device('cuda'),
                cpu_buf_size=MaybeOffloadedTensor.LARGE_BUF_SIZE
            )
        if 'indices' in init_names:
            self.indices = MaybeOffloadedTensor(
                f'attn.indices',
                self.layer_num, torch.uint8,
                torch.device('cuda'),
                cpu_buf_size=MaybeOffloadedTensor.MEDIUM_BUF_SIZE
            )

    def complete_cur_layer(self):
        if self.indices is not None:
            self.indices.complete_cur_layer()
        if self.counts is not None:
            self.counts.complete_cur_layer()
        if self.out_cache is not None:
            self.out_cache.complete_cur_layer()
        if self.lse_constants is not None:
            self.lse_constants.complete_cur_layer()


    def get_indices(self):
        if self.indices is None:
            return None
        return self.indices.get_loaded_value()

    def set_indices(self, indices: Tensor):
        if self.indices is None:
            self.indices = MaybeOffloadedTensor('attn.indices', self.layer_num, indices.dtype, indices.device)
        self.indices.offload(indices)

    def get_counts(self):
        if self.counts is None:
            return None
        return self.counts.get_loaded_value()

    def set_counts(self, counts: Tensor):
        if self.counts is None:
            self.counts = MaybeOffloadedTensor('attn.counts', self.layer_num, counts.dtype, counts.device)
        self.counts.offload(counts)

    def get_out_cache(self):
        if self.out_cache is None:
            return None
        return self.out_cache.get_loaded_value()

    def set_out_cache(self, out_cache: Tensor):
        if self.out_cache is None:
            self.out_cache = MaybeOffloadedTensor('attn.out_cache', self.layer_num, out_cache.dtype, out_cache.device)
        self.out_cache.offload(out_cache)

    def get_lse_constants(self):
        if self.lse_constants is None:
            return None
        return self.lse_constants.get_loaded_value()
    
    def set_lse_constants(self, lse_constants: Tensor):
        if self.lse_constants is None:
            tensor = lse_constants[0] if isinstance(lse_constants, tuple) else lse_constants
            self.lse_constants = MaybeOffloadedTensor('attn.lse_constants', self.layer_num, tensor.dtype, tensor.device)
        self.lse_constants.offload(lse_constants)

    def load_async(self):
        if self.indices is not None:
            self.indices.load_async()
        if self.counts is not None:
            self.counts.load_async()
        if self.out_cache is not None:
            self.out_cache.load_async()
        if self.lse_constants is not None:
            self.lse_constants.load_async()

    def load_async_wait(self):
        if self.indices is not None:
            self.indices.load_async_wait()
        if self.counts is not None:
            self.counts.load_async_wait()
        if self.out_cache is not None:
            self.out_cache.load_async_wait()
        if self.lse_constants is not None:
            self.lse_constants.load_async_wait()

class LayerStorage:
    def __init__(self, layer_num: int):
        self.layer_num = layer_num

        self.mlp = MlpStorage(layer_num)
        self.attn = AttnStorage(layer_num)


    def load_async(self):
        self.mlp.load_async()
        self.attn.load_async()

    def load_async_wait(self):
        self.mlp.load_async_wait()
        self.attn.load_async_wait()