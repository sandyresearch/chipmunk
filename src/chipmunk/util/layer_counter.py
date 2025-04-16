from ..util.config import GLOBAL_CONFIG

class LayerCounter:
    def __init__(self, num_layers: int, num_sparse_submodules_per_layer: int):
        self.num_layers = num_layers
        self.num_submodules_per_layer = num_sparse_submodules_per_layer

        self.cur_inference_step = 0
        self.cur_layer = 0
        self.cur_layer_submodule = 0

    def should_do_full_mlp_step(self):
        return self.cur_inference_step % GLOBAL_CONFIG['mlp']['full_step_every'] == 0
    
    def should_do_full_attn_step(self):
        return self.cur_inference_step % GLOBAL_CONFIG['attn']['full_step_every'] == 0

    def increment(self):
        cur_coord = (self.cur_inference_step, self.cur_layer, self.cur_layer_submodule)
        self.cur_layer_submodule += 1
        if self.cur_layer_submodule == self.num_submodules_per_layer:
            self.cur_layer_submodule = 0
            self.cur_layer += 1
            if self.cur_layer == self.num_layers:
                self.cur_layer = 0
                self.cur_inference_step += 1
        
        return cur_coord
    
    def reset(self):
        self.cur_inference_step = 0
        self.cur_layer = 0
        self.cur_layer_submodule = 0

    def get_cur_coord(self):
        return (self.cur_inference_step, self.cur_layer, self.cur_layer_submodule)
