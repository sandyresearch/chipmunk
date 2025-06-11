from ..util.config import GLOBAL_CONFIG

class LayerCounter:
    def __init__(self, num_layers: int, num_sparse_submodules_per_layer: int):
        self.num_layers = num_layers
        self.num_submodules_per_layer = num_sparse_submodules_per_layer
        self.has_mlp_sparsity = False
        self.has_attn_sparsity = False

        self.cur_inference_step = 0
        self.cur_model_invocation_per_step = 0
        self.cur_layer = 0
        self.cur_layer_submodule = 0

    @staticmethod
    def build_for_layer(is_mlp_sparse: bool = False, is_attn_sparse: bool = False):
        old_layer_num = singleton.num_layers
        singleton.num_layers += 1
        if is_attn_sparse and not singleton.has_attn_sparsity:
            singleton.has_attn_sparsity = True
            singleton.num_submodules_per_layer += 1
        if is_mlp_sparse and not singleton.has_mlp_sparsity:
            singleton.has_mlp_sparsity = True
            singleton.num_submodules_per_layer += 1
        return old_layer_num, singleton

    def should_do_full_mlp_step(self):
        return self.cur_inference_step % GLOBAL_CONFIG['mlp']['full_step_every'] == 0
    
    def should_do_full_attn_step(self):
        manual_full_step_schedule = GLOBAL_CONFIG['attn']['full_step_schedule']
        if manual_full_step_schedule is not None:
            return self.cur_inference_step in manual_full_step_schedule
        else:
            return self.cur_inference_step < 2 or self.cur_inference_step % GLOBAL_CONFIG['attn']['full_step_every'] == 0

    def increment(self):
        # Current coordinate to be returned
        cur_coord = (self.cur_inference_step, self.cur_layer, self.cur_layer_submodule)

        # Increment coordinate
        self.cur_layer_submodule += 1
        if self.cur_layer_submodule == self.num_submodules_per_layer:
            self.cur_layer_submodule = 0
            self.cur_layer += 1
            if self.cur_layer == self.num_layers:
                self.cur_layer = 0
                self.cur_model_invocation_per_step += 1
                if self.cur_model_invocation_per_step == GLOBAL_CONFIG['num_model_invocations_per_inference_step']:
                    self.cur_model_invocation_per_step = 0
                    self.cur_inference_step += 1
        
        if self.cur_inference_step == GLOBAL_CONFIG['steps'] - 1 and \
            self.cur_layer == self.num_layers - 1 and \
            self.cur_layer_submodule == self.num_submodules_per_layer - 1 and \
            self.cur_model_invocation_per_step == GLOBAL_CONFIG['num_model_invocations_per_inference_step'] - 1:
            self.reset()
        
        return cur_coord
    
    def reset(self):
        self.cur_inference_step = 0
        self.cur_layer = 0
        self.cur_layer_submodule = 0
        self.cur_model_invocation_per_step = 0

    def get_cur_coord(self):
        return (self.cur_inference_step, self.cur_layer, self.cur_layer_submodule)
    
    def __repr__(self) -> str:
        return f"LayerCounter(cur_inference_step={self.cur_inference_step}/{GLOBAL_CONFIG['steps']}, cur_layer={self.cur_layer}/{self.num_layers}, cur_layer_submodule={self.cur_layer_submodule}/{self.num_submodules_per_layer}, cur_model_invocation_per_step={self.cur_model_invocation_per_step}/{GLOBAL_CONFIG['num_model_invocations_per_inference_step']})"

singleton = LayerCounter(0, 0)
