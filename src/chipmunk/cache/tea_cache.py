import torch
import numpy as np
from typing import Tuple, Optional, Dict, Any
from chipmunk.util.config import GLOBAL_CONFIG

COEFFICIENTS = {
    # https://github.com/ali-vilab/TeaCache/blob/main/TeaCache4HunyuanVideo/teacache_sample_video.py#L102
    'hunyuan': [7.33226126e+02, -4.01131952e+02,  6.75869174e+01, -3.14987800e+00, 9.61237896e-02],
    # https://github.com/ali-vilab/TeaCache/blob/main/TeaCache4FLUX/teacache_flux.py#L113C32-L113C115
    'flux': [4.98651651e+02, -2.83781631e+02,  5.58554382e+01, -3.82021401e+00, 2.64230861e-01],
    # https://github.com/ali-vilab/TeaCache/blob/main/TeaCache4Wan2.1/teacache_generate.py#L892
    'wan': [-5784.54975374,  5449.50911966, -1811.16591783,   256.27178429, -13.02252404],
    # https://github.com/ali-vilab/TeaCache/blob/main/TeaCache4Mochi/teacache_mochi.py#L70C32-L70C116
    'mochi': [-3.51241319e+03,  8.11675948e+02, -6.09400215e+01,  2.42429681e+00, 3.05291719e-03],
}

class TeaCache:
    """
    TeaCache implementation for skipping computation on certain diffusion steps.
    Stores intermediate results that can be reused in subsequent steps.
    """
    def __init__(self):
        self.input_cache = None
        self.output_cache = None

        self.accumulated_error = 0

        print(f'GLOBAL_CONFIG: {GLOBAL_CONFIG["tea_cache"]}')
        self.enabled = GLOBAL_CONFIG['tea_cache']['is_enabled']
        self.threshold = GLOBAL_CONFIG['tea_cache']['threshold']

        # For debugging
        self.debug = GLOBAL_CONFIG['tea_cache']['debug']
        self.step_counter = 1

    def error(self, modulated: torch.Tensor) -> float:
        # https://github.com/ali-vilab/TeaCache/blob/main/TeaCache4HunyuanVideo/teacache_sample_video.py#L104
        rescale_func = np.poly1d(COEFFICIENTS[GLOBAL_CONFIG['model_name']])
        return rescale_func(((modulated - self.input_cache).abs().mean() / self.input_cache.abs().mean()).cpu().item())
        
    def step(self, modulated: torch.Tensor) -> bool:
        """Determine if current step should be skipped based on configuration."""
        # Disabled
        if not self.enabled:
            return False

        # First step
        if self.input_cache is None:
            self.input_cache = modulated.clone()
            return False

        # All other steps
        self.accumulated_error += self.error(modulated)
        if self.debug:  
            print(f'TeaCache Step {self.step_counter} accumulated error: {self.accumulated_error}')
        if self.accumulated_error > self.threshold:
            self.input_cache = modulated.clone()
            self.accumulated_error = 0
            return False
        if self.debug:
            print(f'TeaCache skipping Step {self.step_counter}')
            self.step_counter += 1
        return True
    
    def store(self, x: torch.Tensor) -> None:
        """Store tensor for future use."""
        if self.enabled:
            self.cache = x.clone()
    
    def load(self) -> Optional[torch.Tensor]:
        """Retrieve cached tensor if available."""
        if self.enabled and self.cache is not None:
            return self.cache
        return None