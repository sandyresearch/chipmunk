import torch
from typing import Tuple, Optional, Dict, Any
from chipmunk.util.config import GLOBAL_CONFIG

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
        return (modulated - self.input_cache).abs().mean() / self.input_cache.abs().mean()
        
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