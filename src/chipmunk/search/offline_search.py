import torch
import json

class OfflineLayerSearch:
    """
    Offline greedy search to determine which layers should be computed in full vs sparse mode.
    Measures error between dense and sparse attention for each layer and optimizes layer selection.
    """
    def __init__(self, num_layers: int, output_path: str = "layer_selection.json"):
        self.num_layers = num_layers
        self.output_path = output_path
        self.errors = torch.zeros(num_layers)
        self.counts = torch.zeros(num_layers)
        self.cur_layer = 0
        self.dense_ref = None
        self.sparse_ref = None
        self.enabled = False  # Set to True when in search mode
        
    def reset_layer_counter(self):
        """Reset layer counter at the beginning of each forward pass."""
        self.cur_layer = 0
        
    @staticmethod
    def compute_error(o1: torch.Tensor, o2: torch.Tensor) -> torch.Tensor:
        """Compute normalized error between two tensors."""
        return torch.norm(o1 - o2) / torch.norm(o1)
        
    def dense(self, o: torch.Tensor):
        """Store dense attention output for current layer."""
        if not self.enabled:
            return
        self.dense_ref = o.detach()
        
    def sparse(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, sparse_fn):
        """
        Compute sparse attention and measure error against dense reference.
        
        Args:
            q, k, v: Query, key, value tensors
            sparse_fn: Function to compute sparse attention
        """
        if not self.enabled or self.dense_ref is None:
            return
            
        # Compute sparse attention
        with torch.no_grad():
            self.sparse_ref = sparse_fn(q, k, v).detach()
            
        # Measure error
        error = self.compute_error(self.dense_ref, self.sparse_ref)
        self.errors[self.cur_layer] += error.item()
        self.counts[self.cur_layer] += 1
        
        self.cur_layer += 1
        
    def finish(self):
        """
        Finalize search and save results.
        Uses greedy algorithm to select which layers should be computed in full.
        """
        if not self.enabled or torch.sum(self.counts) == 0:
            return
            
        # Compute average error per layer
        avg_errors = self.errors / self.counts
        
        # Sort layers by error (highest first)
        sorted_indices = torch.argsort(avg_errors, descending=True)
        
        # Select layers for full computation based on error
        full_layers = sorted_indices[:int(self.num_layers * 0.3)].tolist()  # Top 30% of layers
        
        # Save results
        results = {
            "full_layers": full_layers,
            "avg_errors": avg_errors.tolist()
        }
        
        with open(self.output_path, 'w') as f:
            json.dump(results, f)
            
        print(f"Layer selection results saved to {self.output_path}")