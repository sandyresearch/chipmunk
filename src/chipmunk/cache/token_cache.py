import torch
from einops import rearrange
from chipmunk.util.config import GLOBAL_CONFIG

class TokenCache:
    """
    Per-layer token caching implementation that selects important tokens based on attention scores
    and other metrics, allowing for sparse computation in MLPs.

    Manages a mask that selects tokens for computation.

    Activation cache managed externally.
    """
    def __init__(self):
        self.enabled = GLOBAL_CONFIG['token_cache']['is_enabled']
        self.cache_ratio = GLOBAL_CONFIG['token_cache']['cache_ratio']
        # [b, n]
        self.inds = None
        # Size of patch/voxel
        self.chunk_size = GLOBAL_CONFIG['attn']['mbm']
        
    def score(self, cs: torch.Tensor) -> None:
        """
        Update token importance scores based on column sums of attention matrix.
        
        Args:
            cs: Attention column sums [b, h, qg, n]
        """
        if not self.enabled:
            return
            
        # Sum across heads and query groups to get per-token importance
        scores = cs.sum(dim=(1, 2))
        b, n = scores.shape
        
        # Initialize mask for token selection
        mask = torch.ones_like(scores, dtype=torch.bool)
        
        # Ensure spatial distribution by selecting top token in each chunk
        chunks = rearrange(scores, 'b (nc c) -> b nc c', c=self.chunk_size)
        chunk_inds = torch.argmax(chunks, dim=-1, keepdim=True)
        mask.view(chunks.shape).scatter_(-1, chunk_inds, False)

        # Select top tokens based on importance scores
        topk = int(n * (1 - self.cache_ratio))
        top_inds = torch.topk(scores * mask, topk, dim=1).indices

        self.inds = torch.cat([chunk_inds[..., 0], top_inds], dim=-1)
        
    def gather(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract tokens that need computation based on importance mask.
        
        Args:
            x: Input tensor [b, n, d]
            
        Returns:
            Tensor containing only tokens that need computation
        """
        if not self.enabled or self.inds is None:
            return x
            
        # Expand mask to match input dimensions
        expanded_inds = self.inds.unsqueeze(-1).expand(x.shape[0], -1, x.shape[-1])
        out = x.gather(dim=-1, index=expanded_inds).contiguous()
        return out

    def scatter(self, sparse_tokens: torch.Tensor, full_cache: torch.Tensor | None = None) -> torch.Tensor:
        """
        Scatter recomputed tokens into full cache.
        
        Args:
            sparse_tokens: Newly computed token features
            full_cache: Cached token features. If None, return sparse tokens.
            
        Returns:
            Complete tensor with computed and cached tokens
        """
        same_shape = sparse_tokens.shape == full_cache.shape
        if not self.enabled or self.inds is None or full_cache is None or same_shape:
            return sparse_tokens
            
        # Initialize result tensor with cached values
        result = full_cache
        
        # Update with newly computed tokens
        expanded_inds = self.inds.unsqueeze(-1).expand(result.shape[0], -1, result.shape[-1])
        result.scatter_(-1, expanded_inds, sparse_tokens)
        return result