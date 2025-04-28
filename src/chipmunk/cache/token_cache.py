import torch
import torch.distributed as dist
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
        if GLOBAL_CONFIG['world_size'] > 1:
            # all reduce sum
            dist.all_reduce(scores, op=dist.ReduceOp.SUM, group=dist.group.WORLD)

            # shard
            rank, world_size = dist.get_rank(), dist.get_world_size()
            # start, end = rank * GLOBAL_CONFIG['cp_seq_len'] // world_size, (rank + 1) * GLOBAL_CONFIG['cp_seq_len'] // world_size
            start, end = rank * GLOBAL_CONFIG['non_text_seqlen'] // world_size, (rank + 1) * GLOBAL_CONFIG['non_text_seqlen'] // world_size
            scores = scores[:, start:end]
        else:
            scores = scores[:, :GLOBAL_CONFIG['non_text_seqlen']]

        b, n = scores.shape
        
        # Initialize mask for token selection
        mask = torch.ones_like(scores, dtype=torch.bool)
        
        # Ensure spatial distribution by selecting top token in each chunk
        n = scores.shape[1]
        n_sliced = (n // self.chunk_size) * self.chunk_size
        chunks = rearrange(scores[:, :n_sliced], 'b (nc c) -> b nc c', c=self.chunk_size)
        chunk_inds = torch.argmax(chunks, dim=-1, keepdim=True)
        mask[..., :n_sliced].view(chunks.shape).scatter_(-1, chunk_inds, False)
        
        # Convert chunk indices to flat indices
        flat_chunk_inds = chunk_inds + torch.arange(0, n // self.chunk_size, 
                                                   device=scores.device).unsqueeze(0).unsqueeze(-1) * self.chunk_size
        flat_chunk_inds = flat_chunk_inds.reshape(b, -1)
        
        # Select top tokens based on importance scores
        topk = int(n * (1 - self.cache_ratio))
        top_inds = torch.topk(scores * mask, topk, dim=1).indices

        self.inds = torch.cat([flat_chunk_inds, top_inds], dim=-1)
        
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

        # Update indices if text tokens are present in MLP (i.e., for single stream blocks)
        cp_seqlen = GLOBAL_CONFIG['non_text_seqlen'] // GLOBAL_CONFIG['world_size']
        if x.shape[-2] > cp_seqlen and (self.inds[0, -1] != (x.shape[-2] - 1)):
            txt_len = x.shape[-2] - cp_seqlen
            txt_inds = torch.arange(txt_len, device=x.device).unsqueeze(0) + cp_seqlen
            self.inds = torch.cat([self.inds, txt_inds], dim=-1)
            
        # Expand mask to match input dimensions
        expanded_inds = self.inds.unsqueeze(-1).expand(x.shape[0], -1, x.shape[-1])
        out = x.gather(dim=-2, index=expanded_inds)
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
        result.scatter_(-2, expanded_inds, sparse_tokens)

        return result