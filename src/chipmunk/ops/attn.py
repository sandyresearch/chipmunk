import torch
from typing import Tuple
from triton import cdiv

def csp_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    indices: torch.Tensor,
    indices_counts: torch.Tensor,
    o_scale: int
) -> None:
    return torch.ops.chipmunk.csp_attn(q, k, v, o, indices, indices_counts, o_scale)

def dense_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    o = torch.empty_like(q)
    lse = torch.empty((q.shape[0], q.shape[1], q.shape[2], 1), device=q.device, dtype=torch.float32)
    torch.ops.chipmunk.dense_attn(q, k, v, o, lse)
    return o, lse

def dense_colsum_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    l: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    o = torch.empty_like(q)
    cs = torch.empty((q.shape[0], q.shape[1], cdiv(q.shape[2], 192), q.shape[2]), device=q.device, dtype=q.dtype)
    lse = torch.empty((q.shape[0], q.shape[1], q.shape[2], 1), device=q.device, dtype=torch.float32)
    
    torch.ops.chipmunk.dense_colsum_attn(q, k, v, l, o, cs, lse)

    return o, cs, lse

__all__ = ['csp_attn', 'dense_attn', 'dense_colsum_attn']