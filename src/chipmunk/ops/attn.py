import torch
from typing import Tuple

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
    return torch.ops.chipmunk.dense_attn(q, k, v)

def dense_colsum_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    l: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return torch.ops.chipmunk.dense_colsum_attn(q, k, v, l)

__all__ = ['csp_attn', 'dense_attn', 'dense_colsum_attn']