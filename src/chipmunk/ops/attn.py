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
    # attn = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    # l_vec = torch.ones((q.shape[0], q.shape[1], q.shape[2]), device=q.device)
    # return attn, l_vec
    return_val = torch.ops.chipmunk.dense_attn(q, k, v)
    if return_val is None or len(return_val) == 0: breakpoint()
    return return_val

def dense_colsum_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    l: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return torch.ops.chipmunk.dense_colsum_attn(q, k, v, l)

__all__ = ['csp_attn', 'dense_attn', 'dense_colsum_attn']