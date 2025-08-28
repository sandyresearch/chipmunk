from typing import Tuple
import torch


def torch_tk_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.chipmunk_tk_kernels.attn_fwd(q, k, v)

def torch_colsum_attn_fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, p: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return torch.ops.chipmunk_tk_kernels.colsum_attn_fwd(q, k, v, p)

def torch_csp_attn_fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, indices: torch.Tensor, indices_counts: torch.Tensor) -> torch.Tensor:
    return torch.ops.chipmunk_tk_kernels.csp_attn_fwd(q, k, v, indices, indices_counts)

def torch_mask_to_indices(mask: torch.Tensor, multiple_of: int, pad_to_multiple_of: int) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.chipmunk_tk_kernels.mask_to_indices(mask, multiple_of, pad_to_multiple_of)
