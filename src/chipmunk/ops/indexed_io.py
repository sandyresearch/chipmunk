import torch
from typing import List

def copy_indices(
    bm_fc1: torch.Tensor, 
    bm_mid_cache: torch.Tensor, 
    indices: torch.Tensor, 
    counts: torch.Tensor
) -> None:
    torch.ops.chipmunk.copy_indices(bm_fc1, bm_mid_cache, indices, counts)


def topk_indices(
    activations: torch.Tensor, 
    indices_out: torch.Tensor, 
    counts_out: torch.Tensor, 
    sparsity_amount: float, 
    multiple_of: int, 
    rk: float
) -> None:
    torch.ops.chipmunk.topk_indices(activations, indices_out, counts_out, sparsity_amount, multiple_of, rk)

def scatter_add(
    packed: torch.Tensor, 
    unpacked: torch.Tensor, 
    indices: torch.Tensor, 
    counts: torch.Tensor, 
    num_sms: int
) -> None:
    torch.ops.chipmunk.csp_scatter_add(packed.unsqueeze(0), unpacked.unsqueeze(0), indices.unsqueeze(0), counts.unsqueeze(0), num_sms)

def mask_to_indices(
    mask: torch.Tensor,
    multiple_of: int,
    pad_to_multiple_of: int
) -> List[torch.Tensor]:
    return torch.ops.chipmunk.mask_to_indices(mask, multiple_of, pad_to_multiple_of)