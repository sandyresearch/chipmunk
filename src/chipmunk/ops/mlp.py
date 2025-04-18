import torch
from chipmunk.triton import csp_mlp_mm2_function_ptr, csp_mlp_mm2
from chipmunk.ops.indexed_io import scatter_add

USE_FUSED_MLP_MATMUL_2 = True

def mm1(
    x: torch.Tensor, 
    fc1w: torch.Tensor, 
    sparse_act_packed: torch.Tensor,
    fc1b: torch.Tensor, 
    sparse_act_T: torch.Tensor,
    indices: torch.Tensor, 
    counts: torch.Tensor, 
) -> None:
    torch.ops.chipmunk.csp_mlp_mm1(x, fc1w, sparse_act_packed, fc1b, sparse_act_T, indices, counts)

def mm2_fused(
    packed: torch.Tensor,
    unpacked_colmajor: torch.Tensor,
    indices: torch.Tensor,
    counts: torch.Tensor,
    mma_a: torch.Tensor,
    mma_b: torch.Tensor,
    mma_c: torch.Tensor,
    num_sms_scatter_add: int
) -> None:
    torch.ops.chipmunk.csp_mlp_mm2_and_scatter_add(packed.unsqueeze(0), unpacked_colmajor.unsqueeze(0), indices.unsqueeze(0), counts.unsqueeze(0), mma_a.unsqueeze(0), mma_b.unsqueeze(0), mma_c.unsqueeze(0), num_sms_scatter_add, csp_mlp_mm2_function_ptr)

def mm2_unfused(
    sparse_act_packed: torch.Tensor,
    fc2wT: torch.Tensor,
    cached_out: torch.Tensor,
    unpacked_colmajor: torch.Tensor,
    indices: torch.Tensor,
    counts: torch.Tensor,
    num_sms_scatter_add: int
) -> None:
    scatter_add(sparse_act_packed, unpacked_colmajor, indices, counts, num_sms_scatter_add)
    csp_mlp_mm2(sparse_act_packed, fc2wT, indices, counts, cached_out, 132-num_sms_scatter_add)

def run_e2e(
    x: torch.Tensor, 
    fc1w: torch.Tensor, 
    fc1b: torch.Tensor,
    fc2w_T: torch.Tensor, 
    indices: torch.Tensor, 
    counts: torch.Tensor, 
    sparse_act_T: torch.Tensor, 
    cached_out: torch.Tensor, 
    num_sms_scatter_add: int
) -> None:
    # Ensure shapes match
    M, K1 = x.shape
    K2, K1_ = fc1w.shape
    assert K1 == K1_, "K1 must match"
    K2_, N = fc2w_T.shape
    assert K2 == K2_, "K2 must match"

    # Packed intermediate activations matrix
    sparse_act_packed = torch.empty((M, K2), device=x.device, dtype=x.dtype)
    
    mm1(x, fc1w, sparse_act_packed, fc1b, sparse_act_T, indices, counts)

    # Fused implementation uses CUDAGraphs under the hood to allocate x certain # of SMs to communication kernel
    # and then uses the rest of the SMs for the actual matmul.
    mm2 = mm2_fused if USE_FUSED_MLP_MATMUL_2 else mm2_unfused
    mm2(sparse_act_packed, sparse_act_T, indices, counts, sparse_act_packed, fc2w_T, cached_out, num_sms_scatter_add)

__all__ = ['mm1', 'mm2_fused', 'mm2_unfused', 'run_e2e']
