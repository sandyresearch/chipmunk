import torch
from chipmunk.triton import csp_mlp_mm2_function_ptr, csp_mlp_mm2, csp_mlp_mm1_fp8
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
    scale_a: torch.Tensor = None,
    scale_b: torch.Tensor = None,
) -> None:
    assert x.dtype == torch.bfloat16
    assert sparse_act_packed.dtype == torch.bfloat16
    assert sparse_act_T.dtype == torch.bfloat16
    
    if fc1w.dtype == torch.float8_e4m3fn:
        csp_mlp_mm1_fp8(x, fc1w.T, fc1b, indices, counts, sparse_act_T, sparse_act_packed, scale_a, scale_b)
    elif fc1w.dtype == torch.bfloat16:
        torch.ops.chipmunk.csp_mlp_mm1(x, fc1w, sparse_act_packed, fc1b, sparse_act_T, indices, counts)
    else:
        raise ValueError(f"Unsupported dtype: {fc1w.dtype}")

def mm2_fused(
    packed: torch.Tensor,
    unpacked_colmajor: torch.Tensor,
    indices: torch.Tensor,
    counts: torch.Tensor,
    sparse_act_packed: torch.Tensor,
    fc2wT: torch.Tensor,
    cached_out: torch.Tensor,
    num_sms_scatter_add: int
) -> None:
    assert sparse_act_packed.dtype == torch.bfloat16
    assert fc2wT.dtype == torch.bfloat16
    assert cached_out.dtype == torch.bfloat16
    torch.ops.chipmunk.csp_mlp_mm2_and_scatter_add(packed.unsqueeze(0), unpacked_colmajor.unsqueeze(0), indices.unsqueeze(0), counts.unsqueeze(0), sparse_act_packed.unsqueeze(0), fc2wT.unsqueeze(0), cached_out.unsqueeze(0), num_sms_scatter_add, csp_mlp_mm2_function_ptr)

def mm2_unfused(
    sparse_act_packed: torch.Tensor,
    fc2wT: torch.Tensor,
    cached_out: torch.Tensor,
    unpacked_colmajor: torch.Tensor,
    indices: torch.Tensor,
    counts: torch.Tensor,
    num_sms_scatter_add: int
) -> None:
    assert sparse_act_packed.dtype == torch.bfloat16
    assert fc2wT.dtype == torch.bfloat16
    assert cached_out.dtype == torch.bfloat16
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
    num_sms_scatter_add: int,
    mm1_scale_a: torch.Tensor = None,
    mm1_scale_b: torch.Tensor = None,
) -> None:
    # Ensure shapes match
    M, K1 = x.shape
    K2, K1_ = fc1w.shape
    assert K1 == K1_, "K1 must match"
    K2_, N = fc2w_T.shape
    assert K2 == K2_, "K2 must match"

    # Packed intermediate activations matrix
    sparse_act_packed = torch.empty((M, K2), device=x.device, dtype=x.dtype)
    
    mm1(x, fc1w, sparse_act_packed, fc1b, sparse_act_T, indices, counts, mm1_scale_a, mm1_scale_b)

    # Fused implementation uses CUDAGraphs under the hood to allocate x certain # of SMs to communication kernel
    # and then uses the rest of the SMs for the actual matmul.
    mm2 = mm2_fused if USE_FUSED_MLP_MATMUL_2 else mm2_unfused
    mm2(sparse_act_packed, sparse_act_T, indices, counts, sparse_act_packed, fc2w_T, cached_out, num_sms_scatter_add)

__all__ = ['mm1', 'mm2_fused', 'mm2_unfused', 'run_e2e']
