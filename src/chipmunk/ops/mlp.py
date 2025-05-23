import torch
import chipmunk.triton
from chipmunk.ops.indexed_io import scatter_add
from chipmunk.util import GLOBAL_CONFIG


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
        # FP8 is not supported in CUDA yet!
        # We only support FP8 in Triton for now.
        chipmunk.triton.csp_mlp_mm1_fp8(x, fc1w.T, fc1b, indices, counts, sparse_act_T, sparse_act_packed, scale_a, scale_b)
    elif fc1w.dtype == torch.bfloat16:
        provider = GLOBAL_CONFIG['mlp']['provider']
        if provider == 'triton':
            chipmunk.triton.csp_mlp_mm1_bf16(x, fc1w, sparse_act_packed, fc1b, sparse_act_T, indices, counts)
        elif provider == 'cuda':
            torch.ops.chipmunk.csp_mlp_mm1(x, fc1w, sparse_act_packed, fc1b, sparse_act_T, indices, counts)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    else:
        raise ValueError(f"Unsupported dtype: {fc1w.dtype}")

def mm2_cuda(
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
    assert GLOBAL_CONFIG['mlp']['provider'] == 'cuda'
    # Fused implementation uses CUDAGraphs under the hood to allocate x certain # of SMs to scatter_add cache writeback kernel
    # and then uses the rest of the SMs for the actual matmul.
    torch.ops.chipmunk.csp_mlp_mm2_and_scatter_add(packed.unsqueeze(0), unpacked_colmajor.unsqueeze(0), indices.unsqueeze(0), counts.unsqueeze(0), sparse_act_packed.unsqueeze(0), fc2wT.unsqueeze(0), cached_out.unsqueeze(0), num_sms_scatter_add, chipmunk.triton.csp_mlp_mm2_bf16_function_ptr)

def mm2_triton(
    sparse_act_packed: torch.Tensor,
    _: torch.Tensor,
    indices: torch.Tensor,
    counts: torch.Tensor,
    __: torch.Tensor,
    fc2wT: torch.Tensor,
    cached_out: torch.Tensor,
    ___: int
) -> None:
    assert sparse_act_packed.dtype == torch.bfloat16
    assert fc2wT.dtype == torch.bfloat16
    assert cached_out.dtype == torch.bfloat16
    assert GLOBAL_CONFIG['mlp']['provider'] == 'triton'
    # scatter-add is fused into the csp_mlp_mm1 kernel, so we don't have to take care of it!
    chipmunk.triton.csp_mlp_mm2_bf16(sparse_act_packed, fc2wT, indices, counts, cached_out)

# REASON FOR DISABLING TORCH COMPILE:
# torch inductor likes to insert a dummy Triton kernel between matmul 1 and 2 that just copies data (`triton_fused_poi_2`) 
# that eats up 68us per invocation :(
# uncompiled is still fast because MLPs are very compute-bound and our custom kernels have everything fused in them!
@torch.compiler.disable
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
    mm2 = mm2_cuda if GLOBAL_CONFIG['mlp']['provider'] == 'cuda' else mm2_triton
    mm2(sparse_act_packed, sparse_act_T, indices, counts, sparse_act_packed, fc2w_T, cached_out, num_sms_scatter_add)

__all__ = ['run_e2e']
