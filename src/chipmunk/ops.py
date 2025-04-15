import torch
import triton.language as tl
from chipmunk.triton import csp_mlp_mm2

@torch.library.register_fake("chipmunk::csp_mlp_mm1")
def chipmunk_op_matmul_1_fake(
    a: torch.Tensor, 
    b_colmajor: torch.Tensor, 
    fc1b: torch.Tensor, 
    sp_inds: torch.Tensor, 
    sp_counts: torch.Tensor, 
    sparse_act_colmajor_in: torch.Tensor, 
    sparse_act_packed_out: torch.Tensor
) -> None:
    pass

@torch.library.register_fake("chipmunk::csp_scatter_add")
def chipmunk_op_scatter_add_fake(
    packed: torch.Tensor, 
    unpacked: torch.Tensor, 
    sp_inds: torch.Tensor, 
    sp_counts: torch.Tensor, 
    completion_progress: torch.Tensor,
    num_sms: int
) -> None:
    pass

@torch.library.register_fake("chipmunk::copy_indices")
def copy_indices_fake(
    bmfc1: torch.Tensor, 
    bm_mid_cache: torch.Tensor, 
    sp_inds: torch.Tensor, 
    sp_counts: torch.Tensor
) -> None:
    pass

@torch.library.register_fake("chipmunk::topk_indices")
def topk_indices_fake(
    activations: torch.Tensor, 
    indices_out: torch.Tensor, 
    counts_out: torch.Tensor, 
    sparsity_amount: float, 
    multiple_of: int, 
    rk: float
) -> None:
    pass

@torch.library.register_fake("chipmunk::graph_scatter_add_matmul_2")
def graph_scatter_add_matmul_2_fake(
    packed: torch.Tensor,
    unpacked_colmajor: torch.Tensor,
    sp_inds: torch.Tensor,
    sp_counts: torch.Tensor,
    completion_progress: torch.Tensor,
    mma_a: torch.Tensor,
    mma_b: torch.Tensor,
    mma_c: torch.Tensor,
    num_sms_scatter_add: int,
    matmul_kernel: int
) -> None:
    pass

csp_mlp_mm2_function_ptr = csp_mlp_mm2(
    torch.randn((256, 256), dtype=torch.bfloat16, device='cuda'), 
    torch.randn((256, 256), dtype=torch.bfloat16, device='cuda'), 
    torch.arange(0, 256, 1, device='cuda', dtype=torch.int32).repeat(2, 1).contiguous(), 
    torch.full((2,), 256, device='cuda', dtype=torch.int32), 
    output=torch.empty((256, 256), device='cuda', dtype=torch.bfloat16)
).function