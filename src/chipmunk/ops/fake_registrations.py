import torch

@torch.library.register_fake("chipmunk::csp_mlp_mm1")
def chipmunk_op_matmul_1_fake(
    a: torch.Tensor, 
    b_colmajor: torch.Tensor, 
    sparse_act_packed_out: torch.Tensor,
    fc1b: torch.Tensor, 
    indices: torch.Tensor, 
    counts: torch.Tensor, 
    sparse_act_colmajor_in: torch.Tensor,
) -> None:
    pass

@torch.library.register_fake("chipmunk::csp_mlp_mm2_and_scatter_add")
def chipmunk_op_csp_mlp_mm2_and_scatter_add_fake(
    packed: torch.Tensor,
    unpacked_colmajor: torch.Tensor,
    indices: torch.Tensor,
    counts: torch.Tensor,
    mma_a: torch.Tensor,
    mma_b: torch.Tensor,
    mma_c: torch.Tensor,
    num_sms_scatter_add: int,
    matmul_kernel: int
) -> None:
    pass

@torch.library.register_fake("chipmunk::csp_attn")
def chipmunk_op_csp_attn_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    indices: torch.Tensor,
    indices_counts: torch.Tensor,
    o_scale: int
) -> None:
    pass

@torch.library.register_fake("chipmunk::dense_attn")
def chipmunk_op_dense_attn_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o_out: torch.Tensor,
    lse_out: torch.Tensor
) -> None:
    pass

@torch.library.register_fake("chipmunk::dense_colsum_attn")
def chipmunk_op_dense_colsum_attn_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    p: torch.Tensor,
    o_out: torch.Tensor,
    cs_out: torch.Tensor,
    lse_out: torch.Tensor
) -> None:
    pass

@torch.library.register_fake("chipmunk::copy_indices")
def chipmunk_op_copy_indices_fake(
    bmfc1: torch.Tensor, 
    bm_mid_cache: torch.Tensor, 
    indices: torch.Tensor, 
    counts: torch.Tensor
) -> None:
    pass

@torch.library.register_fake("chipmunk::topk_indices")
def chipmunk_op_topk_indices_fake(
    activations: torch.Tensor, 
    indices_out: torch.Tensor, 
    counts_out: torch.Tensor, 
    sparsity_amount: float, 
    multiple_of: int, 
    rk: float
) -> None:
    pass

@torch.library.register_fake("chipmunk::csp_scatter_add")
def chipmunk_op_scatter_add_fake(
    packed: torch.Tensor, 
    unpacked: torch.Tensor, 
    indices: torch.Tensor, 
    counts: torch.Tensor, 
    num_sms: int
) -> None:
    pass

@torch.library.register_fake("chipmunk::mask_to_indices")
def chipmunk_op_mask_to_indices_fake(
    mask: torch.Tensor,
    multiple_of: int,
    indices_out: torch.Tensor,
    counts_out: torch.Tensor
) -> None:
    pass
