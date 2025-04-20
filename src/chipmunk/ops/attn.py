import torch
from typing import Tuple
from einops import rearrange

# def csp_attn(
#     q: torch.Tensor,
#     k: torch.Tensor,
#     v: torch.Tensor,
#     indices: torch.Tensor,
#     indices_counts: torch.Tensor,
# ) -> torch.Tensor:
#     q = q.contiguous()
#     k = k.contiguous()
#     v = v.contiguous()
#     indices = indices.contiguous()
#     indices_counts = indices_counts.contiguous()
#     return torch.ops.chipmunk_tk_kernels.csp_attn_fwd(q, k, v, indices, indices_counts)

# def dense_attn(
#     q: torch.Tensor,
#     k: torch.Tensor,
#     v: torch.Tensor
# ) -> Tuple[torch.Tensor, torch.Tensor]:
#     q = q.contiguous()
#     k = k.contiguous()
#     v = v.contiguous()
#     return torch.ops.chipmunk_tk_kernels.attn_fwd(q, k, v)

# def dense_colsum_attn(
#     q: torch.Tensor,
#     k: torch.Tensor,
#     v: torch.Tensor,
#     l: torch.Tensor
# ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#     q = q.contiguous()
#     k = k.contiguous()
#     v = v.contiguous()
#     l = l.contiguous()
#     return torch.ops.chipmunk_tk_kernels.colsum_attn_fwd(q, k, v, l)

def dense_attn(q, k, v):
    """
    Compute variable length attention in ThunderKittens.
    """
    return_l = True
    pm = 192
    if q.shape[-2] % pm == 0:
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        # o, l = torch.ops.chipmunk_tk_kernels.attn_fwd(q, k, v)
        o, l = torch.ops.chipmunk_tk_kernels.attn_fwd(q, k, v)
        if return_l:
            return o, l
        else:
            return o

    # pad
    n = q.shape[-2]
    padded_n = ((n + pm - 1) // pm) * pm
    qp = torch.empty(q.shape[:-2] + (padded_n, q.shape[-1]), dtype=q.dtype, device=q.device)
    qp[..., :n, :] = q

    # contiguous
    qp = qp.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    # compute
    # o, l = torch_tk_attn(qp, k, v)
    o, l = torch.ops.chipmunk_tk_kernels.attn_fwd(qp, k, v)

    # unpad
    o = o[..., :n, :].contiguous()
    # leave l padded to pass back in
    l[..., n:, :] = 0

    if return_l:
        return o, l
    else:
        return o

def dense_colsum_attn(q, k, v, p):
        """
        Compute variable length attention in ThunderKittens.
        """
        fuse_reduce = True
        wq = 16   # queries per warp
        pm = 192  # queries per producer (16 * 4 * 3)
        if q.shape[-2] % pm == 0:
            # o, cs, l = torch_colsum_attn_fwd(q, k, v, p)
            o, cs, l = torch.ops.chipmunk_tk_kernels.colsum_attn_fwd(q, k, v, p)
            if not fuse_reduce:
                cs = rearrange(cs, 'b h (m r) n -> b h m r n', r=pm//wq).sum(dim=-2)
            return o, cs, l

        # pad
        n = q.shape[-2]
        padded_n = ((n + pm - 1) // pm) * pm
        qp = torch.empty(q.shape[:-2] + (padded_n, q.shape[-1]), dtype=q.dtype, device=q.device)
        qp[..., :n, :] = q

        # contiguous
        qp = qp.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        p = p.contiguous()

        assert p.shape[-2] == padded_n

        # compute
        # o, cs, l = torch_colsum_attn_fwd(qp, k, v, p)
        o, cs, l = torch.ops.chipmunk_tk_kernels.colsum_attn_fwd(qp, k, v, p)

        # unpad
        o = o[..., :n, :].contiguous()
        # leave l padded to pass back in
        l[..., n:, :] = 0
        if fuse_reduce:
            # use k since on single gpu q is padded to 119056
            kseq = k.shape[-2]
            kgroups = (kseq + pm - 1) // pm
            cs = cs[..., :kgroups, :kseq]
        else:
            cs = rearrange(cs, 'b h (m r) n -> b h m r n', r=pm//wq).sum(dim=-2)[..., :n]
        return o, cs, l

def csp_attn(q, k, v, indices, indices_counts):
    if q.shape[-2] % 192 == 0:
        # return torch_csp_attn_fwd(q, k, v, indices, indices_counts)
        return torch.ops.chipmunk_tk_kernels.csp_attn_fwd(q, k, v, indices, indices_counts)

    # pad
    n = q.shape[-2]
    padded_n = ((n + 192 - 1) // 192) * 192
    qp = torch.zeros(q.shape[:-2] + (padded_n, q.shape[-1]), dtype=q.dtype, device=q.device)
    qp[..., :n, :] = q

    # pad indices to multiple of 16
    indicesp = indices

    # contiguous
    qp = qp.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    indicesp = indicesp.contiguous()

    # o = torch_csp_attn_fwd(qp, k, v, indicesp, indices_counts)
    o = torch.ops.chipmunk_tk_kernels.csp_attn_fwd(qp, k, v, indicesp, indices_counts)

    # unpad
    o = o[..., :n, :].contiguous()
    return o

__all__ = ['csp_attn', 'dense_attn', 'dense_colsum_attn']