import torch
from typing import Tuple
from einops import rearrange
import chipmunk
from chipmunk.util import get_kernel_config_attn, GLOBAL_CONFIG

def dense_attn(q, k, v):
    return_l = True
    pad_to = get_kernel_config_attn()['bm']
    provider = GLOBAL_CONFIG['attn']['provider']
    
    if q.shape[-2] % pad_to == 0:
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        if provider == 'cuda':
            o, l = torch.ops.chipmunk.dense_attn(q, k, v)
        else:
            o, l = chipmunk.triton.dense_attn(q, k, v)
        
        if return_l:
            return o, l
        else:
            return o

    # pad
    n = q.shape[-2]
    padded_n = ((n + pad_to - 1) // pad_to) * pad_to
    qp = torch.empty(q.shape[:-2] + (padded_n, q.shape[-1]), dtype=q.dtype, device=q.device)
    qp[..., :n, :] = q

    # contiguous
    qp = qp.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    # compute
    if provider == 'cuda':
        o, l = torch.ops.chipmunk.dense_attn(qp, k, v)
        l[..., n:, :] = 0
    else:
        o, l = chipmunk.triton.dense_attn(qp, k, v)
        l[0][..., n:, :] = 0
        l[1][..., n:, :] = 0

    # unpad
    o = o[..., :n, :].contiguous()

    return o, l

def dense_colsum_attn(q, k, v, p):
    """
    Compute variable length attention in ThunderKittens.
    """
    fuse_reduce = True
    wq = 16   # queries per warp
    pad_to = get_kernel_config_attn()['bm']
    provider = GLOBAL_CONFIG['attn']['provider']

    if q.shape[-2] % pad_to == 0:
        if provider == 'cuda':
            o, cs, l = torch.ops.chipmunk.dense_colsum_attn(q, k, v, p)
        else:
            o, cs, l = chipmunk.triton.dense_colsum_attn(q, k, v, p)
        
        if not fuse_reduce:
            cs = rearrange(cs, 'b h (m r) n -> b h m r n', r=pad_to//wq).sum(dim=-2)
        return o, cs, l

    # pad
    n = q.shape[-2]
    padded_n = ((n + pad_to - 1) // pad_to) * pad_to
    qp = torch.empty(q.shape[:-2] + (padded_n, q.shape[-1]), dtype=q.dtype, device=q.device)
    qp[..., :n, :] = q

    # contiguous
    qp = qp.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    p = p.contiguous()
    assert p.shape[-2] == padded_n

    # compute
    if provider == 'cuda':
        o, cs, l = torch.ops.chipmunk.dense_colsum_attn(qp, k, v, p)
        l[..., n:, :] = 0
    else:
        o, cs, l = chipmunk.triton.dense_colsum_attn(qp, k, v, p)
        l[0][..., n:, :] = 0
        l[1][..., n:, :] = 0

    # unpad
    o = o[..., :n, :].contiguous()
    if fuse_reduce:
        kseq = k.shape[-2]
        kgroups = (kseq + pad_to - 1) // pad_to
        cs = cs[..., :kgroups, :kseq]
    else:
        cs = rearrange(cs, 'b h (m r) n -> b h m r n', r=pad_to//wq).sum(dim=-2)[..., :n]
    return o, cs, l

def csp_attn(q, k, v, indices, indices_counts):
    pad_to = get_kernel_config_attn()['bm']
    provider = GLOBAL_CONFIG['attn']['provider']

    if q.shape[-2] % pad_to == 0:
        if provider == 'cuda':
            return torch.ops.chipmunk.csp_128_attn(q, k, v, indices, indices_counts)
        else:
            o, _ = chipmunk.triton.csp_attn(q, k, v, indices, indices_counts)
            return o

    # pad
    n = q.shape[-2]
    padded_n = ((n + pad_to - 1) // pad_to) * pad_to
    qp = torch.zeros(q.shape[:-2] + (padded_n, q.shape[-1]), dtype=q.dtype, device=q.device)
    qp[..., :n, :] = q

    if indices.shape[-1] % pad_to == 0:
        indicesp = indices
    else:
        indicesp = torch.empty((indices.shape[0], indices.shape[1], indices.shape[2], padded_n), device=indices.device, dtype=indices.dtype)
        indicesp[..., :indices.shape[-1]] = indices

    # contiguous
    qp = qp.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    indicesp = indicesp.contiguous()

    if provider == 'cuda':
        o = torch.ops.chipmunk.csp_128_attn(qp, k, v, indicesp, indices_counts)
    else:
        o, _ = chipmunk.triton.csp_attn(qp, k, v, indicesp, indices_counts)
    # unpad
    o = o[..., :n, :].contiguous()
    return o

__all__ = ['csp_attn', 'dense_attn', 'dense_colsum_attn']