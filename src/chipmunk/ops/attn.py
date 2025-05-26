import torch
from typing import Tuple
from einops import rearrange
import chipmunk
from chipmunk.util import get_kernel_config_attn, GLOBAL_CONFIG
import torch.nn.functional as F

def pad_qkvo_tensor(tensor, pad_to):
    n = tensor.shape[-2]
    padded_n = ((n + pad_to - 1) // pad_to) * pad_to
    padded_tensor = torch.zeros(tensor.shape[:-2] + (padded_n, tensor.shape[-1]), dtype=tensor.dtype, device=tensor.device)
    padded_tensor[..., :n, :] = tensor
    return padded_tensor

def dense_attn(q, k, v):
    # if GLOBAL_CONFIG['attn']['provider'] == 'triton':
    #     return chipmunk.triton.dense_attn(q, k, v)

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
    should_pad_kv = GLOBAL_CONFIG['attn']['provider'] == 'triton'
    if should_pad_kv:
        qp = q
        kp = pad_qkvo_tensor(k, pad_to)
        vp = pad_qkvo_tensor(v, pad_to)
    else:
        qp = pad_qkvo_tensor(q, pad_to)
        kp = k
        vp = v

    # contiguous
    qp = qp.contiguous()
    kp = kp.contiguous()
    vp = vp.contiguous()

    # compute
    if provider == 'cuda':
        op, l = torch.ops.chipmunk.dense_attn(qp, kp, vp)
        l[..., n:, :] = 0
    else:
        op, l = chipmunk.triton.dense_attn(qp, kp, vp)
        l[0][..., n:, :] = 0
        l[1][..., n:, :] = 0

    # unpad
    o = op[..., :n, :].contiguous()

    return o, l

def dense_colsum_attn(q, k, v, p):
    """
    Compute variable length attention in ThunderKittens.
    """
    # if GLOBAL_CONFIG['attn']['provider'] == 'triton':
    #     return chipmunk.triton.dense_colsum_attn(q, k, v, p)

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
    should_pad_kv = GLOBAL_CONFIG['attn']['provider'] == 'triton'
    if should_pad_kv:
        qp = q
        kp = pad_qkvo_tensor(k, pad_to)
        vp = pad_qkvo_tensor(v, pad_to)
    else:
        qp = pad_qkvo_tensor(q, pad_to)
        kp = k
        vp = v

    # contiguous
    qp = qp.contiguous()
    kp = kp.contiguous()
    vp = vp.contiguous()

    # compute
    if provider == 'cuda':
        assert type(p) == torch.Tensor
        p = p.contiguous()
        assert p.shape[-2] == padded_n
        o, cs, l = torch.ops.chipmunk.dense_colsum_attn(qp, kp, vp, p)
        l[..., n:, :] = 0
    else:
        assert type(p) == tuple
        p = (p[0].contiguous(), p[1].contiguous())
        # assert p[0].shape[-2] == padded_n
        # assert p[1].shape[-2] == padded_n
        o, cs, l = chipmunk.triton.dense_colsum_attn(qp, kp, vp, p)
        l[0][..., n:, :] = 0
        l[1][..., n:, :] = 0

    # unpad
    o = o[..., :n, :].contiguous()
    if fuse_reduce:
        if provider == 'cuda' or True:
            kseq = k.shape[-2]
            kgroups = (kseq + pad_to - 1) // pad_to
            cs = cs[..., :kgroups, :kseq]
        else:
            fuse_amt = get_kernel_config_attn()['bm'] // 64
            # cs = rearrange(cs, 'b h (m r) n -> b h m r n', r=fuse_amt).sum(dim=-2)
    else:
        cs = rearrange(cs, 'b h (m r) n -> b h m r n', r=pad_to//wq).sum(dim=-2)[..., :n]
    return o, cs, l

def csp_attn(q, k, v, indices, indices_counts):
    if GLOBAL_CONFIG['attn']['provider'] == 'triton':
        o, _ = chipmunk.triton.csp_attn(q, k, v, indices, indices_counts)
        return o

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
    should_pad_kv = GLOBAL_CONFIG['attn']['provider'] == 'triton'
    if should_pad_kv:
        qp = q
        kp = k
        vp = v
    else:
        qp = pad_qkvo_tensor(q, pad_to)
        kp = k
        vp = v

    if indices.shape[-1] % pad_to == 0:
        indicesp = indices
    else:
        indicesp = torch.empty((indices.shape[0], indices.shape[1], indices.shape[2], padded_n), device=indices.device, dtype=indices.dtype)
        indicesp[..., :indices.shape[-1]] = indices

    # contiguous
    qp = qp.contiguous()
    kp = kp.contiguous()
    vp = vp.contiguous()
    indicesp = indicesp.contiguous()

    if provider == 'cuda':
        op = torch.ops.chipmunk.csp_128_attn(qp, kp, vp, indicesp, indices_counts)
    else:
        op, _ = chipmunk.triton.csp_attn(qp, kp, vp, indicesp, indices_counts)
    # unpad
    o = op[..., :n, :].contiguous()
    return o

__all__ = ['csp_attn', 'dense_attn', 'dense_colsum_attn']