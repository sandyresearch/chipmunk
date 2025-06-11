import torch
from typing import Tuple
from einops import rearrange
import chipmunk
from chipmunk.util import get_kernel_config_attn, GLOBAL_CONFIG
import torch.nn.functional as F

def pad_qkvo_tensor(tensor, pad_to):
    n = tensor.shape[-2]
    padded_n = ((n + pad_to - 1) // pad_to) * pad_to
    padded_tensor = torch.randn(tensor.shape[:-2] + (padded_n, tensor.shape[-1]), dtype=tensor.dtype, device=tensor.device)
    padded_tensor[..., :n, :] = tensor
    return padded_tensor

def dense_attn(q, k, v):
    assert q.shape == k.shape and q.shape == v.shape, "Input shape mismatch - q: {}, k: {}, v: {}".format(q.shape, k.shape, v.shape)
    
    if GLOBAL_CONFIG['attn']['provider'] == 'triton':
        kp = pad_qkvo_tensor(k, get_kernel_config_attn()['bm'])
        vp = pad_qkvo_tensor(v, get_kernel_config_attn()['bm'])
        o, lse = chipmunk.triton.dense_attn(q, kp, vp)
        assert lse[0].shape == (q.shape[0], q.shape[1], q.shape[2], 1), "LSE shape mismatch"
        assert lse[1].shape == (q.shape[0], q.shape[1], q.shape[2], 1), "LSE shape mismatch"
    else:
        o, lse = torch.ops.chipmunk.dense_attn(q, k,v )
        assert lse.shape == (q.shape[0], q.shape[1], q.shape[2], 1), "LSE shape mismatch"
    
    assert o.shape == q.shape, "Output shape mismatch"
    
    return o, lse

def dense_colsum_attn(q, k, v, p):
    """
    Compute variable length attention in ThunderKittens.
    """
    assert q.shape == k.shape and q.shape == v.shape, "Input shape mismatch - q: {}, k: {}, v: {}".format(q.shape, k.shape, v.shape)
    assert p.shape == (q.shape[0], q.shape[1], q.shape[2], 1), "P shape mismatch - p: {}".format(p.shape)
    
    fuse_reduce = True
    wq = 16   # queries per warp
    pad_to = get_kernel_config_attn()['bm']
    provider = GLOBAL_CONFIG['attn']['provider']
    
    if provider == 'cuda':
        return torch.ops.chipmunk.dense_colsum_attn(q, k, v, p)

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
        kseq = k.shape[-2]
        kgroups = (kseq + pad_to - 1) // pad_to
        cs = cs[..., :kgroups, :kseq]
    else:
        cs = rearrange(cs, 'b h (m r) n -> b h m r n', r=pad_to//wq).sum(dim=-2)[..., :n]
    return o, cs, l

def csp_attn(q, k, v, indices, indices_counts, o, o_scale):
    assert q.shape == k.shape and q.shape == v.shape, "Input shape mismatch - q: {}, k: {}, v: {}".format(q.shape, k.shape, v.shape)
    # Ignore the n_groups dimension in Python - the kernel will also double check for us!
    assert indices.shape == (q.shape[0], q.shape[1], indices.shape[2], q.shape[-2]), "Indices shape mismatch - indices: {}, q: {}".format(indices.shape, q.shape)
    assert indices_counts.shape == indices.shape[:-1], "Indices counts shape mismatch - indices_counts: {}, indices: {}".format(indices_counts.shape, indices.shape)
    
    if GLOBAL_CONFIG['attn']['provider'] == 'triton':
        kp = pad_qkvo_tensor(k, get_kernel_config_attn()['bm'])
        vp = pad_qkvo_tensor(v, get_kernel_config_attn()['bm'])
        o_delta, _ = chipmunk.triton.csp_attn(q, kp, vp, indices, indices_counts)
        assert o_delta.shape == o.shape, "Output delta shape mismatch - o_delta: {}, o: {}".format(o_delta.shape, o.shape)
        o = o + o_delta * o_scale
    else:
        torch.ops.chipmunk.csp_attn(q, k, v, o, indices, indices_counts, o_scale)
    
    assert o.shape == q.shape, "Output shape mismatch - o: {}, q: {}".format(o.shape, q.shape)
    return o

__all__ = ['csp_attn', 'dense_attn', 'dense_colsum_attn']