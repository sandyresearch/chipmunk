import torch
import chipmunk
from chipmunk.util import get_kernel_config_attn, GLOBAL_CONFIG

def pad_qkvo_tensor(tensor, pad_to):
    n = tensor.shape[-2]
    padded_n = ((n + pad_to - 1) // pad_to) * pad_to
    # IMPORTANT: Do not use torch.empty here, it will cause NaN in the Triton kernel!
    padded_tensor = torch.zeros(tensor.shape[:-2] + (padded_n, tensor.shape[-1]), dtype=tensor.dtype, device=tensor.device)
    padded_tensor[..., :n, :] = tensor
    return padded_tensor

def dense_attn(q, k, v):
    assert q.shape == k.shape and q.shape == v.shape, "Input shape mismatch - q: {}, k: {}, v: {}".format(q.shape, k.shape, v.shape)
    
    if GLOBAL_CONFIG['attn']['provider'] == 'triton':
        pad_to = get_kernel_config_attn()['bm']
        o, lse = chipmunk.triton.dense_attn(q, pad_qkvo_tensor(k, pad_to), pad_qkvo_tensor(v, pad_to))
        
        assert type(lse) == tuple, "LSE must be a tuple"
        assert lse[0].shape == (q.shape[0], q.shape[1], q.shape[2], 1), "LSE shape mismatch"
        assert lse[1].shape == (q.shape[0], q.shape[1], q.shape[2], 1), "LSE shape mismatch"
    else:
        o, lse = torch.ops.chipmunk.dense_attn(q, k, v)
        assert lse.shape == (q.shape[0], q.shape[1], q.shape[2], 1), "LSE shape mismatch"
    
    assert o.shape == q.shape, "Output shape mismatch"
    
    return o, lse


def dense_colsum_attn(q, k, v, p):
    """
    Compute variable length attention in ThunderKittens.
    """
    assert q.shape == k.shape and q.shape == v.shape, "Input shape mismatch - q: {}, k: {}, v: {}".format(q.shape, k.shape, v.shape)
    
    provider = GLOBAL_CONFIG['attn']['provider']
    pad_to = get_kernel_config_attn()['bm']
    
    if provider == 'cuda':
        # CUDA implementation
        assert p.shape == (q.shape[0], q.shape[1], q.shape[2], 1), "P shape mismatch - p: {}, q: {}".format(p.shape, q.shape)
        o, cs, l = torch.ops.chipmunk.dense_colsum_attn(q, k, v, p)
        assert l.shape == (q.shape[0], q.shape[1], q.shape[2], 1), "L shape mismatch - l: {}, q: {}".format(l[0].shape, q.shape)
        
    else:
        # Triton implementation
        assert type(p) == tuple
        assert p[0].is_contiguous() and p[1].is_contiguous(), "P must be contiguous"
        assert p[0].shape == (q.shape[0], q.shape[1], q.shape[2], 1), "P shape mismatch - p[0]: {}".format(p[0].shape)
        assert p[1].shape == (q.shape[0], q.shape[1], q.shape[2], 1), "P shape mismatch - p[1]: {}".format(p[1].shape)
        
        if q.shape[-2] % pad_to == 0:
            o, cs, l = chipmunk.triton.dense_colsum_attn(q, k, v, p)
        else:
            o, cs, l = chipmunk.triton.dense_colsum_attn(q, pad_qkvo_tensor(k, pad_to), pad_qkvo_tensor(v, pad_to), p)
        
        assert l[0].shape == (q.shape[0], q.shape[1], q.shape[2], 1), "L shape mismatch - l: {}, q: {}".format(l[0].shape, q.shape)
        assert l[1].shape == (q.shape[0], q.shape[1], q.shape[2], 1), "L shape mismatch - l: {}, q: {}".format(l[1].shape, q.shape)
        
    assert o.shape == q.shape, "Output shape mismatch - o: {}, q: {}".format(o.shape, q.shape)
    assert cs.shape == (q.shape[0], q.shape[1], (q.shape[-2] + pad_to - 1) // pad_to, q.shape[2]), "CS shape mismatch - cs: {}, q: {}".format(cs.shape, q.shape)
    
    return o, cs, l

def csp_attn(q, k, v, indices, indices_counts, o, o_scale):
    assert q.shape == k.shape and q.shape == v.shape, "Input shape mismatch - q: {}, k: {}, v: {}".format(q.shape, k.shape, v.shape)
    # Ignore the n_groups dimension in Python - the kernel will also double check for us!
    assert indices.shape == (q.shape[0], q.shape[1], indices.shape[2], q.shape[-2]), "Indices shape mismatch - indices: {}, q: {}".format(indices.shape, q.shape)
    assert indices_counts.shape == indices.shape[:-1], "Indices counts shape mismatch - indices_counts: {}, indices: {}".format(indices_counts.shape, indices.shape)
    assert o.shape == q.shape, "Output shape mismatch - o: {}, q: {}".format(o.shape, q.shape)
    
    if GLOBAL_CONFIG['attn']['provider'] == 'triton':
        pad_to = get_kernel_config_attn()['bm']
        o_delta, _ = chipmunk.triton.csp_attn(q, pad_qkvo_tensor(k, pad_to), pad_qkvo_tensor(v, pad_to), indices, indices_counts)
        assert o_delta.shape == o.shape, "Output delta shape mismatch - o_delta: {}, o: {}".format(o_delta.shape, o.shape)
        o = o + o_delta * o_scale
    else:
        torch.ops.chipmunk.csp_attn(q, k, v, o, indices, indices_counts, o_scale)
    
    return o

__all__ = ['csp_attn', 'dense_attn', 'dense_colsum_attn']