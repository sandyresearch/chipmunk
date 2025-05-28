import importlib.metadata
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from einops import rearrange

try:
    import flash_attn
    from flash_attn.flash_attn_interface import _flash_attn_forward
    from flash_attn.flash_attn_interface import flash_attn_varlen_func
except ImportError:
    flash_attn = None
    flash_attn_varlen_func = None
    _flash_attn_forward = None

from hyvideo.modules.head_parallel import all_to_all_collect_tokens, all_to_all_collect_heads, all_gather


MEMORY_LAYOUT = {
    "flash": (
        lambda x: x.view(x.shape[0] * x.shape[1], *x.shape[2:]),
        lambda x: x,
    ),
    "torch": (
        lambda x: x.transpose(1, 2),
        lambda x: x.transpose(1, 2),
    ),
    "vanilla": (
        lambda x: x.transpose(1, 2),
        lambda x: x.transpose(1, 2),
    ),
}


def get_cu_seqlens(text_mask, img_len):
    """Calculate cu_seqlens_q, cu_seqlens_kv using text_mask and img_len

    Args:
        text_mask (torch.Tensor): the mask of text
        img_len (int): the length of image

    Returns:
        torch.Tensor: the calculated cu_seqlens for flash attention
    """
    batch_size = text_mask.shape[0]
    text_len = text_mask.sum(dim=1)
    max_len = text_mask.shape[1] + img_len

    cu_seqlens = torch.zeros([2 * batch_size + 1], dtype=torch.int32, device="cuda")

    for i in range(batch_size):
        s = text_len[i] + img_len
        s1 = i * max_len + s
        s2 = (i + 1) * max_len
        cu_seqlens[2 * i + 1] = s1
        cu_seqlens[2 * i + 2] = s2

    return cu_seqlens


def attention(
    q,
    k,
    v,
    mode="torch",
    drop_rate=0,
    attn_mask=None,
    causal=False,
    cu_seqlens_q=None,
    cu_seqlens_kv=None,
    max_seqlen_q=None,
    max_seqlen_kv=None,
    batch_size=1,
    attn=None,
    inference_step=None,
):
    """
    Perform QKV self attention.

    Args:
        q (torch.Tensor): Query tensor with shape [b, s, a, d], where a is the number of heads.
        k (torch.Tensor): Key tensor with shape [b, s1, a, d]
        v (torch.Tensor): Value tensor with shape [b, s1, a, d]
        mode (str): Attention mode. Choose from 'self_flash', 'cross_flash', 'torch', and 'vanilla'.
        drop_rate (float): Dropout rate in attention map. (default: 0)
        attn_mask (torch.Tensor): Attention mask with shape [b, s1] (cross_attn), or [b, a, s, s1] (torch or vanilla).
            (default: None)
        causal (bool): Whether to use causal attention. (default: False)
        cu_seqlens_q (torch.Tensor): dtype torch.int32. The cumulative sequence lengths of the sequences in the batch,
            used to index into q.
        cu_seqlens_kv (torch.Tensor): dtype torch.int32. The cumulative sequence lengths of the sequences in the batch,
            used to index into kv.
        max_seqlen_q (int): The maximum sequence length in the batch of q.
        max_seqlen_kv (int): The maximum sequence length in the batch of k and v.

    Returns:
        torch.Tensor: Output tensor after self attention with shape [b, s, ad]
    """
    pre_attn_layout, post_attn_layout = MEMORY_LAYOUT[mode]
    q = pre_attn_layout(q)
    k = pre_attn_layout(k)
    v = pre_attn_layout(v)

    if mode == "torch":
        if attn_mask is not None and attn_mask.dtype != torch.bool:
            attn_mask = attn_mask.to(q.dtype)

        # need to not attend to post text
        if cu_seqlens_kv is not None:
            qit = q[:, :, :cu_seqlens_q[1], :]
            kit = k[:, :, :cu_seqlens_kv[1], :]
            vit = v[:, :, :cu_seqlens_kv[1], :]
            x = attn(qit, kit, vit)
            # tail is ignored so we can cat anything
            x = torch.cat([x, q[:, :, cu_seqlens_q[1]:, :]], dim=2)
        else:
            x = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, dropout_p=drop_rate, is_causal=causal
            )
    elif mode == "flash":
        x = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_kv,
            max_seqlen_q,
            max_seqlen_kv,
        )
        # x with shape [(bxs), a, d]
        x = x.view(
            batch_size, max_seqlen_q, x.shape[-2], x.shape[-1]
        )  # reshape x to [b, s, a, d]
    elif mode == "vanilla":
        scale_factor = 1 / math.sqrt(q.size(-1))

        b, a, s, _ = q.shape
        s1 = k.size(2)
        attn_bias = torch.zeros(b, a, s, s1, dtype=q.dtype, device=q.device)
        if causal:
            # Only applied to self attention
            assert (
                attn_mask is None
            ), "Causal mask and attn_mask cannot be used together"
            temp_mask = torch.ones(b, a, s, s, dtype=torch.bool, device=q.device).tril(
                diagonal=0
            )
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
            attn_bias.to(q.dtype)

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
            else:
                attn_bias += attn_mask

        # TODO: Maybe force q and k to be float32 to avoid numerical overflow
        attn = (q @ k.transpose(-2, -1)) * scale_factor
        attn += attn_bias
        attn = attn.softmax(dim=-1)
        attn = torch.dropout(attn, p=drop_rate, train=True)
        x = attn @ v
    else:
        raise NotImplementedError(f"Unsupported attention mode: {mode}")

    x = post_attn_layout(x)
    b, s, a, d = x.shape
    out = x.reshape(b, s, -1)
    return out


def parallel_attention(
    hybrid_seq_parallel_attn,
    q,
    k,
    v,
    img_q_len,
    img_kv_len,
    cu_seqlens_q,
    cu_seqlens_kv
):
    attn1 = hybrid_seq_parallel_attn(
        None,
        q[:, :img_q_len, :, :],
        k[:, :img_kv_len, :, :],
        v[:, :img_kv_len, :, :],
        dropout_p=0.0,
        causal=False,
        joint_tensor_query=q[:,img_q_len:cu_seqlens_q[1]],
        joint_tensor_key=k[:,img_kv_len:cu_seqlens_kv[1]],
        joint_tensor_value=v[:,img_kv_len:cu_seqlens_kv[1]],
        joint_strategy="rear",
    )
    if flash_attn.__version__ >= '2.7.0':
        attn2, *_ = _flash_attn_forward(
            q[:,cu_seqlens_q[1]:],
            k[:,cu_seqlens_kv[1]:],
            v[:,cu_seqlens_kv[1]:],
            dropout_p=0.0,
            softmax_scale=q.shape[-1] ** (-0.5),
            causal=False,
            window_size_left=-1,
            window_size_right=-1,
            softcap=0.0,
            alibi_slopes=None,
            return_softmax=False,
        )
    else:
        attn2, *_ = _flash_attn_forward(
            q[:,cu_seqlens_q[1]:],
            k[:,cu_seqlens_kv[1]:],
            v[:,cu_seqlens_kv[1]:],
            dropout_p=0.0,
            softmax_scale=q.shape[-1] ** (-0.5),
            causal=False,
            window_size=(-1, -1),
            softcap=0.0,
            alibi_slopes=None,
            return_softmax=False,
        )
    attn = torch.cat([attn1, attn2], dim=1)
    b, s, a, d = attn.shape
    attn = attn.reshape(b, s, -1)

    return attn

@torch.compiler.disable
def head_parallel_attention(
    attn,
    q,
    k,
    v,
    img_q_len,
    img_kv_len,
    cu_seqlens_q,
    cu_seqlens_kv,
    inference_step
):
    """
    q: [b, s, a, d]
    k: [b, s, a, d]
    v: [b, s, a, d]

    -> [b, s, ad]
    """
    rank, world_size = dist.get_rank(), dist.get_world_size()

    num_local_heads = q.shape[2] // world_size
    local_head_slice = slice(rank * num_local_heads, (rank + 1) * num_local_heads)

    # [b, s // world_size, a, d] -> [b, a // world_size, s, d]
    qi = q[:, :img_q_len]
    ki = k[:, :img_kv_len]
    vi = v[:, :img_kv_len]
    qt = q[:, img_q_len:cu_seqlens_q[1], local_head_slice]
    kt = k[:, img_kv_len:cu_seqlens_kv[1], local_head_slice]
    vt = v[:, img_kv_len:cu_seqlens_kv[1], local_head_slice]
    qt = rearrange(qt, 'b s la d -> b la s d')
    kt = rearrange(kt, 'b s la d -> b la s d')
    vt = rearrange(vt, 'b s la d -> b la s d')
    qe = q[:, cu_seqlens_q[1]:]
    ke = k[:, cu_seqlens_kv[1]:]
    ve = v[:, cu_seqlens_kv[1]:]
    qe = rearrange(qe, 'b s la d -> b la s d')
    ke = rearrange(ke, 'b s la d -> b la s d')
    ve = rearrange(ve, 'b s la d -> b la s d')

    qi, ki, vi = all_to_all_collect_tokens(torch.stack([qi, ki, vi]))

    qit = torch.cat([qi, qt], dim=2)
    kit = torch.cat([ki, kt], dim=2)
    vit = torch.cat([vi, vt], dim=2)

    oit = attn(qit, kit, vit, inference_step)

    oi = oit[:, :, :img_q_len * world_size]
    ot = oit[:, :, img_q_len * world_size:]

    # [b, a // world_size, s, d] -> [b, s // world_size, a * d]
    oi = all_to_all_collect_heads(oi)

    ot = all_gather(ot)
    ot = rearrange(ot, '(G b) la s d -> b s (G la d)', G=world_size)

    oe = F.scaled_dot_product_attention(qe, ke, ve)
    oe = rearrange(oe, 'b la s d -> b s (la d)')

    o = torch.cat([oi, ot, oe], dim=1).contiguous()

    return o
