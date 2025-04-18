import math
import torch
from einops import rearrange
from typing import Tuple

# x: [b, ah, t, h, w, d]
#
# return
#   chunked: [b, ah, (num_chunks * 4^3), d]
#   mask: [b, ah, (num_chunks * 4^3)]
def voxel_chunk(x, voxel_shape=(4, 4, 4)):
    # Get dimensions
    b, ah, t, h, w, d = x.shape
    
    vt, vh, vw = voxel_shape
    # Calculate padding needed for each dimension
    pad_t = (vt - (t % vt)) % vt
    pad_h = (vh - (h % vh)) % vh
    pad_w = (vw - (w % vw)) % vw
    
    # Pad the spatial dimensions
    x_padded = torch.nn.functional.pad(x, (0, 0,    # d dimension
                                         0, pad_w,   # w dimension
                                         0, pad_h,   # h dimension
                                         0, pad_t))  # t dimension

    x_flat = rearrange(x_padded, "b ah t h w d -> b ah (t h w) d")
    
    # Flatten and reshape into 4x4x4 chunks
    x_chunks = rearrange(x_flat, "b ah (nt ct nh ch nw cw) d -> b ah (nt nh nw) (ct ch cw) d",
                        nt=((t+pad_t)//vt), ct=vt,
                        nh=((h+pad_h)//vh), ch=vh,
                        nw=((w+pad_w)//vw), cw=vw)
    
    # Flatten chunks to final shape
    x_chunk_flat = rearrange(x_chunks, "b ah nc c d -> b ah (nc c) d")
    
    # Create padding mask
    mask = torch.ones_like(x_padded[:, :, :, :, :, 0], dtype=torch.bool)
    mask[:, :, t:, :, :] = False  # temporal padding
    mask[:, :, :, h:, :] = False  # height padding
    mask[:, :, :, :, w:] = False  # width padding
    
    # Reshape mask to match chunked data
    mask_flat = rearrange(mask, "b ah t h w -> b ah (t h w)")
    mask_chunks = rearrange(mask_flat, "b ah (nt ct nh ch nw cw) -> b ah (nt nh nw) (ct ch cw)",
                           nt=((t+pad_t)//vt), ct=vt,
                           nh=((h+pad_h)//vh), ch=vh,
                           nw=((w+pad_w)//vw), cw=vw)
    
    # Flatten mask chunks
    mask_chunk_flat = rearrange(mask_chunks, "b ah nc c -> b ah (nc c)")

    if x_chunk_flat.size(2) % 64 != 0:
        padding = 64 - x_chunk_flat.size(2) % 64
        x_chunk_flat = torch.cat([x_chunk_flat, torch.zeros((b, ah, padding, d), dtype=x_chunk_flat.dtype, device="cuda")], dim=2)
        mask_chunk_flat = torch.cat([mask_chunk_flat, torch.zeros((b, ah, padding), dtype=mask_chunk_flat.dtype, device="cuda")], dim=2)
    
    return x_chunk_flat.contiguous(), mask_chunk_flat[0, 0].contiguous()

def reverse_voxel_chunk(x_chunk_flat, original_shape, voxel_shape=(4, 4, 4)):
    b, ah, t, h, w, d = original_shape
    
    vt, vh, vw = voxel_shape
    # Calculate padding needed for each dimension
    pad_t = (vt - (t % vt)) % vt
    pad_h = (vh - (h % vh)) % vh
    pad_w = (vw - (w % vw)) % vw
    
    # Reshape flat chunks back into separate chunks
    num_voxels = ((t+pad_t)//vt) * ((h+pad_h)//vh) * ((w+pad_w)//vw)
    # Without final 64 padding
    x_chunks = rearrange(x_chunk_flat[:, :, :num_voxels * (vt * vh * vw)], "b ah (nc c) d -> b ah nc c d",
                        nc=num_voxels)
    
    # Reshape chunks back into padded volume
    x_padded = rearrange(x_chunks, "b ah (nt nh nw) (ct ch cw) d -> b ah (nt ct) (nh ch) (nw cw) d",
                        nt=((t+pad_t)//vt), ct=vt,
                        nh=((h+pad_h)//vh), ch=vh,
                        nw=((w+pad_w)//vw), cw=vw)
    
    # Remove padding
    x = x_padded[:, :, :t, :h, :w, :]
    
    return x.contiguous()

# x: [b, ah, t, h, w, d]
#
# return
#   chunked: [b, ah, (num_chunks * 4^3), d]
#   mask: [b, ah, (num_chunks * 4^3)]
def voxel_chunk_no_padding(x, voxel_shape=(4, 4, 4)):
    # Get dimensions
    b, ah, t, h, w, d = x.shape
    vt, vh, vw = voxel_shape

    # 1) Determine the largest multiple in each dimension that we can chunk fully.
    T_full = (t // vt) * vt
    H_full = (h // vh) * vh
    W_full = (w // vw) * vw

    # 2) Extract and chunk the main region.
    x_main = x[:, :, :T_full, :H_full, :W_full, :]
    x_main = rearrange(
        x_main,
        "b ah (nt vt) (nh vh) (nw vw) d -> b ah (nt nh nw) (vt vh vw) d",
        vt=vt, vh=vh, vw=vw
    )
    x_main = rearrange(x_main, "b ah nc c d -> b ah (nc c) d")
    # print(f'x_main shape: {x_main.shape}')

    # 3) Extract and chunk the tail region.
    xt_tail = x[:, :, T_full:, :, :, :]
    xh_tail = x[:, :, :T_full, H_full:, :, :]
    xw_tail = x[:, :, :T_full, :H_full, W_full:, :]
    # tail_size = (t - T_full) * h * w + T_full * (h - H_full) * w + T_full * H_full * (w - W_full)
    # print(f'tail_size: {tail_size}')
    # tails = []
    # if T_full < t:
    #     tails.append(rearrange(xt_tail, "b ah tt th tw d -> b ah (tt th tw) d"))
    # if H_full < h:
    #     tails.append(rearrange(xh_tail, "b ah tt th tw d -> b ah (tt th tw) d"))
    # if W_full < w:
    #     tails.append(rearrange(xw_tail, "b ah tt th tw d -> b ah (tt th tw) d"))
    # x_tail = torch.cat(tails, dim=2)
    x_tail = torch.cat([
        rearrange(xt_tail, "b ah tt th tw d -> b ah (tt th tw) d"),
        rearrange(xh_tail, "b ah tt th tw d -> b ah (tt th tw) d"),
        rearrange(xw_tail, "b ah tt th tw d -> b ah (tt th tw) d"),
    ], dim=2)
    # print(f'x_tail shape: {x_tail.shape}')

    # 4) Concat
    x_flat = torch.cat([x_main, x_tail], dim=2).contiguous()
    assert x_flat.shape[2] == t * h * w
    
    return x_flat

def reverse_voxel_chunk_no_padding(x_chunk_flat, original_shape, voxel_shape=(4, 4, 4)):
    b, ah, t, h, w, d = original_shape
    vt, vh, vw = voxel_shape

    # 1) Determine the largest multiple in each dimension that we can chunk fully.
    T_full = (t // vt) * vt
    H_full = (h // vh) * vh
    W_full = (w // vw) * vw

    # 2) Extract and reverse chunk the main region.
    x_main = x_chunk_flat[:, :, :T_full * H_full * W_full]
    x_main = rearrange(x_main, "b ah (nt nh nw ct ch cw) d -> b ah (nt ct) (nh ch) (nw cw) d",
                       nt=T_full // vt, ct=vt,
                       nh=H_full // vh, ch=vh,
                       nw=W_full // vw, cw=vw)
    
    # 3) Extract and reverse chunk the tail region.
    x_tail = x_chunk_flat[:, :, T_full * H_full * W_full:]
    # x_tail = rearrange(x_tail, "b ah (tt th tw) d -> b ah tt th tw d",
    #                    tt=t-T_full, th=h-H_full, tw=w-W_full)
    
    # 4) Concat along dims (2, 3, 4)
    x_out = torch.zeros(original_shape, dtype=x_chunk_flat.dtype, device=x_chunk_flat.device)
    x_out[:, :, :T_full, :H_full, :W_full, :] = x_main
    tail_offs = 0
    if T_full < t:
        offs = (t-T_full) * h * w
        tail = x_tail[:, :, tail_offs:tail_offs+offs, :]
        x_out[:, :, T_full:, :, :, :] = rearrange(tail, "b ah (tt th tw) d -> b ah tt th tw d",
                                                   tt=t-T_full, th=h, tw=w)
        tail_offs += offs
    if H_full < h:
        offs = T_full * (h-H_full) * w
        tail = x_tail[:, :, tail_offs:tail_offs+offs, :]
        x_out[:, :, :T_full, H_full:, :, :] = rearrange(tail, "b ah (tt th tw) d -> b ah tt th tw d",
                                                   tt=T_full, th=h-H_full, tw=w)
        tail_offs += offs
    if W_full < w:
        offs = T_full * H_full * (w-W_full)
        tail = x_tail[:, :, tail_offs:tail_offs+offs, :]
        x_out[:, :, :T_full, :H_full, W_full:, :] = rearrange(tail, "b ah (tt th tw) d -> b ah tt th tw d",
                                                   tt=T_full, th=H_full, tw=w-W_full)
    
    return x_out

def offsets(base_coord, full_size, offset_range):
    toffsl = [-i for i in range(1, offset_range + 1) if base_coord - i >= 0]
    toffsr = [i for i in range(1, offset_range + 1) if base_coord + i < full_size]

    if len(toffsl) < offset_range:
        for _ in range(offset_range - len(toffsl)):
            toffsr.append(toffsr[-1] + 1)
    elif len(toffsr) < offset_range:
        for _ in range(offset_range - len(toffsr)):
            toffsl.append(toffsl[-1] - 1)

    toffsl.append(0)
    return sorted(toffsl + toffsr)

def get_local_voxel_indices(full_shape, local_shape):
    """
    Args:
        full_shape : [ t,  h, w ]
        local_shape: [lt, lh, lw]

    Returns:
        inds: (t * h * w, (lt + 1) * (lh + 1) * (lw + 1))

    For each voxel in the full shape, return the indices of the larger local voxel that contains it.
    """

    t, h, w = full_shape
    lt, lh, lw = local_shape

    inds = torch.zeros((t * h * w, (lt + 1) * (lh + 1) * (lw + 1)), dtype=torch.int64)
    if lt == 0 or lh == 0 or lw == 0:
        return inds

    # BASE COORDS
    for bt in range(t): 
        toffs = offsets(bt, t, lt // 2)
        for bh in range(h):
            hoffs = offsets(bh, h, lh // 2)
            for bw in range(w):
                woffs = offsets(bw, w, lw // 2)

                bc = bt * h * w + bh * w + bw
                # LOCAL COORDS
                # print(f'coord: {(bt, bh, bw)}')
                # print(f'toffs: {toffs}')
                # print(f'hoffs: {hoffs}')
                # print(f'woffs: {woffs}')
                for ic, i in enumerate(toffs):
                    for jc, j in enumerate(hoffs):
                        for kc, k in enumerate(woffs):
                            lc = ic * (lh + 1) * (lw + 1) + jc * (lw + 1) + kc
                            ut = (bt + i) * h * w
                            uh = (bh + j) * w
                            uw = (bw + k)
                            # print(f'bc: {bc}, lc: {lc}, ut: {ut}, uh: {uh}, uw: {uw}')
                            inds[bc, lc] = ut + uh + uw

    return inds

# @torch.compile
def masktoinds(mask, multiple=None):
    """
    Compute the per-row nonzero indices and counts of the mask.

    Args:
        mask     : [..., m, n]
        multiple : int

    Returns:
        inds     : [..., m, n]
        counts   : [..., m]
    """

    if multiple is not None:
        counts = ((mask.sum(dim=-1).to(torch.int32) + multiple - 1) // multiple) * multiple
    else:
        counts = mask.sum(dim=-1).to(torch.int32)
    inds = mask.char().argsort(dim=-1, descending=True)
    # return None, counts.contiguous().to(torch.int32)
    return inds.contiguous().to(torch.int32), counts.contiguous().to(torch.int32)

def merge_indices(a, b, full_shape):
    """
    Merge two sets of indices, handling overlaps.

    Args:
        a          : [..., m, r]
        b          : [..., m, s]
        full_shape : [..., m, n]

    Returns:
        inds       : [..., m, n]
        counts     : [..., m]
    """
    # everything except the last dim is the same
    assert a.shape[:-1] == b.shape[:-1]
    assert a.shape[:-1] == full_shape.shape[:-1]

    mask = torch.zeros(full_shape, device=a.device, dtype=torch.bool)
    mask.scatter_(dim=-1, index=a, value=True)
    mask.scatter_(dim=-1, index=b, value=True)

    inds, counts = masktoinds(mask)
    return inds, counts

def get_local_indices_with_text(
    vid_shape,
    txt_len,
    voxel_shape,
    local_shape,
    full_tail_from_attn=False,
    full_tail_to_attn=False,
    rk=0,
    kv_tile_size=128,
    device=torch.device('cuda')
):
    cdiv = lambda x, y: ((x + y - 1) // y)

    # square away our shapes
    tt, th, tw = vid_shape
    lt, lh, lw = local_shape
    vt, vh, vw = voxel_shape
    vid_seqlen = tt * th * tw
    vid_txt_seqlen = vid_seqlen + txt_len
    voxel_size = vt * vh * vw
    n_voxels = cdiv(vid_txt_seqlen, voxel_size)
    # txt_groups = (txt_len // voxel_size) + 1

    mask = torch.zeros((n_voxels, vid_txt_seqlen), device=device, dtype=torch.bool)
    # Text attends to everything (rounded down to the nearest multiple of voxel_size)
    # mask[-1 * (txt_len // voxel_size + 1):, -1 * ((vid_txt_seqlen // kv_tile_size) * kv_tile_size):] = True
    # All queries attend to text.
    mask[:, vid_seqlen:] = True

    # [lt, lh, lw] cube of (lt * lh * lw) voxels with the query voxel in the center.
    # vtt, vth, vtw = cdiv(tt, vt), cdiv(th, vh), cdiv(tw, vw)
    vtt, vth, vtw = tt // vt, th // vh, tw // vw
    n_img_voxels = vtt * vth * vtw
    # print(f'getting local indices for {vtt, vth, vtw} with {lt, lh, lw}')
    local_indices = get_local_voxel_indices((vtt, vth, vtw), (lt, lh, lw)).to(device)
    # print(f'got local indices')

    # prepare for merge with full mask
    local_mask = torch.zeros((n_img_voxels, n_img_voxels), device=device, dtype=torch.bool)
    # print(f'scattering local indices')
    local_mask.scatter_(-1, local_indices, True)
    # print(f'scattered local indices')
    # print(f'local mask shape before expand: {local_mask.shape}')
    # r_local_mask = rearrange(local_mask, "(t h w) (tt th tw) -> t h w tt th tw", t=vtt, h=vth, w=vtw, tt=vtt, th=vth, tw=vtw)
    # print(f'local_mask[0, 0, 0]: {r_local_mask[0, 0, 0]}')
    # print(f'local_mask[0, 0, 1]: {r_local_mask[0, 0, 1]}')
    # print(f'local_mask[0, 0, 2]: {r_local_mask[0, 0, 2]}')
    # print(f'local_mask[0, 0, 3]: {r_local_mask[0, 0, 3]}')
    # print(f'local_mask[1, 2, 3]: {r_local_mask[1, 2, 3]}')
    # print(f'local_mask[2, 0, 3]: {r_local_mask[2, 0, 3]}')
    # print(f'local_mask[6, 4, 5]: {r_local_mask[6, 4, 5]}')
    local_mask = rearrange(
        local_mask[:, :, None].expand(-1, -1, voxel_size),
        'm n r -> m (n r)'
    )[:mask.shape[0], :mask.shape[-1]]
    # print(f'reshaped local mask')
    # print(f'local mask shape after expand: {local_mask.shape}')
    pad0 = mask.shape[0] - local_indices.shape[0]
    if pad0 > 0:
        local_mask = torch.cat([
            local_mask,
            torch.zeros((pad0, local_mask.shape[1]), device=device, dtype=torch.bool)],
            dim=0
        )
    pad1 = mask.shape[1] - local_mask.shape[1]
    if pad1 > 0:
        if full_tail_to_attn:
            local_mask = torch.cat([
                local_mask,
                # attend to all raster order tokens in the tail of the 3d vid dimensions
                torch.ones((local_mask.shape[0], pad1), device=device, dtype=torch.bool)],
                dim=1
            )
        else:
            local_mask = torch.cat([
                local_mask,
                torch.zeros((local_mask.shape[0], pad1), device=device, dtype=torch.bool)],
                dim=1
            )
    # local window for tail of 3d vid dimensions unaccounted for in local voxel window
    local_size = voxel_size * lt * lh * lw
    if local_size > 0:
        local_mask[local_mask.shape[0] - pad0:, -local_size:] = True
    # print(f'padded local mask')
    mask = mask | local_mask
    mask[-1 * (txt_len // voxel_size + 1):, -1 * ((vid_txt_seqlen // kv_tile_size) * kv_tile_size):] = True
    if full_tail_from_attn and pad0 > 0:
        # print(f'pad0: {pad0}')
        mask[-1 * pad0:, -1 * ((vid_txt_seqlen // kv_tile_size) * kv_tile_size):] = True
    if rk > 0:
        rand = torch.rand(mask.shape, device=device) < rk
        if full_tail_from_attn and pad0 > 0:
            rand[-1 * pad0:, :] = False
        rand[-1 * (txt_len // voxel_size + 1):, :] = False
        mask = mask | rand

    # print(f'merged mask')
    inds, counts = masktoinds(mask, multiple=kv_tile_size)
    return mask, inds, counts

@torch.compile(dynamic=False)
def bitpack(mask: torch.Tensor):
    r"""
    Compresses a boolean tensor into a bit-packed uint8 tensor in parallel on the GPU.
    Each output byte encodes 8 bits (True or False) from the input tensor, in little-endian order.

    Args:
        mask (torch.Tensor): A boolean tensor to compress. Must be on the GPU.

    Returns:
        (torch.Tensor, Tuple[int, ...]):
            A tuple of:
            - A 1-D torch.uint8 tensor of length ceil(numel(mask) / 8)
              storing the packed bits on the GPU.
            - The original shape of the mask tensor (for later unpacking).
    """
    original_shape = mask.shape
    # Flatten the tensor
    flat_mask = mask.flatten()
    n = flat_mask.numel()

    # Number of bits we need to pad so that we can reshape into 8 columns
    pad_size = (-n) % 8  # same as: (8 - (n % 8)) % 8

    # Zero-pad if necessary
    # if pad_size > 0:
    flat_mask = torch.cat([flat_mask, flat_mask.new_zeros(pad_size)])

    # Reshape to [N/8, 8], cast to uint8
    flat_mask = flat_mask.view(-1, 8).to(torch.uint8)

    # For each column j, we multiply by 2^j and sum across columns
    # shifts = [1, 2, 4, 8, 16, 32, 64, 128]
    shifts = (2 ** torch.arange(8, dtype=torch.uint8, device=flat_mask.device)).view(1, -1)
    packed = (flat_mask * shifts).sum(dim=1, dtype=torch.uint8).contiguous()  # [N/8]

    return packed, original_shape


@torch.compile(dynamic=False)
def bitunpack(packed: torch.Tensor, original_shape: Tuple[int, ...]):
    r"""
    Decompresses a bit-packed tensor (uint8) back to a boolean tensor in parallel on the GPU.

    Args:
        packed (torch.Tensor): A 1-D bit-packed tensor of type torch.uint8 on the GPU.
        original_shape (Tuple[int, ...]): The original shape of the boolean tensor.

    Returns:
        torch.Tensor: A boolean tensor of shape original_shape.
    """
    # Compute total number of bits needed
    total_bits = 1
    for dim in original_shape:
        total_bits *= dim

    # Expand the packed bytes to 8 bits each
    # shifts = [1, 2, 4, 8, 16, 32, 64, 128]
    shifts = (2 ** torch.arange(8, dtype=torch.uint8, device=packed.device)).view(1, -1)
    
    # (packed.unsqueeze(1) >> shift) & 1 gives bits; shape => [N_bytes, 8]
    bits_2d = ((packed.unsqueeze(1) & shifts) > 0).to(torch.bool)

    # Flatten and truncate if there was padding
    bits = bits_2d.view(-1)[:total_bits]

    # Reshape to the original shape
    return bits.view(*original_shape)

