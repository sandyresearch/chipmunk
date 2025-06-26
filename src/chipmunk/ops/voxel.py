import torch
from einops import rearrange

from chipmunk.util.config import GLOBAL_CONFIG

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
    full_tail_from_attn=None,
    full_tail_to_attn=None,
    rk=0,
    kv_tile_size=128,
    device=torch.device('cuda')
):
    if full_tail_from_attn is None:
        full_tail_from_attn = GLOBAL_CONFIG['attn']['should_make_tail_dense']
    if full_tail_to_attn is None:
        full_tail_to_attn = GLOBAL_CONFIG['attn']['should_make_tail_dense']

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