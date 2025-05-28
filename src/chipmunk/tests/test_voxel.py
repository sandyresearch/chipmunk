import torch
from einops import rearrange

from chipmunk import (
    get_local_voxel_indices,
    voxel_chunk_no_padding,
    reverse_voxel_chunk_no_padding,
    get_local_indices_with_text,
    masktoinds
)

def test_voxel_chunk_no_padding():
    b = 1
    ah = 3
    t = 4
    h = 6
    w = 9
    d = 1
    
    # x = torch.randn(b, ah, t, h, w, d)
    x = torch.arange(t * h * w).unsqueeze(-1).unsqueeze(0).unsqueeze(0).expand(b, ah, -1, 1)
    x = rearrange(x, 'b ah (t h w) d -> b ah t h w d', t=t, h=h, w=w)

    # print(f'x shape: {x.shape}')
    # print(f't * h * w: {t * h * w}')
    vx = voxel_chunk_no_padding(x, voxel_shape=(4, 4, 4))
    # print(f'vx shape: {vx.shape}')
    torch.set_printoptions(profile='full')
    # print(f't, h, w: {t, h, w}')
    # print(f'x: {x[0, 0, :4, :4, :4, 0]}')
    # print(f'vx: {vx[0, 0, :64, 0]}')

    # Chunk 1
    ref = x[:, :, :4, :4, :4, :].flatten()
    act = vx[:, :, :64, 0].flatten()
    # print(f'Checking chunk 1...')
    assert torch.allclose(ref, act)
    # print(f'Chunk 1 passed!')

    # Chunk 2
    ref = x[:, :, :4, :4, 4:8, :].flatten()
    act = vx[:, :, 64:128, 0].flatten()
    # print(f'Checking chunk 2...')
    assert torch.allclose(ref, act)
    # print(f'Chunk 2 passed!')

    # Chunk 3
    # ref = x[:, :, :4, 4:8, :4, :].flatten()
    # act = vx[:, :, 128:192, 0].flatten()
    # print(f'Checking chunk 3...')
    # assert torch.allclose(ref, act)
    # print(f'Chunk 3 passed!')


    # vxr = rearrange(vx, 'b ah (n c) d -> b ah n c d', n=t * h * w // (4 * 4 * 4), c=4 * 4 * 4)[..., 0]
    # print(f'vx: {vxr[0, 0]}')
    torch.set_printoptions(profile='default')
    # print(f'vx shape: {vx.shape}')
    # assert not torch.allclose(x, vx)
    uvx = reverse_voxel_chunk_no_padding(vx, (b, ah, t, h, w, d), voxel_shape=(4, 4, 4))
    assert torch.allclose(x, uvx)
    print()
    print(f'Voxel chunk no padding test passed!')
    print()

def test_voxel_chunk_no_padding_hunyuan_shape():
    b = 1
    ah = 1
    t = 33
    h = 45
    w = 10
    d = 1
    
    x = torch.arange(t * h * w).unsqueeze(-1).unsqueeze(0).unsqueeze(0).expand(b, ah, -1, 1)
    x = rearrange(x, 'b ah (t h w) d -> b ah t h w d', t=t, h=h, w=w)

    vx = voxel_chunk_no_padding(x, voxel_shape=(4, 4, 4))
    # Chunk 1
    ref = x[:, :, :4, :4, :4, :].flatten()
    act = vx[:, :, :64, 0].flatten()
    print(f'Checking chunk 1...')
    assert torch.allclose(ref, act)
    print(f'Chunk 1 passed!')

    # Chunk 2
    ref = x[:, :, :4, :4, 4:8, :].flatten()
    act = vx[:, :, 64:128, 0].flatten()
    print(f'Checking chunk 2...')
    assert torch.allclose(ref, act)
    print(f'Chunk 2 passed!')

    uvx = reverse_voxel_chunk_no_padding(vx, (b, ah, t, h, w, d), voxel_shape=(4, 4, 4))

    diff_indices = (x != uvx).nonzero()
    if diff_indices.numel() > 0:
        print(f'diff_indices: {diff_indices}')
        print(f'diff_indices shape: {diff_indices.shape}')

    print(f'uvx: {uvx[0, 0, :4, :4, :4, 0]}')
    assert torch.allclose(x, uvx)
    print()
    print(f'Voxel chunk no padding hunyuan shape test passed!')
    print()

def test_voxel_chunk_no_padding_hunyuan_shape_4_6_8():
    b = 1
    ah = 1
    t = 33
    h = 45
    w = 10
    d = 1
    
    x = torch.arange(t * h * w).unsqueeze(-1).unsqueeze(0).unsqueeze(0).expand(b, ah, -1, 1)
    x = rearrange(x, 'b ah (t h w) d -> b ah t h w d', t=t, h=h, w=w)

    vx = voxel_chunk_no_padding(x, voxel_shape=(4, 6, 8))
    # Chunk 1
    ref = x[:, :, :4, :6, :8, :].flatten()
    act = vx[:, :, :192, 0].flatten()
    print(f'Checking chunk 1...')
    assert torch.allclose(ref, act)
    print(f'Chunk 1 passed!')

    # Chunk 2
    ref = x[:, :, :4, 6:12, :8, :].flatten()
    act = vx[:, :, 192:384, 0].flatten()
    print(f'Checking chunk 2...')
    assert torch.allclose(ref, act)
    print(f'Chunk 2 passed!')

    uvx = reverse_voxel_chunk_no_padding(vx, (b, ah, t, h, w, d), voxel_shape=(4, 6, 8))

    diff_indices = (x != uvx).nonzero()
    if diff_indices.numel() > 0:
        print(f'diff_indices: {diff_indices}')
        print(f'diff_indices shape: {diff_indices.shape}')

    print(f'uvx: {uvx[0, 0, :4, :6, :8, 0]}')
    assert torch.allclose(x, uvx)
    print()
    print(f'Voxel chunk no padding hunyuan shape 4 6 8 test passed!')
    print()


def test_voxel_local_indices():
    b = 1
    t = 4
    h = 8
    w = 12
    d = 1

    lt, lh, lw = 2, 2, 2
    vt, vh, vw = 1, 1, 1

    actual = get_local_voxel_indices((t, h, w), (lt, lh, lw))

    ### (0, 0, 0) ####
    indices = []
    bt, bh, bw = 0, 0, 0
    for i in [0, 1, 2]:
        for j in [0, 1, 2]:
            for k in [0, 1, 2]:
                ut = (bt + i) * h * w
                uh = (bh + j) * w
                uw = (bw + k)
                indices.append(ut + uh + uw)
    expected = torch.tensor(indices)
    print(f'(0, 0, 0) expected: {expected}')
    print(f'(0, 0, 0) actual  : {actual[0]}')
    print()
    #################

    ### (0, 0, 1) ####
    indices = []
    bt, bh, bw = 0, 0, 1
    for i in [0, 1, 2]:
        for j in [0, 1, 2]:
            for k in [-1, 0, 1]:
                ut = (bt + i) * h * w
                uh = (bh + j) * w
                uw = (bw + k)
                indices.append(ut + uh + uw)
    expected = torch.tensor(indices)
    print(f'(0, 0, 1) expected: {expected}')
    print(f'(0, 0, 1) actual  : {actual[1]}')
    print()
    #################

    ### (1, 1, 1) ####
    indices = []
    bt, bh, bw = 1, 1, 1
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            for k in [-1, 0, 1]:
                ut = (bt + i) * h * w
                uh = (bh + j) * w
                uw = (bw + k)
                indices.append(ut + uh + uw)
    expected = torch.tensor(indices)
    print(f'(1, 1, 1) expected: {expected}')
    print(f'(1, 1, 1) actual  : {actual[h * w + w + 1]}')
    print()
    #################

    ### (3, 7, 11) ####
    indices = []
    bt, bh, bw = 3, 7, 11
    for i in [-2, -1, 0]:
        for j in [-2, -1, 0]:
            for k in [-2, -1, 0]:
                ut = (bt + i) * h * w
                uh = (bh + j) * w
                uw = (bw + k)
                indices.append(ut + uh + uw)
    expected = torch.tensor(indices)
    print(f'(3, 7, 11) expected: {expected}')
    print(f'(3, 7, 11) actual  : {actual[3 * h * w + 7 * w + 11]}')
    print()
    #################

    # print(f'local_inds: {local_inds}')
    local_mask = torch.zeros((t * h * w, t * h * w), dtype=torch.bool)
    local_mask.scatter_(-1, actual, True)
    # local_mask = rearrange(local_mask.unsqueeze(-1).expand(-1, -1, 2), 'm n c -> m (n c)')
    ar = torch.arange(t * h * w).unsqueeze(0) * local_mask
    local_mask = rearrange(local_mask, 'n (t h w) -> n t h w', t=t, h=h, w=w)
    ar = rearrange(ar, 'n (t h w) -> n t h w', t=t, h=h, w=w)
    torch.set_printoptions(profile='full')
    print(f'ar: {ar[0]}')
    print(f'ar: {ar[1]}')
    print(f'ar: {ar[2]}')
    print(f'ar: {ar[3]}')
    print(f'ar: {ar[2 * w + 2]}')
    torch.set_printoptions(profile='default')

    # x = torch.arange(t * h * w).unsqueeze(0).unsqueeze(-1)
    # print(f'x: {x}')
    # print(f'x shape: {x.shape}')

def test_voxel_local_indices_non_multiple():
    b = 1
    t = 4
    h = 6
    w = 9
    d = 1

    lt, lh, lw = 2, 2, 2
    vt, vh, vw = 1, 1, 1

    actual = get_local_voxel_indices((t, h, w), (lt, lh, lw))

    ### (0, 0, 0) ####
    indices = []
    bt, bh, bw = 0, 0, 0
    for i in [0, 1, 2]:
        for j in [0, 1, 2]:
            for k in [0, 1, 2]:
                ut = (bt + i) * h * w
                uh = (bh + j) * w
                uw = (bw + k)
                indices.append(ut + uh + uw)
    expected = torch.tensor(indices)
    print(f'(0, 0, 0) expected: {expected}')
    print(f'(0, 0, 0) actual  : {actual[0]}')
    print()
    #################

    ### (0, 0, 1) ####
    indices = []
    bt, bh, bw = 0, 0, 1
    for i in [0, 1, 2]:
        for j in [0, 1, 2]:
            for k in [-1, 0, 1]:
                ut = (bt + i) * h * w
                uh = (bh + j) * w
                uw = (bw + k)
                indices.append(ut + uh + uw)
    expected = torch.tensor(indices)
    print(f'(0, 0, 1) expected: {expected}')
    print(f'(0, 0, 1) actual  : {actual[1]}')
    print()
    #################

    ### (1, 1, 1) ####
    indices = []
    bt, bh, bw = 1, 1, 1
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            for k in [-1, 0, 1]:
                ut = (bt + i) * h * w
                uh = (bh + j) * w
                uw = (bw + k)
                indices.append(ut + uh + uw)
    expected = torch.tensor(indices)
    print(f'(1, 1, 1) expected: {expected}')
    print(f'(1, 1, 1) actual  : {actual[h * w + w + 1]}')
    print()
    #################

    ### (3, 5, 8) ####
    indices = []
    bt, bh, bw = 3, 5, 8
    for i in [-2, -1, 0]:
        for j in [-2, -1, 0]:
            for k in [-2, -1, 0]:
                ut = (bt + i) * h * w
                uh = (bh + j) * w
                uw = (bw + k)
                indices.append(ut + uh + uw)
    expected = torch.tensor(indices)
    print(f'(3, 5, 8) expected: {expected}')
    print(f'(3, 5, 8) actual  : {actual[3 * h * w + 5 * w + 8]}')
    print()
    #################

    # print(f'local_inds: {local_inds}')
    local_mask = torch.zeros((t * h * w, t * h * w), dtype=torch.bool)
    local_mask.scatter_(-1, actual, True)
    # local_mask = rearrange(local_mask.unsqueeze(-1).expand(-1, -1, 2), 'm n c -> m (n c)')
    ar = torch.arange(t * h * w).unsqueeze(0) * local_mask
    local_mask = rearrange(local_mask, 'n (t h w) -> n t h w', t=t, h=h, w=w)
    ar = rearrange(ar, 'n (t h w) -> n t h w', t=t, h=h, w=w)
    torch.set_printoptions(profile='full')
    print(f'ar: {ar[0]}')
    print(f'ar: {ar[1]}')
    print(f'ar: {ar[2]}')
    print(f'ar: {ar[3]}')
    print(f'ar: {ar[2 * w + 2]}')
    torch.set_printoptions(profile='default')

    # x = torch.arange(t * h * w).unsqueeze(0).unsqueeze(-1)
    # print(f'x: {x}')
    # print(f'x shape: {x.shape}')

def test_get_local_indices_with_text():
    vid_shape = (33, 45, 80)
    txt_len = 13
    voxel_shape = (4, 6, 8)
    local_shape = (0, 0, 0)
    # local_shape = (3, 3, 3)
    device = torch.device('cpu') if not torch.cuda.is_available() else torch.device('cuda')

    mask, actual_inds, actual_counts = get_local_indices_with_text(
        vid_shape,
        txt_len,
        voxel_shape,
        local_shape,
        kv_tile_size=112,
        full_tail_from_attn=True,
        full_tail_to_attn=True,
        rk=0,
        device=device,
    )
    torch.set_printoptions(profile='full')
    print(f'actual_inds[0]: {actual_inds[0, :actual_counts[0]].sort().values}')
    print(f'actual_inds[-1]: {actual_inds[-1, :actual_counts[0]].sort().values[:200]}')
    print(f'actual_counts: {actual_counts}')
    torch.set_printoptions(profile='default')
    print(f'actual inds shape: {actual_inds.shape}')
    print(f'actual counts shape: {actual_counts.shape}')
    print()
    print(f'Voxel local indices with text test passed!')
    print()
    print(f'size of actual_inds: {actual_inds.numel() * actual_inds.element_size() / 1e9} GB')
    print(f'size of actual_counts: {actual_counts.numel() * actual_counts.element_size() / 1e9} GB')


    # import triton
    # ms = triton.testing.do_bench(lambda: masktoinds(mask, multiple=128), warmup=100, rep=1000)
    # print(f'masktoinds time: {ms}')
    # mask = mask.unsqueeze(0).expand(50, -1, -1)

    # print(f'mask shape: {mask.shape}')
    print(f'mask sparsity: {mask.sum() / mask.numel()}')
    # print(f'size of mask: {mask.numel() * mask.element_size() / 1e9} GB')
    # print()

if __name__ == '__main__':
    # test_voxel_local_indices()
    # test_voxel_local_indices_non_multiple()
    # test_voxel_chunk_no_padding()
    # test_voxel_chunk_no_padding_hunyuan_shape()
    # test_voxel_chunk_no_padding_hunyuan_shape_4_6_8()
    test_get_local_indices_with_text()
