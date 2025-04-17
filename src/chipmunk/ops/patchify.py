from einops import rearrange
from chipmunk.util.config import GLOBAL_CONFIG

is_patchify_enabled = GLOBAL_CONFIG['patchify']['is_enabled']
chunk_size_1 = GLOBAL_CONFIG['patchify']['chunk_size_1']
chunk_size_2 = GLOBAL_CONFIG['patchify']['chunk_size_2']

def patchify(x):
    assert x.ndim == 3, "Input tensor must have 3 dimensions (b, h, w)."
    b, h, w = x.shape
    assert h % chunk_size_1 == 0, "Height must be divisible by chunk_size."
    assert w % chunk_size_1 == 0, "Width must be divisible by chunk_size."
    assert h % chunk_size_2 == 0, "Height must be divisible by chunk_size_2."
    assert w % chunk_size_2 == 0, "Width must be divisible by chunk_size_2."
    assert chunk_size_1 % chunk_size_2 == 0, "chunk_size_1 must be divisible by chunk_size_2."
    
    # flatten
    x_chunks = rearrange(x, "b h w -> b (h w)")
    # patch 1
    x_chunks = rearrange(
        x_chunks,
        "b (nh ch nw cw) -> b (nh nw) (ch cw)",
        nh=h // chunk_size_1, ch=chunk_size_1,
        nw=w // chunk_size_1, cw=chunk_size_1
    )
    # patch 2
    x_chunks = rearrange(
        x_chunks,
        "b n (nh ch nw cw) -> b n (nh nw) (ch cw)",
        nh=chunk_size_1 // chunk_size_2, ch=chunk_size_2,
        nw=chunk_size_1 // chunk_size_2, cw=chunk_size_2
    )
    # Flatten patches into final shape: [b, num_patches * (chunk_size^2)]
    x_chunk_flat = rearrange(x_chunks, "b n nc c -> b (n nc c)")
    
    return x_chunk_flat

def unpatchify(x_chunk_flat, original_shape):
    b, h, w = original_shape
    # Number of patches
    num_patches_1 = (h // chunk_size_1) * (w // chunk_size_1)
    num_patches_2 = (chunk_size_1 // chunk_size_2) ** 2
    # Reshape flat patches back to the patch grid
    x_chunks = rearrange(x_chunk_flat, "b (n nc c) -> b n nc c", n=num_patches_1, nc=num_patches_2)
    
    # unpatch 2
    x_unpatched = rearrange(
        x_chunks,
        "b n (nh nw) (ch cw) -> b n (nh ch) (nw cw)",
        nh=chunk_size_1 // chunk_size_2, ch=chunk_size_2,
        nw=chunk_size_1 // chunk_size_2, cw=chunk_size_2
    )
    x_unpatched = rearrange(x_unpatched, "b n nc c -> b n (nc c)")
        
    # unpatch 1
    x_unpatched = rearrange(
        x_unpatched,
        "b (nh nw) (ch cw) -> b (nh ch) (nw cw)",
        nh=h // chunk_size_1, ch=chunk_size_1,
        nw=w // chunk_size_1, cw=chunk_size_1
    )
    
    return x_unpatched
