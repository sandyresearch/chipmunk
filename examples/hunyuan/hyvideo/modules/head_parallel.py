import torch
import torch.distributed as dist
from einops import rearrange

DIST_GROUP = None
DIST_RANK = None
DIST_WORLD_SIZE = None


def setup_dist(dist_group, dist_rank, dist_world_size):
    global DIST_GROUP, DIST_RANK, DIST_WORLD_SIZE
    print(f"Setting up dist with group: {dist_group}, rank: {dist_rank}, world size: {dist_world_size}")
    DIST_GROUP = dist_group
    DIST_RANK = dist_rank
    DIST_WORLD_SIZE = dist_world_size

def get_dist():
    return DIST_GROUP, DIST_RANK, DIST_WORLD_SIZE

def all_gather_into_tensor(x: torch.Tensor, group: dist.ProcessGroup):
    group_size = dist.get_world_size(group)

    x = x.contiguous()
    output = torch.empty(group_size * x.size(0), *x.shape[1:], dtype=x.dtype, device=x.device)
    dist.all_gather_into_tensor(output, x, group=group)
    return output


def all_gather(tensor: torch.Tensor) -> torch.Tensor:
    if not DIST_GROUP:
        return tensor

    return all_gather_into_tensor(tensor, DIST_GROUP)

@torch.compiler.disable()
def _all_to_all_single(output, input, group):
    # Disable compilation since torch compile changes contiguity.
    assert input.is_contiguous(), "Input tensor must be contiguous."
    assert output.is_contiguous(), "Output tensor must be contiguous."
    return dist.all_to_all_single(output, input, group=group)

def collect_tokens(qkv: torch.Tensor, group: dist.ProcessGroup, num_heads: int):
    """Redistribute heads and receive tokens.

    Args:
        qkv: query, key or value. Shape: [3, b, ls, h, d]

    Returns:
        qkv: shape: [3, b, lh, s, d]

    where ls is the number of local tokens,
    s = cp_size * ls is the number of global tokens,
    lh = h // cp_size is the number of local heads.
    """
    assert num_heads % DIST_WORLD_SIZE == 0
    local_heads = num_heads // DIST_WORLD_SIZE

    qkv = rearrange(
        qkv,
        "qkv b ls (G lh) d -> G ls lh b (qkv d)",
        qkv=3,
        G=DIST_WORLD_SIZE,
        lh=local_heads,
    ).contiguous()

    output_chunks = torch.empty_like(qkv)
    _all_to_all_single(output_chunks, qkv, group=group)

    return rearrange(output_chunks, "G ls lh b (qkv d) -> qkv b lh (G ls) d", qkv=3)

# x       : [3, b, ls, h, d]
# returns : [3, b, lh, s, d]
#
# Give heads to all GPUs for our token chunk,
# and receive all tokens for our head chunk.
def all_to_all_collect_tokens(x: torch.Tensor) -> torch.Tensor:
    num_heads = x.size(-2)
    if not DIST_GROUP:
        # Move QKV dimension to the front.
        #   B M (3 H d) -> 3 B M H d
        _, b, s, a, d = x.size()
        x = x.view(b, s, 3, a, d)
        return x.permute(2, 0, 1, 3, 4)

    return collect_tokens(x, DIST_GROUP, num_heads)

def collect_heads(x: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    """Redistribute tokens and receive heads.

    Args:
        x: Output of attention. Shape: [b, lh, s, d]

    Returns:
        Shape: [b, ls, h * d]
    """
    local_heads = x.size(2)
    head_dim = x.size(3)
    group_size = dist.get_world_size(group)
    x = rearrange(x, "b lh (G ls) d -> G lh ls b d", G=group_size).contiguous()
    output = torch.empty_like(x)
    _all_to_all_single(output, x, group=group)
    del x
    return rearrange(output, "G lh ls b d -> b ls (G lh d)")

# x       : [b, lh, s, d]
# returns : [b, ls, h * d]
#
# Give tokens to all GPUs for our head chunk,
# and receive all heads for our token chunk.
def all_to_all_collect_heads(x: torch.Tensor) -> torch.Tensor:
    if not DIST_GROUP:
        # Merge heads.
        return x.view(x.size(0), x.size(1), x.size(2) * x.size(3))

    return collect_heads(x, DIST_GROUP)
