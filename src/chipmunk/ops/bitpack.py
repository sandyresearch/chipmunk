import torch
from typing import Tuple

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