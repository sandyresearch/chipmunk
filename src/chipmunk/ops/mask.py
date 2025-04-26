import torch
from typing import Optional

def compute_coverage_mask(
    cs: torch.Tensor,
    coverage: float,
    mask: Optional[torch.Tensor],
    rows_per_chunk: int = 1024,
    *,
    device_for_sort: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Parameters
    ----------
    cs
        Attention scores.  Shape  (B, H, Q, K)  **or**  (B, H, K)
        (we only assume the last dimension are the keys).
    coverage
        Fraction in (0, 1].  Minimal mass we want to collect.
    rows_per_chunk
        How many (B*H*Q) rows are processed in a single mini-batch.
        Tune this until it fits your GPU memory.
    device_for_sort
        Move every chunk to this device for sorting.  Useful when
        the model runs on GPU but sorting can be done on CPU RAM.
    """
    orig_shape   = cs.shape
    num_keys     = cs.shape[-1]
    flat_cs      = cs.view(-1, num_keys)           # (R, K)  where  R = B*H*Q
    R            = flat_cs.size(0)

    mask_flat    = mask.view(-1, num_keys)
    needed       = (flat_cs.sum(dim=-1, dtype=flat_cs.dtype) * coverage)   # (R,)

    # Optional: do the heavy work on a different device (CPU)
    sort_dev = device_for_sort or cs.device

    for start in range(0, R, rows_per_chunk):
        end          = min(start + rows_per_chunk, R)
        rows         = flat_cs[start:end].to(sort_dev)

        # 1) full sort *per row* (rows_per_chunk × K) only
        values, idx  = rows.sort(dim=-1, descending=True)    # (r, K)

        # 2) cumulative mass and Boolean prefix that reaches coverage
        cumsum       = values.cumsum(dim=-1, dtype=values.dtype)
        keep         = cumsum < needed[start:end].unsqueeze(-1)  # (r, K)
        keep         |= keep.roll(1, dims=-1)                   # include boundary
        keep[:, 0]   = True                                     # always keep first

        # 3) map back to original key positions
        selected_idx = torch.where(keep, idx, torch.tensor(0, device=sort_dev))
        chunk_mask   = torch.zeros_like(rows, dtype=torch.bool)
        chunk_mask.scatter_(-1, selected_idx, True)             # (r, K)

        # 4) write result back (move to original device if necessary)
        mask_flat[start:end] |= chunk_mask.to(mask_flat.device)

    return mask_flat.view(orig_shape)
