# Chipmunk Kernels Specification

Come and hack with Chipmunk kernels with us! Chipmunk kernels are written in ThunderKittens whenever possible for ease of use and building.

## 1. Attention Kernels (`csrc/attn/`)

### 1.1. Column-Sparse Attention

**Signature:** `csp_attn(q, k, v, o, indices, counts, o_scale) -> None`

This function is the meat of Chipmunk! It computes column-sparse attention given q, k, and v matrices, and it will accumulate the results such that `o += attn_output * o_scale` where `o_scale` is either `1` or `-1`.

Shapes:
- q/k/v/o: (B, H, N, D)
- indices: (B, H, ⌈N/192⌉, N)
- counts: (B, H, ⌈N/192⌉, N)
- o_scale: int64 scalar (1 or -1)

### 1.2. Dense Attention
**Signature:** `dense_attn(q, k, v) -> o, lse`

Computes standard non-causal 128-headdim attention but also outputs softmax LSE constants to be consumed by _dense_colsum_attn_ in a future inference step. 

Shapes:
- q/k/v/o: (B, H, N, D)
- lse: (B, H, N)

### 1.3. Dense Fused Column-Sum Attention

**Signature:** `dense_colsum_attn(q, k, v, lse_prev) -> o, col_sums, lse_cur`

Computes standard non-causal 128-headdim attention for q, k, and v, and writes output to o. 
We also compute _softmax(qk^T)_ using the _lse_prev_ constants supplied, and then sum the post-softmax intermediate matrix along column-chunks of length 192.

Shapes:
- q/k/v/o: (B, H, N, D)
- col_sums: (B, H, ⌈N / 192⌉, N)
- lse: (B, H, N)

## 2. MLP Kernels (`csrc/mlp/`)

To do!

### 2.1. Column-Sparse Matmul 1

### 2.1. Column-Sparse Matmul 2 + Scatter Add

## 3. Indexed IO Kernels (`csrc/indexed_io/`)

To do!

### 3.1. Copy Indices

### 3.2. Mask to Indices

### 3.3. Approximate Top-K Indices

### 3.4. Scatter-Add (Cache Writeback)

