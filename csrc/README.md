# **Chipmunk CUDA Kernels**

<p align="center"><img src="../assets/images/kittens-2.png" width="60%" /></p>
<center><i>We're big fans of ThunderKittens! Chipmunk kernels are written in ThunderKittens wherever possible.</i></center>

Chipmunk ships all custom kernels as PyTorch extensions available under the `chipmunk.cuda` package. All tensors are expected to be **CUDA‑resident**, contiguous (except attention QKV inputs, which can be strided except for the embedding dimension). Shapes below omit batch B when the kernel is designed for B = 1; otherwise the full shape is given.

### Update (6/15/2025)

⚡️ Exciting news! Our attention kernels now support completely unpadded and arbitrarily strided inputs for Q, K, and V (i.e., BHND no longer need to be contiguous - only D does). This can save 5-10% of the E2E video generation time which was previously spent on making tensors contiguous and adding padding to them! _Note: If you play around with the indices and LSE tensors, you'll notice that they are always padded to a multiple of `N%4 == 0`. This is to align the TMA stride to a multiple of 16 bytes. Since this is never exposed to user-facing APIs, you don't need to worry about this unlness you're hacking on Chipmunk internals!_

### **Kernel Summary Table**

| Category   | Kernel                      | Purpose                                                        |
| ---------- | --------------------------- | -------------------------------------------------------------- |
| Attention  | csp_attn                    | Column‑sparse 192‑chunk attention                              |
|            | dense_attn                  | Dense attention \+ log-sum-exp constants                       |
|            | dense_colsum_attn           | Dense attention \+ 192‑col sums                                |
| MLP        | csp_mlp_mm1                 | FC1: Column-sparse A × Bᵀ with bias/GeLU/post-activation cache |
|            | csp_mlp_mm2_and_scatter_add | Cache writeback \+ FC2: second matmul                          |
| Indexed IO | copy_indices                | Fast gather‑scatter between buffers                            |
|            | mask_to_indices             | Boolean mask → (indices,counts)                                |
|            | topk_indices                | Approx. top‑k / quantile sparsity                              |
|            | csp_scatter_add             | Column‑major post-activation cache writeback                   |

---

## 1 Attention Kernels (`csrc/attn/`)

### 1.1 Column‑Sparse Attention

```
csp_attn(q, k, v,               // (B, H, N, D)
         o,                      // (B, H, N, D)  (accumulates)
         indices, counts,        // (B, H, ⌈N/192⌉, N)  /  (B, H, ⌈N/192⌉)
         int64_t o_scale)        // 1 or ‑1
```

- **The meat of Chipmunk!** Implements “192‑column” dynamic sparsity.

- Computes o \+= Softmax(Q Kᵀ) ⋅ V × o_scale with column-sparse matrix multiplications

- indices holds the **physical column IDs** that participate in each 192‑wide chunk; counts gives the number of valid columns per chunk.

### **1.2 Dense Attention**

### **dense_attn**

```
o, lse = dense_attn(q, k, v)     // Q/K/V/O: (B, H, N, D), lse: (B, H, N)
```

- Standard **non‑causal** 128‑head‑dim attention written in ThunderKittens.

- Returns “log‑sum‑exp” vectors (lse) so that a later kernel can fuse column‑sums without re‑computing the soft‑max denominator.

### 1.3 Dense + Column‑Sum Fusion

```
o, col_sums, lse_cur = dense_colsum_attn(q, k, v, lse_prev)
   // q/k/v/o:  (B,H,N,D)
   // col_sums: (B,H,⌈N/192⌉,N)
   // lse_prev   : (B,H,N)
```

- Reuses lse_prev to keep numerical ranges aligned with earlier blocks.

- Produces **192‑wide column sums** used by the column‑sparse backward pass or PA‑cache logic in inference.

---

## 2 MLP Kernels (`csrc/mlp/`)

Chipmunk’s MLP is a **two‑stage, column‑sparse feed‑forward** built around 64×64 ThunderKittens matmuls.

### **2.1 Column‑Sparse Matmul 1**

### **Stage 1**

```
csp_mlp_mm1(a,                  // (M, K)  row‑major activations
            b_colmajor,         // (N, K)  weight matrix column‑major
            c,                  // (M, N)  output
            bias,               // (N)
            pa_cache_colmajor,  // (N, M)  previous‑activation cache (col‑major)
            indices,            // (⌈M/128⌉, N)  per‑128‑row dynamic column map
            indices_counts)     // (⌈M/128⌉)
```

- Each 128‑row **producer warp‑group** gathers the subset of N columns it actually needs (via indices) before the compute warp‑groups run mixed‑precision MMA.

- Fused **bias add → GeLU → PA‑cache subtraction** (all optional at compile time).

- Handles arbitrary M×K×N, but achieves peak throughput when M and N are multiples of 64 × M_BLOCK and 64 × N_BLOCK, respectively (defaults: M_BLOCK = 2, N_BLOCK = 4).

### 2.2 Column‑Sparse Matmul + Scatter Add

```
csp_mlp_mm2_and_scatter_add(
        packed, unpacked_colmajor,          // (M, MLP_BM, F)
        sp_inds, sp_counts,                 // (M, F)  /  (M)
        mma_a, mma_b, mma_c,                // stage‑2 GEMM operands
        int   num_ctas_scatter_add,         // CTAs to dedicate to scatter
        void* matmul_kernel)                // opaque CUfunction handle
```

1. **Scatter‑Add**

   - Kernel scatter_add_kernel converts **packed** row‑major tiles (R×F) into an **unpacked column‑major** layout using sp_inds.

   - Each thread‑block handles one **M‑row**; dynamic shared memory (up to 231 KB) tilts bandwidth in favour of global‑to‑shared cp.async.

2. **Matmul‑2**

   - Runs concurrently on a second stream; points to a Triton matmul kernel (see `src/chipmunk/triton/bf16/csp_mlp_mm2.py`)

3. Both pieces are stitched together in a single **CUDA Graph**, so launch overhead is amortised.

---

## 3 Indexed‑I/O Kernels (`csrc/indexed_io/`)

These helper kernels move sparse data between dense staging buffers and kernel‑friendly layouts.

### **3.1 Copy Indices**

```
copy_indices(bmfc1, bm_mid_cache,  // (B, MR, F)  in/out
             sp_inds, sp_counts)   // (B, M, F) / (B, M)
```

- Fast scatter‑copy: for each row r the first sp_counts\[b,m\] columns listed in sp_inds\[b,m,:\] are copied from bmfc1 → bm_mid_cache.

- One thread‑block per (B × M × R) row, 256 threads; supports **bf16 / fp16 / fp32**.

### **3.2 Mask → Indices**

```
indices, counts = mask_to_indices(mask, multiple_of, pad_to_multiple_of=192)
    // mask   : (B, H, M, N)  boolean
    // indices: (B, H, M, pad_N)
    // counts : (B, H, M)
```

- One warp converts each boolean row into compact indices.

- Ensures counts\[b,h,m\] is **rounded up** to multiple_of by padding with any unused column IDs so later kernels can assume 32/64/192‑wide tiles.

### **3.3 Approximate Top‑K Indices**

```
topk_indices(activation, indices, counts,
             sparsity_amount,     // 0 → keep all, 1 → keep none
             multiple_of,         // pad so counts % multiple_of == 0
             random_amount=0.0)   // stochastic “extra” keep‑probability
```

- Uses **Block‑Merge‑Sort** algorithm on a 1024‑thread block to estimate the value at the given quantile (sparsity_amount) and keep everything above it.

- Pads the result so counts is a multiple of e.g. 64 or 192\.

- Supports optional random retention (random_amount) for _exploration_ style sparsity schemes.

### **3.4 Scatter‑Add (Cache Writeback)**

### **csp_scatter_add**

```
csp_scatter_add(packed,               // (B, MMLP_BM, F)  row‑major
                unpacked_colmajor,    // (B, F, MMLP_BM)  col‑major output
                sp_inds, sp_counts,   // (B, M, F) / (B, M)
                int num_sms)          // CTAs = \#SMS used
```

- Scatters and adds each packed tile back to a **column‑major post-activation cache** based on the sparsity indices

- Uses TMA `cp.async.reduce.bulk` to accumulate atomics into global memory efficiently (3x faster than naive register-based accumulators)

---
