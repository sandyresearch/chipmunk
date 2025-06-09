## Hacking on Chipmunk

### Running WAN
First, follow the quickstart guide in the main README to get weights set up.

#### Commands
`cd examples/wan`

Run the script to generate a video:
`./run.sh`

which will run:

~~~
CONFIG_DIR=./

CUDA_VISIBLE_DEVICES=7 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python generate.py \
--task t2v-14B \
--size 832*480 \
--ckpt_dir ./Wan2.1-T2V-14B \
--prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." \
--base_seed 42 \
--offload_model True \
--chipmunk-config ${CONFIG_DIR}/chipmunk-config.yml \
--output-dir ${CONFIG_DIR}/media
~~~

#### System Requirements
Default WAN at 480p consumed ~66 GB of memory on a single H100. Chipmunk adds a bit extra peak memory
due to activation caching and large intermediate states for dynamic sparsity masks, reaching a peak of ~77 GB.
For further memory optimization, a good next step would be hacking around with weight offloading to free up some
additional memory. Chipmunk's `MaybeOffloadedTensor` can provide a good abstraction for this -- it maintains
an async pipeline of activation cache from CPU to GPU and could be extended to support weights.


### Under the Hood (the `src/chipmunk/modules/attn.py` file)

Chipmunk caches activations and computes sparse deltas against the cache.

#### Computing the sparsity `mask`

The sparsity pattern determines which tokens each query can attend to. Chipmunk can combine static and dynamic sparsity:

**`initialize_static_mask: (seq_shape: tuple(int), txt_len: int) -> bool[b, h, n / qg, n]`**
- Creates the base sparsity pattern that remains fixed across inference steps
- If the input sequence has been rearranged into 3D voxel chunks, `get_local_indices_with_text` computes a sliding local voxel mask around each query group (e.g., 4×6×8 voxels for `bm=192`)
- `lw1d` (local 1D window) computes a simpler 1D sliding window mask around each query group - each group of 192 queries attends to nearby tokens in sequence order
- `rk` (random keys) controls an optional amount of randomness to add to the static mask
- Text tokens are always included in the attention pattern

**`random_and_topk: (colsum: bf16[b, h, n / qg, n], topk: int) -> bool[b, h, n / qg, n]`**
- Adds dynamic sparsity based on attention column sums from the previous step
- Takes the top-k highest attention weights per query group
- Combines with random sampling
- Merges with the static mask using logical OR: `(topk_mask | random_mask | static_mask)`

**`mask_to_indices: (mask: bool[b, h, n / qg, n]) -> inds: int32[b, h, n / qg, n], counts: int32[b, h, n / qg]`**
- Converts boolean masks into packed integer indices for efficient kernel consumption
- `inds`: which key/value indices each query group should attend to
- `counts`: how many keys each query group attends to

#### Reading/writing the cache and sparsity mask: `AttnStorage` + `LayerCounter`

**Per-layer storage (`AttnStorage`):**
- `indices`: Packed sparsity indices (which k/v tokens to attend to)
- `counts`: Number of attended tokens per query group
- `out_cache`: Cached attention output from full computation steps (dense - sparse)
- `lse_constants`: Log-sum-exp constants for numerically stable attention

**Global singletons:**
- `singleton_static_mask`: The base sparsity pattern shared across all layers
- `singleton_video_query_groups`: Which query groups use sparse attention vs dense (for handling full cross attention to a subset of tokens)

**`LayerCounter` coordinates the sparse attention pipeline:**
- `cur_model_invocation_step`: Which model call within an inference step (for classifier free guidance)
- `cur_inference_step`: Current inference iteration (0=first, 1=recompute masks, 2+=sparse)
- `cur_layer`: Current transformer layer being processed
- `cur_layer_submodule`: Sub-component within layer (0=attn, 1=mlp)

**`MaybeOffloadedTensor` manages the async CPU-GPU pipeline for offloading:**
- Uses **pipeline depth of 2** to overlap GPU compute with CPU-GPU memory transfers
- Dedicated CUDA streams (`global_offload_stream`, `global_load_stream`) prevent blocking the main compute stream
- **Pinned CPU memory** for high-bandwidth async transfers
- Call `load_async()` before you need the tensor, then `load_async_wait()` to synchronize

#### The Kernels: `dense_attn`, `dense_colsum_attn`, `csp_attn`

Chipmunk uses three custom CUDA/Triton kernels for different phases:

**`dense_attn(q, k, v) -> (output, lse_constants)`**
- Standard dense attention for the first inference step
- Returns LSE constants needed for subsequent sparse steps
- Used for layers < `first_n_dense_layers` (always dense)

**`dense_colsum_attn(q, k, v, prev_lse) -> (output, column_sums, lse_constants)`**
- Dense attention + column sum computation for mask recomputation
- `column_sums`: Attention weights summed over queries (used for top-k selection)
- `prev_lse`: LSE constants from previous step to use for column sum
- Only runs on inference step 1 or when `recompute_mask=True`

**`csp_attn(q, k, v, indices, counts) -> sparse_output`**
- **Column Sparse** attention - the core sparse kernel
- Only computes attention for the sparse indices specified in `indices`
- `counts` tells the kernel how many k/v pairs each query group attends to
- Can run in additive mode (`sparse_output = cached_output + sparse_delta`)

#### The Sparse Attention Algorithm Flow

```
Inference Step 0: FULL DENSE
├── dense_attn(q, k, v) → output, lse_constants
└── Cache: output, lse_constants

Inference Step 1: RECOMPUTE MASKS  
├── dense_colsum_attn(q, k, v, lse_constants) → output, column_sums, new_lse
├── random_and_topk(column_sums, topk) → sparse_mask
├── mask_to_indices(sparse_mask) → indices, counts  
├── csp_attn(q, k, v, indices, counts) → sparse_output
└── Cache: (output - sparse_output), indices, counts

Inference Step 2+: SPARSE ONLY
├── Load cached: output_cache, indices, counts
├── csp_attn(q, k, v, indices, counts) → sparse_delta
└── Return: output_cache + sparse_delta

...

Inference Step 10: RECOMPUTE MASKS  
├── dense_colsum_attn(q, k, v, lse_constants) → output, column_sums, new_lse
├── random_and_topk(column_sums, topk) → sparse_mask
├── mask_to_indices(sparse_mask) → indices, counts  
├── csp_attn(q, k, v, indices, counts) → sparse_output
└── Cache: (output - sparse_output), indices, counts

...
```

### Hacking on the Algorithm

**To modify sparsity patterns:**
- Edit `initialize_static_mask()` to change the static attention pattern
- Adjust `local_voxels`, `local_1d_window`, `random_keys`, `top_keys` in config
- Modify `random_and_topk()` to change dynamic sparsity selection

**To add new cached tensors:**
- Add storage in `AttnStorage.__init__()`
- Implement getter/setter methods following the pattern
- Add to `MaybeOffloadedTensor` offloading config if needed

**Memory optimization:**
- Enable/disable offloading per tensor type in the global config
