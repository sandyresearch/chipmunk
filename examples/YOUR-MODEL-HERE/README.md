# How to Chipmunkify your model!

<p align="center"><img src="https://github.com/sandyresearch/chipmunk/blob/master/assets/images/kittens.png" width="25%"></p>

🐿️⚡️ Chipmunk is very easy to add to any open-source diffusion transformer (DiT) codebase! To see your end-to-end speedup, you only need to sparsify your attention layer! The rest of the steps are optional, but can help you get even more speedups, optimize memory usage, or maximize generation quality.

We've also put together a video version of this tutorial for the Mochi (start at 19:30).

<p align="center"><a href="https://www.youtube.com/watch?v=Rg9enIRSXmo"><img src="../../assets/images/yt-thumbnail.png" width="75%"></a></p>

## Step 1: Sparsify your Attention Layer

### Step 1.1: Load Chipmunk's Configuration

Copy over a chipmunk-config.yml file from one of our example models, depending on the type of task you're working on. For text2image tasks without clasifier-free guidance (CFG), Flux is a good starting point. For text2video tasks, Mochi is a good starting point if you have CFG or Hunyuan without CFG. This config will apply globally to all Chipmunk code in your model. Remember to do this in every process for multi-GPU inference (we recommend putting it at the top of your inference script).

```python
import chipmunk.util.config
chipmunk.util.config.load_from_file("/path/to/your/config.yml")
```

### Step 1.2: Sparsify your Attention Layer

Sparsifying attention can provide a significant speedup on models with large sequence lengths. In your `Attention` module, instantiate a `SparseDiffAttn` object like this:

```python
from chipmunk.modules import SparseDiffAttn
from chipmunk.util import LayerCounter

class FullSelfAttention(nn.Module):
    def __init__(self, ...):
        ...
        # If you also sparsify the MLP of the model, set is_mlp_sparse=True
        self.chipmunk_attention = SparseDiffAttn(LayerCounter.build_for_layer(is_attn_sparse=True, is_mlp_sparse=False))

    def forward(self, q, k, v):
        ## NEW CODE
        return self.chipmunk_attention(q, k, v)
        ## OLD CODE
        return F.scaled_dot_product_attention(q, k, v)
```

## Optional Steps

These steps are optional - for most DiT models, you only need to sparsify your attention layer to get a speedup while maintaining strong quality. However, there are a few other things you can do to further optimize the model or improve quality:

- Sparsify your MLP layer
- Reduce Chipmunk's GPU memory footprint
- Add a one-time reordering of tokens into patches (text2image tasks) or voxels (text2video tasks)

### Step 2: Sparsify your MLP Layer (Optional)

On most models with large sequence lengths, MLP takes a much smaller proportion of the total computation time than attention, so it's not as important to sparsify it. However, it can be very valuable for models with small sequence lengths.

```python
from chipmunk.modules import SparseDiffMLP
from chipmunk.util import LayerCounter

class MLP(nn.Module):
    def __init__(self, ...):
        ...
        # Note: If you're only sparsifying attention or only sparsifying MLP, you should set is_attn_sparse=True or is_mlp_sparse=True, respectively.
        layer_num, layer_counter = LayerCounter.build_for_layer(is_attn_sparse=True, is_mlp_sparse=True)
        # Note: self.activation MUST be a GELU module! This is what our kernels implement under the hood.
        self.chipmunk_mlp = SparseDiffMLP(layer_num, layer_counter, self.fc1, self.activation, self.fc2)

    def forward(self, x):
        ## NEW CODE
        return self.chipmunk_mlp(x)
        ## OLD CODE
        return self.fc2(self.activation(self.fc1(x)))
```

### Step 3: Reduce Chipmunk's GPU Memory Footprint (Optional)

This section is only relevant for large models with long sequence lengths that are running inference on a single GPU. If you are running inference on multiple GPUs or don't care about memory usage, you can skip this step. In your main DiT `forward` method, modify your forward loop to load the sparse attention and MLP data asynchronously and keep only a "sliding window" of the attention and MLP weights in GPU memory. Please see [the Flux code](https://github.com/sandyresearch/chipmunk/blob/master/examples/flux/src/flux/model.py#L124) for a complete example.

```python
from chipmunk.util import GLOBAL_CONFIG
from chipmunk.util.storage.offloaded_tensor import PIPELINE_DEPTH

for i, block in enumerate(self.blocks):
    # Add these 4 lines to the beginning of your forward loop
    # Note: If you're only sparsifying attention or only sparsifying MLP, you can only include one of them below!
    if not GLOBAL_CONFIG['offloading']['global_disable_offloading']:
        next_block = self.blocks[(i + PIPELINE_DEPTH - 1) % len(self.blocks)]
        for storage in [next_block.chipmunk_mlp.storage, next_block.chipmunk_attention.storage]: storage.load_async()
        for storage in [     block.chipmunk_mlp.storage,      block.chipmunk_attention.storage]: storage.load_async_wait()
    # Keep your old block logic unmodified - example below:
    x = block(x, vec=vec, pe=pe)

```

### Step 4: Token Reordering (Optional)

Token reordering is a method that improves the quality of model generations by ensuring that the chunks of contiguous tokens share a similar brightness/color in the final output generation. Since chunks of contiguous tokens share the same sparsity pattern, this means that tokens that look similar will share the same sparsity pattern (i.e., attend to the same keys/values for attention or activate the same neurons in W1/W2 for MLPs). Please see Section 3.2 and Alg. 1 of [our paper](https://arxiv.org/abs/2506.03275) for more details.

#### Text2Image (Patchifying)

Add this to your main DiT model's `forward` method. For an example, see [the Flux code](https://github.com/sandyresearch/chipmunk/blob/master/examples/flux/src/flux/sampling.py#L265).

```python
from chipmunk.util import GLOBAL_CONFIG
from chipmunk.ops import patchify, unpatchify, patchify_rope
from einops import rearrange

def forward(self, img, txt, img_ids, txt_ids, vec, height, width, guidance, step_fn=None):
    # ... same as before, initialize the vector, timesteps, embeddings, etc.
    latent_width, latent_height = height // 2, width // 2
    if GLOBAL_CONFIG['patchify']['is_enabled']:
        img = rearrange(img, "b (h w) c -> (b c) h w", h=latent_height, w=latent_width)
        img = patchify(img)
        img = rearrange(img, "(b c) x -> b x c", b=1)
        if not hasattr(self, 'rope_patchified'):
            ids = torch.cat((txt_ids, img_ids), dim=1)
            self.rope_patchified = patchify_rope(img.shape, self.pe_embedder(ids), latent_width, latent_height)

    # ... same as before, our main DiT loop that calls each of self.blocks ...

    if GLOBAL_CONFIG['patchify']['is_enabled']:
        img = rearrange(img, "b np c -> (b c) np")
        img = unpatchify(img, (1, latent_height, latent_width))
        img = rearrange(img, "(b c) h w -> b (h w) c", b=1)
```

#### Text2Video (Voxelizing)

Add this to your main DiT model's `forward` method. For an example, see [the Mochi code](https://github.com/sandyresearch/chipmunk/blob/master/examples/mochi/src/genmo/mochi_preview/dit/joint_model/asymm_models_joint.py#L709).

```python
from chipmunk.util import GLOBAL_CONFIG
import chipmunk

def forward(self, x, sigma, y_feat, y_mask, rope_cos, rope_sin, num_ff_checkpoint, num_qkv_checkpoint, num_post_attn_checkpoint):
    # ... same as before, initialize the vector, timesteps, embeddings, etc.

    if GLOBAL_CONFIG['patchify']['is_enabled']:
        voxel_shape = (4, 6, 8) # 192

        x = rearrange(x.unsqueeze(1), 'b nh (t h w) d -> b nh t h w d', t=T, h=H // self.patch_size, w=W // self.patch_size)
        x = chipmunk.ops.voxel.voxel_chunk_no_padding(x, voxel_shape=voxel_shape).squeeze(1)

        rope_cos = rearrange(rope_cos.unsqueeze(0), 'b (t h w) nh c -> b nh t h w c', t=T, h=H // self.patch_size, w=W // self.patch_size)
        rope_sin = rearrange(rope_sin.unsqueeze(0), 'b (t h w) nh c -> b nh t h w c', t=T, h=H // self.patch_size, w=W // self.patch_size)
        rope_cos = chipmunk.ops.voxel.voxel_chunk_no_padding(rope_cos, voxel_shape=voxel_shape)
        rope_sin = chipmunk.ops.voxel.voxel_chunk_no_padding(rope_sin, voxel_shape=voxel_shape)
        rope_cos = rearrange(rope_cos.squeeze(0), 'nh n c -> n nh c')
        rope_sin = rearrange(rope_sin.squeeze(0), 'nh n c -> n nh c')

    # ... same as before, our main DiT loop that calls each of self.blocks ...

    if GLOBAL_CONFIG['patchify']['is_enabled']:
        og_shape = (1, 1, T, H // self.patch_size, W // self.patch_size, self.patch_size ** 2 * self.out_channels)
        x = chipmunk.ops.voxel.reverse_voxel_chunk_no_padding(x.unsqueeze(1), og_shape, voxel_shape=voxel_shape).squeeze(1)
        x = rearrange(x, "b t h w d -> b (t h w) d")

```
